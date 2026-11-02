"""Raw (recorder-level) episode format.

One directory per episode::

    <raw_root>/<robot_id>/<session_id>/<episode_id>/
        meta.json            EpisodeMeta (robot, task, operator, sensor config, calibration, labels...)
        streams.zarr/        Zarr v3 group, one sub-group per stream:
            <stream>/t_device  int64 ns  sensor/driver clock (ROS header.stamp)
            <stream>/t_host    int64 ns  recorder receive time (ROS node clock)
            <stream>/seq       int64     per-stream message counter (gaps => drops at source)
            <stream>/t_true    int64 ns  ground-truth capture time (sim / virtual robot only)
            <stream>/data      low-dim (N, D) float32 | depth (N, H, W) uint16 [mm]
        video/<stream>.mp4   RGB streams; frame i <-> row i of <stream>/t_*

Streams are kept at their native rates and *unsynchronised*: alignment happens later
(robodata.sync) so that sync quality can be measured and re-done without re-recording.
"""

from __future__ import annotations

import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import zarr
from pydantic import BaseModel, Field

from robodata.config import RobotConfig, StreamConfig
from robodata.io.video import VideoWriter

RAW_FORMAT_VERSION = "robodata-raw/1.0"


class EpisodeMeta(BaseModel):
    episode_id: str
    robot_id: str
    robot_type: str
    session_id: str
    scene_id: str = "unknown"
    task: str
    operator: str = "unknown"
    is_sim: bool = False
    source: str = "live"  # live | hf:<repo>@<rev>#ep<idx> | sim:<generator>
    success: bool | None = None
    sensor_config_hash: str = ""
    calibration_version: str = "uncalibrated"
    sync_mode: str = "free_run"
    has_ground_truth_clock: bool = False
    recorded_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    format_version: str = RAW_FORMAT_VERSION
    streams: dict[str, dict] = Field(default_factory=dict)  # stream name -> StreamConfig dump + counts
    injected_faults: list[dict] = Field(default_factory=list)  # set by fault injector (test ground truth)
    extra: dict = Field(default_factory=dict)


class _StreamBuf:
    def __init__(self, name: str, cfg: StreamConfig, group: zarr.Group, video_dir: Path, video_cfg: dict):
        self.name, self.cfg, self.lock = name, cfg, threading.Lock()
        self.t_device: list[int] = []
        self.t_host: list[int] = []
        self.t_true: list[int] = []
        self.seq: list[int] = []
        self.lowdim: list[np.ndarray] = []
        self.depth_pending: list[np.ndarray] = []
        self.group = group.require_group(name)
        self.video: VideoWriter | None = None
        self.depth_arr = None
        # images are encoded/compressed on one ordered worker per stream so callers (ROS callbacks) return fast
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"rd-{name}") if cfg.kind in ("rgb", "depth") else None
        if cfg.kind == "rgb":
            h, w = cfg.shape[:2]
            self.video = VideoWriter(video_dir / f"{name}.mp4", cfg.rate_hz, w, h, codec=video_cfg.get("codec", "libx264"),
                                     crf=video_cfg.get("crf", 18), preset=video_cfg.get("preset", "veryfast"),
                                     gop=video_cfg.get("gop", 30))
        elif cfg.kind == "depth":
            h, w = cfg.shape[:2]
            self.depth_arr = self.group.create_array("data", shape=(0, h, w), chunks=(8, h, w), dtype="uint16")

    def push(self, t_device: int, t_host: int, data, t_true: int | None, seq: int | None):
        with self.lock:
            self.t_device.append(int(t_device))
            self.t_host.append(int(t_host))
            self.seq.append(len(self.seq) if seq is None else int(seq))
            if t_true is not None:
                self.t_true.append(int(t_true))
            if self.video is not None:
                self.worker.submit(self.video.write, np.array(data, copy=True))
            elif self.depth_arr is not None:
                self.depth_pending.append(np.array(data, dtype=np.uint16).reshape(self.cfg.shape[:2]))
                if len(self.depth_pending) >= 8:
                    batch, self.depth_pending = self.depth_pending, []
                    self.worker.submit(self._append_depth, batch)
            else:
                self.lowdim.append(np.asarray(data, dtype=np.float32).reshape(-1))

    def _append_depth(self, batch: list[np.ndarray]):
        self.depth_arr.append(np.stack(batch), axis=0)

    def close(self) -> int:
        with self.lock:
            if self.depth_arr is not None and self.depth_pending:
                self.worker.submit(self._append_depth, self.depth_pending)
                self.depth_pending = []
            if self.worker is not None:
                self.worker.shutdown(wait=True)  # drains queued frames in order
            if self.video is not None:
                self.video.close()
            g = self.group
            g.create_array("t_device", data=np.asarray(self.t_device, dtype=np.int64))
            g.create_array("t_host", data=np.asarray(self.t_host, dtype=np.int64))
            g.create_array("seq", data=np.asarray(self.seq, dtype=np.int64))
            if self.t_true:
                g.create_array("t_true", data=np.asarray(self.t_true, dtype=np.int64))
            if self.cfg.kind not in ("rgb", "depth"):
                dim = int(np.prod(self.cfg.shape))
                arr = np.stack(self.lowdim) if self.lowdim else np.zeros((0, dim), np.float32)
                g.create_array("data", data=arr.astype(np.float32), chunks=(max(1, min(len(arr), 4096)), dim))
            return len(self.t_device)


class RawEpisodeWriter:
    """Thread-safe writer used by the ROS 2 recorder node, the virtual robot and the sim converter."""

    def __init__(self, raw_root: str | Path, robot: RobotConfig, meta: EpisodeMeta, streams: list[str] | None = None):
        self.robot = robot
        self.meta = meta
        self.dir = Path(raw_root) / meta.robot_id / meta.session_id / meta.episode_id
        if self.dir.exists():
            shutil.rmtree(self.dir)
        self.tmp = self.dir.with_name(self.dir.name + ".partial")
        if self.tmp.exists():
            shutil.rmtree(self.tmp)
        (self.tmp / "video").mkdir(parents=True)
        self.root = zarr.open_group(str(self.tmp / "streams.zarr"), mode="w")
        names = streams if streams is not None else list(robot.streams)
        self.bufs = {n: _StreamBuf(n, robot.streams[n], self.root, self.tmp / "video", robot.raw_video) for n in names}

    def push(self, stream: str, t_device_ns: int, t_host_ns: int, data, t_true_ns: int | None = None,
             seq: int | None = None) -> None:
        self.bufs[stream].push(t_device_ns, t_host_ns, data, t_true_ns, seq)

    def close(self, success: bool | None = None) -> Path:
        counts = {n: b.close() for n, b in self.bufs.items()}
        if success is not None:
            self.meta.success = success
        self.meta.sensor_config_hash = self.meta.sensor_config_hash or self.robot.sensor_config_hash()
        self.meta.sync_mode = self.robot.sync.mode
        self.meta.streams = {
            n: {**self.robot.streams[n].model_dump(exclude={"clock"}), "count": counts[n]} for n in self.bufs
        }
        (self.tmp / "meta.json").write_text(self.meta.model_dump_json(indent=2))
        self.tmp.rename(self.dir)  # atomic publish: a crashed recording never looks complete
        return self.dir

    def abort(self) -> None:
        for b in self.bufs.values():
            try:
                if b.worker is not None:
                    b.worker.shutdown(wait=True)
                if b.video is not None:
                    b.video.close()
            except Exception:
                pass
        shutil.rmtree(self.tmp, ignore_errors=True)


class RawEpisode:
    """Read-only accessor for a raw episode directory."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.meta = EpisodeMeta(**json.loads((self.path / "meta.json").read_text()))
        self.root = zarr.open_group(str(self.path / "streams.zarr"), mode="r")

    @property
    def id(self) -> str:
        return self.meta.episode_id

    @property
    def streams(self) -> list[str]:
        return list(self.root.group_keys())

    def kind(self, stream: str) -> str:
        return self.meta.streams[stream]["kind"]

    def times(self, stream: str, which: str = "t_device") -> np.ndarray:
        g = self.root[stream]
        return np.asarray(g[which][:]) if which in g else None

    def data(self, stream: str) -> np.ndarray:
        return np.asarray(self.root[stream]["data"][:])

    def video_path(self, stream: str) -> Path:
        return self.path / "video" / f"{stream}.mp4"

    def has(self, stream: str, key: str) -> bool:
        return stream in self.root and key in self.root[stream]

    def sidecar(self, name: str) -> dict | None:
        p = self.path / name
        return json.loads(p.read_text()) if p.exists() else None

    def write_sidecar(self, name: str, payload: dict) -> None:
        (self.path / name).write_text(json.dumps(payload, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def list_episodes(raw_root: str | Path, robot_id: str | None = None) -> list[Path]:
    root = Path(raw_root)
    pattern = f"{robot_id}/*/*/meta.json" if robot_id else "*/*/*/meta.json"
    return sorted(p.parent for p in root.glob(pattern))
