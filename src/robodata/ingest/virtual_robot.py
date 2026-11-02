"""Virtual robot: replays open-source episodes through the recorder as if they were live sensors.

Open-source datasets are already resampled to one clock, which hides every real-world
timing problem. The virtual robot re-introduces them per stream (clock offset, drift,
jitter, transport latency, dropped messages - see ``configs/robots/*.yaml: clock``) and
records through the exact same :class:`EpisodeRecorder` code path the ROS 2 node uses.
The original source timing is kept as ``t_true`` so sync accuracy can be measured.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, Field

from robodata.config import CONFIG_DIR, RobotConfig, load_robot, resolve, source_path
from robodata.ingest.lerobot_source import LeRobotSource, SourceEpisode
from robodata.recorder.clock import NS, stamp
from robodata.recorder.core import EpisodeRecorder

SESSION_EPOCH_NS = 1_767_225_600 * NS  # 2026-01-01T00:00:00Z, deterministic replay epoch


class StreamSource(BaseModel):
    key: str
    decode: dict | None = None  # depth decoding for 8-bit visualised depth videos
    index: list[int] | None = None  # select dims of a vector column


class SourceSpec(BaseModel):
    name: str
    repo_id: str
    revision: str
    license: str
    robot: str
    local_dir: str
    episodes: list[int] | str = "all"
    session_size: int = 10
    operator: str = "unknown"
    scene_id: str = "unknown"
    success: bool | None = None
    streams: dict[str, StreamSource]
    notes: str = ""
    extra: dict = Field(default_factory=dict)


def load_source(name: str) -> SourceSpec:
    with open(resolve(name, "sources")) as f:
        return SourceSpec(**yaml.safe_load(f))


def _seed(*parts) -> int:
    return int(hashlib.sha256("/".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def decode_depth(rgb: np.ndarray, spec: dict) -> np.ndarray:
    """8-bit visualised depth -> uint16 millimetres (0 = invalid)."""
    mode = spec.get("mode", "gray_linear")
    if mode != "gray_linear":
        raise ValueError(f"unknown depth decode mode {mode}")
    v = rgb[..., 0].astype(np.float32)
    mm = v / 255.0 * spec["max_m"] * 1000.0
    mm[v <= spec.get("invalid_below", 0)] = 0
    return mm.astype(np.uint16)


def replay_episode(src: SourceSpec, robot: RobotConfig, ep: SourceEpisode, ordinal: int, raw_root: Path,
                   apply_clock: bool = True, session: str | None = None, salt: str = "",
                   extra: dict | None = None) -> Path:
    """``session``/``salt`` let the same source episode be recorded again (a re-upload) with new clock noise."""
    session = session or f"{src.name}_s{ordinal // src.session_size:03d}"
    rec = EpisodeRecorder(robot, raw_root, session_id=session, operator=src.operator, scene_id=src.scene_id)
    rec.start(task=ep.task, episode_index=ordinal,
              source=f"hf:{src.repo_id}@{src.revision[:12]}#ep{ep.index}",
              success=src.success, has_ground_truth_clock=True,
              extra={"source_episode_index": ep.index, "source_fps": ep.fps, "source_license": src.license,
                     **(extra or {})})
    epoch = SESSION_EPOCH_NS + ordinal * 600 * NS
    n = len(ep)
    t_true = (np.arange(n) * (NS / ep.fps)).astype(np.int64)
    try:
        for stream, ss in src.streams.items():
            cfg = robot.streams[stream]
            # salt only for deliberate re-recordings; a plain ingest keeps the original seed (reproducible)
            rng = np.random.default_rng(_seed(src.name, ep.index, stream, *([salt] if salt else [])))
            clock = cfg.clock if apply_clock else type(cfg.clock)()
            t_dev, t_host, keep = stamp(t_true, clock, rng, epoch_ns=epoch)
            if cfg.kind in ("rgb", "depth"):
                values = ep.iter_video(ss.key)
            else:
                col = ep.column(ss.key)
                values = iter(col[:, ss.index] if ss.index else col)
            for i, v in enumerate(values):
                if i >= n:
                    break
                if not keep[i]:
                    continue
                if cfg.kind == "depth":
                    v = decode_depth(v, ss.decode or {})
                rec.on_message(stream, int(t_dev[i]), v, t_host_ns=int(t_host[i]), t_true_ns=int(t_true[i] + epoch), seq=i)
    except Exception:
        rec.discard()
        raise
    return rec.stop()


def _worker(src_name: str, ordinal: int, episode_index: int, raw_root: str, apply_clock: bool) -> str:
    src = load_source(src_name)
    robot = load_robot(src.robot)
    lr = LeRobotSource(source_path(src.local_dir))
    return str(replay_episode(src, robot, lr.by_index(episode_index), ordinal, Path(raw_root), apply_clock))


def ingest(src_name: str, raw_root: str | Path, max_episodes: int | None = None, workers: int = 4,
           apply_clock: bool = True) -> list[str]:
    src = load_source(src_name)
    lr = LeRobotSource(source_path(src.local_dir))
    idxs = lr.episode_indices if src.episodes == "all" else [i for i in src.episodes if i in set(lr.episode_indices)]
    if max_episodes:
        idxs = idxs[:max_episodes]
    out = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_worker, src_name, o, i, str(raw_root), apply_clock): i for o, i in enumerate(idxs)}
        for f in as_completed(futs):
            out.append(f.result())
            print(f"[ingest:{src_name}] {len(out)}/{len(idxs)} {Path(out[-1]).name}", flush=True)
    return sorted(out)


__all__ = ["ingest", "load_source", "replay_episode", "decode_depth", "CONFIG_DIR"]
