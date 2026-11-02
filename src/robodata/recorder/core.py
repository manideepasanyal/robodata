"""ROS-agnostic episode recorder.

The ROS 2 node (ros2_ws/src/robodata_ros) is a thin adapter: it converts messages to
numpy and calls :meth:`EpisodeRecorder.on_message`. The same class is driven
in-process by the virtual robot (open-source replay) and the sim converter, so the
code path that writes raw data is identical for live, replayed and simulated data.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from robodata.config import RobotConfig
from robodata.io.raw_episode import EpisodeMeta, RawEpisodeWriter
from robodata.schema import episode_id as make_episode_id


class EpisodeRecorder:
    def __init__(self, robot: RobotConfig, raw_root: str | Path, session_id: str | None = None,
                 operator: str = "unknown", scene_id: str = "unknown", is_sim: bool = False):
        self.robot = robot
        self.raw_root = Path(raw_root)
        self.session_id = session_id or datetime.now(timezone.utc).strftime("session_%Y%m%dT%H%M%S")
        self.operator, self.scene_id, self.is_sim = operator, scene_id, is_sim
        self._writer: RawEpisodeWriter | None = None
        self._lock = threading.Lock()
        self._counter = 0
        self.dropped_while_idle = 0

    @property
    def recording(self) -> bool:
        return self._writer is not None

    def start(self, task: str, episode_index: int | None = None, source: str = "live", **meta_kw) -> str:
        with self._lock:
            if self._writer is not None:
                raise RuntimeError("already recording")
            idx = self._counter if episode_index is None else episode_index
            self._counter = idx + 1
            fields = dict(operator=self.operator, scene_id=self.scene_id, is_sim=self.is_sim)
            fields.update(meta_kw)
            meta = EpisodeMeta(
                episode_id=make_episode_id(self.robot.robot_id, self.session_id, idx),
                robot_id=self.robot.robot_id, robot_type=self.robot.robot_type, session_id=self.session_id,
                task=task, source=source, calibration_version=self.robot.calibration_version, **fields)
            self._writer = RawEpisodeWriter(self.raw_root, self.robot, meta)
            return meta.episode_id

    def on_message(self, stream: str, t_device_ns: int, data, t_host_ns: int | None = None,
                   t_true_ns: int | None = None, seq: int | None = None) -> None:
        w = self._writer
        if w is None:
            self.dropped_while_idle += 1
            return
        if t_host_ns is None:
            t_host_ns = time.time_ns()
        w.push(stream, t_device_ns, t_host_ns, data, t_true_ns, seq)

    def stop(self, success: bool | None = None) -> Path:
        with self._lock:
            if self._writer is None:
                raise RuntimeError("not recording")
            w, self._writer = self._writer, None
        return w.close(success)

    def discard(self) -> None:
        with self._lock:
            w, self._writer = self._writer, None
        if w is not None:
            w.abort()
