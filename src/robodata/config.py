"""Typed configuration models (robot, health thresholds, curation, dataset recipes).

Every stage of the pipeline is driven by YAML files under ``configs/``. The same
recorder / health-check / build code runs on any robot: only the robot YAML changes.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
DATA_ROOT = Path(os.environ.get("ROBODATA_DATA", REPO_ROOT / "data"))


def source_path(local_dir: str | Path) -> Path:
    """Source configs say `data/sources/<x>`; resolve it under DATA_ROOT so Docker's /data volume keeps it."""
    p = Path(local_dir)
    if p.is_absolute():
        return p
    return DATA_ROOT / p.relative_to("data") if p.parts and p.parts[0] == "data" else REPO_ROOT / p

StreamKind = Literal["rgb", "depth", "joint_state", "pose", "wrench", "pose_cmd", "joint_cmd", "twist_cmd", "scalar"]
InterpKind = Literal["linear", "pose", "hold", "lowpass_linear", "nearest"]


class ClockModel(BaseModel):
    """Imperfections of a sensor clock + transport, used by the virtual robot / sim converter.

    t_device = t_true * (1 + drift_ppm * 1e-6) + offset_ms + N(0, jitter_ms)
    t_host   = t_true + latency_ms + Exp(latency_jitter_ms)
    """

    offset_ms: float = 0.0
    drift_ppm: float = 0.0
    jitter_ms: float = 0.0
    latency_ms: float = 0.0
    latency_jitter_ms: float = 0.0
    drop_prob: float = 0.0


class StreamConfig(BaseModel):
    kind: StreamKind
    topic: str | None = None  # ROS 2 topic (recorder node)
    msg_type: str | None = None  # e.g. sensor_msgs/msg/Image
    rate_hz: float
    shape: list[int]
    dtype: str = "float32"
    names: list[str] | None = None
    required: bool = True
    frame_id: str | None = None
    depth_unit: Literal["mm", "m"] | None = None  # depth streams are stored as uint16 mm in raw
    known_latency_ms: float = 0.0  # calibrated constant latency (exposure / driver), removed during sync
    value_range: list[list[float]] | None = None  # per-dim [lo, hi] for plausibility checks
    clock: ClockModel = Field(default_factory=ClockModel)  # only used when replaying/simulating


class FeatureMap(BaseModel):
    """Maps a raw stream onto a training feature (LeRobot key)."""

    stream: str
    interp: InterpKind = "linear"
    names: list[str] | None = None
    angle_dims: list[int] | None = None  # dims holding angles (unwrapped before interpolation)


class SyncConfig(BaseModel):
    mode: Literal["hardware_trigger", "software_trigger", "free_run"] = "free_run"
    reference_stream: str
    clock_fit: Literal["lower_envelope", "least_squares", "none"] = "lower_envelope"
    max_frame_tolerance_ms: float = 20.0


class RobotConfig(BaseModel):
    robot_id: str
    robot_type: str
    description: str = ""
    target_fps: int
    sync: SyncConfig
    streams: dict[str, StreamConfig]
    features: dict[str, FeatureMap]
    calibration_version: str = "uncalibrated"
    raw_video: dict = Field(default_factory=lambda: {"codec": "libx264", "crf": 18, "preset": "veryfast", "gop": 30})
    health_overrides: dict = Field(default_factory=dict)  # per-robot threshold overrides (e.g. very short episodes)

    @model_validator(mode="after")
    def _check(self):
        if self.sync.reference_stream not in self.streams:
            raise ValueError(f"reference_stream {self.sync.reference_stream!r} not in streams")
        for key, fm in self.features.items():
            if fm.stream not in self.streams:
                raise ValueError(f"feature {key!r} references unknown stream {fm.stream!r}")
        return self

    @property
    def camera_streams(self) -> list[str]:
        return [k for k, s in self.streams.items() if s.kind in ("rgb", "depth")]

    @property
    def required_streams(self) -> list[str]:
        return [k for k, s in self.streams.items() if s.required]

    def sensor_config_hash(self) -> str:
        """Hash of everything that defines the sensor setup (not the virtual clock)."""
        payload = {
            k: s.model_dump(exclude={"clock"}) for k, s in sorted(self.streams.items())
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


class HealthThresholds(BaseModel):
    max_drop_ratio: float = 0.02
    max_gap_factor: float = 3.0  # a gap > factor * nominal period counts as a burst drop
    max_burst_gap_ms: float = 250.0
    max_rate_deviation: float = 0.10
    max_drift_ppm: float = 1000.0
    min_drift_duration_s: float = 1.0
    max_sync_residual_p95_ms: float = 15.0
    max_sync_residual_warn_ms: float = 8.0
    max_corrupt_frames: int = 0
    max_frozen_run: int = 15  # consecutive identical frames
    black_frame_mean: float = 4.0
    max_nan_ratio: float = 0.0
    min_duration_s: float = 2.0
    max_duration_s: float = 600.0
    max_timestamp_jump_ms: float = 100.0
    max_video_ts_mismatch: int = 0
    max_out_of_range_ratio: float = 0.01
    max_idle_ratio: float = 0.9
    require_calibration: Literal["fail", "warn", "ignore"] = "warn"


class CurationConfig(BaseModel):
    min_quality: float = 0.5
    require_success: bool = False
    dedup_traj_threshold: float = 0.05  # DTW distance on per-robot z-scored trajectories
    dedup_phash_threshold: float = 0.35  # DTW-aligned RMS between z-normalised frame thumbnails
    dedup_keyframes: int = 4
    split_by: str = "session_id"  # group key: scene_id | session_id | operator
    split_ratios: dict[str, float] = Field(default_factory=lambda: {"train": 0.8, "val": 0.1, "eval": 0.1})
    eval_groups: list[str] = Field(default_factory=list)  # force these groups into eval (held-out scenes)
    seed: str = "robodata-v1"


class DatasetSource(BaseModel):
    robot_id: str
    max_episodes: int | None = None


class DatasetRecipe(BaseModel):
    name: str
    repo_id: str
    fps: int
    image_size: list[int]  # [H, W]
    robot_type: str
    sources: list[DatasetSource]
    features: dict[str, dict]  # output feature key -> {"shape": [...], "names": [...]} (cameras inferred)
    cameras: list[str]  # canonical camera names expected in every source
    depth_cameras: list[str] = Field(default_factory=list)
    depth: dict = Field(default_factory=lambda: {"depth_min": 0.05, "depth_max": 3.0, "use_log": False})
    rgb_encoder: dict = Field(default_factory=lambda: {"vcodec": "libsvtav1", "crf": 30, "g": 2})
    phases: bool = True
    license: str = "apache-2.0"
    curation: CurationConfig = Field(default_factory=CurationConfig)


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(path: str | Path, subdir: str) -> Path:
    """Accept either a path or a bare config name (looked up under configs/<subdir>/)."""
    p = Path(path)
    if p.exists():
        return p
    cand = CONFIG_DIR / subdir / (p.name if p.suffix else f"{p.name}.yaml")
    if cand.exists():
        return cand
    raise FileNotFoundError(f"config {path!r} not found (also tried {cand})")


def load_robot(name_or_path: str | Path) -> RobotConfig:
    return RobotConfig(**_load_yaml(resolve(name_or_path, "robots")))


def load_health(name_or_path: str | Path = "default") -> HealthThresholds:
    return HealthThresholds(**(_load_yaml(resolve(name_or_path, "health")) or {}))


def load_recipe(name_or_path: str | Path) -> DatasetRecipe:
    return DatasetRecipe(**_load_yaml(resolve(name_or_path, "datasets")))
