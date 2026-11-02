"""Sim-to-dataset converter: native simulator logs -> raw episodes in the real-robot schema.

The mapping (stream names, unit / convention transforms) lives in
``configs/sources/<sim>.yaml``, so other simulators (Isaac Sim, Gazebo, ACT/robomimic
HDF5 exports) only need a new mapping file. Output goes through the same
:class:`EpisodeRecorder` as real data, including per-sensor clock models, and every
episode is flagged ``is_sim=True``.
"""

from __future__ import annotations

import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import yaml
from pydantic import BaseModel

from robodata.config import load_robot, resolve
from robodata.recorder.clock import NS, stamp
from robodata.recorder.core import EpisodeRecorder

SIM_EPOCH_NS = 1_767_225_600 * NS


class SimStream(BaseModel):
    key: str  # dataset path inside the HDF5 log
    t: str  # timestamp dataset (seconds, sim clock)
    transform: str | None = None  # m_to_mm | wxyz_to_xyzw
    sensor_model: str | None = None  # rgb_read_noise | depth_realsense (renders are noise-free)


class SimSourceSpec(BaseModel):
    name: str
    generator: str
    license: str
    robot: str
    streams: dict[str, SimStream]


def load_sim_source(name: str) -> SimSourceSpec:
    with open(resolve(name, "sources")) as f:
        return SimSourceSpec(**yaml.safe_load(f))


def camera_noise_rgb(img: np.ndarray, rng: np.random.Generator, read_sigma: float = 2.0) -> np.ndarray:
    """Sensor read noise: rendered images are noise-free, real ones never are (and frozen-frame checks rely on it)."""
    return np.clip(img.astype(np.float32) + rng.normal(0, read_sigma, img.shape), 0, 255).astype(np.uint8)


def camera_noise_depth(depth_m: np.ndarray, rng: np.random.Generator, k: float = 0.0015, dropout: float = 0.003) -> np.ndarray:
    """RealSense-like depth noise: sigma grows with z^2, 1 mm quantisation, random invalid pixels (0)."""
    d = depth_m.astype(np.float32)
    d = d + rng.normal(0, 1, d.shape).astype(np.float32) * k * d * d
    d = np.round(d * 1000) / 1000
    d[rng.random(d.shape) < dropout] = 0.0
    return d


SENSOR_MODELS = {"rgb_read_noise": camera_noise_rgb, "depth_realsense": camera_noise_depth}


def _transform(x: np.ndarray, how: str | None) -> np.ndarray:
    if how is None:
        return x
    if how == "m_to_mm":
        mm = np.asarray(x, np.float32) * 1000.0
        mm[(mm <= 0) | (mm > 65000) | ~np.isfinite(mm)] = 0  # beyond far plane / invalid -> 0 (sensor convention)
        return mm.astype(np.uint16)
    if how == "wxyz_to_xyzw":
        return np.concatenate([x[..., :3], x[..., 4:7], x[..., 3:4]], axis=-1)
    raise ValueError(how)


def convert_log(log_path: str | Path, spec: SimSourceSpec, raw_root: str | Path, ordinal: int,
                apply_clock: bool = True) -> Path:
    robot = load_robot(spec.robot)
    log_path = Path(log_path)
    with h5py.File(log_path, "r") as f:
        a = dict(f.attrs)
        rec = EpisodeRecorder(robot, raw_root, session_id=f"{a['scene_id']}", operator=str(a["operator"]),
                              scene_id=str(a["scene_id"]), is_sim=True)
        ph_t = f["sensors/phase/t"][:]
        ph = f["sensors/phase/data"][:]
        change = np.concatenate([[0], np.where(np.diff(ph) != 0)[0] + 1])
        rec.start(task=str(a["task"]), episode_index=ordinal, source=f"sim:{spec.generator}:{log_path.name}",
                  success=bool(a["success"]), has_ground_truth_clock=True,
                  extra={"sim_seed": int(a["seed"]), "sim": str(a["sim"]), "mujoco_version": str(a["mujoco_version"]),
                         "depth_reached_m": float(a["depth_reached_m"]), "misalign_bias_mm": float(a["misalign_bias_mm"]),
                         "gt_phase_names": str(a["phase_names"]).split(","),
                         "gt_phase_change_s": ph_t[change].tolist(), "gt_phase_index": ph[change].astype(int).tolist()})
        epoch = SIM_EPOCH_NS + ordinal * 600 * NS
        try:
            for stream, ss in spec.streams.items():
                t_true = np.round(f[ss.t][:] * NS).astype(np.int64)
                ds = f[ss.key]
                # zlib.crc32, not hash(): str hashes are randomised per process, which made sim data irreproducible
                rng = np.random.default_rng(int(a["seed"]) * 1000 + zlib.crc32(stream.encode()) % 997)
                clock = robot.streams[stream].clock if apply_clock else type(robot.streams[stream].clock)()
                t_dev, t_host, keep = stamp(t_true, clock, rng, epoch_ns=epoch)
                big = ds.ndim >= 3
                data = None if big else _transform(ds[:], ss.transform)
                for i in range(len(t_true)):
                    if not keep[i]:
                        continue
                    if big:
                        raw = ds[i]
                        if ss.sensor_model:
                            raw = SENSOR_MODELS[ss.sensor_model](raw, rng)
                        v = _transform(raw, ss.transform)
                    else:
                        v = data[i]
                    rec.on_message(stream, int(t_dev[i]), v, t_host_ns=int(t_host[i]), t_true_ns=int(t_true[i] + epoch),
                                   seq=i)
        except Exception:
            rec.discard()
            raise
    return rec.stop()


def _worker(spec_name: str, path: str, ordinal: int, raw_root: str, apply_clock: bool) -> str:
    return str(convert_log(path, load_sim_source(spec_name), raw_root, ordinal, apply_clock))


def convert_dir(spec_name: str, logs_dir: str | Path, raw_root: str | Path, workers: int = 8,
                apply_clock: bool = True) -> list[str]:
    logs = sorted(Path(logs_dir).rglob("*.hdf5"))
    out = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_worker, spec_name, str(p), i, str(raw_root), apply_clock) for i, p in enumerate(logs)]
        for fu in as_completed(futs):
            out.append(fu.result())
            print(f"[sim2raw] {len(out)}/{len(logs)} {Path(out[-1]).name}", flush=True)
    return sorted(out)
