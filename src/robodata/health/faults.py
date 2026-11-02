"""Fault injection: derive broken copies of clean raw episodes to test the health checks.

Each faulty copy records what was injected in ``meta.injected_faults`` so tests and
the report can compute detection recall / false-alarm rates per fault type.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import zarr

from robodata.io.raw_episode import RawEpisode
from robodata.io.video import VideoWriter, iter_frames, probe


def _rewrite_array(g: zarr.Group, name: str, data: np.ndarray) -> None:
    if name in g:
        del g[name]
    chunks = (min(len(data), 8),) + data.shape[1:] if data.ndim == 3 else None
    if len(data) == 0:
        g.create_array(name, shape=data.shape, dtype=data.dtype)
    elif chunks:
        g.create_array(name, data=data, chunks=chunks)
    else:
        g.create_array(name, data=data)


def _reencode(path: Path, frames: list[np.ndarray], fps: float) -> None:
    h, w = frames[0].shape[:2]
    tmp = path.with_suffix(".tmp.mp4")
    with VideoWriter(tmp, fps, w, h) as vw:
        for f in frames:
            vw.write(f)
    tmp.replace(path)


def _rows(ep_dir: Path, stream: str, keep: np.ndarray, meta: dict) -> None:
    """Keep only rows `keep` of a stream (timestamps, data and video frames)."""
    g = zarr.open_group(str(ep_dir / "streams.zarr"), mode="r+")[stream]
    for k in ("t_device", "t_host", "seq", "t_true", "data"):
        if k in g:
            _rewrite_array(g, k, np.asarray(g[k][:])[keep])
    vp = ep_dir / "video" / f"{stream}.mp4"
    if vp.exists():
        frames = list(iter_frames(vp))
        _reencode(vp, [frames[i] for i in np.where(keep)[0]], meta["streams"][stream]["rate_hz"])


FAULTS = {}
DUPLICATE_FAULTS = {"near_duplicate", "exact_duplicate", "reencode_duplicate", "trimmed_duplicate", "video_swap"}


def fault(fn):
    FAULTS[fn.__name__] = fn
    return fn


@fault
def drop_frames(ep_dir, meta, rng, stream, ratio=0.1, burst=False):
    n = meta["streams"][stream]["count"]
    keep = np.ones(n, bool)
    if burst:
        L = max(1, int(n * ratio))
        s0 = rng.integers(n // 4, max(n // 4 + 1, n - L - 1))
        keep[s0:s0 + L] = False
    else:
        keep[rng.choice(np.arange(1, n - 1), int(n * ratio), replace=False)] = False
    _rows(ep_dir, stream, keep, meta)


@fault
def timestamp_offset(ep_dir, meta, rng, stream, offset_ms=80.0, at=0.5):
    g = zarr.open_group(str(ep_dir / "streams.zarr"), mode="r+")[stream]
    t = np.asarray(g["t_device"][:])
    k = int(len(t) * at)
    t[k:] += int(offset_ms * 1e6)
    _rewrite_array(g, "t_device", t)


@fault
def clock_drift(ep_dir, meta, rng, stream, ppm=3000.0):
    g = zarr.open_group(str(ep_dir / "streams.zarr"), mode="r+")[stream]
    t = np.asarray(g["t_device"][:])
    t = t + ((t - t[0]) * ppm * 1e-6).astype(np.int64)
    _rewrite_array(g, "t_device", t)


@fault
def corrupt_frames(ep_dir, meta, rng, stream, n=5, mode="noise"):
    kind = meta["streams"][stream]["kind"]
    if kind == "depth":
        g = zarr.open_group(str(ep_dir / "streams.zarr"), mode="r+")[stream]
        d = np.asarray(g["data"][:])
        d[rng.choice(len(d), n, replace=False)] = 0
        _rewrite_array(g, "data", d)
        return
    vp = ep_dir / "video" / f"{stream}.mp4"
    frames = list(iter_frames(vp))
    for i in rng.choice(np.arange(5, len(frames) - 5), n, replace=False):
        frames[i] = (rng.integers(0, 256, frames[i].shape, dtype=np.uint8) if mode == "noise"
                     else np.zeros_like(frames[i]))
    _reencode(vp, frames, meta["streams"][stream]["rate_hz"])


@fault
def frozen_camera(ep_dir, meta, rng, stream, length=60):
    vp = ep_dir / "video" / f"{stream}.mp4"
    frames = list(iter_frames(vp))
    s0 = len(frames) // 3
    for i in range(s0, min(len(frames), s0 + length)):
        frames[i] = frames[s0]
    _reencode(vp, frames, meta["streams"][stream]["rate_hz"])


@fault
def truncate_video(ep_dir, meta, rng, stream, n=10):
    vp = ep_dir / "video" / f"{stream}.mp4"
    frames = list(iter_frames(vp))
    _reencode(vp, frames[:-n], meta["streams"][stream]["rate_hz"])


@fault
def remove_modality(ep_dir, meta, rng, stream):
    root = zarr.open_group(str(ep_dir / "streams.zarr"), mode="r+")
    del root[stream]
    (ep_dir / "video" / f"{stream}.mp4").unlink(missing_ok=True)
    meta["streams"].pop(stream, None)


@fault
def nan_values(ep_dir, meta, rng, stream, ratio=0.05):
    g = zarr.open_group(str(ep_dir / "streams.zarr"), mode="r+")[stream]
    d = np.asarray(g["data"][:]).copy()
    d[rng.choice(len(d), max(1, int(len(d) * ratio)), replace=False), 0] = np.nan
    _rewrite_array(g, "data", d)


@fault
def near_duplicate(ep_dir, meta, rng, stream=None, noise=1e-4):
    """Copy with tiny state noise: should be caught by dedup, not by health."""
    root = zarr.open_group(str(ep_dir / "streams.zarr"), mode="r+")
    for s, info in meta["streams"].items():
        if info["kind"] not in ("rgb", "depth") and s in root and "data" in root[s]:
            d = np.asarray(root[s]["data"][:])
            noisy = np.clip(d + rng.normal(0, noise, d.shape), d.min(0), d.max(0))  # stay inside the original range
            _rewrite_array(root[s], "data", noisy.astype(np.float32))


@fault
def exact_duplicate(ep_dir, meta, rng, stream=None):
    """Byte-identical copy (the copy itself is done by inject)."""


@fault
def reencode_duplicate(ep_dir, meta, rng, stream=None, crf=30, noise=1e-4):
    """Same footage re-encoded at another quality + tiny sensor noise: a re-exported copy."""
    for s, info in meta["streams"].items():
        vp = ep_dir / "video" / f"{s}.mp4"
        if info["kind"] == "rgb" and vp.exists():
            frames = list(iter_frames(vp))
            h, w = frames[0].shape[:2]
            tmp = vp.with_suffix(".tmp.mp4")
            with VideoWriter(tmp, info["rate_hz"], w, h, crf=crf, preset="fast") as vw:
                for f in frames:
                    vw.write(f)
            tmp.replace(vp)
    near_duplicate(ep_dir, meta, rng, noise=noise)


@fault
def trimmed_duplicate(ep_dir, meta, rng, stream=None, frac=0.08):
    """Same recording with the first `frac` of the episode cut off (every stream, by time)."""
    root = zarr.open_group(str(ep_dir / "streams.zarr"), mode="r")
    t0 = min(int(root[s]["t_host"][0]) for s in meta["streams"] if s in root and root[s]["t_host"].shape[0])
    t1 = max(int(root[s]["t_host"][-1]) for s in meta["streams"] if s in root and root[s]["t_host"].shape[0])
    cut = t0 + int(frac * (t1 - t0))
    for s in list(meta["streams"]):
        if s in root:
            _rows(ep_dir, s, np.asarray(root[s]["t_host"][:]) >= cut, meta)


@fault
def video_swap(ep_dir, meta, rng, stream=None, video_from: str = ""):
    """Keep this episode's trajectory but replace its camera videos with another episode's footage."""
    other = Path(video_from)
    root = zarr.open_group(str(ep_dir / "streams.zarr"), mode="r+")
    oroot = zarr.open_group(str(other / "streams.zarr"), mode="r")
    for s, info in meta["streams"].items():
        if info["kind"] == "rgb" and (other / "video" / f"{s}.mp4").exists():
            shutil.copy(other / "video" / f"{s}.mp4", ep_dir / "video" / f"{s}.mp4")
            for k in ("t_device", "t_host", "seq", "t_true"):  # timestamps travel with the frames
                if k in oroot[s]:
                    _rewrite_array(root[s], k, np.asarray(oroot[s][k][:]))


def inject(src_dir: str | Path, faults: list[dict], tag: str | None = None, seed: int = 0,
           session: str | None = None) -> Path:
    """Copy episode ``src_dir`` and apply ``faults`` ([{"type": "drop_frames", "stream": "front", ...}]).

    ``session`` places the copy in another recording session (i.e. potentially another split).
    """
    src = RawEpisode(src_dir)
    tag = tag or "+".join(f["type"] for f in faults)
    new_id = f"{src.id}__fault-{tag}"
    parent = Path(src_dir).parent if session is None else Path(src_dir).parent.parent / session
    parent.mkdir(parents=True, exist_ok=True)
    dst = parent / new_id
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src_dir, dst)
    for side in ("sync.json", "health.json"):
        (dst / side).unlink(missing_ok=True)
    meta = json.loads((dst / "meta.json").read_text())
    rng = np.random.default_rng(seed)
    for f in faults:
        params = {k: v for k, v in f.items() if k != "type"}
        FAULTS[f["type"]](dst, meta, rng, **params)
        if f["type"] != "remove_modality" and params.get("stream") in meta["streams"]:
            g = zarr.open_group(str(dst / "streams.zarr"), mode="r")[params["stream"]]
            meta["streams"][params["stream"]]["count"] = int(g["t_device"].shape[0])
    meta["episode_id"] = new_id
    if session is not None:
        meta["session_id"] = session
    meta["injected_faults"] = meta.get("injected_faults", []) + faults
    if any(f["type"] in DUPLICATE_FAULTS for f in faults):
        meta["extra"]["duplicate_of"] = src.id
    (dst / "meta.json").write_text(json.dumps(meta, indent=2))
    return dst


# Default fault suite used by `robodata faults` and the tests: (tag, faults, expected failing check)
def default_suite(cam: str, depth: str | None, lowdim: str, wrench: str | None) -> list[tuple[str, list[dict], str]]:
    suite = [
        ("drop10", [{"type": "drop_frames", "stream": cam, "ratio": 0.10}], "drop_ratio"),
        ("dropburst", [{"type": "drop_frames", "stream": cam, "ratio": 0.08, "burst": True}], "max_gap_ms"),
        ("tsjump", [{"type": "timestamp_offset", "stream": cam, "offset_ms": 80}], "clock_jumps"),
        ("drift", [{"type": "clock_drift", "stream": lowdim, "ppm": 3000}], "drift_ppm"),
        ("noiseframes", [{"type": "corrupt_frames", "stream": cam, "n": 4, "mode": "noise"}], "corrupt_frames"),
        ("blackframes", [{"type": "corrupt_frames", "stream": cam, "n": 4, "mode": "black"}], "corrupt_frames"),
        ("frozen", [{"type": "frozen_camera", "stream": cam, "length": 45}], "frozen_run"),
        ("truncated", [{"type": "truncate_video", "stream": cam, "n": 12}], "video_ts_mismatch"),
        ("missing", [{"type": "remove_modality", "stream": wrench or lowdim}], "missing_modality"),
        ("nan", [{"type": "nan_values", "stream": lowdim, "ratio": 0.05}], "nan_ratio"),
    ]
    if depth:
        suite.append(("depthholes", [{"type": "corrupt_frames", "stream": depth, "n": 3}], "corrupt_frames"))
    return suite


__all__ = ["inject", "FAULTS", "default_suite", "probe"]
