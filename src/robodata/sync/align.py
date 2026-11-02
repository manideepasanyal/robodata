"""Resample multi-rate streams onto a common, fixed-rate grid (training clock)."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation, Slerp


def make_grid(starts_ns: list[int], ends_ns: list[int], fps: float, phase_ref_ns: np.ndarray | None = None) -> np.ndarray:
    """Uniform grid over the time span covered by *all* streams.

    If ``phase_ref_ns`` (reference camera frame times) is given, the grid starts on the
    first reference frame inside the common span so the reference camera has ~zero
    residual and other streams are measured relative to it.
    """
    start, end = max(starts_ns), min(ends_ns)
    if end <= start:
        return np.zeros(0, np.int64)
    if phase_ref_ns is not None and len(phase_ref_ns):
        inside = phase_ref_ns[phase_ref_ns >= start]
        if len(inside):
            start = int(inside[0])
    step = 1e9 / fps
    n = int(np.floor((end - start) / step)) + 1
    return (start + np.round(np.arange(n) * step)).astype(np.int64)


def nearest_index(t_src: np.ndarray, t_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest source sample per grid tick and the residual |t_src - t_grid| in ms."""
    j = np.searchsorted(t_src, t_grid)
    j = np.clip(j, 1, len(t_src) - 1)
    left, right = t_src[j - 1], t_src[j]
    pick = np.where(np.abs(t_grid - left) <= np.abs(right - t_grid), j - 1, j)
    return pick, np.abs(t_src[pick] - t_grid) / 1e6


def _to_seconds(t: np.ndarray, t0: int) -> np.ndarray:
    return (np.asarray(t, np.int64) - t0) / 1e9


def interp_linear(t_src, x, t_grid, angle_dims=None):
    x = np.asarray(x, np.float64).copy()
    if angle_dims:
        x[:, angle_dims] = np.unwrap(x[:, angle_dims], axis=0)
    t0 = int(t_grid[0])
    ts, tg = _to_seconds(t_src, t0), _to_seconds(t_grid, t0)
    out = np.stack([np.interp(tg, ts, x[:, k]) for k in range(x.shape[1])], 1)
    if angle_dims:
        out[:, angle_dims] = (out[:, angle_dims] + np.pi) % (2 * np.pi) - np.pi
    return out.astype(np.float32)


def interp_hold(t_src, x, t_grid, tol_frac: float = 0.25):
    """Zero-order hold: last command issued at or before each tick (commands are piecewise constant).

    Clock-corrected stamps carry ~1 ms of residual error, so a command issued exactly at a tick can
    appear a hair *after* it; a plain "<= tick" then picks the previous command for whole episodes
    (an off-by-one action shift caught by `robodata audit`). Commands within a quarter of their own
    period after the tick are treated as issued at it - jitter tolerance, not look-ahead.
    """
    t_src = np.asarray(t_src, np.int64)
    period = float(np.median(np.diff(t_src))) if len(t_src) > 1 else 0.0
    j = np.searchsorted(t_src, np.asarray(t_grid, np.int64) + int(tol_frac * period), side="right") - 1
    return np.asarray(x)[np.clip(j, 0, len(t_src) - 1)].astype(np.float32)


def interp_pose(t_src, x, t_grid):
    """xyz linear + quaternion (x, y, z, w) slerp."""
    x = np.asarray(x, np.float64)
    pos = interp_linear(t_src, x[:, :3], t_grid)
    q = x[:, 3:7].copy()
    q /= np.linalg.norm(q, axis=1, keepdims=True).clip(1e-9)
    flip = np.concatenate([[1.0], np.sign(np.sum(q[1:] * q[:-1], 1) + 1e-12)])  # hemisphere continuity
    q *= np.cumprod(flip)[:, None]
    t0 = int(t_grid[0])
    ts, tg = _to_seconds(t_src, t0), _to_seconds(t_grid, t0)
    ts, uniq = np.unique(ts, return_index=True)
    slerp = Slerp(ts, Rotation.from_quat(q[uniq]))
    qg = slerp(np.clip(tg, ts[0], ts[-1])).as_quat()
    return np.concatenate([pos, qg], 1).astype(np.float32)


def interp_lowpass(t_src, x, t_grid, target_fps: float):
    """Anti-aliased downsampling for high-rate signals (F/T @ 500 Hz -> 30 Hz)."""
    t_src = np.asarray(t_src, np.int64)
    if len(t_src) < 16:
        return interp_linear(t_src, x, t_grid)
    src_rate = 1e9 / np.median(np.diff(t_src))
    if src_rate < 2.5 * target_fps:
        return interp_linear(t_src, x, t_grid)
    # re-grid at the native rate (removes jitter), zero-phase Butterworth, then sample
    t0 = int(t_src[0])
    uni = (t0 + np.arange(int((t_src[-1] - t0) / (1e9 / src_rate)) + 1) * (1e9 / src_rate)).astype(np.int64)
    xu = interp_linear(t_src, x, uni).astype(np.float64)
    sos = butter(4, 0.4 * target_fps, fs=src_rate, output="sos")
    xf = sosfiltfilt(sos, xu, axis=0) if len(xu) > 30 else xu
    return interp_linear(uni, xf, t_grid)


def resample(kind: str, t_src, x, t_grid, target_fps: float, angle_dims=None) -> np.ndarray:
    if kind == "linear":
        return interp_linear(t_src, x, t_grid, angle_dims)
    if kind == "hold":
        return interp_hold(t_src, x, t_grid)
    if kind == "pose":
        return interp_pose(t_src, x, t_grid)
    if kind == "lowpass_linear":
        return interp_lowpass(t_src, x, t_grid, target_fps)
    if kind == "nearest":
        idx, _ = nearest_index(np.asarray(t_src, np.int64), t_grid)
        return np.asarray(x)[idx].astype(np.float32)
    raise ValueError(kind)
