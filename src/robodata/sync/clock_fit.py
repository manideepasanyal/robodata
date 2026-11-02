"""Per-stream clock estimation: map sensor (device) timestamps onto the recorder (host) clock.

Each message gives one pair (t_device, t_host) with
    t_host - t_device = offset + skew * t + latency,   latency >= latency_min > 0.
Transport latency is one-sided, so least squares is biased by the *mean* latency and
by its jitter. Following passive clock-sync methods (e.g. TICSync, Harrison & Newman,
ICRA 2011) we instead fit the *lower envelope* of the offsets: the line
y = c + d x that lies below every point while being as tight as possible (an LP).
The remaining constant (min latency: exposure/driver delay) is not observable from
timestamps alone and is removed with the calibrated ``known_latency_ms`` per stream.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import linprog

NS_PER_MS = 1e6


@dataclass
class ClockFit:
    method: str
    c_ms: float  # host - device offset at t0 (ms)
    d_ms_per_s: float  # offset slope (ms/s) == -drift_ppm / 1000
    t0_dev_ns: int
    known_latency_ms: float
    n: int
    envelope_p50_ms: float  # distribution of transport delay above the envelope
    envelope_p95_ms: float
    envelope_max_ms: float

    @property
    def drift_ppm(self) -> float:
        return -self.d_ms_per_s * 1000.0

    def to_host(self, t_device_ns: np.ndarray) -> np.ndarray:
        x = (np.asarray(t_device_ns, dtype=np.int64) - self.t0_dev_ns) / 1e9
        corr_ms = self.c_ms + self.d_ms_per_s * x - self.known_latency_ms
        return np.asarray(t_device_ns, dtype=np.int64) + np.round(corr_ms * NS_PER_MS).astype(np.int64)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["drift_ppm"] = self.drift_ppm
        return d


def _envelope_lp(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    # maximise sum(c + d x_i) s.t. c + d x_i <= y_i   (tightest supporting line from below)
    res = linprog(c=[-len(x), -float(x.sum())], A_ub=np.stack([np.ones_like(x), x], 1), b_ub=y,
                  bounds=[(None, None), (None, None)], method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return float(res.x[0]), float(res.x[1])


def fit_clock(t_device: np.ndarray, t_host: np.ndarray, method: str = "lower_envelope",
              known_latency_ms: float = 0.0, max_points: int = 20000) -> ClockFit:
    t_device = np.asarray(t_device, dtype=np.int64)
    t_host = np.asarray(t_host, dtype=np.int64)
    if len(t_device) < 2:
        raise ValueError("need >= 2 samples to fit a clock")
    t0 = int(t_device[0])
    x = (t_device - t0) / 1e9
    y = (t_host - t_device) / NS_PER_MS
    if method == "none":
        c, d = 0.0, 0.0
    elif method == "least_squares":
        d, c = np.polyfit(x, y, 1)
    elif method == "lower_envelope":
        sel = np.linspace(0, len(x) - 1, min(len(x), max_points)).astype(int)
        c, d = _envelope_lp(x[sel], y[sel])
    else:
        raise ValueError(method)
    resid = y - (c + d * x)
    return ClockFit(method, float(c), float(d), t0, float(known_latency_ms), int(len(x)),
                    float(np.percentile(resid, 50)), float(np.percentile(resid, 95)), float(resid.max()))


def find_timestamp_jumps(t_device: np.ndarray, t_host: np.ndarray, threshold_ms: float) -> np.ndarray:
    """Indices i where the device clock jumped relative to the host clock between i-1 and i.

    Compares per-message increments on both clocks; transport jitter only perturbs
    single samples, whereas a clock step persists. We therefore test the running
    median of (t_host - t_device) on either side of i.
    """
    y = (np.asarray(t_host, np.int64) - np.asarray(t_device, np.int64)) / NS_PER_MS
    n = len(y)
    if n < 3:
        return np.zeros(0, int)
    # ~1 s windows: a host-load burst raises latency for a few hundred ms, a clock step persists
    period = float(np.median(np.diff(np.asarray(t_device, np.int64)))) / 1e9
    w = int(max(15, round(1.0 / period))) if period > 0 else 15
    if n < 2 * w + 1:
        return np.zeros(0, int)
    from numpy.lib.stride_tricks import sliding_window_view

    mins = sliding_window_view(y, w).min(axis=1)  # envelope per window: robust to positive latency jitter
    before = mins[: n - 2 * w + 1]
    after = mins[w: n - w + 1]
    step = np.abs(after - before)
    idx = np.where(step > threshold_ms)[0]
    if len(idx) == 0:
        return idx
    # collapse consecutive detections to one per jump
    groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    out = []
    for g in groups:  # the min-envelope detects robustly; localise with a two-sided median split
        lo, hi = max(w, int(g[0])), min(n - w, int(g[-1]) + 2 * w)
        cand = np.arange(lo, hi + 1)
        score = [abs(np.median(y[i - w:i]) - np.median(y[i:i + w])) for i in cand]
        out.append(int(cand[int(np.argmax(score))]))
    return np.array(out, int)
