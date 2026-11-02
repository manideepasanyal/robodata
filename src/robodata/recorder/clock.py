"""Sensor clock + transport model.

Real multi-sensor rigs have one clock per device. Each device stamps a message with
its own (offset, drifting, jittery) clock, and the message reaches the recorder after
a variable, strictly positive transport latency. We simulate exactly that when
replaying open-source data or sim logs through the recorder, keeping the true capture
time (t_true) so the sync stage can be scored against ground truth.
"""

from __future__ import annotations

import numpy as np

from robodata.config import ClockModel

NS = 1_000_000_000
MS = 1_000_000


def stamp(t_true_ns: np.ndarray, clock: ClockModel, rng: np.random.Generator,
          epoch_ns: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t_device, t_host, keep_mask) for true capture times.

    epoch_ns is the session start on the *host* clock; device clocks are offset from it.
    """
    t = np.asarray(t_true_ns, dtype=np.int64)
    n = len(t)
    rel = (t - t[0]).astype(np.float64) if n else np.zeros(0)
    drift = rel * clock.drift_ppm * 1e-6
    jitter = rng.normal(0.0, clock.jitter_ms, n) * MS if clock.jitter_ms > 0 else np.zeros(n)
    t_device = t + (drift + clock.offset_ms * MS + jitter).astype(np.int64) + epoch_ns
    lat = clock.latency_ms * MS + (rng.exponential(clock.latency_jitter_ms, n) * MS if clock.latency_jitter_ms > 0 else 0)
    t_host = t + np.asarray(lat, dtype=np.int64) + epoch_ns
    t_host = np.maximum.accumulate(t_host)  # FIFO transport: per-stream arrival order is preserved
    keep = rng.random(n) >= clock.drop_prob if clock.drop_prob > 0 else np.ones(n, bool)
    return t_device, t_host, keep
