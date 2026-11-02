"""Task-phase segmentation from kinematics + force (no labels needed).

Boundaries:
  b1  end of the transport motion  - EE speed drops below 25% of its peak after the peak
  b2  contact onset                - |F - F_rest| exceeds max(4 sigma_rest, 1.5 N) for 3 ticks
                                     (falls back to the start of the final descent without F/T)
  b3  motion end                   - last tick with speed above 10% of peak
Phases: 0 approach [0, b1), 1 align [b1, b2), 2 contact/insert [b2, b3), 3 settle [b3, end].
Validated against the simulator's ground-truth operator phases (see build report).
"""

from __future__ import annotations

import numpy as np

PHASE_NAMES = ["approach", "align", "contact", "settle"]


def _smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    if len(x) < k:
        return x
    return np.convolve(np.pad(x, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def segment_phases(t_s: np.ndarray, ee_pos: np.ndarray, force: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    n = len(t_s)
    phase = np.zeros(n, np.int64)
    if n < 10:
        return phase, {"b": [n, n, n]}
    v = np.linalg.norm(np.gradient(ee_pos, t_s, axis=0), axis=1)
    v = _smooth(v, 7)
    peak_i = int(np.argmax(v))
    peak = float(v[peak_i]) + 1e-9
    after = np.where(v[peak_i:] < 0.25 * peak)[0]
    b1 = peak_i + int(after[0]) if len(after) else peak_i
    moving = np.where(v > 0.10 * peak)[0]
    b3 = int(moving[-1]) + 1 if len(moving) else n
    b2 = None
    method = "none"
    if force is not None and len(force) == n:
        f = np.linalg.norm(force[:, :3], axis=1)
        rest = f[: max(5, min(b1, n // 5))]
        thr = max(4 * float(rest.std()), 1.5)
        dev = np.abs(f - np.median(rest)) > thr
        run = np.convolve(dev.astype(int), np.ones(3, int), mode="valid") == 3
        hits = np.where(run[b1:])[0]
        if len(hits):
            b2, method = b1 + int(hits[0]), "force"
    if b2 is None:  # last descent start: z velocity turns negative and stays negative until b3
        vz = _smooth(np.gradient(ee_pos[:, 2], t_s), 7)
        neg = vz[b1:b3] < -0.1 * np.abs(vz[b1:b3]).max() if b3 > b1 else np.zeros(0, bool)
        idx = np.where(~neg)[0]
        b2 = b1 + (int(idx[-1]) + 1 if len(idx) else 0)
        method = "descent"
    b1, b2, b3 = sorted([min(b1, n), min(b2, n), min(b3, n)])
    phase[b1:b2], phase[b2:b3], phase[b3:] = 1, 2, 3
    return phase, {"b": [b1, b2, b3], "contact_method": method, "peak_speed": peak}


def boundary_errors(t_s: np.ndarray, bounds: list[int], gt_change_s: list[float]) -> dict:
    """Compare detected boundaries (ticks) to ground-truth phase change times (seconds, relative)."""
    det = [float(t_s[min(b, len(t_s) - 1)] - t_s[0]) for b in bounds]
    gt = list(gt_change_s[1:4]) + [np.nan] * (3 - len(gt_change_s[1:4]))
    return {f"b{k + 1}_err_s": (abs(d - g) if np.isfinite(g) else None) for k, (d, g) in enumerate(zip(det, gt))}
