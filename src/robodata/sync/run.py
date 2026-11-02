"""Episode-level synchronisation: clock fits, grid alignment, residual sync error, data-driven lag check.

Writes ``sync.json`` next to the raw episode. Downstream stages (health, build) only
read that file plus the raw timestamps, so sync can be re-run with new calibration
(known latencies, method) without touching recorded data.
"""

from __future__ import annotations

import numpy as np

from robodata.config import RobotConfig
from robodata.io.raw_episode import RawEpisode
from robodata.io.video import iter_frames
from robodata.sync.align import make_grid, nearest_index
from robodata.sync.clock_fit import ClockFit, find_timestamp_jumps, fit_clock

JUMP_THRESHOLD_MS = 20.0
MIN_SEGMENT = 10


def _pct(a: np.ndarray, q: float) -> float | None:
    return float(np.percentile(a, q)) if len(a) else None


def fit_stream(t_dev: np.ndarray, t_host: np.ndarray, method: str, known_latency_ms: float) -> tuple[list[dict], np.ndarray]:
    """Piecewise clock fit (a new segment after every detected device-clock jump)."""
    jumps = find_timestamp_jumps(t_dev, t_host, JUMP_THRESHOLD_MS)
    bounds = sorted({0, *[int(j) for j in jumps if 0 < j < len(t_dev)], len(t_dev)})
    segs, t_est = [], np.empty(len(t_dev), np.int64)
    prev: ClockFit | None = None
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a >= MIN_SEGMENT or prev is None:
            fit = fit_clock(t_dev[a:b], t_host[a:b], method if b - a >= 2 else "none", known_latency_ms)
        else:  # too short to fit: reuse the previous drift, re-anchor the offset
            y = (t_host[a:b] - t_dev[a:b]) / 1e6
            fit = ClockFit(prev.method, float(y.min()), prev.d_ms_per_s, int(t_dev[a]), known_latency_ms, b - a,
                           0.0, 0.0, 0.0)
        t_est[a:b] = fit.to_host(t_dev[a:b])
        segs.append({"start": a, "end": b, **fit.as_dict()})
        prev = fit
    return segs, t_est


def corrected_times(ep: RawEpisode, stream: str, sync: dict) -> np.ndarray:
    """Host-clock capture times for a stream, using the fits stored in sync.json."""
    t_dev = ep.times(stream, "t_device")
    out = np.empty(len(t_dev), np.int64)
    for seg in sync["streams"][stream]["segments"]:
        fit = ClockFit(seg["method"], seg["c_ms"], seg["d_ms_per_s"], seg["t0_dev_ns"], seg["known_latency_ms"],
                       seg["n"], 0, 0, 0)
        out[seg["start"]:seg["end"]] = fit.to_host(t_dev[seg["start"]:seg["end"]])
    return out


def motion_energy(video_path, t_frames: np.ndarray, size: int = 64) -> np.ndarray:
    import cv2

    prev, e = None, []
    for fr in iter_frames(video_path):
        g = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY), (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
        e.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
        prev = g
    e = np.asarray(e[: len(t_frames)])
    return e


def xcorr_lag(t_a: np.ndarray, a: np.ndarray, t_b: np.ndarray, b: np.ndarray, max_lag_ms: float = 150,
              rate_hz: float = 200) -> tuple[float, float]:
    """Lag (ms) of signal a relative to b maximising normalised cross-correlation (+ = a late)."""
    t0, t1 = max(t_a[0], t_b[0]), min(t_a[-1], t_b[-1])
    if t1 - t0 < 2e9:
        return float("nan"), 0.0
    g = np.arange(t0, t1, 1e9 / rate_hz)
    ua = np.interp(g, t_a, a)
    ub = np.interp(g, t_b, b)
    ua = (ua - ua.mean()) / (ua.std() + 1e-9)
    ub = (ub - ub.mean()) / (ub.std() + 1e-9)
    L = int(max_lag_ms * rate_hz / 1000)
    lags = np.arange(-L, L + 1)
    cc = np.array([np.mean(ua[max(0, k):len(ua) + min(0, k)] * ub[max(0, -k):len(ub) + min(0, -k)]) for k in lags])
    k = int(np.argmax(cc))
    off = 0.0
    if 0 < k < len(cc) - 1:  # parabolic sub-sample refinement
        den = cc[k - 1] - 2 * cc[k] + cc[k + 1]
        off = 0.5 * (cc[k - 1] - cc[k + 1]) / den if den != 0 else 0.0
    return float((lags[k] + off) * 1000 / rate_hz), float(cc[k])


def _speed_signal(ep: RawEpisode, robot: RobotConfig, t_est: dict) -> tuple[np.ndarray, np.ndarray] | None:
    # EE translational speed tracks image motion best (validated on sim with ground-truth clocks:
    # ~1 ms vs ~20 ms bias for joint-space speed); joint speed is the fallback.
    for kind in ("pose", "joint_state"):
        for s, cfg in robot.streams.items():
            if cfg.kind == kind and s in t_est and ep.has(s, "data") and len(ep.times(s)) > 10:
                x = ep.data(s).astype(np.float64)
                dims = slice(0, 3) if kind == "pose" else slice(0, max(1, x.shape[1] - 1))
                t = t_est[s]
                v = np.linalg.norm(np.diff(x[:, dims], axis=0), axis=1) / np.maximum(np.diff(t) / 1e9, 1e-4)
                return (t[1:] + t[:-1]) // 2, v  # a finite difference belongs to the interval midpoint
    return None


def sync_episode(ep: RawEpisode, robot: RobotConfig, do_xcorr: bool = True) -> dict:
    method = robot.sync.clock_fit
    rep: dict = {"episode_id": ep.id, "method": method, "sync_mode": robot.sync.mode, "streams": {}}
    t_est: dict[str, np.ndarray] = {}
    for s in ep.streams:
        if s not in robot.streams:
            continue
        t_dev, t_host = ep.times(s, "t_device"), ep.times(s, "t_host")
        if len(t_dev) < 2:
            rep["streams"][s] = {"n": int(len(t_dev)), "segments": []}
            continue
        segs, est = fit_stream(t_dev, t_host, method, robot.streams[s].known_latency_ms)
        t_est[s] = est
        d = np.diff(est) / 1e6
        info = {"n": int(len(t_dev)), "segments": segs, "clock_jumps": len(segs) - 1,
                "drift_ppm": segs[0]["drift_ppm"],
                "rate_hz": float((len(est) - 1) / ((est[-1] - est[0]) / 1e9)) if est[-1] > est[0] else None,
                "transport_delay_p95_ms": segs[0]["envelope_p95_ms"]}
        t_true = ep.times(s, "t_true")
        if t_true is not None and len(t_true) == len(est):
            err = (est - t_true) / 1e6
            info["gt_clock_error_ms"] = {"mean": float(err.mean()), "p95_abs": _pct(np.abs(err), 95),
                                         "max_abs": float(np.abs(err).max())}
        rep["streams"][s] = info

    # --- alignment onto the training grid ---------------------------------------------
    ref = robot.sync.reference_stream
    required = [s for s in robot.required_streams if s in t_est]
    if ref in t_est and required:
        grid = make_grid([int(t_est[s][0]) for s in required], [int(t_est[s][-1]) for s in required],
                         robot.target_fps, phase_ref_ns=t_est[ref])
        rep["grid"] = {"start_ns": int(grid[0]) if len(grid) else None, "n": int(len(grid)), "fps": robot.target_fps}
        cams, true_err = {}, []
        for s in robot.camera_streams:
            if s not in t_est or not len(grid):
                continue
            idx, res = nearest_index(t_est[s], grid)
            cams[s] = {"residual_p50_ms": _pct(res, 50), "residual_p95_ms": _pct(res, 95), "residual_max_ms": float(res.max()),
                       "ticks_over_tolerance": int((res > robot.sync.max_frame_tolerance_ms).sum())}
            t_true = ep.times(s, "t_true")
            if t_true is not None and len(t_true) == len(t_est[s]):
                te = np.abs(t_true[idx] - grid) / 1e6
                cams[s]["gt_error_p95_ms"] = _pct(te, 95)
                true_err.append(te)
        rep["cameras"] = cams
        if cams:
            rep["residual_sync_error_p95_ms"] = max(c["residual_p95_ms"] for c in cams.values())
            if true_err:
                rep["gt_sync_error_p95_ms"] = float(np.percentile(np.concatenate(true_err), 95))
        if len(cams) >= 2 and len(grid):
            sel = {s: t_est[s][nearest_index(t_est[s], grid)[0]] for s in cams}
            arr = np.stack(list(sel.values()))
            rep["inter_camera_skew_p95_ms"] = _pct((arr.max(0) - arr.min(0)) / 1e6, 95)

    # --- data-driven check: does camera motion line up with robot motion? ----------------
    if do_xcorr:
        cam = next((s for s in robot.camera_streams if robot.streams[s].kind == "rgb" and "wrist" in s and s in t_est),
                   None)
        sp = _speed_signal(ep, robot, t_est)
        if cam and sp is not None and ep.video_path(cam).exists():
            try:
                e = motion_energy(ep.video_path(cam), t_est[cam])
                te = t_est[cam][: len(e)]
                mid = (te[1:] + te[:-1]) // 2  # frame difference i-1 -> i belongs to the midpoint
                lag, peak = xcorr_lag(mid, e[1:], sp[0], sp[1])
                rep["xcorr"] = {"camera": cam, "lag_ms": lag, "peak_corr": peak}
            except Exception as exc:  # corrupt video: reported by health checks
                rep["xcorr"] = {"camera": cam, "error": str(exc)}
    ep.write_sidecar("sync.json", rep)
    return rep
