"""Per-episode health checks with pass / warn / fail rules (thresholds: configs/health/*.yaml)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from robodata.config import HealthThresholds, RobotConfig
from robodata.io.raw_episode import RawEpisode
from robodata.io.video import iter_frames
from robodata.sync.run import corrected_times, sync_episode

PASS, WARN, FAIL = "pass", "warn", "fail"
GLITCH_HF_RATIO = 5.0
# absolute floor: on near-textureless views the episode median is tiny and keyframe-vs-P-frame grain alone
# exceeds 5x (measured false positives <= 196); injected noise / garbage frames measured >= 2640
GLITCH_HF_FLOOR = 1000.0


@dataclass
class Check:
    check: str
    status: str
    stream: str | None = None
    value: float | int | str | None = None
    threshold: float | int | str | None = None
    detail: str = ""


def _cmp(name, value, thr, stream=None, warn_thr=None, detail="", higher_is_bad=True) -> Check:
    bad = value > thr if higher_is_bad else value < thr
    status = FAIL if bad else PASS
    if not bad and warn_thr is not None and (value > warn_thr if higher_is_bad else value < warn_thr):
        status = WARN
    return Check(name, status, stream, round(float(value), 4), thr, detail)


def video_stats(path, n_expected: int, h: HealthThresholds) -> dict:
    """Decode the whole video; count black, frozen and glitch (corrupt) frames."""
    smalls, hf, error = [], [], None
    try:
        for fr in iter_frames(path):
            g = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)
            hf.append(float(cv2.Laplacian(cv2.resize(g, (160, 120), interpolation=cv2.INTER_AREA), cv2.CV_32F).var()))
            smalls.append(cv2.resize(g, (64, 48), interpolation=cv2.INTER_AREA).astype(np.int16))
    except Exception as exc:  # truncated / corrupt bitstream
        error = f"{type(exc).__name__}: {exc}"
    n = len(smalls)
    dark = np.array([s.mean() < h.black_frame_mean for s in smalls], bool)
    # an isolated black frame is a dropout (corrupt); a run of dark frames is an occluded lens (warn only)
    prev_ok = np.concatenate([[True], ~dark[:-1]]) if n else dark
    next_ok = np.concatenate([~dark[1:], [True]]) if n else dark
    black = int((dark & prev_ok & next_ok).sum())
    dark_run = int(dark.sum() - black)
    frozen_run, glitches = 0, 0
    if n >= 3:
        s = np.stack(smalls)
        d = np.abs(np.diff(s, axis=0)).mean(axis=(1, 2))  # d[i] = |f[i+1]-f[i]|
        # lossy codecs make a frozen stream *almost* identical (and keyframes add small blips), so
        # threshold far below real sensor noise (>= ~0.01 grey levels here) and bridge blips of <= 2 frames
        still = d < 0.006
        run = best = blip = 0
        for v in still:
            if v:
                run, blip = run + 1 + blip, 0
            elif run and blip < 2:
                blip += 1
            else:
                run, blip = 0, 0
            best = max(best, run)
        frozen_run = best
        # glitch: a frame far from both neighbours while its neighbours agree with each other
        # garbage / noise frames carry far more high-frequency energy than the episode's own frames
        # (measured: injected noise >= 7.5x the episode median, clean real frames <= 2.5x)
        # ...and are isolated spikes (sharp-vs-blurry real footage changes gradually over many frames)
        r = np.asarray(hf) / (np.median(hf) + 1e-6)
        nb = np.maximum(np.concatenate([[0.0], r[:-1]]), np.concatenate([r[1:], [0.0]]))
        glitches = int(np.sum((r > GLITCH_HF_RATIO) & (r > 2.0 * nb) & (np.asarray(hf) > GLITCH_HF_FLOOR)))
    return {"decoded": n, "expected": n_expected, "black": black, "dark_run": dark_run, "frozen_run": frozen_run,
            "glitches": glitches, "error": error}


def depth_stats(arr) -> dict:
    n = arr.shape[0]
    invalid = np.array([(arr[i] == 0).mean() for i in range(n)]) if n else np.zeros(0)
    return {"frames": n, "all_invalid": int((invalid > 0.98).sum()),
            "invalid_ratio_mean": float(invalid.mean()) if n else 1.0}


def check_episode(ep: RawEpisode, robot: RobotConfig, h: HealthThresholds, decode_video: bool = True,
                  resync: bool = False) -> dict:
    checks: list[Check] = []
    if robot.health_overrides:
        h = h.model_copy(update=robot.health_overrides)
    sync = None if resync else ep.sidecar("sync.json")
    if sync is None:
        sync = sync_episode(ep, robot)
    present = set(ep.streams)

    # --- modalities ------------------------------------------------------------------
    for s in robot.required_streams:
        n = len(ep.times(s)) if s in present else 0
        checks.append(Check("missing_modality", FAIL if n == 0 else PASS, s, n, ">0"))

    durations = []
    for s, cfg in robot.streams.items():
        if s not in present:
            continue
        t_dev = ep.times(s, "t_device")
        n = len(t_dev)
        if n < 2:
            continue
        t = corrected_times(ep, s, sync)
        dt = np.diff(t) / 1e6
        period = 1000.0 / cfg.rate_hz
        dur = (t[-1] - t[0]) / 1e9
        durations.append(dur)
        # timestamp integrity
        checks.append(_cmp("timestamp_non_monotonic", int((np.diff(t_dev) <= 0).sum()), 0, s))
        seq = ep.times(s, "seq")
        seq_gaps = int(np.clip(np.diff(seq) - 1, 0, None).sum()) if seq is not None else 0
        expected = int(round(dur * cfg.rate_hz)) + 1
        drops = max(expected - n, seq_gaps)
        # one isolated drop is tolerated even in very short episodes (25 frames: 1 drop = 4 %)
        drop_thr = max(h.max_drop_ratio, 1.0 / max(expected, 1))
        checks.append(_cmp("drop_ratio", drops / max(expected, 1), drop_thr, s,
                           detail=f"{drops} dropped of ~{expected} (seq gaps {seq_gaps})"))
        # a burst = several consecutive periods missing; at 5 Hz one drop is already a 400 ms gap
        checks.append(_cmp("max_gap_ms", float(dt.max()), max(h.max_burst_gap_ms, 3.5 * period), s,
                           warn_thr=h.max_gap_factor * period))
        rate = (n - 1) / dur if dur > 0 else 0.0
        checks.append(_cmp("rate_deviation", abs(rate - cfg.rate_hz) / cfg.rate_hz, h.max_rate_deviation, s,
                           detail=f"measured {rate:.2f} Hz vs nominal {cfg.rate_hz}"))
        info = sync["streams"].get(s, {})
        checks.append(_cmp("clock_jumps", info.get("clock_jumps", 0), 0, s,
                           detail="device clock stepped relative to host clock"))
        # drift is judged only when the estimate is precise enough: slope noise ~ delay spread / duration
        # (4 ms jitter over a 1.7 s episode = ~2400 ppm of noise; 0.3 ms over 4 s = ~75 ppm)
        drift = abs(info.get("drift_ppm", 0.0))
        noise_ppm = 1000.0 * float(info.get("transport_delay_p95_ms") or 0.0) / max(dur, 1e-3)
        if dur >= h.min_drift_duration_s and noise_ppm < 0.5 * h.max_drift_ppm:
            checks.append(_cmp("drift_ppm", drift, h.max_drift_ppm, s, detail=f"estimate noise ~{noise_ppm:.0f} ppm"))
        else:
            checks.append(Check("drift_ppm", PASS, s, round(drift, 1), h.max_drift_ppm,
                                f"not judged: estimate noise ~{noise_ppm:.0f} ppm over {dur:.1f} s"))
        # content checks
        if cfg.kind == "rgb" and decode_video:
            vs = video_stats(ep.video_path(s), n, h)
            checks.append(_cmp("video_ts_mismatch", abs(vs["decoded"] - n), h.max_video_ts_mismatch, s,
                               detail=f"decoded {vs['decoded']} frames, {n} timestamps"
                               + (f"; decode error {vs['error']}" if vs["error"] else "")))
            checks.append(_cmp("corrupt_frames", vs["black"] + vs["glitches"] + (1 if vs["error"] else 0),
                               h.max_corrupt_frames, s, detail=f"black={vs['black']} glitch={vs['glitches']}"))
            checks.append(_cmp("frozen_run", vs["frozen_run"], h.max_frozen_run, s, detail="longest run of identical frames"))
            if vs["dark_run"]:
                checks.append(Check("dark_frames", WARN, s, vs["dark_run"], 0, "consecutive dark frames (occlusion?)"))
        elif cfg.kind == "depth":
            ds = depth_stats(ep.root[s]["data"])
            checks.append(_cmp("video_ts_mismatch", abs(ds["frames"] - n), h.max_video_ts_mismatch, s))
            checks.append(_cmp("corrupt_frames", ds["all_invalid"], h.max_corrupt_frames, s,
                               detail=f"mean invalid ratio {ds['invalid_ratio_mean']:.2f}"))
        elif ep.has(s, "data"):
            x = ep.data(s)
            checks.append(_cmp("nan_ratio", float((~np.isfinite(x)).any(axis=1).mean()), h.max_nan_ratio, s))
            if cfg.value_range:
                lo = np.array([r[0] for r in cfg.value_range])
                hi = np.array([r[1] for r in cfg.value_range])
                with np.errstate(invalid="ignore"):
                    oor = ((x < lo) | (x > hi)).any(axis=1).mean()
                checks.append(_cmp("out_of_range_ratio", float(oor), h.max_out_of_range_ratio, s))
            if cfg.kind in ("pose", "pose_cmd") and x.shape[1] == 7:
                qn = np.abs(np.linalg.norm(x[:, 3:7], axis=1) - 1)
                checks.append(_cmp("quaternion_norm_err_ratio", float((qn > 0.05).mean()), 0.01, s))

    # --- episode level -----------------------------------------------------------------
    if durations:
        dur = float(np.median(durations))
        checks.append(_cmp("duration_s", dur, h.min_duration_s, higher_is_bad=False))
        checks.append(_cmp("duration_s_max", dur, h.max_duration_s))
    if "residual_sync_error_p95_ms" in sync:
        checks.append(_cmp("sync_residual_p95_ms", sync["residual_sync_error_p95_ms"], h.max_sync_residual_p95_ms,
                           warn_thr=h.max_sync_residual_warn_ms,
                           detail=f"gt={sync.get('gt_sync_error_p95_ms')}"))
    xc = sync.get("xcorr") or {}
    if xc.get("peak_corr", 0) > 0.5 and abs(xc["lag_ms"]) > 0.5 * 1000 / robot.target_fps:
        # data-driven estimate (heuristic): flags a suspicious latency calibration, never quarantines
        checks.append(Check("motion_lag_ms", WARN, xc.get("camera"), round(xc["lag_ms"], 1),
                            round(0.5 * 1000 / robot.target_fps, 1),
                            f"camera motion vs robot motion, peak corr {xc['peak_corr']:.2f}"))
    if h.require_calibration != "ignore" and ep.meta.calibration_version == "uncalibrated":
        checks.append(Check("calibration", FAIL if h.require_calibration == "fail" else WARN, None,
                            ep.meta.calibration_version, "calibrated"))
    idle = _idle_ratio(ep, robot)
    if idle is not None:
        checks.append(_cmp("idle_ratio", idle, h.max_idle_ratio))

    statuses = [c.status for c in checks]
    status = FAIL if FAIL in statuses else WARN if WARN in statuses else PASS
    report = {
        "episode_id": ep.id, "robot_id": ep.meta.robot_id, "status": status,
        "failed": sorted({f"{c.check}:{c.stream}" if c.stream else c.check for c in checks if c.status == FAIL}),
        "warned": sorted({f"{c.check}:{c.stream}" if c.stream else c.check for c in checks if c.status == WARN}),
        "sync": {k: sync.get(k) for k in ("residual_sync_error_p95_ms", "gt_sync_error_p95_ms",
                                          "inter_camera_skew_p95_ms")},
        "checks": [asdict(c) for c in checks],
    }
    ep.write_sidecar("health.json", report)
    return report


def _idle_ratio(ep: RawEpisode, robot: RobotConfig) -> float | None:
    for s, cfg in robot.streams.items():
        if cfg.kind in ("joint_state", "pose") and s in ep.streams and ep.has(s, "data"):
            x = ep.data(s)
            if len(x) < 3 or x.shape[1] < 3:
                continue
            x = x[:, :3] if cfg.kind == "pose" else x[:, :-1] if x.shape[1] > 1 else x
            v = np.linalg.norm(np.diff(x, axis=0), axis=1) * cfg.rate_hz
            thr = 0.005 if cfg.kind == "pose" else 0.01  # m/s or rad/s
            return float((v < thr).mean())
    return None
