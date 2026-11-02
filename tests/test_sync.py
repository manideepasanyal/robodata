import numpy as np

from robodata.config import ClockModel
from robodata.io.raw_episode import RawEpisode
from robodata.recorder.clock import NS, stamp
from robodata.sync.align import interp_hold, interp_lowpass, interp_pose, make_grid, nearest_index
from robodata.sync.clock_fit import find_timestamp_jumps, fit_clock
from robodata.sync.run import sync_episode


def test_lower_envelope_recovers_drift_and_beats_least_squares():
    rng = np.random.default_rng(0)
    t_true = (np.arange(0, 60, 1 / 30) * NS).astype(np.int64)
    clock = ClockModel(offset_ms=12, drift_ppm=80, jitter_ms=0.1, latency_ms=20, latency_jitter_ms=8)
    t_dev, t_host, _ = stamp(t_true, clock, rng)
    env = fit_clock(t_dev, t_host, "lower_envelope", known_latency_ms=20)
    ls = fit_clock(t_dev, t_host, "least_squares", known_latency_ms=20)
    assert abs(env.drift_ppm - 80) < 15
    err_env = np.abs(env.to_host(t_dev) - t_true) / 1e6
    err_ls = np.abs(ls.to_host(t_dev) - t_true) / 1e6
    assert np.percentile(err_env, 95) < 1.0  # sub-millisecond despite 8 ms exponential latency jitter
    assert np.median(err_ls) > 4 * np.median(err_env)  # least squares is biased by mean latency


def test_timestamp_jump_detected():
    t_true = (np.arange(0, 10, 1 / 30) * NS).astype(np.int64)
    t_dev, t_host, _ = stamp(t_true, ClockModel(latency_ms=10, latency_jitter_ms=2), np.random.default_rng(1))
    t_dev[150:] += 80 * 1_000_000
    jumps = find_timestamp_jumps(t_dev, t_host, 20)
    assert len(jumps) == 1 and abs(jumps[0] - 150) <= 2


def test_alignment_primitives():
    t = (np.arange(0, 2, 0.01) * NS).astype(np.int64)
    grid = make_grid([int(t[0])], [int(t[-1])], 30)
    idx, res = nearest_index(t, grid)
    assert res.max() <= 5.0 + 1e-6
    cmd = np.arange(len(t), dtype=np.float32)[:, None]
    held = interp_hold(t, cmd, grid)
    assert np.all(t[held[:, 0].astype(int)] <= grid)  # zero-order hold never looks into the future
    q = np.tile([0, 0, 0, 1.0], (len(t), 1))
    q[len(t) // 2:] *= -1  # sign flip = same rotation; slerp must not spin
    out = interp_pose(t, np.concatenate([np.zeros((len(t), 3)), q], 1), grid)
    assert np.allclose(np.abs(out[:, 6]), 1, atol=1e-5)
    hi = (np.arange(0, 2, 1 / 500) * NS).astype(np.int64)
    sig = np.sin(2 * np.pi * 120 * hi / NS)[:, None]  # 120 Hz vibration must not alias into a 30 Hz grid
    out = interp_lowpass(hi, sig, make_grid([int(hi[0])], [int(hi[-1])], 30), 30)
    assert np.abs(out[3:-3]).max() < 0.03  # > 30 dB attenuation away from the filtfilt edge transients


def test_sync_episode_ground_truth(robot, clean_episode):
    rep = sync_episode(RawEpisode(clean_episode), robot, do_xcorr=False)
    assert rep["gt_sync_error_p95_ms"] < 2.0
    assert rep["residual_sync_error_p95_ms"] < 5.0
    assert all(s.get("clock_jumps", 0) == 0 for s in rep["streams"].values())
