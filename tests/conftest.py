"""Synthetic robot + episodes so the test-suite runs without downloads, GPUs or ROS."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from robodata.config import load_robot
from robodata.recorder.clock import NS, stamp
from robodata.recorder.core import EpisodeRecorder

ROBOT = {
    "robot_id": "test_bot", "robot_type": "testarm", "target_fps": 30, "calibration_version": "test_bot/v0",
    "sync": {"mode": "free_run", "reference_stream": "cam", "clock_fit": "lower_envelope"},
    "streams": {
        "cam": {"kind": "rgb", "rate_hz": 30, "shape": [48, 64, 3], "dtype": "uint8", "known_latency_ms": 20,
                "clock": {"offset_ms": 5, "drift_ppm": 40, "jitter_ms": 0.2, "latency_ms": 20, "latency_jitter_ms": 3}},
        "wrist_cam": {"kind": "rgb", "rate_hz": 30, "shape": [48, 64, 3], "dtype": "uint8", "known_latency_ms": 25,
                      "clock": {"offset_ms": -3, "drift_ppm": -30, "jitter_ms": 0.2, "latency_ms": 25, "latency_jitter_ms": 3}},
        "depth": {"kind": "depth", "rate_hz": 30, "shape": [48, 64], "dtype": "uint16", "depth_unit": "mm",
                  "known_latency_ms": 22, "clock": {"offset_ms": 5, "drift_ppm": 40, "latency_ms": 22, "latency_jitter_ms": 3}},
        "joints": {"kind": "joint_state", "rate_hz": 100, "shape": [7], "known_latency_ms": 1,
                   "value_range": [[-7, 7]] * 6 + [[0, 1]], "clock": {"drift_ppm": 2, "latency_ms": 1, "latency_jitter_ms": 0.3}},
        "ee": {"kind": "pose", "rate_hz": 100, "shape": [7], "known_latency_ms": 1,
               "clock": {"drift_ppm": 2, "latency_ms": 1, "latency_jitter_ms": 0.3}},
        "ft": {"kind": "wrench", "rate_hz": 200, "shape": [6], "known_latency_ms": 1,
               "clock": {"offset_ms": 1, "drift_ppm": 10, "latency_ms": 1, "latency_jitter_ms": 0.3}},
        "action": {"kind": "joint_cmd", "rate_hz": 100, "shape": [7], "clock": {"latency_ms": 0.5}},
    },
    "features": {
        "observation.images.cam": {"stream": "cam"}, "observation.images.wrist_cam": {"stream": "wrist_cam"},
        "observation.images.cam_depth": {"stream": "depth"},
        "observation.state": {"stream": "joints"}, "observation.ee_pose": {"stream": "ee", "interp": "pose"},
        "observation.force_torque": {"stream": "ft", "interp": "lowpass_linear"},
        "action": {"stream": "action", "interp": "hold"},
    },
}


@pytest.fixture(scope="session")
def robot_yaml(tmp_path_factory):
    p = tmp_path_factory.mktemp("cfg") / "test_bot.yaml"
    p.write_text(yaml.safe_dump(ROBOT))
    return p


@pytest.fixture(scope="session")
def robot(robot_yaml):
    return load_robot(robot_yaml)


def _trajectory(t, amp=1.0):
    """Approach -> descend with contact -> hold (amp scales the motion: a different demo)."""
    x = np.where(t < 2, 0.3 * amp * (t / 2), 0.3 * amp)
    z = np.where(t < 2, 0.3, np.where(t < 3.2, 0.3 - 0.2 * amp * (t - 2) / 1.2, 0.3 - 0.2 * amp))
    return x, z


def _image(t, rng):
    yy, xx = np.mgrid[0:48, 0:64]
    cx = 10 + 40 * min(t / 2, 1.0)
    img = np.stack([100 + 80 * np.exp(-((xx - cx) ** 2 + (yy - 24) ** 2) / 60), 90 + xx, 120 + yy * 0.5], -1)
    return np.clip(img + rng.normal(0, 6.0, img.shape), 0, 255).astype(np.uint8)  # survives H.264 at 64x48


def make_episode(robot, raw_root, index=0, session="s0", seed=0, duration=4.0, shift=0.0, amp=1.0):
    rng = np.random.default_rng(seed)
    rec = EpisodeRecorder(robot, raw_root, session_id=session, operator="tester", scene_id=f"scene_{session}")
    rec.start(task="test insertion", episode_index=index, success=True, has_ground_truth_clock=True)
    epoch = 1_700_000_000 * NS + index * 100 * NS
    for name, cfg in robot.streams.items():
        n = int(duration * cfg.rate_hz)
        t = np.arange(n) / cfg.rate_hz
        t_true = (t * NS).astype(np.int64)
        t_dev, t_host, _ = stamp(t_true, cfg.clock, rng, epoch_ns=epoch)
        x, z = _trajectory(t + shift, amp)
        for i in range(n):
            if cfg.kind == "rgb":
                v = _image(t[i] + shift, rng)
            elif cfg.kind == "depth":
                v = np.full((48, 64), 800, np.uint16) + rng.integers(0, 5, (48, 64)).astype(np.uint16)
            elif cfg.kind in ("joint_state", "joint_cmd"):
                v = np.array([x[i], z[i], 0.1, 0.2, 0.3, 0.4, 1.0], np.float32)
            elif cfg.kind == "pose":
                v = np.array([x[i], 0.0, z[i], 0, 0, 0, 1], np.float32)
            else:
                f = 5.0 if 3.0 < t[i] + shift < 3.4 else 0.0
                v = np.array([0, 0, -2 - f, 0, 0, 0], np.float32) + rng.normal(0, 0.05, 6).astype(np.float32)
            rec.on_message(name, int(t_dev[i]), v, t_host_ns=int(t_host[i]), t_true_ns=int(t_true[i] + epoch), seq=i)
    return rec.stop()


@pytest.fixture(scope="session")
def clean_episode(robot, tmp_path_factory):
    return make_episode(robot, tmp_path_factory.mktemp("raw"))
