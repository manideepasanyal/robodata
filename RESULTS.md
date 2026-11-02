# Results from the run in this repository

Everything below was produced by the code in this repo. Reports are in `data/reports/`, datasets in
`data/datasets/` and version manifests in `data/versions/`.

## Data volumes

| stage | contents | size |
|---|---|---|
| sources (pinned HF revisions) | catheter (100 eps), droid_100 (100 eps) | 0.8 GB |
| sim native logs | 60 MuJoCo episodes (6 scenes × 10, 2 operator styles) | 3.7 GB |
| raw (recorder format) | 100 catheter + 100 DROID + 60 sim + 47 fault-injected copies | 7.1 GB |
| **`ur5_insertion_mixed`** v`c8c29b21d775f7cf` | 144 eps / 55,605 frames @ 30 Hz: 100 real + 44 sim, 3 RGB + 3 depth + state, EE pose, F/T, teleop, action | 2.26 GB |
| `ur5_insertion_mixed_rgb` v`742c5c4a294e0b1d` | derived RGB-only view, for policies with 3-channel backbones | 0.36 GB |
| **`franka_droid100`** v`e8d6ad2c637dac25` | 98 eps @ 15 Hz, 3 RGB + Cartesian state, gripper, velocity action | 0.44 GB |

## 1. Capture (ROS 2)

ROS 2 end-to-end test in Docker (Jazzy): a raw sim episode is replayed as live topics in real time,
recorded by `robodata_ros/recorder`, then synced and health-checked.

| | result |
|---|---|
| messages published / recorded | 10,085 / 10,083 (2 best-effort camera frames lost in DDS) |
| health of the ROS-recorded episode | **warn** (no failures); residual sync p95 **1.73 ms**, inter-camera skew p95 1.68 ms |
| receive latency, Python recorder vs C++ `stamp_monitor` | joint_states p50 1.55 vs 1.84 ms · F/T 1.43 vs 1.57 ms · wrist image 26.4 vs 26.7 ms |
| C++ `stamp_monitor` | 4,647/4,647 F/T messages at 499 Hz, 1,162/1,162 joint states, zero deserialisation |

Real recorder bugs this test surfaced and fixed:
1. `now()` in Python callbacks stamped 125/500 Hz messages up to **~770 ms late** behind image
   callbacks. Fixed by using the middleware `received_timestamp`.
2. A 500 Hz F/T stream lost **627/4647** messages to queue overflow. Fixed with queue depth of about
   4 s of messages per stream.
3. A reentrant callback group **reordered depth frames** (non-monotonic timestamps). Fixed with one
   mutually-exclusive group per stream.
4. Image encoding moved off the callback thread, to an ordered worker per stream.

## 2. Synchronisation and calibration

Clock models applied to every replayed stream: offsets up to 11 ms, drift up to ±60 ppm, latency
1–41 ms with exponential jitter up to 6 ms, and random drops. Errors are measured against the
ground-truth capture time:

| robot | clock error vs ground truth, p95 per episode (median / worst) | grid residual p95 (median / worst) |
|---|---|---|
| ur5_catheter (real, replayed) | 0.65 / 1.52 ms | 1.49 / 2.67 ms |
| ur5_sim | 1.12 / 1.55 ms | 2.22 / 2.96 ms |
| franka_droid (real, replayed) | 0.24 / 1.19 ms | 0.83 / 2.63 ms |

* The lower-envelope fit beats least squares by more than 4× in median error under one-sided latency
  jitter (`tests/test_sync.py`).
* **Data-driven finding:** in the *source* catheter recordings, wrist-camera motion lags end-effector
  motion by about 17 ms. This is measured with ground-truth replay clocks, so it is a property of the
  original recording, about half a frame. The same estimator gives about 1 ms on sim, where the truth
  is 0. It is shipped as a warning in the health report.

Simulated camera calibration (`configs/calibration/ur5_sim/v1.yaml`), scored against the MuJoCo model:

| camera | reprojection RMS | intrinsics error | extrinsics / hand-eye error |
|---|---|---|---|
| wrist (eye-in-hand) | 0.045 px (65 views) | fx 0.88 %, cx 0.4 px | hand-eye **2.0 mm / 1.8°** |
| base | 0.148 px (33 views) | fx 0.40 %, cx 2.2 px | **2.1 mm / 0.55°** |
| front | 0.575 px (20 views) | fx 4.9 %, cy 11 px | 30 mm / 2.2° (weakest: distant camera, small board, few detections) |

## 3. Health checks

307 raw episodes were checked: 260 clean and 47 fault-injected copies.

| injected fault | episodes | detected | by check |
|---|---:|---:|---|
| dropped frames (10 % random, 8 % burst) | 8 | 8 | drop_ratio, max_gap_ms, rate_deviation, sync_residual |
| timestamp jump (+80 ms step) | 4 | 4 | clock_jumps |
| clock drift (3000 ppm) | 4 | 4 | drift_ppm |
| corrupt frames (noise, black, depth holes) | 11 | 11 | corrupt_frames |
| frozen camera (45–60 frames) | 4 | 4 | frozen_run |
| truncated video | 4 | 4 | video_ts_mismatch |
| missing modality | 4 | 4 | missing_modality |
| NaN values | 4 | 4 | nan_ratio |
| near-duplicate (must *pass* health) | 4 | 0 (correct) | caught 4/4 by dedup instead |

* **43/43 real faults detected. False alarms: 2/260 clean episodes**, and both are legitimate. One is a
  short DROID episode whose replay clock model dropped more than 2 % of camera frames. The other is a
  1.4 s idle DROID episode. 45 episodes were quarantined.
* The detectors were tuned on real footage. The first version produced 15 false alarms: occluded
  wrist cameras, blurry-vs-sharp frames, and noise-free sim renders that looked frozen. All of them are
  now handled by measured rules, described in `docs/ARCHITECTURE.md`.

## 4. Curation and splits

| dataset | in | selected | excluded | duplicate pairs | leakage |
|---|---:|---:|---|---:|---|
| ur5_insertion_mixed | 163 | 144 | 16 low-quality (failed sim demos), 3 near-duplicates | 3 (all injected) | none |
| franka_droid100 | 99 | 98 | 1 near-duplicate | 1 (injected) | none |

`ur5_insertion_mixed` splits (by session; sim session = scene; `sim_scene_05` forced into eval):

| split | real sessions | sim scenes | real eps | sim eps |
|---|---|---|---:|---:|
| train | catheter_s000–003, s005–008 | sim_scene_00–03 | 80 | 29 |
| val | catheter_s009 | sim_scene_04 | 10 | 7 |
| eval | catheter_s004 | sim_scene_05 (held-out scene) | 10 | 8 |

## 5. Synthetic data

* 60 episodes: careful operator 27/30 successes, fast operator 17/30 (73 % overall). Failures happen
  naturally when lateral misalignment exceeds about 3 mm and the peg lands on the block top.
* Same schema as the real UR5 data (stream names, feature keys, units, quaternion convention, depth in
  mm). `is_sim` is set on every frame.
* **Mixing verified.** The training smoke test samples a requested 30 % sim fraction and measures
  **30.06 %** per batch. Validation loss is reported per domain.

## 6. Preprocessing

* Phase segmentation vs the simulator's ground-truth operator phases (44 sim episodes, median absolute
  error): approach→align boundary **0.21 s**, settle boundary **0.40 s**. The contact boundary is
  defined as *force onset*, which differs by design from the sim's "insert" phase start (12 mm above
  the hole; median offset 2.7 s). Contact was detected from F/T in 140/144 episodes, and from descent
  in the other 4.
* Inspection report: `data/reports/inspect/ur5_insertion_mixed/index.html` (contact sheets with depth,
  signal plots with phase shading and sync residual, synced mosaic videos).

## 7. Training-loop compatibility

| test | result |
|---|---|
| `robodata train-smoke ur5_insertion_mixed` (LeRobotDataset, 3 RGB + 3 depth + state + F/T, 10-step action chunks) | loss 0.347 → 0.0037 in 300 steps · val real 0.0020 / sim 0.0102 · 154 samples/s |
| `lerobot-train --policy.type=act` on `ur5_insertion_mixed_rgb` (stock LeRobot, unmodified) | runs, loss 26.6 → 5.66 in 60 steps |
| `robodata train-smoke franka_droid100` | loss 0.42 → 0.32 in 150 steps · 742 samples/s |

## 8. Storage

See `docs/STORAGE_AND_ENCODING.md` for the Parquet/Zarr/HDF5 and codec/CRF/GOP measurements.

A dataset-integrity bug was found and fixed. LeRobot's *streaming* encoder drops frames when its queue
is full, which is reasonable for live capture. In an offline build it silently removed about **5 % of
depth frames** and shifted episode boundaries. The inspection contact sheet showed sim depth at the end
of a real episode. The build now uses batch encoding, and every build runs `verify_videos` (packet
counts against episode lengths and segment durations), failing if anything is off.
