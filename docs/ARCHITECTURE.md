# Architecture

## Design principles

1. **Record raw, align later.** The recorder stores each stream at its native rate with *two* timestamps:
   the sensor/driver clock (`header.stamp`) and the recorder's receive time. Alignment is a separate,
   re-runnable stage. Better calibration or a new sync method never requires re-recording, and sync
   quality is measurable instead of being hidden by early resampling.
2. **One code path for live, replayed and simulated data.** The ROS 2 node, the open-source *virtual
   robot* and the sim converter all write through `EpisodeRecorder` → `RawEpisodeWriter`.
3. **Everything is configuration.** Robots (`configs/robots`), sources (`configs/sources`), health
   thresholds (`configs/health`) and dataset recipes (`configs/datasets`). Adding a robot means adding a
   YAML file.
4. **Never silently drop data.** Failing episodes are quarantined with a reason, not deleted. Curation
   exclusions are recorded in the catalog.
5. **Output a standard format.** The training set is LeRobot v3.0, written with LeRobot's own writer.
   robodata extras live under `meta/robodata/` and as ordinary features (`is_sim`, `phase_index`,
   `observation.sync_residual_ms`).

## Raw episode (recorder output)

```
data/raw/<robot_id>/<session_id>/<episode_id>/
  meta.json        episode_id, robot, task, operator, scene, session, is_sim, source, success,
                   sensor_config_hash, calibration_version, sync_mode, per-stream config + counts,
                   injected_faults (tests), extra (e.g. sim ground-truth phases)
  streams.zarr/    <stream>/{t_device, t_host, seq[, t_true], data}
  video/<cam>.mp4  RGB, frame i ↔ row i of <cam>/t_*
  sync.json        written by `robodata sync`/`health`: clock fits, grid, residuals, xcorr
  health.json      per-check results
```

Episode ids are `<robot>__<session>__ep<NNNNNN>`. Episodes are written to `*.partial` and renamed when
complete, so a crashed recording never looks valid.

## Open-source replay: the "virtual robot"

Public datasets ship already resampled to one clock, which hides every timing problem the pipeline must
handle. `robodata.ingest.virtual_robot` replays each source episode through the recorder and applies a
per-stream **clock model** from the robot YAML:

```
t_device = t_true·(1 + drift_ppm·1e-6) + offset + N(0, jitter)        # sensor clock
t_host   = t_true + latency + Exp(latency_jitter)  (FIFO per stream)  # transport, one-sided
dropped with probability drop_prob                                    # transport loss
```

`t_true` is kept, so every sync estimate can be scored against ground truth. The same model is applied to
sim data by the converter.

## Synchronisation (`robodata.sync`)

1. **Per-stream clock fit.** Pairs `(t_device, t_host)` satisfy `t_host − t_device = c + d·t + latency`
   with `latency ≥ L_min`. Least squares is biased by mean latency and its jitter. Instead, the
   **tightest line under all points** (a 2-variable LP, solved with HiGHS) gives offset and drift. This
   is the one-way-delay envelope idea from TICSync. `L_min` (exposure and driver delay) cannot be
   observed from timestamps, so the calibrated `known_latency_ms` is subtracted.
2. **Clock jumps.** A jump detector (running window minima, then median localisation) splits a stream
   into segments, each fitted separately.
3. **Grid alignment.** A uniform grid at the recipe fps covers the span where all required streams
   exist. It is phase-locked to the reference camera. Cameras take the nearest frame. Low-dim streams
   are interpolated: linear, slerp for quaternions, zero-order hold for commands (no look-ahead), or
   Butterworth low-pass then sample for high-rate F/T (anti-aliasing).
4. **Residual sync error per episode.** `residual_sync_error_p95_ms` is the worst camera's p95
   |frame − tick|. `inter_camera_skew_p95_ms` and `gt_sync_error_p95_ms` (against ground-truth clocks,
   when available) are also reported. The per-frame worst residual is shipped as
   `observation.sync_residual_ms`.
5. **Data-driven cross-check.** Wrist-camera motion energy is cross-correlated with end-effector speed.
   The lag shows mis-calibrated latencies even without ground truth. Validated in sim: about 1 ms with
   EE speed, against about 20 ms bias with joint-space speed. It is a *warning only*.

Hardware or software triggering is declared per robot (`sync.mode`). The `trigger` node publishes a
stamped tick for software-triggered rigs.

## Calibration (`robodata.calib`)

* Intrinsics: chessboard (`findChessboardCornersSB`) with `calibrateCamera`.
* Static extrinsics: `solvePnP` on a board at a known pose.
* Hand-eye (eye-in-hand): `calibrateHandEye` (Park), from flange poses and board observations.
* Board orientation is ambiguous (mirrored or 180° flipped corner order). It is resolved by trying all
  four orderings and keeping the one whose board normal faces the camera and whose first outer square
  is black. Without this, hand-eye errors were 90°.
* `robodata sim calibrate` renders views in the MuJoCo twin and scores every estimate against the exact
  model parameters. The result is stored as `configs/calibration/<robot>/<name>.yaml` with a content
  hash. Episodes reference it via `calibration_version`.

## Health checks (`robodata.health`)

| check | fails when | catches |
|---|---|---|
| `missing_modality` | a required stream is absent or empty | unplugged sensor, crashed driver |
| `drop_ratio` | > 2 % expected samples missing (time span and seq gaps) | dropped frames or messages |
| `max_gap_ms` | any gap > 250 ms (warn at 3× period) | burst drops, USB resets |
| `rate_deviation` | measured rate off nominal by > 10 % | wrong camera mode, throttling |
| `timestamp_non_monotonic` | device stamps go backwards | driver bugs |
| `clock_jumps` | device clock steps relative to host | NTP/PTP step, device reconnect |
| `drift_ppm` | fitted drift > 1000 ppm | broken clock, wrong time base |
| `video_ts_mismatch` | decoded frames ≠ timestamps | truncated or corrupt video, writer bugs |
| `corrupt_frames` | undecodable, isolated black frame, or isolated high-frequency spike (≥ 5× median Laplacian energy) | bit-rot, sensor glitches; all-invalid depth frames |
| `frozen_run` | > 15 near-identical consecutive frames | stalled camera |
| `nan_ratio`, `out_of_range_ratio`, `quaternion_norm` | invalid values | serialisation bugs, limits |
| `sync_residual_p95_ms` | > 15 ms (warn > 8) | unsynchronisable episode |
| `duration_s`, `idle_ratio` | too short or no motion | false starts |
| `calibration` (warn) | no calibration for the rig | unusable for 3D geometry |
| `motion_lag_ms`, `dark_frames` (warn) | heuristics | latency calibration, occluded lens |

The detectors were tuned on real footage: frozen and noise frames are separated by measured margins,
and isolation rules stop sharp-vs-blurry real footage being flagged. They were validated with the fault
injector (`robodata faults`): drops, bursts, clock jumps, drift, noise and black frames, frozen camera,
truncated video, missing modality, NaNs, depth holes, and near-duplicates (which must *pass* health and
be caught by dedup).

## Curation (`robodata.curation`)

* **Catalog** (`catalog.parquet`): every raw episode with its metadata, health, sync residual,
  smoothness, idle ratio, quality components, duplicate cluster, exclusion reason and split.
* **Quality** = health (1 / 0.85 / 0) × success (1 / 0.8 unknown / 0.2) × idle × smoothness, where
  smoothness is the robot-relative log dimensionless jerk. Episodes below `min_quality` are excluded.
* **Near-duplicates.** Both must match: DTW distance between z-scored EE trajectories (64 points), and
  mean pHash distance of 4 keyframes. Union-find clusters; the best-quality member of each is kept.
* **Splits.** By group (`session_id`: recording session for real data, scene for sim), stratified by
  `(robot, is_sim)`. Groups are ordered by hash with greedy deficit filling, so splits are
  deterministic and stable when data is added. Specific groups can be forced into eval (here a held-out
  sim scene). The leakage report checks that no group or duplicate cluster spans splits.

## Simulation (`robodata.sim`)

* The scene is built in code (MJCF): UR5 kinematics (Menagerie offsets), a held peg, a hole block, and
  front, base (`targetbody`) and wrist cameras. An F/T site is at the flange. There are 6 discrete
  randomised scenes (lighting, table colour, block placement, camera jitter); the block pose varies per
  episode.
* **Operator.** Min-jerk Cartesian segments (approach, align, insert, hold) with OU tremor, 80 ms
  reaction lag and per-operator speed/misalignment styles. A force-feedback "feel" stops pushing once
  axial contact force exceeds 10 N. Commands are smoothed (servoL-style), solved with damped
  least-squares IK at 125 Hz, interpolated at 500 Hz and tracked by position servos with gravity
  compensation.
* **Outputs.** Sensors at native rates (F/T 500 Hz, joints/EE/commands 125 Hz, teleop and cameras
  30 Hz), ground-truth phases and the success label (insertion depth and lateral error). Failure
  episodes arise naturally (about 27 %).
* **Native log → converter.** The mapping file renames streams, converts units (depth m → uint16 mm,
  quaternion wxyz → xyzw) and applies the sensor model (RGB read noise; RealSense-like depth noise
  ∝ z², 1 mm quantisation, dropouts). Without it, frozen-frame detection fires on perfectly static
  renders, which is a real sim-to-real gap the health checks surfaced.

## Build (`robodata.preprocess.build`)

Episodes are ordered train | val | eval and split into contiguous shards. The shards are written in
parallel with `LeRobotDataset.create/add_frame/save_episode` and merged with `aggregate_datasets`, so
LeRobot's `info.json["splits"]` ranges are exact. Depth uses LeRobot's native depth support
(`is_depth_map`, 12-bit lossless HEVC, millimetres).
