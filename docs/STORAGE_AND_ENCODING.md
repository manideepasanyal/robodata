# Storage formats, video encoding and versioning: decisions and measurements

All numbers come from `robodata bench formats` and `robodata bench video`, run on this
repository's data (`data/reports/bench/*.md` hold the full tables).

## 1. Two storage layers, two different jobs

| layer | written by | format | why |
|---|---|---|---|
| **raw** (`data/raw/<robot>/<session>/<episode>/`) | recorder (ROS 2 node, virtual robot, sim converter) | **Zarr v3** for timestamps, low-dim and depth (`streams.zarr`), **H.264 MP4** for RGB, `meta.json` | multi-rate, unsynchronised, append-heavy; must survive crashes; re-syncable |
| **training** (`data/datasets/<name>/`) | `robodata build` via LeRobot's writer | **LeRobot v3.0**: Parquet for tabular frames, AV1 MP4 for RGB, 12-bit lossless HEVC for depth | fixed-rate, aligned, what a training loop reads; one ecosystem-standard format |

### Parquet vs Zarr vs HDF5 for multimodal time series

Measured on 50 raw episodes (sim: 500 Hz F/T and 125 Hz joints; real: 30 Hz). The figures cover low-dim
streams only: 8.7 MB uncompressed in total.

| format | size MB | compression | write s | full read s | random 64-sample window ms |
|---|---:|---:|---:|---:|---:|
| Parquet (zstd, one table per stream) | 9.08 | 0.92x | 0.13 | 0.27 | 0.31 |
| Zarr v3 (zstd, 1024-row chunks) | 6.23 | 1.34x | 1.79 | 0.67 | 2.36 |
| HDF5 gzip (shuffle) | 6.58 | 1.26x | 0.17 | 0.07 | 0.24 |
| HDF5 lzf | 7.13 | 1.17x | 0.10 | 0.06 | 0.21 |

How to read this:
* **Low-dim data is tiny.** A whole episode's state, action, F/T and poses take less space than one second
  of video, so size does not decide the format. Access patterns, robustness and ecosystem do.
* **HDF5** reads fastest from local disk, but it is one file per writer. Concurrent or parallel writes
  and partial writes are fragile, because a crash mid-write can corrupt the whole file. Object-storage
  access is also poor. We use it only for the **native simulator log**, following the robomimic and ACT
  convention (one process, written once).
* **Zarr** has the best compression and writes chunk by chunk to independent objects, so a crashed
  recording loses only its last chunk. It supports concurrent writers (one per sensor callback), ragged
  multi-rate groups and cloud object stores. Small random reads are slower here because of
  per-chunk-file overhead on local disk. That is irrelevant for recording, which is append-only. This
  is why Zarr is the **raw recording format**, as in the Diffusion Policy replay buffer. Raw episodes
  are published atomically (`*.partial` → rename).
* **Parquet** is columnar with zero-copy Arrow, has row-group random access and is the native table
  format of LeRobot and HF `datasets`. It is somewhat larger on noisy float sensor data, but it is
  what the **training** format must be for "loads in a LeRobot-style training loop" with no
  conversion.

## 2. Video encoding (RGB and depth)

2 episodes (1 real, 1 sim), 300 wrist frames each. PSNR and SSIM are measured against the raw archive
frames. *Random access* is the median latency to seek to and decode one random frame, which is what a
shuffled data loader pays per sample.

| config | kB/frame | compression | PSNR dB | SSIM | enc fps | dec fps | random access ms | all frames intact |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| H.264 crf18 g30 (raw archive) | 3.63 | 254x | 43.9 | 0.993 | 315 | 3458 | 2.36 | yes |
| H.264 crf23 g30 | 2.05 | 451x | 41.5 | 0.989 | 379 | 3702 | 2.35 | yes |
| H.264 crf28 g30 | 1.04 | 890x | 38.9 | 0.982 | 410 | 5007 | 1.98 | yes |
| H.264 crf23 **g2** | 5.95 | 155x | 41.1 | 0.988 | 406 | 3770 | **1.37** | yes |
| HEVC crf28 g30 | **0.61** | 1523x | 39.8 | 0.984 | 293 | 4135 | 4.19 | yes |
| AV1 crf30 g2 (LeRobot default) | 5.30 | 174x | 41.6 | 0.987 | 84 | 1853 | 2.05 | yes |
| AV1 crf30 g30 | 1.15 | 799x | 41.4 | 0.987 | 387 | 2156 | 5.11 | yes |
| AV1 crf40 g30 | 0.66 | 1405x | 40.3 | 0.985 | 352 | 2287 | 3.40 | yes |
| FFV1 lossless (yuv444) | 60.3 | 15x | 53.2* | 0.998 | 212 | 1287 | 5.72 | yes |

\* FFV1 is lossless; the residual error comes from the RGB → YUV444 conversion.

| depth (uint16 mm) | kB/frame | compression | MAE mm | enc fps | dec fps | random access ms |
|---|---:|---:|---:|---:|---:|---:|
| Zarr + zstd (raw store) | 33.4 | 18x | 0 | – | – | chunk read |
| FFV1 gray16le lossless | 23.3 | 26x | 0 | 507 | 1995 | 4.9 |
| HEVC gray12 lossless (LeRobot depth) | 29.7 | 21x | 0 (≤ 4.095 m) | 135 | 2506 | 6.7 |
| HEVC gray12 crf20 | 1.7 | 366x | 4.2 | 268 | 3506 | 5.1 |

Decisions:
* **Keyframe interval (GOP) is the main lever for training I/O.** g=2 roughly halves random-access
  latency compared with g=30, at 3–5x the size. LeRobot's default (AV1, crf 30, g=2) is a sensible
  training-time choice. For **archival raw** we use H.264 crf18 g30, which is 13x smaller per quality
  point than g=2 and far faster to encode live on a robot PC.
* **AV1 has the best size/quality at equal GOP** (crf30 g30: 41.4 dB at 1.15 kB/frame). Its encode cost
  only matters with short GOPs.
* **Depth must stay lossless or near-lossless.** HEVC crf20 saves 17x but adds about 4 mm MAE, with
  much larger errors at depth edges. We keep lossless Zarr in raw and LeRobot's lossless 12-bit HEVC in
  the training set, which caps range at 4.095 m in mm units.
* **Frame integrity** is verified for every configuration: decoded frame count equals input count, and
  the health checks re-verify it per episode (`video_ts_mismatch`).

## 3. Versioning and reproducibility

* `robodata version <dataset>` writes `data/versions/<dataset>/<version>.json`. The manifest holds
  SHA-256 hashes of every file, a root hash (the version id) and the **lineage**: pinned HF source
  revisions (`data/sources/lock.json`), hashes of every config file and the code version (git commit,
  or a source-tree hash).
* `robodata verify <dataset> [--version v]` re-hashes the files on disk and reports missing, extra and
  changed files. Experiments log the version id, so a result can be traced back to its exact bytes.
* **DVC** provides remote storage and deduplication on top of the same directories:

  ```bash
  git init && dvc init
  dvc add data/datasets/ur5_insertion_mixed      # creates data/datasets/ur5_insertion_mixed.dvc
  dvc remote add -d store s3://bucket/robodata && dvc push
  git add data/datasets/*.dvc data/versions && git commit -m "dataset ur5_insertion_mixed@<version>"
  ```
  The robodata manifest stays useful under DVC because it also records *how* the bytes were produced.
