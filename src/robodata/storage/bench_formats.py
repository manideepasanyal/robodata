"""Parquet vs Zarr vs HDF5 for multimodal low-dim time series (state, action, F/T, poses, timestamps).

Measures what matters for robot learning: size on disk, write time, full-episode read,
random-window read (a training sampler pulling 64-step chunks), and column projection
(reading one modality only).
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from robodata.config import load_robot
from robodata.io.raw_episode import RawEpisode, list_episodes


def _episode_tables(raw_root, n: int) -> list[dict[str, np.ndarray]]:
    eps = []
    for robot_id in ("ur5_sim", "ur5_catheter"):
        for p in [p for p in list_episodes(raw_root, robot_id) if "__fault-" not in p.name][:n]:
            ep = RawEpisode(p)
            robot = load_robot(ep.meta.robot_id)
            cols = {}
            for s, cfg in robot.streams.items():
                if cfg.kind not in ("rgb", "depth") and s in ep.streams:
                    cols[f"{s}/t"] = ep.times(s)
                    cols[f"{s}/data"] = ep.data(s)
            eps.append(cols)
    return eps


def _dir_size(p: Path) -> int:
    return p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def run_format_bench(raw_root, out_dir, n_episodes: int = 3, window: int = 64, reps: int = 200) -> str:
    out_dir = Path(out_dir)
    work = out_dir / "fmt_tmp"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    eps = _episode_tables(raw_root, n_episodes)
    rng = np.random.default_rng(0)
    rows = []
    raw_bytes = sum(v.nbytes for e in eps for v in e.values())

    # ---- writers ---------------------------------------------------------------------
    def w_parquet():
        for i, e in enumerate(eps):  # one table per stream (multi-rate), fixed-size list columns
            for k in {k.split("/")[0] for k in e}:
                t, d = e[f"{k}/t"], e[f"{k}/data"]
                tbl = pa.table({"t": t, "data": pa.FixedSizeListArray.from_arrays(pa.array(d.ravel()), d.shape[1])})
                pq.write_table(tbl, work / f"pq_{i}_{k}.parquet", compression="zstd")

    def w_zarr():
        for i, e in enumerate(eps):
            g = zarr.open_group(str(work / f"ep{i}.zarr"), mode="w")
            for k, v in e.items():
                g.create_array(k, data=v, chunks=(min(len(v), 1024),) + v.shape[1:])

    def w_hdf5(comp):
        def f():
            with h5py.File(work / f"all_{comp}.h5", "w") as h:
                for i, e in enumerate(eps):
                    for k, v in e.items():
                        h.create_dataset(f"ep{i}/{k}", data=v, chunks=(min(len(v), 1024),) + v.shape[1:],
                                         compression=comp, shuffle=True)
        return f

    # ---- readers -----------------------------------------------------------------------
    def r_full(fmt):
        if fmt == "parquet":
            return lambda i, k: np.stack(pq.read_table(work / f"pq_{i}_{k}.parquet", columns=["data"]).column(0).to_numpy(zero_copy_only=False))
        if fmt == "zarr":
            return lambda i, k: zarr.open_group(str(work / f"ep{i}.zarr"), mode="r")[f"{k}/data"][:]
        comp = fmt.split("_")[1]
        return lambda i, k: h5py.File(work / f"all_{comp}.h5", "r")[f"ep{i}/{k}/data"][:]

    def r_window(fmt):
        if fmt == "parquet":
            def f(i, k, a):
                pf = pq.ParquetFile(work / f"pq_{i}_{k}.parquet")
                return pf.read(columns=["data"]).slice(a, window)  # row groups are whole-episode here
            return f
        if fmt == "zarr":
            return lambda i, k, a: zarr.open_group(str(work / f"ep{i}.zarr"), mode="r")[f"{k}/data"][a:a + window]
        comp = fmt.split("_")[1]
        return lambda i, k, a: h5py.File(work / f"all_{comp}.h5", "r")[f"ep{i}/{k}/data"][a:a + window]

    formats = {"parquet": w_parquet, "zarr": w_zarr, "hdf5_gzip": w_hdf5("gzip"), "hdf5_lzf": w_hdf5("lzf")}
    for fmt, writer in formats.items():
        t0 = time.perf_counter()
        writer()
        wt = time.perf_counter() - t0
        size = sum(_dir_size(p) for p in work.iterdir() if p.name.startswith(
            {"parquet": "pq_", "zarr": "ep", "hdf5_gzip": "all_gzip", "hdf5_lzf": "all_lzf"}[fmt]))
        streams = sorted({k.split("/")[0] for k in eps[0]})
        rf, rw = r_full(fmt), r_window(fmt)
        t0 = time.perf_counter()
        for i in range(len(eps)):
            for k in streams:
                if f"{k}/data" in eps[i]:
                    rf(i, k)
        full = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(reps):
            i = int(rng.integers(len(eps)))
            k = "action" if "action/data" in eps[i] else streams[0]
            n = len(eps[i][f"{k}/data"])
            rw(i, k, int(rng.integers(0, max(1, n - window))))
        win = (time.perf_counter() - t0) / reps
        rows.append({"format": fmt, "size_MB": size / 1e6, "compression_x": raw_bytes / size, "write_s": wt,
                     "full_read_s": full, "random_window_read_ms": win * 1000})
    shutil.rmtree(work)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "format_bench.csv", index=False)
    md = ["# Storage format benchmark (low-dim multimodal time series)", "",
          f"{len(eps)} raw episodes (sim 500 Hz F/T + 125 Hz joints, real 30 Hz), {raw_bytes / 1e6:.1f} MB uncompressed; "
          f"random window = {window} consecutive samples of one stream, {reps} reps.", "", df.round(3).to_markdown(index=False)]
    (out_dir / "format_bench.md").write_text("\n".join(md) + "\n")
    return str(out_dir / "format_bench.md")
