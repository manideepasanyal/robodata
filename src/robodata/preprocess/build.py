"""Build a training-ready LeRobot (v3.0) dataset from curated raw episodes.

Per episode: clock-corrected times (sync.json) -> fixed-rate grid at recipe fps ->
cameras by nearest frame, low-dim by the configured interpolation -> resize -> phase
segmentation -> written with LeRobot's own writer (so the output loads in any
LeRobot-style training loop). Episodes are written ordered train | val | eval so the
standard ``info.json["splits"]`` ranges are exact; robodata metadata goes to
``meta/robodata/``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from robodata import schema
from robodata.config import DatasetRecipe, load_robot
from robodata.io.raw_episode import RawEpisode
from robodata.io.video import iter_frames
from robodata.preprocess.segment import PHASE_NAMES, boundary_errors, segment_phases
from robodata.sync.align import make_grid, nearest_index, resample
from robodata.sync.run import corrected_times, sync_episode

SPLIT_ORDER = ["train", "val", "eval"]


class SequentialFrames:
    """Random access by non-decreasing index over a sequential decoder (no full-episode buffering)."""

    def __init__(self, path: Path):
        self.it = iter_frames(path)
        self.i, self.cur = -1, None

    def get(self, idx: int) -> np.ndarray:
        while self.i < idx:
            self.cur = next(self.it)
            self.i += 1
        return self.cur


def lerobot_features(recipe: DatasetRecipe) -> dict:
    h, w = recipe.image_size
    feats = {}
    for cam in recipe.cameras:
        feats[schema.image_key(cam)] = {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
    for cam in recipe.depth_cameras:
        feats[schema.depth_key(cam)] = {"dtype": "video", "shape": (h, w, 1), "names": ["height", "width", "channels"],
                                        "info": {"is_depth_map": True}}
    for key, spec in recipe.features.items():
        feats[key] = {"dtype": "float32", "shape": (len(spec["names"]),), "names": list(spec["names"])}
    feats[schema.IS_SIM] = {"dtype": "bool", "shape": (1,), "names": None}
    feats[schema.SYNC_RESIDUAL] = {"dtype": "float32", "shape": (1,), "names": None}
    if recipe.phases:
        feats[schema.PHASE] = {"dtype": "int64", "shape": (1,), "names": None}
    return feats


def align_episode(ep: RawEpisode, recipe: DatasetRecipe) -> dict:
    """Everything except image pixels, on the training grid."""
    robot = load_robot(ep.meta.robot_id)
    sync = ep.sidecar("sync.json") or sync_episode(ep, robot, do_xcorr=False)
    t_est = {s: corrected_times(ep, s, sync) for s in sync["streams"] if sync["streams"][s].get("segments")}
    req = [s for s in robot.required_streams if s in t_est]
    grid = make_grid([int(t_est[s][0]) for s in req], [int(t_est[s][-1]) for s in req], recipe.fps,
                     phase_ref_ns=t_est[robot.sync.reference_stream])
    out = {"grid": grid, "lowdim": {}, "frame_idx": {}, "residual": {}}
    for key, fm in robot.features.items():
        s = fm.stream
        kind = robot.streams[s].kind
        if kind in ("rgb", "depth"):
            idx, res = nearest_index(t_est[s], grid)
            out["frame_idx"][key] = idx
            out["residual"][key] = res
        elif key in recipe.features:
            out["lowdim"][key] = resample(fm.interp, t_est[s], ep.data(s), grid, recipe.fps, fm.angle_dims)
    out["sync_residual_ms"] = (np.max(np.stack(list(out["residual"].values())), 0) if out["residual"]
                               else np.zeros(len(grid))).astype(np.float32)
    # phases from EE position (+ force if present)
    ee_key = next((k for k, f in robot.features.items() if robot.streams[f.stream].kind == "pose"), None)
    ft_key = next((k for k, f in robot.features.items() if robot.streams[f.stream].kind == "wrench"), None)
    t_s = (grid - grid[0]) / 1e9
    if ee_key and ee_key in out["lowdim"]:
        phase, info = segment_phases(t_s, out["lowdim"][ee_key][:, :3].astype(np.float64),
                                     out["lowdim"].get(ft_key) if ft_key else None)
    else:
        phase, info = np.zeros(len(grid), np.int64), {"b": []}
    out["phase"], out["phase_info"] = phase, info
    gt = ep.meta.extra.get("gt_phase_change_s")
    if gt and info["b"]:
        t_true0 = ep.times(robot.sync.reference_stream, "t_true")
        if t_true0 is not None:
            rel = (grid - t_true0[0]) / 1e9
            out["phase_eval"] = boundary_errors(rel, info["b"], gt)
    return out


def _resize(img: np.ndarray, hw: list[int], depth: bool = False) -> np.ndarray:
    h, w = hw
    if img.shape[:2] == (h, w):
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST if depth else cv2.INTER_AREA)


def _create(recipe: DatasetRecipe, root: Path, repo_id: str):
    from lerobot.configs.video import DepthEncoderConfig, RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset.create(
        repo_id=repo_id, fps=recipe.fps, features=lerobot_features(recipe), root=root,
        robot_type=recipe.robot_type, use_videos=True,
        rgb_encoder=RGBEncoderConfig(**recipe.rgb_encoder),
        # closed GOP, no B-frames: with x265's default open GOP, a seek can drop leading pictures and the
        # requested frame becomes undecodable (0.17 % of random accesses failed before this change)
        depth_encoder=DepthEncoderConfig(**{"extra_options": {"x265-params": "lossless=1:open-gop=0:bframes=0"},
                                            **recipe.depth}) if recipe.depth_cameras else None,
        # Batch (not streaming) encoding: LeRobot's streaming encoder drops frames under back-pressure,
        # which is right for live capture but silently corrupts an offline build (caught by verify_videos).
        image_writer_threads=4, streaming_encoding=False)


def _write_shard(recipe_json: str, shard: list[tuple[str, str, str]], root: str, repo_id: str) -> list[dict]:
    """Write (split, episode_id, raw_path) episodes, in order, into one LeRobot dataset."""
    recipe = DatasetRecipe.model_validate_json(recipe_json)
    ds = _create(recipe, Path(root), repo_id)
    rgb_keys = {schema.image_key(c) for c in recipe.cameras}
    depth_keys = {schema.depth_key(c) for c in recipe.depth_cameras}
    records = []
    for local, (split, eid, path) in enumerate(shard):
        ep = RawEpisode(path)
        robot = load_robot(ep.meta.robot_id)
        al = align_episode(ep, recipe)
        n = len(al["grid"])
        readers, depth_arrays = {}, {}
        for key, fm in robot.features.items():
            kind = robot.streams[fm.stream].kind
            if kind == "rgb" and key in rgb_keys:
                readers[key] = SequentialFrames(ep.video_path(fm.stream))
            elif kind == "depth" and key in depth_keys:
                depth_arrays[key] = ep.root[fm.stream]["data"]
        missing = (rgb_keys - set(readers)) | (depth_keys - set(depth_arrays))
        if missing:
            raise ValueError(f"{eid}: robot {robot.robot_id} lacks recipe cameras {sorted(missing)}")
        for i in range(n):
            frame = {"task": ep.meta.task}
            for key, rd in readers.items():
                frame[key] = _resize(rd.get(int(al["frame_idx"][key][i])), recipe.image_size)
            for key, arr in depth_arrays.items():
                frame[key] = _resize(np.asarray(arr[int(al["frame_idx"][key][i])]), recipe.image_size, depth=True)[..., None]
            for key in recipe.features:
                frame[key] = al["lowdim"][key][i].astype(np.float32)
            frame[schema.IS_SIM] = np.array([bool(ep.meta.is_sim)])
            frame[schema.SYNC_RESIDUAL] = np.array([al["sync_residual_ms"][i]], np.float32)
            if recipe.phases:
                frame[schema.PHASE] = np.array([al["phase"][i]], np.int64)
            ds.add_frame(frame)
        ds.save_episode()
        records.append({"episode_id": eid, "split": split, "frames": n,
                        "phase_boundaries": al["phase_info"]["b"],
                        "phase_contact_method": al["phase_info"].get("contact_method"),
                        **{f"phase_{k}": v for k, v in (al.get("phase_eval") or {}).items()},
                        "sync_residual_p95_ms": float(np.percentile(al["sync_residual_ms"], 95))})
        print(f"[build:{recipe.name}:{Path(root).name}] {split} {local + 1}/{len(shard)} {eid} frames={n}", flush=True)
    ds.finalize()
    return records


def build_dataset(recipe: DatasetRecipe, curated_dir: str | Path, out_root: str | Path, overwrite: bool = True,
                  limit_per_split: int | None = None, shards: int = 1) -> dict:
    """Shards are contiguous slices of the ordered episode list, written in parallel processes and
    merged in order with LeRobot's own ``aggregate_datasets`` (so split ranges stay contiguous)."""
    from concurrent.futures import ProcessPoolExecutor

    curated_dir = Path(curated_dir)
    catalog = pd.read_parquet(curated_dir / "catalog.parquet").set_index("episode_id")
    splits = json.loads((curated_dir / "splits.json").read_text())
    root = Path(out_root) / recipe.name
    if root.exists():
        if not overwrite:
            raise FileExistsError(root)
        shutil.rmtree(root)
    ordered = []
    for s in SPLIT_ORDER:
        ids = splits["splits"].get(s, [])
        ordered += [(s, i, str(catalog.loc[i, "path"])) for i in (ids[:limit_per_split] if limit_per_split else ids)]
    shards = max(1, min(shards, len(ordered)))
    bounds = np.linspace(0, len(ordered), shards + 1).astype(int)
    parts = [ordered[a:b] for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    if len(parts) == 1:
        records = _write_shard(recipe.model_dump_json(), parts[0], str(root), recipe.repo_id)
    else:
        from lerobot.datasets.aggregate import aggregate_datasets

        tmp = Path(out_root) / f".{recipe.name}_shards"
        shutil.rmtree(tmp, ignore_errors=True)
        roots = [tmp / f"shard{k:02d}" for k in range(len(parts))]
        repo_ids = [f"{recipe.repo_id}_shard{k:02d}" for k in range(len(parts))]
        with ProcessPoolExecutor(max_workers=len(parts)) as ex:
            futs = [ex.submit(_write_shard, recipe.model_dump_json(), part, str(r), rid)
                    for part, r, rid in zip(parts, roots, repo_ids)]
            shard_records = [f.result() for f in futs]
        aggregate_datasets(repo_ids, recipe.repo_id, roots=roots, aggr_root=root)
        records = [r for recs in shard_records for r in recs]
        shutil.rmtree(tmp)
    ranges = {}
    for cursor, r in enumerate(records):
        r["episode_index"] = cursor
        ranges.setdefault(r["split"], [cursor, cursor])
        ranges[r["split"]][1] = cursor + 1

    # --- robodata metadata alongside the standard LeRobot files -----------------------
    info_p = root / "meta" / "info.json"
    info = json.loads(info_p.read_text())
    info["splits"] = {s: f"{a}:{b}" for s, (a, b) in ranges.items()}
    info_p.write_text(json.dumps(info, indent=4))
    meta_dir = root / "meta" / "robodata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    rec_df = pd.DataFrame(records)
    cat = catalog.reset_index()
    # catalog values are pre-build (raw) metrics; build records win, raw ones are kept with a raw_ prefix
    overlap = [c for c in rec_df.columns if c in cat.columns and c != "episode_id"]
    cat = cat.drop(columns=["path", *[c for c in overlap if c == "split"]], errors="ignore")
    cat = cat.rename(columns={c: f"raw_{c}" for c in overlap if c != "split"})
    ep_df = rec_df.merge(cat, on="episode_id", how="left")
    ep_df.to_parquet(meta_dir / "episodes.parquet", index=False)
    (meta_dir / "splits.json").write_text(json.dumps(
        {"group_key": splits["group_key"], "ranges": info["splits"],
         "episode_ids": {s: [r["episode_id"] for r in records if r["split"] == s] for s in SPLIT_ORDER},
         "leakage": splits["leakage"]}, indent=2, default=str))
    (meta_dir / "phases.json").write_text(json.dumps({"names": PHASE_NAMES, "feature": schema.PHASE}, indent=2))
    (meta_dir / "recipe.json").write_text(recipe.model_dump_json(indent=2))
    for name in ("curation_summary.json", "duplicates.json"):
        if (curated_dir / name).exists():
            shutil.copy(curated_dir / name, meta_dir / name)
    integrity = verify_videos(root)
    (meta_dir / "video_integrity.json").write_text(json.dumps(integrity, indent=2))
    if not integrity["ok"]:
        raise RuntimeError(f"video integrity check failed: {integrity['problems'][:5]}")
    return {"root": str(root), "episodes": len(records), "ranges": info["splits"], "records": records,
            "integrity": {k: v for k, v in integrity.items() if k != "problems"}}


def verify_videos(root: str | Path) -> dict:
    """Every video file must hold exactly the frames its episodes claim, and every episode segment
    must span length / fps seconds. Counts packets (no decoding), so it is cheap enough to always run."""
    import av

    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = info["fps"]
    eps = pd.concat([pd.read_parquet(p) for p in sorted((root / "meta" / "episodes").rglob("*.parquet"))])
    problems, checked = [], 0
    for key, f in info["features"].items():
        if f["dtype"] != "video":
            continue
        ck, fk = f"videos/{key}/chunk_index", f"videos/{key}/file_index"
        for (c, fi), g in eps.groupby([ck, fk]):
            path = root / info["video_path"].format(video_key=key, chunk_index=int(c), file_index=int(fi))
            with av.open(str(path)) as cont:
                n_pkts = sum(1 for p in cont.demux(cont.streams.video[0]) if p.size)
            expected = int(g["length"].sum())
            checked += 1
            if n_pkts != expected:
                problems.append({"video": str(path.relative_to(root)), "frames": n_pkts, "expected": expected})
            span = (g[f"videos/{key}/to_timestamp"] - g[f"videos/{key}/from_timestamp"]) * fps
            bad = g[(span - g["length"]).abs() > 0.5]
            for _, r in bad.iterrows():
                problems.append({"video": key, "episode": int(r.episode_index), "segment_frames": float(span[r.name]),
                                 "length": int(r.length)})
    return {"ok": not problems, "files_checked": checked, "problems": problems}
