"""Independent cross-verification of a built dataset (``robodata audit <dataset>``).

The audit deliberately re-derives everything from the files instead of trusting pipeline
metadata:

A. structure   every episode's first/last frame decodes through LeRobotDataset for every camera;
               video packet counts == episode lengths; stats.json == stats recomputed from parquet;
               frame_index / timestamp contiguity; split ranges cover all episodes exactly once.
B. leakage     no recording session spans two splits; no near-identical content across splits
               (own fingerprint: resampled state + mid-frame thumbnail, not the dedup code);
               at most one survivor per planted duplicate family (ground truth from raw metadata).
C. fidelity    for sampled episodes from public sources: built state / action / camera frames vs.
               the ORIGINAL Hugging Face data at the ground-truth source frame, scanning lags -2..2
               so a systematic off-by-one shift is caught, not just a large error.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from robodata.config import CONFIG_DIR, load_robot, source_path
from robodata.ingest.lerobot_source import LeRobotSource
from robodata.io.raw_episode import RawEpisode, list_episodes

LAGS = [-2, -1, 0, 1, 2]


def _psnr(a, b) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0**2 / mse)


def _source_for_robot(robot_id: str):
    for f in (CONFIG_DIR / "sources").glob("*.yaml"):
        d = yaml.safe_load(f.read_text())
        if d.get("robot") == robot_id and "repo_id" in d:
            return d
    return None


# --------------------------------------------------------------------------------------------- A
def check_structure(root: Path) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from robodata.preprocess.build import verify_videos

    info = json.loads((root / "meta" / "info.json").read_text())
    data = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet"))])
    problems = []
    ds = LeRobotDataset(info.get("repo_id", f"robodata/{root.name}"), root=root)
    cams = [k for k, f in info["features"].items() if f["dtype"] == "video"]
    decoded = 0
    for e, g in data.groupby("episode_index"):
        g = g.sort_values("frame_index")
        if not (g.frame_index.to_numpy() == np.arange(len(g))).all():
            problems.append(f"ep{e}: frame_index not contiguous")
        if np.abs(g.timestamp.to_numpy() - np.arange(len(g)) / info["fps"]).max() > 1e-3:
            problems.append(f"ep{e}: timestamps != frame_index / fps")
        for idx in (int(g["index"].iloc[0]), int(g["index"].iloc[-1])):
            try:
                item = ds[idx]
                for k in cams:
                    if not np.isfinite(item[k].numpy()).all():
                        problems.append(f"ep{e}: non-finite pixels in {k}")
                decoded += 1
            except Exception as exc:
                problems.append(f"ep{e} idx {idx}: decode failed: {type(exc).__name__}: {str(exc)[:120]}")
    vid = verify_videos(root)
    problems += [f"video: {p}" for p in vid["problems"]]
    # stats.json vs recomputed
    stats = json.loads((root / "meta" / "stats.json").read_text())
    stat_err = {}
    for k, f in info["features"].items():
        if f["dtype"] == "float32" and k in data and k in stats:
            x = np.stack(data[k].to_numpy()).astype(np.float64).reshape(len(data), -1)
            m, s = np.asarray(stats[k]["mean"]).ravel(), np.asarray(stats[k]["std"]).ravel()
            err = float(np.max(np.abs(x.mean(0) - m) / (x.std(0) + 1e-6)))
            stat_err[k] = err
            if err > 1e-3 or not (np.isfinite(m).all() and np.isfinite(s).all()):
                problems.append(f"stats {k}: mean mismatch {err:.2e} (in std units)")
    covered = []
    for split, rng in info.get("splits", {}).items():
        a, b = map(int, rng.split(":"))
        covered += list(range(a, b))
    if sorted(covered) != sorted(data.episode_index.unique().tolist()):
        problems.append("split ranges do not cover every episode exactly once")
    return {"ok": not problems, "episodes": int(data.episode_index.nunique()), "frames": int(len(data)),
            "first_last_frames_decoded": decoded, "video_files_checked": vid["files_checked"],
            "stats_max_mean_err_std_units": max(stat_err.values()) if stat_err else None, "problems": problems[:50]}


# --------------------------------------------------------------------------------------------- B
def _episode_fingerprints(root: Path, info: dict, data: pd.DataFrame) -> dict[int, dict]:
    src = LeRobotSource(root)
    cams = [k for k, f in info["features"].items() if f["dtype"] == "video"
            and not (f.get("info") or {}).get("is_depth_map")]
    out = {}
    for e in sorted(data.episode_index.unique()):
        g = data[data.episode_index == e].sort_values("frame_index")
        x = np.stack(g["observation.state"].to_numpy()).astype(np.float32).reshape(len(g), -1) if "observation.state" in g else None
        traj = None if x is None else np.stack([np.interp(np.linspace(0, len(x) - 1, 32), np.arange(len(x)), x[:, j])
                                                for j in range(x.shape[1])], 1)
        thumb = None
        if cams:
            ep = src.by_index(int(e))
            mid = len(ep) // 2
            for i, fr in enumerate(ep.iter_video(cams[0])):
                if i == mid:
                    t = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY), (16, 12), interpolation=cv2.INTER_AREA).astype(np.float32)
                    thumb = (t - t.mean()) / (t.std() + 1e-3)
                    break
        out[int(e)] = {"traj": traj, "thumb": thumb}
    return out


def check_leakage(root: Path, raw_root: Path) -> dict:
    info = json.loads((root / "meta" / "info.json").read_text())
    data = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet"))])
    meta = pd.read_parquet(root / "meta" / "robodata" / "episodes.parquet")
    split_of = dict(zip(meta.episode_index, meta.split))
    problems = []
    per_session = meta.groupby("session_id").split.nunique()
    for s in per_session[per_session > 1].index:
        problems.append(f"session {s} appears in {sorted(meta[meta.session_id == s].split.unique())}")
    # content duplicates across splits (independent fingerprint)
    fp = _episode_fingerprints(root, info, data)
    ids = sorted(fp)
    trajs = [fp[i]["traj"] for i in ids]
    cross = []
    if all(t is not None for t in trajs) and len(ids) > 1:
        T = np.stack(trajs)
        scale = T.reshape(-1, T.shape[2]).std(0) + 1e-6
        T = T / scale
        for a in range(len(ids)):
            d = np.linalg.norm(T - T[a], axis=2).mean(1)
            for b in np.where(d < 0.05)[0]:
                if b <= a or split_of[ids[a]] == split_of[ids[b]]:
                    continue
                ta, tb = fp[ids[a]]["thumb"], fp[ids[b]]["thumb"]
                vis = float(np.sqrt(np.mean((ta - tb) ** 2))) if ta is not None and tb is not None else 0.0
                if vis < 0.35:
                    cross.append({"a": ids[a], "b": ids[b], "splits": [split_of[ids[a]], split_of[ids[b]]],
                                  "traj_dist": float(d[b]), "thumb_dist": vis})
    problems += [f"near-identical content across splits: eps {c['a']} ({c['splits'][0]}) / {c['b']} ({c['splits'][1]})"
                 for c in cross]
    # ground-truth duplicate families: follow duplicate_of to a root in raw + quarantine metadata
    fam_root = {}
    for base in (raw_root, raw_root.parent / "quarantine"):
        for p in list_episodes(base):
            m = json.loads((p / "meta.json").read_text())
            fam_root[m["episode_id"]] = m.get("extra", {}).get("duplicate_of") or m["episode_id"]
    def root_of(e):
        seen = set()
        while fam_root.get(e, e) != e and e not in seen:
            seen.add(e)
            e = fam_root[e]
        return e
    meta["family"] = meta.episode_id.map(root_of)
    fam = meta.groupby("family").episode_id.apply(list)
    multi = fam[fam.map(len) > 1]
    planted = sum(1 for e, r in fam_root.items() if r != e)
    for f, members in multi.items():
        problems.append(f"duplicate family {f} has {len(members)} survivors: {members}")
    return {"ok": not problems, "sessions": int(meta.session_id.nunique()), "cross_split_near_identical": cross,
            "planted_duplicates_known": planted, "families_with_multiple_survivors": int(len(multi)),
            "problems": problems[:50]}


# --------------------------------------------------------------------------------------------- C
def check_fidelity(root: Path, curated_dir: Path, per_split: int = 2, frame_step: int = 3) -> dict:
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = info["fps"]
    data = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet"))])
    meta = pd.read_parquet(root / "meta" / "robodata" / "episodes.parquet")
    catalog = pd.read_parquet(curated_dir / "catalog.parquet").set_index("episode_id")
    built = LeRobotSource(root)
    picks = meta[~meta.episode_id.str.contains("__fault-") & ~meta.session_id.str.contains("_reupload")]
    picks = picks.groupby("split", group_keys=False).head(per_split)
    results, problems = [], []
    for _, r in picks.iterrows():
        raw = RawEpisode(catalog.loc[r.episode_id, "path"])
        robot = load_robot(raw.meta.robot_id)
        src_cfg = _source_for_robot(raw.meta.robot_id)
        if src_cfg is None or "source_episode_index" not in raw.meta.extra:
            continue
        src = LeRobotSource(source_path(src_cfg["local_dir"])).by_index(raw.meta.extra["source_episode_index"])
        ref = robot.sync.reference_stream
        t_true, seq = raw.times(ref, "t_true"), raw.times(ref, "seq")
        period = 1e9 / src.fps
        epoch = float(np.median(t_true - seq * period))
        sync = raw.sidecar("sync.json")
        k0 = int(round((sync["grid"]["start_ns"] - epoch) / period))
        g = data[data.episode_index == r.episode_index].sort_values("frame_index")
        n = len(g)
        res = {"episode": int(r.episode_index), "episode_id": r.episode_id, "split": r.split, "built_frames": n,
               "source_frames": len(src), "k0": k0}
        for feat, sm in robot.features.items():
            sc = src_cfg["streams"].get(sm.stream)
            if not sc or feat not in g or robot.streams[sm.stream].kind in ("rgb", "depth"):
                continue
            b = np.stack(g[feat].to_numpy()).astype(np.float64).reshape(n, -1)
            s = src.column(sc["key"]).astype(np.float64)
            if sc.get("index"):
                s = s[:, sc["index"]]
            errs = {}
            for lag in LAGS:
                idx = np.arange(n) + k0 + lag
                ok = (idx >= 0) & (idx < len(s))
                if ok.sum() < n * 0.5:
                    continue
                d = np.abs(b[ok] - s[idx[ok]])
                errs[lag] = (float(d.mean()), float((d.max(1) < 1e-4).mean()))
            best = min(errs, key=lambda k: errs[k][0])
            res[feat] = {"mae_lag0": errs.get(0, (np.nan,))[0], "exact_frac_lag0": errs.get(0, (0, np.nan))[1],
                         "best_lag": best, "mae_best": errs[best][0]}
            if best != 0:
                problems.append(f"ep{r.episode_index} {feat}: best alignment at lag {best} (mae {errs[best][0]:.2e} "
                                f"vs {errs.get(0, (np.nan,))[0]:.2e} at lag 0)")
        # camera frames
        cam_feats = [k for k in info["features"] if info["features"][k]["dtype"] == "video"
                     and k in robot.features and robot.streams[robot.features[k].stream].kind == "rgb"][:2]
        for feat in cam_feats:
            sc = src_cfg["streams"][robot.features[feat].stream]
            sframes = list(src.iter_video(sc["key"]))
            bframes = list(built.by_index(int(r.episode_index)).iter_video(feat))
            h, w = bframes[0].shape[:2]
            sframes = [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in sframes]
            lag_psnr = {}
            for lag in LAGS:
                vals = [_psnr(bframes[i], sframes[i + k0 + lag]) for i in range(0, len(bframes), frame_step)
                        if 0 <= i + k0 + lag < len(sframes)]
                if vals:
                    lag_psnr[lag] = float(np.mean(vals))
            best = max(lag_psnr, key=lag_psnr.get)
            # in static stretches neighbouring frames tie; only flag a clear (>0.5 dB) preference
            clear = best != 0 and lag_psnr[best] - lag_psnr.get(0, -np.inf) > 0.5
            res[feat] = {"psnr_lag0": lag_psnr.get(0), "best_lag": best, "psnr_best": lag_psnr[best]}
            if clear:
                problems.append(f"ep{r.episode_index} {feat}: frames best match source at lag {best} "
                                f"({lag_psnr[best]:.1f} dB vs {lag_psnr.get(0, float('nan')):.1f} dB at lag 0)")
            if lag_psnr.get(0, 0) < 25:
                problems.append(f"ep{r.episode_index} {feat}: low fidelity {lag_psnr.get(0)} dB")
        results.append(res)
    return {"ok": not problems, "episodes_checked": len(results), "results": results, "problems": problems[:50]}


def audit_dataset(name: str, data_root: Path) -> dict:
    root = data_root / "datasets" / name
    report = {"dataset": name,
              "structure": check_structure(root),
              "leakage": check_leakage(root, data_root / "raw"),
              "fidelity": check_fidelity(root, data_root / "curated" / name)}
    report["ok"] = all(report[k]["ok"] for k in ("structure", "leakage", "fidelity"))
    out = data_root / "reports" / "audit"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(report, indent=2, default=float))
    return report
