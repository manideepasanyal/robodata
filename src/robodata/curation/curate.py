"""Curation: catalog -> quality score -> filter -> dedup -> leakage-free splits."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from robodata.config import DatasetRecipe, load_robot
from robodata.curation.dedup import (clusters, content_hash, find_duplicates, group_rate, robust_scale, trajectory_samples,
                                     trajectory_sequence, trajectory_signature, visual_signature)
from robodata.curation.splits import assign_splits, leakage_report
from robodata.io.raw_episode import RawEpisode, list_episodes

HEALTH_FACTOR = {"pass": 1.0, "warn": 0.85, "fail": 0.0}


def log_dimensionless_jerk(t_ns: np.ndarray, pos: np.ndarray) -> float | None:
    """Smoothness of the end-effector path (Hogan & Sternad 2009); more negative = jerkier."""
    if len(t_ns) < 10:
        return None
    t = (t_ns - t_ns[0]) / 1e9
    g = np.arange(0, t[-1], 1 / 30)
    if len(g) < 10:
        return None
    p = np.stack([np.interp(g, t, pos[:, k]) for k in range(pos.shape[1])], 1)
    v = np.gradient(p, g, axis=0)
    j = np.gradient(np.gradient(v, g, axis=0), g, axis=0)
    vmax = np.linalg.norm(v, axis=1).max()
    dur = g[-1]
    if vmax < 1e-6:
        return None
    return float(-np.log(dur**3 / vmax**2 * np.trapezoid(np.sum(j**2, 1), g)))


def _episode_row(path: str) -> dict:
    ep = RawEpisode(path)
    robot = load_robot(ep.meta.robot_id)
    m = ep.meta
    health = ep.sidecar("health.json") or {}
    sync = ep.sidecar("sync.json") or {}
    ref = robot.sync.reference_stream
    t_ref = ep.times(ref) if ref in ep.streams else np.zeros(0)
    idle = next((c["value"] for c in health.get("checks", []) if c["check"] == "idle_ratio"), None)
    ldj = None
    for s, cfg in robot.streams.items():
        if cfg.kind == "pose" and s in ep.streams and ep.has(s, "data"):
            ldj = log_dimensionless_jerk(ep.times(s), ep.data(s)[:, :3].astype(np.float64))
            break
    traj = trajectory_signature(ep, robot)
    samples = trajectory_samples(ep, robot)
    vis, grey = visual_signature(ep, robot, with_frames=True)
    chash = content_hash(ep, robot)
    return {
        "episode_id": m.episode_id, "path": str(path), "robot_id": m.robot_id, "robot_type": m.robot_type,
        "session_id": m.session_id, "scene_id": m.scene_id, "task": m.task, "operator": m.operator,
        "is_sim": m.is_sim, "source": m.source, "success": m.success, "sensor_config_hash": m.sensor_config_hash,
        "calibration_version": m.calibration_version, "recorded_at": m.recorded_at,
        "duration_s": float((t_ref[-1] - t_ref[0]) / 1e9) if len(t_ref) > 1 else 0.0,
        "n_ref_frames": int(len(t_ref)), "health": health.get("status", "unchecked"),
        "health_warnings": ";".join(health.get("warned", [])),
        "sync_residual_p95_ms": sync.get("residual_sync_error_p95_ms"), "idle_ratio": idle, "ldj": ldj,
        "injected_faults": ";".join(f["type"] for f in m.injected_faults),
        "_traj": traj, "_samples": samples, "_vis": vis, "_grey": grey, "_hash": chash,
    }


def build_catalog(raw_root: str | Path, robot_ids: list[str], workers: int = 8) -> pd.DataFrame:
    paths = [str(p) for r in robot_ids for p in list_episodes(raw_root, r)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_episode_row, paths, chunksize=4))
    return pd.DataFrame(rows)


def score_quality(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["q_health"] = df.health.map(HEALTH_FACTOR).fillna(0.5)
    df["q_success"] = df.success.map({True: 1.0, False: 0.2}).fillna(0.8)
    df["q_idle"] = (1 - (df.idle_ratio.fillna(0).astype(float) - 0.6).clip(lower=0)).clip(0, 1)
    q_smooth = pd.Series(1.0, index=df.index)
    for _, g in df.groupby("robot_id"):  # smoothness relative to the robot's own distribution
        ldj = g.ldj.astype(float)
        med = ldj.median()
        mad = (ldj - med).abs().median() + 1e-6
        z = (med - ldj) / mad  # positive = jerkier than typical
        q_smooth[g.index] = (1 - 0.1 * (z - 3).clip(lower=0)).clip(0.5, 1.0).fillna(1.0)
    df["q_smooth"] = q_smooth
    df["quality"] = df.q_health * df.q_success * df.q_idle * df.q_smooth
    return df


def curate(recipe: DatasetRecipe, raw_root: str | Path, out_dir: str | Path, workers: int = 8) -> dict:
    cfg = recipe.curation
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = build_catalog(raw_root, [s.robot_id for s in recipe.sources], workers)
    if df.empty:
        raise RuntimeError("no raw episodes found for recipe sources")
    # cap episodes per source (deterministic: earliest ids first)
    keep = []
    for s in recipe.sources:
        ids = sorted(df.loc[df.robot_id == s.robot_id, "episode_id"])
        keep += ids[: s.max_episodes] if s.max_episodes else ids
    df = df[df.episode_id.isin(keep)].reset_index(drop=True)
    df = score_quality(df)

    # --- dedup (within robot + task) -----------------------------------------------------
    sigs = {}
    for robot_id, g in df.groupby("robot_id"):
        trajs = [t for t in g._traj if t is not None]
        sc = robust_scale(trajs) if trajs else None
        rate = group_rate([float(s_[0][-1]) for s_ in g._samples if s_ is not None] or [1.0])
        for _, r in g.iterrows():
            seq = trajectory_sequence(r._samples, rate)
            smp = None if r._samples is None else (r._samples[0], r._samples[1] / sc)
            sigs[r.episode_id] = {"traj": None if r._traj is None else r._traj / sc, "samples": smp,
                                  "seq": None if seq is None else seq / sc,
                                  "vis": r._vis, "frames": r._grey, "hash": r._hash, "group": robot_id,
                                  "duration": float(r.duration_s), "n_frames": int(r.n_ref_frames)}
    pairs, nn = find_duplicates(sigs, cfg.dedup_traj_threshold, cfg.dedup_phash_threshold)
    # copies (exact / near) collapse to their best member; a video-reuse pair means one of the two labels
    # is wrong and we cannot tell which, so both go to review instead of silently keeping either
    copy_pairs = [p for p in pairs if p["kind"] in ("exact", "near")]
    reuse = {e for p in pairs if p["kind"] == "video_reuse" for e in (p["a"], p["b"])}
    df["dup_cluster"] = df.episode_id.map(clusters(df.episode_id.tolist(), copy_pairs))
    df["nn_traj_dist"] = df.episode_id.map(lambda e: nn.get(e, {}).get("nn_traj_dist"))
    df["nn_vis_dist"] = df.episode_id.map(lambda e: nn.get(e, {}).get("nn_vis_dist"))
    # keep the best-quality member of each cluster
    df["dup_keep"] = df.groupby("dup_cluster").quality.transform(lambda q: q == q.max())
    df.loc[df.duplicated(["dup_cluster", "dup_keep"], keep="first") & df.dup_keep, "dup_keep"] = False

    # --- filter ------------------------------------------------------------------------
    reasons = pd.Series("", index=df.index)
    reasons[df.health == "fail"] += "health_fail;"
    reasons[df.quality < cfg.min_quality] += "low_quality;"
    if cfg.require_success:
        reasons[df.success != True] += "not_success;"  # noqa: E712
    reasons[~df.dup_keep] += "near_duplicate;"
    reasons[df.episode_id.isin(reuse)] += "video_reuse;"
    # one recording with conflicting labels: send its whole copy-family to review (deterministic,
    # independent of which member would otherwise have been kept)
    conflicted = set(df.loc[df.episode_id.isin(reuse), "dup_cluster"])
    reasons[df.dup_cluster.isin(conflicted) & ~df.episode_id.isin(reuse)] += "video_reuse_family;"
    df["excluded"] = reasons.str.rstrip(";")
    sel = df[df.excluded == ""].copy()

    # --- splits ------------------------------------------------------------------------
    sel["split"] = assign_splits(sel, cfg.split_by, cfg.split_ratios, cfg.seed, cfg.eval_groups)
    df["split"] = sel["split"]
    leak = leakage_report(sel, cfg.split_by)
    # duplicates that were dropped must not have a sibling in another split either (they are dropped, so ok)

    public = df.drop(columns=["_traj", "_samples", "_vis", "_grey", "_hash"])
    public.to_parquet(out_dir / "catalog.parquet", index=False)
    splits = {s: sorted(g.episode_id) for s, g in sel.groupby("split")}
    (out_dir / "splits.json").write_text(json.dumps({"group_key": cfg.split_by, "seed": cfg.seed, "splits": splits,
                                                     "leakage": leak}, indent=2, default=str))
    (out_dir / "duplicates.json").write_text(json.dumps(pairs, indent=2))
    summary = {
        "episodes_total": int(len(df)), "selected": int(len(sel)),
        "excluded": df.excluded[df.excluded != ""].str.split(";").explode().value_counts().to_dict(),
        "duplicate_pairs": len(copy_pairs), "video_reuse_pairs": len(pairs) - len(copy_pairs), "leakage_ok": leak["ok"], "per_split": leak["episodes_per_split"],
        "domain_per_split": leak["domain_per_split"],
    }
    (out_dir / "curation_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
