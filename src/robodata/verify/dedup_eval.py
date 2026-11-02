"""Score deduplication against planted ground truth (robodata dupes) for curated datasets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _kind(eid: str, session: str) -> str | None:
    if "_reupload" in session:
        return "reupload"
    for k in ("dup_exact_xsession", "dup_exact", "dup_reencode", "dup_trim", "videoswap"):
        if f"__fault-{k}" in eid:
            return k
    return None


def evaluate(curated_root: Path, names: list[str], raw_root: Path) -> dict:
    rows, fams = [], []
    for name in names:
        cat = pd.read_parquet(curated_root / name / "catalog.parquet")
        dup_of = {}
        for p in cat.path:
            m = json.loads((Path(p) / "meta.json").read_text())
            dup_of[m["episode_id"]] = m.get("extra", {}).get("duplicate_of")
        cat["kind"] = [_kind(e, s) for e, s in zip(cat.episode_id, cat.session_id)]
        cat["dup_of"] = cat.episode_id.map(dup_of)
        cat["kept"] = cat.excluded.fillna("") == ""
        planted = cat[cat.kind.notna()]
        roots = set(planted[planted.kind != "videoswap"].dup_of.dropna())
        donors = set(planted[planted.kind == "videoswap"].dup_of.dropna())  # trajectory reused by a swap copy
        cluster = dict(zip(cat.episode_id, cat.dup_cluster))
        for k, g in planted.groupby("kind"):
            if k == "videoswap":
                ok = g.excluded.fillna("").str.contains("video_reuse")
            else:  # detected = linked into the same duplicate cluster as its original
                ok = pd.Series([cluster.get(e) is not None and cluster.get(e) == cluster.get(o)
                                for e, o in zip(g.episode_id, g.dup_of)], index=g.index)
            rows.append({"dataset": name, "kind": k, "planted_in_catalog": len(g), "caught": int(ok.sum())})
        for r in roots:  # exactly one survivor per family (root + exact/near copies), unless reuse sent all to review
            fam = cat[(cat.episode_id == r) | ((cat.dup_of == r) & (cat.kind != "videoswap"))]
            fams.append({"dataset": name, "family": r, "members": len(fam), "survivors": int(fam.kept.sum()),
                         "reuse_review": bool(fam.excluded.fillna("").str.contains("video_reuse").any())})
        originals = cat[cat.kind.isna() & ~cat.episode_id.isin(roots | donors) & ~cat.episode_id.str.contains("__fault-")]
        fp = originals[originals.excluded.fillna("").str.contains("near_duplicate|video_reuse")]
        rows.append({"dataset": name, "kind": "FALSE_POSITIVE(originals)", "planted_in_catalog": len(originals),
                     "caught": len(fp), "ids": fp.episode_id.tolist()})
        # margins: planted near-copy distances vs nearest-neighbour distances among genuine originals only
        pairs = pd.DataFrame(json.loads((curated_root / name / "duplicates.json").read_text()) or [])
        ok_ids = set(originals.episode_id)
        if len(pairs):
            near = pairs[pairs.kind == "near"]
            planted_ids = set(planted.episode_id)
            pn = near[near.a.isin(planted_ids) | near.b.isin(planted_ids)]
            rows[-1]["planted_near_traj_max"] = float(pn.traj_dist.max()) if len(pn) else None
            rows[-1]["planted_near_vis_max"] = float(pn.vis_dist.max()) if len(pn) else None
        nnd = originals.nn_traj_dist.astype(float)
        rows[-1]["orig_nn_traj_min"] = float(np.nanmin(nnd)) if len(nnd) else None
        rows[-1]["orig_nn_traj_p5"] = float(np.nanpercentile(nnd, 5)) if len(nnd) else None
        rows[-1]["n_ok"] = len(ok_ids)
    df = pd.DataFrame(rows)
    fam = pd.DataFrame(fams)
    return {"per_kind": df, "families": fam}
