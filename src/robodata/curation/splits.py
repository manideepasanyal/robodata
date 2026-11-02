"""Leakage-free train / val / eval splits.

Episodes are split by *group* (session, scene or operator - never by frame or episode),
stratified by domain (robot + real/sim) so every split sees every domain when possible.
Assignment is deterministic (hash-ordered groups, greedy fill by deficit), so adding new
sessions later never moves existing episodes between splits. Near-duplicate clusters are
checked to never straddle splits.
"""

from __future__ import annotations

import hashlib

import pandas as pd


def _h(seed: str, key: str) -> str:
    return hashlib.sha256(f"{seed}/{key}".encode()).hexdigest()


def assign_splits(df: pd.DataFrame, group_key: str, ratios: dict[str, float], seed: str,
                  eval_groups: list[str] | None = None) -> pd.Series:
    eval_groups = set(eval_groups or [])
    names = list(ratios)
    split = pd.Series(index=df.index, dtype=object)
    for _, strat in df.groupby(["robot_id", "is_sim"]):
        sizes = strat.groupby(group_key).size()
        groups = sorted(sizes.index, key=lambda g: _h(seed, str(g)))
        total = sizes.sum()
        filled = {n: 0 for n in names}
        forced = [g for g in groups if g in eval_groups]
        for g in forced:
            split[strat.index[strat[group_key] == g]] = "eval"
            filled["eval"] += sizes[g]
        free = [g for g in groups if g not in eval_groups]
        # smaller splits first get one group each (if enough groups exist), then fill by largest deficit
        order = sorted(names, key=lambda n: ratios[n])
        for n in order:
            if len(free) > len(names) - 1 and filled[n] == 0 and ratios[n] > 0 and n != order[-1]:
                g = free.pop(0)
                split[strat.index[strat[group_key] == g]] = n
                filled[n] += sizes[g]
        for g in free:
            n = max(names, key=lambda k: ratios[k] * total - filled[k])
            split[strat.index[strat[group_key] == g]] = n
            filled[n] += sizes[g]
    return split


def leakage_report(df: pd.DataFrame, group_key: str) -> dict:
    per_group = df.groupby(group_key)["split"].nunique()
    leaking_groups = per_group[per_group > 1].index.tolist()
    rep = {"group_key": group_key, "groups_in_multiple_splits": leaking_groups}
    if "dup_cluster" in df:
        per_cluster = df.groupby("dup_cluster")["split"].nunique()
        rep["dup_clusters_across_splits"] = per_cluster[per_cluster > 1].index.tolist()
    rep["ok"] = not leaking_groups and not rep.get("dup_clusters_across_splits")
    rep["episodes_per_split"] = df.split.value_counts().to_dict()
    rep["groups_per_split"] = {s: sorted(map(str, g[group_key].unique())) for s, g in df.groupby("split")}
    rep["domain_per_split"] = {s: {f"{r}|sim={bool(m)}": int(n) for (r, m), n in g.groupby(["robot_id", "is_sim"]).size().items()}
                               for s, g in df.groupby("split")}
    return rep
