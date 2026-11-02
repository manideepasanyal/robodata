import json

import numpy as np
import pandas as pd

from robodata.config import load_robot
from robodata.curation.dedup import (clusters, content_hash, dtw, find_duplicates, group_rate, robust_scale,
                                     trajectory_samples, trajectory_sequence, trajectory_signature, visual_signature)
from robodata.curation.splits import assign_splits, leakage_report
from robodata.health.faults import inject
from robodata.io.raw_episode import RawEpisode
from robodata.preprocess.segment import segment_phases
from robodata.versioning.manifest import create_version, verify

from .conftest import make_episode


def _catalog(n_groups=12, per_group=5, sim=False, robot="r"):
    rows = [{"episode_id": f"{robot}_{g}_{i}", "session_id": f"{robot}_s{g}", "robot_id": robot, "is_sim": sim}
            for g in range(n_groups) for i in range(per_group)]
    return pd.DataFrame(rows)


def test_splits_group_disjoint_and_stable():
    df = pd.concat([_catalog(), _catalog(6, 4, True, "sim")], ignore_index=True)
    ratios = {"train": 0.8, "val": 0.1, "eval": 0.1}
    df["split"] = assign_splits(df, "session_id", ratios, "seed", eval_groups=["sim_s5"])
    rep = leakage_report(df, "session_id")
    assert rep["ok"] and set(df.split) == {"train", "val", "eval"}
    assert set(df[df.session_id == "sim_s5"].split) == {"eval"}
    for (_, sim), g in df.groupby(["robot_id", "is_sim"]):  # every domain present in every split
        assert set(g.split) == {"train", "val", "eval"}
    again = assign_splits(df, "session_id", ratios, "seed", eval_groups=["sim_s5"])
    assert (again == df.split).all()  # deterministic


def test_dedup_variants(robot, clean_episode, tmp_path):
    other = make_episode(robot, tmp_path, index=7, seed=9, shift=0.3, amp=0.5)  # a genuinely different demo
    copies = {
        "exact": inject(clean_episode, [{"type": "exact_duplicate"}], "exact"),
        "reencode": inject(clean_episode, [{"type": "reencode_duplicate"}], "reenc", session="s_other"),
        "trimmed": inject(clean_episode, [{"type": "trimmed_duplicate", "frac": 0.08}], "trim", session="s_other"),
        "swap": inject(other, [{"type": "video_swap", "video_from": str(clean_episode)}], "swap"),
    }
    eps = [clean_episode, other, *copies.values()]
    trajs = {p.name: trajectory_signature(RawEpisode(p), robot) for p in eps}
    sc = robust_scale(list(trajs.values()))
    samples = {p.name: trajectory_samples(RawEpisode(p), robot) for p in eps}
    rate = group_rate([float(s[0][-1]) for s in samples.values()])
    sig = {}
    for p in eps:
        e = RawEpisode(p)
        vis, grey = visual_signature(e, robot, with_frames=True)
        sig[p.name] = {"traj": trajs[p.name] / sc, "seq": trajectory_sequence(samples[p.name], rate) / sc,
                       "vis": vis, "frames": grey, "hash": content_hash(e, robot), "group": "g"}
    pairs, nn = find_duplicates(sig, 0.05, 0.35)
    found = {(p["a"], p["b"]): p["kind"] for p in pairs}
    key = lambda x, y: tuple(sorted([x.name, y.name]))  # noqa: E731
    assert found.get(key(clean_episode, copies["exact"])) == "exact"
    assert found.get(key(clean_episode, copies["reencode"])) == "near"
    assert found.get(key(clean_episode, copies["trimmed"])) == "near"
    assert found.get(key(clean_episode, copies["swap"])) == "video_reuse"  # same footage, other trajectory
    assert key(clean_episode, other) not in found  # distinct demo is never a duplicate
    roots = clusters(list(sig), pairs)
    assert len({roots[p.name] for p in (clean_episode, *list(copies.values())[:3])}) == 1
    assert dtw(trajs[clean_episode.name], trajs[clean_episode.name]) == 0


def test_phase_segmentation_on_known_profile():
    t = np.arange(0, 4, 1 / 30)
    x = np.where(t < 2, 0.3 * (10 * (t / 2) ** 3 - 15 * (t / 2) ** 4 + 6 * (t / 2) ** 5), 0.3)
    z = np.where(t < 2.4, 0.3, np.where(t < 3.4, 0.3 - 0.1 * (t - 2.4), 0.2))
    f = np.zeros((len(t), 6))
    f[(t > 3.0), 2] = -8
    phase, info = segment_phases(t, np.stack([x, np.zeros_like(x), z], 1), f)
    b1, b2, b3 = (t[b] for b in info["b"])
    assert 1.2 < b1 < 2.1 and abs(b2 - 3.0) < 0.15 and 3.3 < b3 < 3.6
    assert list(np.unique(phase)) == [0, 1, 2, 3] and np.all(np.diff(phase) >= 0)


def test_manifest_detects_changes(tmp_path):
    ds = tmp_path / "ds"
    (ds / "meta").mkdir(parents=True)
    (ds / "meta" / "info.json").write_text(json.dumps({"a": 1}))
    (ds / "data.bin").write_bytes(b"x" * 100)
    v = create_version(ds, tmp_path / "versions")
    assert verify(ds, tmp_path / "versions")["ok"]
    (ds / "data.bin").write_bytes(b"y" * 100)
    r = verify(ds, tmp_path / "versions", v["version"])
    assert not r["ok"] and r["changed"] == ["data.bin"]


def test_robot_configs_are_valid():
    for name in ("ur5_catheter", "ur5_sim", "franka_droid"):
        r = load_robot(name)
        assert r.sync.reference_stream in r.streams and r.required_streams
    real, sim = load_robot("ur5_catheter"), load_robot("ur5_sim")
    assert set(real.features) == set(sim.features)  # same schema -> mixable
