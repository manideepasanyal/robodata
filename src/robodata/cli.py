"""robodata command line: every pipeline stage is one sub-command (see README for the full flow)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from robodata.config import CONFIG_DIR, DATA_ROOT, REPO_ROOT, source_path

DATA = DATA_ROOT
RAW, QUAR, REPORTS = DATA / "raw", DATA / "quarantine", DATA / "reports"


def _p(obj):
    print(json.dumps(obj, indent=2, default=str))


def cmd_fetch(a):
    """Download every source exactly as its config pins it: same revision, same episode subset."""
    import yaml

    from robodata.ingest.onboard import download_source

    names = sorted(p.stem for p in (CONFIG_DIR / "sources").glob("*.yaml")) if a.source == "all" else [a.source]
    lock_p = DATA / "sources" / "lock.json"
    lock = json.loads(lock_p.read_text()) if lock_p.exists() else {}
    for n in names:
        spec = yaml.safe_load((CONFIG_DIR / "sources" / f"{n}.yaml").read_text())
        if "repo_id" not in spec:
            continue  # generated data (the simulator), nothing to download
        r = download_source(spec["repo_id"], spec["revision"], spec["local_dir"], spec.get("episodes", "all"))
        lock[spec["repo_id"]] = {"revision": spec["revision"], "path": str(source_path(spec["local_dir"])), **r}
        print(f"{n}: {spec['repo_id']}@{spec['revision'][:12]} -> {source_path(spec['local_dir'])} ({r['files']} files)")
    lock_p.parent.mkdir(parents=True, exist_ok=True)
    lock_p.write_text(json.dumps(lock, indent=2))


def cmd_onboard(a):
    from robodata.ingest.onboard import onboard

    for repo in a.repo_ids:
        r = onboard(repo, max_episodes=a.max_episodes, name=a.name if len(a.repo_ids) == 1 else None,
                    session_size=a.session_size, download=not a.no_download, revision=a.revision)
        print(f"{repo}: {r['episodes']} eps, {r['files']} files, cams={r['cameras']} fps={r['fps']} "
              f"licence={r['license']} -> name '{r['name']}'")


def cmd_ingest(a):
    from robodata.ingest.virtual_robot import ingest

    out = ingest(a.source, RAW, max_episodes=a.max_episodes, workers=a.workers, apply_clock=not a.no_clock)
    print(f"{len(out)} raw episodes written under {RAW}")


def cmd_sim(a):
    if a.sim_cmd == "generate":
        from robodata.sim.generate import generate

        res = generate(a.out, episodes_per_scene=a.episodes_per_scene, workers=a.workers, base_seed=a.seed)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "generation_summary.json").write_text(json.dumps(res, indent=1, default=float))
        print(f"{len(res)} sim episodes, success rate {sum(r['success'] for r in res) / max(1, len(res)):.2f}")
    elif a.sim_cmd == "convert":
        from robodata.sim.convert import convert_dir

        out = convert_dir(a.spec, a.logs, RAW, workers=a.workers, apply_clock=not a.no_clock)
        print(f"{len(out)} sim episodes converted to raw")
    elif a.sim_cmd == "calibrate":
        from robodata.calib.sim import run_sim_calibration

        r = run_sim_calibration(n_views=a.views, save_as=a.name)
        _p({c: {"rms_px": v.get("rms_px"), "gt_error": v.get("gt_error")} for c, v in r["cameras"].items() if "K" in v})
        print("saved", r.get("path"))


def cmd_calibrate(a):
    import cv2

    from robodata.calib.core import calibrate_intrinsics, save_calibration
    import numpy as np

    imgs = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in sorted(Path(a.images).glob("*.png"))
            + sorted(Path(a.images).glob("*.jpg"))]
    cols, rows = a.pattern
    obj = np.array([[c * a.square, r * a.square, 0] for r in range(rows) for c in range(cols)], np.float32)
    r = calibrate_intrinsics(imgs, (cols, rows), obj)
    path = save_calibration(a.robot, a.name, {a.camera: {"K": r["K"], "dist": r["dist"], "rms_px": r["rms_px"],
                                                         "width": r["size"][0], "height": r["size"][1]}},
                            {"intrinsics": "opencv chessboard", "pattern": [cols, rows], "square_m": a.square})
    print(f"rms {r['rms_px']:.3f}px from {len(r['views_used'])} views -> {path}")


def cmd_sync(a):
    from concurrent.futures import ProcessPoolExecutor

    paths = [str(p) for p in __import__("robodata.io.raw_episode", fromlist=["x"]).list_episodes(RAW, a.robot)]
    with ProcessPoolExecutor(a.workers) as ex:
        reps = list(ex.map(_sync_one, paths))
    import numpy as np

    res = [r["residual_sync_error_p95_ms"] for r in reps if r.get("residual_sync_error_p95_ms") is not None]
    gt = [r["gt_sync_error_p95_ms"] for r in reps if r.get("gt_sync_error_p95_ms") is not None]
    print(f"synced {len(reps)} episodes; residual p95 median {np.median(res):.2f} ms"
          + (f"; vs ground truth median {np.median(gt):.2f} ms" if gt else ""))


def _sync_one(path):
    from robodata.config import load_robot
    from robodata.io.raw_episode import RawEpisode
    from robodata.sync.run import sync_episode

    ep = RawEpisode(path)
    return sync_episode(ep, load_robot(ep.meta.robot_id))


def cmd_faults(a):
    from robodata.config import load_robot
    from robodata.health.faults import default_suite, inject
    from robodata.io.raw_episode import list_episodes

    robot = load_robot(a.robot)
    clean = [p for p in list_episodes(RAW, a.robot) if "__fault-" not in p.name][: a.episodes]
    rgb = [s for s, c in robot.streams.items() if c.kind == "rgb"]
    cam = next((s for s in rgb if "wrist" in s), rgb[0])
    depth = next((s for s, c in robot.streams.items() if c.kind == "depth"), None)
    lowdim = next(s for s, c in robot.streams.items() if c.kind in ("joint_state", "pose", "joint_cmd"))
    wrench = next((s for s, c in robot.streams.items() if c.kind == "wrench"), None)
    n = 0
    for i, src in enumerate(clean):
        for tag, faults, _ in default_suite(cam, depth, lowdim, wrench):
            inject(src, faults, tag, seed=i)
            n += 1
        inject(src, [{"type": "near_duplicate"}], "dup", seed=i)
        n += 1
    print(f"injected {n} faulty / duplicate episodes from {len(clean)} clean ones")


def cmd_dupes(a):
    """Plant duplicates with ground truth (meta.extra.duplicate_of) to measure dedup recall/precision."""
    import yaml

    from robodata.config import load_robot
    from robodata.health.faults import inject
    from robodata.ingest.lerobot_source import LeRobotSource
    from robodata.ingest.virtual_robot import load_source, replay_episode
    from robodata.io.raw_episode import RawEpisode, list_episodes

    robot = load_robot(a.robot)
    eps = [p for p in list_episodes(RAW, a.robot) if "__fault-" not in p.name and "_reupload" not in p.parent.name]
    sessions = sorted({p.parent.name for p in eps})
    src_name = next((f.stem for f in (CONFIG_DIR / "sources").glob("*.yaml")
                     if (d := yaml.safe_load(f.read_text())).get("robot") == a.robot and "repo_id" in d), None)
    made = 0
    for k, src in enumerate(eps[: a.n]):
        other_session = next(s for s in reversed(sessions) if s != src.parent.name) if len(sessions) > 1 else None
        donor = eps[-(k + 1)]  # a different episode whose footage gets swapped in
        jobs = [("dup_exact", [{"type": "exact_duplicate"}], None),
                ("dup_exact_xsession", [{"type": "exact_duplicate"}], other_session),
                ("dup_reencode", [{"type": "reencode_duplicate"}], other_session),
                ("dup_trim", [{"type": "trimmed_duplicate", "frac": 0.08}], other_session)]
        for tag, faults, sess in jobs:
            inject(src, faults, tag, seed=k, session=sess)
            made += 1
        inject(donor, [{"type": "video_swap", "video_from": str(src)}], "videoswap", seed=k)
        made += 1
        if src_name:  # the same source recording ingested a second time ("uploaded twice")
            spec = load_source(src_name)
            meta = RawEpisode(src).meta
            lr = LeRobotSource(source_path(spec.local_dir))
            replay_episode(spec, robot, lr.by_index(meta.extra["source_episode_index"]), 900 + k, RAW,
                           session=f"{src_name}_reupload", salt="reupload",
                           extra={"duplicate_of": meta.episode_id, "reupload": True})
            made += 1
    print(f"{a.robot}: planted {made} duplicates from {min(a.n, len(eps))} source episodes")


def cmd_health(a):
    from robodata.config import load_health
    from robodata.health.report import quarantine, run_health

    h = load_health(a.config)
    df = run_health(RAW, REPORTS / "health", h, robot_id=a.robot, workers=a.workers)
    print(df.status.value_counts().to_string())
    print(f"report: {REPORTS / 'health' / 'health_report.md'}")
    if a.quarantine:
        moved = quarantine(df, RAW, QUAR)
        print(f"quarantined {len(moved)} episodes -> {QUAR}")


def cmd_unquarantine(a):
    import shutil

    from robodata.io.raw_episode import list_episodes

    n = 0
    for src in list_episodes(QUAR, a.robot):
        dst = RAW / src.relative_to(QUAR)
        (src / "QUARANTINE.json").unlink(missing_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        n += 1
    print(f"restored {n} episodes to {RAW}")


def cmd_curate(a):
    from robodata.config import load_recipe
    from robodata.curation.curate import curate

    recipe = load_recipe(a.recipe)
    _p(curate(recipe, RAW, DATA / "curated" / recipe.name, workers=a.workers))


def cmd_build(a):
    from robodata.config import load_recipe
    from robodata.preprocess.build import build_dataset

    recipe = load_recipe(a.recipe)
    r = build_dataset(recipe, DATA / "curated" / recipe.name, DATA / "datasets", limit_per_split=a.limit_per_split,
                      shards=a.shards)
    print(f"built {r['episodes']} episodes -> {r['root']} splits={r['ranges']}")


def cmd_inspect(a):
    from robodata.preprocess.inspect import inspect_dataset

    out = inspect_dataset(DATA / "datasets" / a.dataset, REPORTS / "inspect" / a.dataset, a.episodes, video=a.video)
    print(f"inspection report: {out}")


def cmd_bench(a):
    if a.what == "video":
        from robodata.storage.bench_video import run_video_bench

        print(run_video_bench(RAW, REPORTS / "bench", n_episodes=a.episodes))
    else:
        from robodata.storage.bench_formats import run_format_bench

        print(run_format_bench(RAW, REPORTS / "bench", n_episodes=a.episodes))


def cmd_derive(a):
    """Derived view without some features (e.g. an RGB-only copy for policies with 3-channel backbones)."""
    import shutil

    from lerobot.datasets.dataset_tools import remove_feature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    src_root = DATA / "datasets" / a.dataset
    info = json.loads((src_root / "meta" / "info.json").read_text())
    drop = list(a.drop)
    if a.drop_depth:
        drop += [k for k, f in info["features"].items() if (f.get("info") or {}).get("is_depth_map")]
    out = DATA / "datasets" / a.name
    shutil.rmtree(out, ignore_errors=True)
    ds = LeRobotDataset(info.get("repo_id", f"robodata/{a.dataset}"), root=src_root)
    remove_feature(ds, drop, output_dir=out, repo_id=f"robodata/{a.name}")
    new_info = json.loads((out / "meta" / "info.json").read_text())
    new_info["splits"] = info.get("splits", {})  # episodes are unchanged, so split ranges carry over
    (out / "meta" / "info.json").write_text(json.dumps(new_info, indent=4))
    if (src_root / "meta" / "robodata").exists():
        shutil.copytree(src_root / "meta" / "robodata", out / "meta" / "robodata", dirs_exist_ok=True)
        (out / "meta" / "robodata" / "derived_from.json").write_text(json.dumps({"dataset": a.dataset, "dropped": drop}))
    print(f"{out}: dropped {drop}")


def cmd_audit(a):
    from robodata.verify.audit import audit_dataset

    for name in a.datasets:
        r = audit_dataset(name, DATA)
        print(f"== {name}: {'PASS' if r['ok'] else 'FAIL'}")
        for part in ("structure", "leakage", "fidelity"):
            x = r[part]
            print(f"  {part:9s} {'ok' if x['ok'] else 'FAIL'}  " + ", ".join(
                f"{k}={v}" for k, v in x.items() if k not in ("ok", "problems", "results", "cross_split_near_identical")))
            for p in x["problems"][:8]:
                print(f"     - {p}")


def cmd_dedup_eval(a):
    """Score dedup against the planted duplicates (robodata dupes) across curated datasets."""
    from robodata.verify.dedup_eval import evaluate

    names = a.datasets or [l.split()[0] for l in (CONFIG_DIR / "hub_datasets.txt").read_text().splitlines()
                           if l.strip() and not l.startswith("#")]
    r = evaluate(DATA / "curated", names, RAW)
    df, fam = r["per_kind"], r["families"]
    (DATA / "reports").mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA / "reports" / "dedup_eval.csv", index=False)
    fam.to_csv(DATA / "reports" / "dedup_families.csv", index=False)
    planted = df[df.kind != "FALSE_POSITIVE(originals)"].groupby("kind")[["planted_in_catalog", "caught"]].sum()
    fp = df[df.kind == "FALSE_POSITIVE(originals)"]
    print(planted.to_string())
    print(f"false positives on genuine episodes: {int(fp.caught.sum())} / {int(fp.planted_in_catalog.sum())}")
    print(f"duplicate families with >1 survivor: {int((fam.survivors > 1).sum())} / {len(fam)}")


def cmd_card(a):
    from robodata.versioning.card import write_dataset_card

    print(write_dataset_card(DATA / "datasets" / a.dataset))


def cmd_version(a):
    from robodata.versioning.manifest import create_version

    _p(create_version(DATA / "datasets" / a.dataset, DATA / "versions", note=a.note))


def cmd_verify(a):
    from robodata.versioning.manifest import verify

    r = verify(DATA / "datasets" / a.dataset, DATA / "versions", a.version)
    _p(r)
    sys.exit(0 if r["ok"] else 1)


def cmd_train_smoke(a):
    from robodata.train.smoke import run_smoke

    _p(run_smoke(DATA / "datasets" / a.dataset, steps=a.steps, batch_size=a.batch_size, sim_weight=a.sim_weight,
                 device=a.device))


def cmd_pipeline(a):
    """Everything after raw data exists: sync -> health (+quarantine) -> curate -> build -> card -> version."""
    ns = argparse.Namespace
    from robodata.config import load_recipe

    recipe = load_recipe(a.recipe)
    for src in recipe.sources:
        cmd_sync(ns(robot=src.robot_id, workers=a.workers))
        cmd_health(ns(config="default", robot=src.robot_id, workers=a.workers, quarantine=True))
    cmd_curate(ns(recipe=a.recipe, workers=a.workers))
    cmd_build(ns(recipe=a.recipe, limit_per_split=None, shards=8))
    cmd_card(ns(dataset=recipe.name))
    cmd_version(ns(dataset=recipe.name, note="pipeline run"))


def main(argv=None):
    p = argparse.ArgumentParser("robodata", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("fetch", help="download pinned open-source datasets")
    s.add_argument("--source", default="all")
    s.set_defaults(fn=cmd_fetch)
    s = sub.add_parser("onboard", help="pin + partially download LeRobot hub datasets and auto-generate configs")
    s.add_argument("repo_ids", nargs="+")
    s.add_argument("--max-episodes", type=int, default=40)
    s.add_argument("--name")
    s.add_argument("--session-size", type=int, default=10)
    s.add_argument("--no-download", action="store_true")
    s.add_argument("--revision", help="pin a Hub commit (default: current head, recorded in the configs)")
    s.set_defaults(fn=cmd_onboard)
    s = sub.add_parser("ingest", help="replay an open-source dataset through the virtual robot into raw episodes")
    s.add_argument("source")
    s.add_argument("--max-episodes", type=int)
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--no-clock", action="store_true", help="disable the per-sensor clock model")
    s.set_defaults(fn=cmd_ingest)
    s = sub.add_parser("sim", help="simulated demonstrations")
    ss = s.add_subparsers(dest="sim_cmd", required=True)
    g = ss.add_parser("generate")
    g.add_argument("--out", default=str(DATA / "sim_logs" / "ur5_insertion_v1"))
    g.add_argument("--episodes-per-scene", type=int, default=10)
    g.add_argument("--workers", type=int, default=6)
    g.add_argument("--seed", type=int, default=1000)
    g = ss.add_parser("convert")
    g.add_argument("--spec", default="sim_ur5")
    g.add_argument("--logs", default=str(DATA / "sim_logs" / "ur5_insertion_v1"))
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--no-clock", action="store_true")
    g = ss.add_parser("calibrate")
    g.add_argument("--views", type=int, default=60)
    g.add_argument("--name", default="v1")
    s.set_defaults(fn=cmd_sim)
    s = sub.add_parser("calibrate", help="intrinsics from real chessboard images")
    s.add_argument("--images", required=True)
    s.add_argument("--robot", required=True)
    s.add_argument("--camera", required=True)
    s.add_argument("--pattern", type=int, nargs=2, default=[8, 5])
    s.add_argument("--square", type=float, default=0.025)
    s.add_argument("--name", default="v1")
    s.set_defaults(fn=cmd_calibrate)
    s = sub.add_parser("sync", help="clock fits + residual sync error for raw episodes")
    s.add_argument("--robot")
    s.add_argument("--workers", type=int, default=8)
    s.set_defaults(fn=cmd_sync)
    s = sub.add_parser("faults", help="inject faults into copies of clean episodes (health-check validation)")
    s.add_argument("--robot", required=True)
    s.add_argument("--episodes", type=int, default=2)
    s.set_defaults(fn=cmd_faults)
    s = sub.add_parser("dupes", help="plant exact / cross-session / re-encoded / trimmed / re-uploaded / video-swap duplicates")
    s.add_argument("--robot", required=True)
    s.add_argument("--n", type=int, default=3)
    s.set_defaults(fn=cmd_dupes)
    s = sub.add_parser("health", help="per-episode health report (+ quarantine)")
    s.add_argument("--config", default="default")
    s.add_argument("--robot")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--quarantine", action="store_true")
    s.set_defaults(fn=cmd_health)
    s = sub.add_parser("unquarantine", help="move quarantined episodes back (e.g. after changing thresholds)")
    s.add_argument("--robot")
    s.set_defaults(fn=cmd_unquarantine)
    s = sub.add_parser("curate", help="quality filter, dedup, leakage-free splits")
    s.add_argument("recipe")
    s.add_argument("--workers", type=int, default=8)
    s.set_defaults(fn=cmd_curate)
    s = sub.add_parser("build", help="write the training-ready LeRobot dataset")
    s.add_argument("recipe")
    s.add_argument("--limit-per-split", type=int)
    s.add_argument("--shards", type=int, default=8, help="parallel writer processes (merged with aggregate_datasets)")
    s.set_defaults(fn=cmd_build)
    s = sub.add_parser("inspect", help="visual inspection report for built episodes")
    s.add_argument("dataset")
    s.add_argument("--episodes", type=int, nargs="*")
    s.add_argument("--video", action="store_true")
    s.set_defaults(fn=cmd_inspect)
    s = sub.add_parser("bench", help="storage benchmarks")
    s.add_argument("what", choices=["video", "formats"])
    s.add_argument("--episodes", type=int, default=3)
    s.set_defaults(fn=cmd_bench)
    s = sub.add_parser("derive", help="derived dataset view with features removed (e.g. --drop-depth)")
    s.add_argument("dataset")
    s.add_argument("--name", required=True)
    s.add_argument("--drop", nargs="*", default=[])
    s.add_argument("--drop-depth", action="store_true")
    s.set_defaults(fn=cmd_derive)
    s = sub.add_parser("audit", help="independent cross-verification of built datasets against their sources")
    s.add_argument("datasets", nargs="+")
    s.set_defaults(fn=cmd_audit)
    s = sub.add_parser("dedup-eval", help="score dedup recall / false positives against planted duplicates")
    s.add_argument("datasets", nargs="*")
    s.set_defaults(fn=cmd_dedup_eval)
    s = sub.add_parser("card", help="write README dataset card")
    s.add_argument("dataset")
    s.set_defaults(fn=cmd_card)
    s = sub.add_parser("version", help="hash manifest -> immutable dataset version")
    s.add_argument("dataset")
    s.add_argument("--note", default="")
    s.set_defaults(fn=cmd_version)
    s = sub.add_parser("verify", help="check a dataset against a version manifest")
    s.add_argument("dataset")
    s.add_argument("--version")
    s.set_defaults(fn=cmd_verify)
    s = sub.add_parser("train-smoke", help="train a small policy through the LeRobot loader (real+sim mix)")
    s.add_argument("dataset")
    s.add_argument("--steps", type=int, default=200)
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--sim-weight", type=float, default=0.5, help="target fraction of sim samples per batch")
    s.add_argument("--device", default="auto")
    s.set_defaults(fn=cmd_train_smoke)
    s = sub.add_parser("pipeline", help="sync -> health -> curate -> build -> card -> version")
    s.add_argument("recipe")
    s.add_argument("--workers", type=int, default=8)
    s.set_defaults(fn=cmd_pipeline)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
