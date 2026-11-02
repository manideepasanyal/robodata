"""Run health checks over a raw root, write the aggregated report and quarantine failures."""

from __future__ import annotations

import html
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from robodata.config import HealthThresholds, load_robot
from robodata.health.checks import check_episode
from robodata.io.raw_episode import RawEpisode, list_episodes


def _run_one(path: str, thresholds: dict, resync: bool) -> dict:
    ep = RawEpisode(path)
    rep = check_episode(ep, load_robot(ep.meta.robot_id), HealthThresholds(**thresholds), resync=resync)
    rep["path"] = path
    rep["injected_faults"] = [f["type"] for f in ep.meta.injected_faults]
    rep["is_sim"] = ep.meta.is_sim
    return rep


def run_health(raw_root: str | Path, report_dir: str | Path, h: HealthThresholds, robot_id: str | None = None,
               workers: int = 8, resync: bool = True) -> pd.DataFrame:
    paths = [str(p) for p in list_episodes(raw_root, robot_id)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        reports = list(ex.map(_run_one, paths, [h.model_dump()] * len(paths), [resync] * len(paths)))
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = [{
        "episode_id": r["episode_id"], "robot_id": r["robot_id"], "is_sim": r["is_sim"], "status": r["status"],
        "failed": ";".join(r["failed"]), "warned": ";".join(r["warned"]),
        "injected_faults": ";".join(r["injected_faults"]),
        "sync_residual_p95_ms": r["sync"]["residual_sync_error_p95_ms"],
        "gt_sync_error_p95_ms": r["sync"]["gt_sync_error_p95_ms"],
        "inter_camera_skew_p95_ms": r["sync"]["inter_camera_skew_p95_ms"], "path": r["path"],
    } for r in reports]
    df = pd.DataFrame(rows).sort_values("episode_id")
    df.to_csv(report_dir / "health_summary.csv", index=False)
    (report_dir / "health_reports.json").write_text(json.dumps(reports, indent=1, default=str))
    (report_dir / "health_report.md").write_text(_markdown(df, h))
    (report_dir / "health_report.html").write_text(_html(df, reports, h))
    return df


def quarantine(df: pd.DataFrame, raw_root: str | Path, quarantine_root: str | Path) -> list[str]:
    """Move failing episodes out of the raw root (kept, not deleted, with the reason alongside)."""
    moved = []
    for _, r in df[df.status == "fail"].iterrows():
        src = Path(r.path)
        if not src.exists():
            continue
        dst = Path(quarantine_root) / src.relative_to(raw_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
        (dst / "QUARANTINE.json").write_text(json.dumps({
            "reason": r.failed.split(";"), "quarantined_at": datetime.now(timezone.utc).isoformat()}, indent=2))
        moved.append(r.episode_id)
    return moved


def _markdown(df: pd.DataFrame, h: HealthThresholds) -> str:
    counts = df.status.value_counts().to_dict()
    lines = ["# Dataset health report", "", f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}", "",
             f"**Episodes:** {len(df)} | pass {counts.get('pass', 0)} | warn {counts.get('warn', 0)} | "
             f"fail {counts.get('fail', 0)}", ""]
    lines += ["## Failures by check", ""]
    fails = df.failed[df.failed != ""].str.split(";").explode()
    for k, v in fails.str.split(":").str[0].value_counts().items():
        lines.append(f"- `{k}`: {v}")
    inj = df[df.injected_faults != ""]
    if len(inj):
        detected = (inj.status == "fail").mean()
        clean = df[df.injected_faults == ""]
        lines += ["", "## Fault-injection validation", "",
                  f"- injected episodes: {len(inj)}, detected (fail): {detected:.0%}",
                  f"- clean episodes failing (false alarms): {(clean.status == 'fail').sum()} / {len(clean)}", ""]
        lines += ["| episode | injected | failed checks |", "|---|---|---|"]
        for _, r in inj.iterrows():
            lines.append(f"| {r.episode_id[-40:]} | {r.injected_faults} | {r.failed or '-'} |")
    sync = df.sync_residual_p95_ms.dropna()
    if len(sync):
        lines += ["", "## Residual sync error (p95 per episode, ms)", "",
                  f"- median {sync.median():.2f}, worst {sync.max():.2f} (fail > {h.max_sync_residual_p95_ms})"]
        gt = df.gt_sync_error_p95_ms.dropna()
        if len(gt):
            lines.append(f"- vs ground truth clock: median {gt.median():.2f}, worst {gt.max():.2f}")
    lines += ["", "## Thresholds", "", "```json", json.dumps(h.model_dump(), indent=2), "```"]
    return "\n".join(lines) + "\n"


def _html(df: pd.DataFrame, reports: list[dict], h: HealthThresholds) -> str:
    colors = {"pass": "#1a7f37", "warn": "#9a6700", "fail": "#cf222e"}
    rows = []
    for r in sorted(reports, key=lambda r: r["episode_id"]):
        bad = [c for c in r["checks"] if c["status"] != "pass"]
        det = "".join(f"<li><b>{html.escape(c['check'])}</b> {html.escape(str(c['stream'] or ''))}: "
                      f"{c['value']} (thr {c['threshold']}) {html.escape(c['detail'])}</li>" for c in bad)
        rows.append(f"<tr><td>{html.escape(r['episode_id'])}</td><td style='color:{colors[r['status']]};font-weight:600'>"
                    f"{r['status']}</td><td>{r['sync']['residual_sync_error_p95_ms'] or ''}</td>"
                    f"<td><ul>{det}</ul></td></tr>")
    counts = df.status.value_counts().to_dict()
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Health report</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;color:#1f2328}}table{{border-collapse:collapse;width:100%}}
td,th{{border-bottom:1px solid #d0d7de;padding:6px;text-align:left;vertical-align:top;font-size:13px}}ul{{margin:0;padding-left:16px}}</style>
</head><body><h1>Dataset health report</h1>
<p>{len(df)} episodes - pass {counts.get('pass', 0)}, warn {counts.get('warn', 0)}, fail {counts.get('fail', 0)}</p>
<table><tr><th>episode</th><th>status</th><th>sync p95 ms</th><th>non-passing checks</th></tr>{''.join(rows)}</table>
</body></html>"""
