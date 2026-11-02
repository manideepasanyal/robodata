"""Visual inspection of built (aligned) episodes, read back through the LeRobot loader.

Per episode: a contact sheet (every camera incl. colour-mapped depth at the phase
boundaries), signal plots with phase shading and per-frame sync residual, and
optionally a tiled synchronised MP4. Everything links from one index.html.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from robodata import schema  # noqa: E402
from robodata.io.video import VideoWriter  # noqa: E402

PHASE_COLORS = ["#dbeafe", "#fef3c7", "#fee2e2", "#dcfce7"]


def _to_img(t, key: str) -> np.ndarray:
    a = t.numpy()
    if key.endswith(schema.DEPTH_SUFFIX):
        d = a[0]
        valid = d > 60
        lo, hi = (np.percentile(d[valid], [2, 98]) if valid.any() else (0, 1))
        u8 = (np.clip((d - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8)
        img = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)[..., ::-1]
        img[~valid] = 0
        return img
    return (np.transpose(a, (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def inspect_dataset(root: str | Path, out_dir: str | Path, episodes: list[int] | None = None, video: bool = False,
                    max_episodes: int = 6) -> Path:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root, out_dir = Path(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = json.loads((root / "meta" / "info.json").read_text())
    ds = LeRobotDataset(info.get("repo_id", f"robodata/{root.name}"), root=root)
    data = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet"))])
    meta_p = root / "meta" / "robodata" / "episodes.parquet"
    emeta = pd.read_parquet(meta_p).set_index("episode_index") if meta_p.exists() else None
    phase_names = (json.loads((root / "meta" / "robodata" / "phases.json").read_text())["names"]
                   if (root / "meta" / "robodata" / "phases.json").exists() else [])
    cam_keys = [k for k, f in ds.meta.features.items() if f["dtype"] == "video"]
    if episodes is None:  # a few from each split / domain
        if emeta is not None:
            episodes = (emeta.reset_index().groupby(["split", "is_sim"]).episode_index.first().tolist())[:max_episodes]
        else:
            episodes = list(range(min(max_episodes, ds.num_episodes)))
    cards = []
    for e in episodes:
        df = data[data.episode_index == e].sort_values("frame_index")
        idx0 = int(df["index"].iloc[0])
        n = len(df)
        t = df.timestamp.to_numpy()
        phase = np.stack(df[schema.PHASE].to_numpy()).ravel() if schema.PHASE in df else np.zeros(n, int)
        bounds = [0] + [int(i) for i in np.where(np.diff(phase) != 0)[0] + 1]
        picks = sorted(set([*bounds, n // 2, n - 1]))[:6]
        # --- contact sheet --------------------------------------------------------------
        rows = []
        for key in cam_keys:
            tiles = []
            for i in picks:
                s = ds[idx0 + i]
                ph = phase_names[phase[i]] if phase_names else str(phase[i])
                tiles.append(_label(_to_img(s[key], key), f"{key.split('.')[-1]} t={t[i]:.2f}s {ph}"))
            rows.append(np.concatenate(tiles, 1))
        sheet = np.concatenate(rows, 0)
        sheet_p = out_dir / f"ep{e:04d}_frames.jpg"
        cv2.imwrite(str(sheet_p), sheet[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
        # --- signals ----------------------------------------------------------------------
        series = [k for k in ("observation.state", "action", "observation.ee_pose", "observation.force_torque",
                              "observation.cartesian_state", schema.SYNC_RESIDUAL) if k in df]
        fig, axes = plt.subplots(len(series), 1, figsize=(11, 2.0 * len(series)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, k in zip(axes, series):
            x = np.stack(df[k].to_numpy()).reshape(n, -1)
            names = ds.meta.features[k].get("names") or [k]
            if k == "observation.ee_pose":
                x, names = x[:, :3], names[:3]
            for j in range(x.shape[1]):
                ax.plot(t, x[:, j], lw=1, label=str(names[j]) if j < 7 else None)
            for a, b in zip(bounds, bounds[1:] + [n]):
                ax.axvspan(t[a], t[min(b, n - 1)], color=PHASE_COLORS[phase[a] % 4], alpha=0.6, lw=0)
            ax.set_ylabel(k.replace("observation.", ""), fontsize=8)
            ax.legend(fontsize=6, ncol=7, loc="upper right")
        axes[-1].set_xlabel("time [s] (phases shaded: " + ", ".join(phase_names) + ")")
        fig.tight_layout()
        sig_p = out_dir / f"ep{e:04d}_signals.png"
        fig.savefig(sig_p, dpi=90)
        plt.close(fig)
        vid_p = None
        if video:
            vid_p = out_dir / f"ep{e:04d}_mosaic.mp4"
            vw = None
            for i in range(n):
                s = ds[idx0 + i]
                tiles = [_label(_to_img(s[k], k), k.split(".")[-1]) for k in cam_keys]
                cols = 3
                while len(tiles) % cols:
                    tiles.append(np.zeros_like(tiles[0]))
                grid = np.concatenate([np.concatenate(tiles[r:r + cols], 1) for r in range(0, len(tiles), cols)], 0)
                grid = _label(grid, f"ep {e} frame {i} t={t[i]:.2f}s phase={phase_names[phase[i]] if phase_names else phase[i]} "
                                    f"sim={bool(s[schema.IS_SIM])} sync_res={float(s[schema.SYNC_RESIDUAL]):.1f}ms")
                if vw is None:
                    vw = VideoWriter(vid_p, ds.fps, grid.shape[1] // 2 * 2, grid.shape[0] // 2 * 2, crf=23, preset="fast")
                vw.write(grid[: grid.shape[0] // 2 * 2, : grid.shape[1] // 2 * 2])
            vw.close()
        m = emeta.loc[e].to_dict() if emeta is not None and e in emeta.index else {}
        cards.append((e, m, sheet_p.name, sig_p.name, vid_p.name if vid_p else None))
    body = []
    for e, m, sheet, sig, vid in cards:
        facts = " | ".join(f"{k}: {html.escape(str(m.get(k)))}" for k in
                           ("episode_id", "split", "is_sim", "success", "quality", "health", "phase_boundaries",
                            "sync_residual_p95_ms") if k in m)
        v = f'<p><a href="{vid}">synchronised mosaic video</a></p>' if vid else ""
        body.append(f"<section><h2>Episode {e}</h2><p class='m'>{facts}</p><img src='{sheet}'><img src='{sig}'>{v}</section>")
    index = out_dir / "index.html"
    index.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>{root.name} inspection</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;color:#1f2328;background:#fff}}img{{max-width:100%;display:block;margin:8px 0}}
.m{{font-size:13px;color:#57606a}}section{{border-bottom:1px solid #d0d7de;padding-bottom:16px}}</style></head>
<body><h1>{root.name} - episode inspection</h1>{''.join(body)}</body></html>""")
    return index
