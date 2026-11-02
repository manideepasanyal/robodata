"""Onboard any LeRobot dataset from the Hub: pin revision, partial download, auto-generate configs.

``robodata onboard lerobot/pusht --max-episodes 50`` writes
  configs/robots/<name>.yaml    streams derived from info.json features (cameras by dtype, low-dim by key)
  configs/sources/<name>.yaml   source -> stream mapping + pinned revision + licence
  configs/datasets/<name>.yaml  build recipe (native fps, cameras downscaled to <= 320 px wide)
and downloads only the parquet / video files that contain the first N episodes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from robodata.config import CONFIG_DIR, DATA_ROOT, source_path

SKIP_KEYS = {"timestamp", "frame_index", "episode_index", "index", "task_index", "next.reward", "next.done",
             "next.success", "observation.is_first", "observation.is_last", "observation.is_terminal"}
CAM_CLOCKS = [  # rotated through cameras so every camera has its own imperfect clock
    {"offset_ms": 3.0, "drift_ppm": 25, "jitter_ms": 0.3, "latency_ms": 28, "latency_jitter_ms": 4, "drop_prob": 0.002},
    {"offset_ms": -6.0, "drift_ppm": -35, "jitter_ms": 0.4, "latency_ms": 34, "latency_jitter_ms": 5, "drop_prob": 0.002},
    {"offset_ms": 9.0, "drift_ppm": 50, "jitter_ms": 0.3, "latency_ms": 25, "latency_jitter_ms": 4, "drop_prob": 0.002},
    {"offset_ms": -2.0, "drift_ppm": -15, "jitter_ms": 0.3, "latency_ms": 31, "latency_jitter_ms": 4, "drop_prob": 0.002},
]
PROPRIO_CLOCK = {"drift_ppm": 3, "jitter_ms": 0.05, "latency_ms": 2, "latency_jitter_ms": 0.5}
CMD_CLOCK = {"latency_ms": 1, "latency_jitter_ms": 0.3}


def slug(repo_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", repo_id.split("/")[-1].lower()).strip("_")


def _flat_names(names, n: int) -> list[str]:
    if isinstance(names, dict):
        names = next(iter(names.values()), None)
    if isinstance(names, list) and len(names) == n and all(isinstance(x, str) for x in names):
        return names
    return [f"d{i}" for i in range(n)]


def _hwc(feat: dict) -> tuple[int, int, int]:
    shape, names = feat["shape"], feat.get("names") or []
    if names and names[0] in ("channel", "channels", "c"):
        return shape[1], shape[2], shape[0]
    return shape[0], shape[1], shape[2]


def _download_meta(repo_id: str, revision: str, local: Path) -> dict:
    snapshot_download(repo_id, repo_type="dataset", revision=revision, local_dir=local, allow_patterns=["meta/*", "meta/**/*", "README.md"])
    return json.loads((local / "meta" / "info.json").read_text())


def _needed_files(info: dict, local: Path, max_episodes: int | None) -> tuple[list[str], list[int]]:
    """Files holding the first N episodes (v3 aggregates many episodes per file)."""
    cams = [k for k, f in info["features"].items() if f["dtype"] == "video"]
    if info["codebase_version"].startswith("v2"):
        eps = [json.loads(l) for l in (local / "meta" / "episodes.jsonl").read_text().splitlines() if l]
        idx = [e["episode_index"] for e in eps][: max_episodes or None]
        cs = info.get("chunks_size", 1000)
        files = [info["data_path"].format(episode_chunk=i // cs, episode_index=i) for i in idx]
        files += [info["video_path"].format(episode_chunk=i // cs, video_key=k, episode_index=i) for i in idx for k in cams]
        return files, idx
    em = pd.concat([pd.read_parquet(p) for p in sorted((local / "meta" / "episodes").rglob("*.parquet"))])
    em = em.sort_values("episode_index").head(max_episodes or len(em))
    files = {info["data_path"].format(chunk_index=int(r["data/chunk_index"]), file_index=int(r["data/file_index"]))
             for _, r in em.iterrows()}
    for k in cams:
        files |= {info["video_path"].format(video_key=k, chunk_index=int(r[f"videos/{k}/chunk_index"]),
                                            file_index=int(r[f"videos/{k}/file_index"])) for _, r in em.iterrows()}
    return sorted(files), em.episode_index.astype(int).tolist()


def download_source(repo_id: str, revision: str, local_dir: str, episodes: list[int] | str = "all") -> dict:
    """Fetch exactly what a source config pins: meta + the files holding its episodes (or everything)."""
    local = source_path(local_dir)
    info = _download_meta(repo_id, revision, local)
    if episodes == "all":
        snapshot_download(repo_id, repo_type="dataset", revision=revision, local_dir=local)
        return {"files": "all"}
    files, _ = _needed_files(info, local, max(episodes) + 1)
    for f in files:
        hf_hub_download(repo_id, f, repo_type="dataset", revision=revision, local_dir=local)
    return {"files": len(files)}


def onboard(repo_id: str, max_episodes: int | None = 40, name: str | None = None, session_size: int = 10,
            max_width: int = 320, download: bool = True, revision: str | None = None) -> dict:
    """``revision`` pins the Hub commit; omitted = the current head (recorded in the generated configs)."""
    api = HfApi()
    meta = api.dataset_info(repo_id, revision=revision)
    revision = revision or meta.sha
    licence = next((t.split(":", 1)[1] for t in (meta.tags or []) if t.startswith("license:")), "unknown")
    name = name or slug(repo_id)
    local_rel = f"data/sources/{repo_id.replace('/', '__')}"
    local = source_path(local_rel)
    info = _download_meta(repo_id, revision, local)
    files, episodes = _needed_files(info, local, max_episodes)
    if download:
        for f in files:
            hf_hub_download(repo_id, f, repo_type="dataset", revision=revision, local_dir=local)
    fps = int(round(info["fps"]))
    feats = info["features"]

    streams, src_streams, robot_feats, recipe_feats, cams = {}, {}, {}, {}, []
    for i, (key, f) in enumerate(sorted((k, v) for k, v in feats.items() if v["dtype"] in ("video", "image"))):
        h, w, c = _hwc(f)
        cam = key.split(".")[-1]
        streams[cam] = {"kind": "rgb", "topic": f"/{cam}/image_raw", "msg_type": "sensor_msgs/msg/Image", "rate_hz": fps,
                        "shape": [h, w, 3], "dtype": "uint8", "known_latency_ms": CAM_CLOCKS[i % 4]["latency_ms"],
                        "clock": CAM_CLOCKS[i % 4]}
        src_streams[cam] = {"key": key}
        robot_feats[f"observation.images.{cam}"] = {"stream": cam}
        cams.append((cam, h, w))
    for key, f in feats.items():
        if key in SKIP_KEYS or not str(f["dtype"]).startswith("float") or len(f["shape"]) != 1:
            continue
        n = int(f["shape"][0])
        names = _flat_names(f.get("names"), n)
        stream = "action" if key == "action" else key.replace("observation.", "").replace(".", "_")
        kind = "joint_cmd" if key == "action" else "joint_state" if key == "observation.state" else "scalar"
        clock = CMD_CLOCK if kind == "joint_cmd" else PROPRIO_CLOCK
        streams[stream] = {"kind": kind, "topic": f"/{stream}", "msg_type": "sensor_msgs/msg/JointState", "rate_hz": fps,
                           "shape": [n], "names": names, "known_latency_ms": clock["latency_ms"], "clock": clock}
        src_streams[stream] = {"key": key}
        robot_feats[key] = {"stream": stream, "interp": "hold" if kind == "joint_cmd" else "linear"}
        recipe_feats[key] = {"names": names}
    if not cams:
        raise ValueError(f"{repo_id}: no camera features")
    # episode lengths drive the duration threshold (some datasets have 1-2 s episodes by design)
    lengths = _episode_lengths(info, local)
    min_len_s = float(np.percentile(lengths, 5)) / fps if len(lengths) else 2.0
    robot = {
        "robot_id": name, "robot_type": info.get("robot_type") or "unknown",
        "description": f"auto-generated from {repo_id}@{revision[:12]} (LeRobot {info['codebase_version']})",
        "target_fps": fps, "calibration_version": "uncalibrated",
        "sync": {"mode": "free_run", "reference_stream": cams[0][0], "clock_fit": "lower_envelope",
                 "max_frame_tolerance_ms": round(0.6 * 1000 / fps, 1)},
        "raw_video": {"codec": "libx264", "crf": 18, "preset": "veryfast", "gop": max(2, fps)},
        "health_overrides": {"min_duration_s": round(max(0.3, 0.5 * min_len_s), 2)},
        "streams": streams, "features": robot_feats,
    }
    source = {"name": name, "repo_id": repo_id, "revision": revision, "license": licence, "robot": name,
              "local_dir": local_rel, "episodes": episodes, "session_size": session_size,
              "operator": "unknown", "scene_id": f"{name}_scene", "success": None, "streams": src_streams}
    _, h0, w0 = cams[0]
    scale = min(1.0, max_width / max(w for _, _, w in cams))
    size = [int(round(h0 * scale / 2) * 2), int(round(w0 * scale / 2) * 2)]
    recipe = {"name": name, "repo_id": f"robodata/{name}", "fps": fps, "image_size": size,
              "robot_type": robot["robot_type"], "license": licence, "sources": [{"robot_id": name}],
              "cameras": [c for c, _, _ in cams], "depth_cameras": [], "features": recipe_feats,
              "rgb_encoder": {"vcodec": "libsvtav1", "crf": 30, "g": 2}, "phases": False,
              "curation": {"min_quality": 0.5, "split_by": "session_id", "split_ratios": {"train": 0.8, "val": 0.1, "eval": 0.1},
                           "seed": "robodata-v1"}}
    out = {}
    for sub, doc in (("robots", robot), ("sources", source), ("datasets", recipe)):
        p = CONFIG_DIR / sub / f"{name}.yaml"
        p.write_text(f"# auto-generated by `robodata onboard {repo_id}`\n" + yaml.safe_dump(doc, sort_keys=False, width=140))
        out[sub] = str(p)
    lock = DATA_ROOT / "sources" / "lock.json"
    lk = json.loads(lock.read_text()) if lock.exists() else {}
    lk[repo_id] = {"revision": revision, "path": str(local), "episodes": len(episodes), "files": len(files)}
    lock.write_text(json.dumps(lk, indent=2))
    return {"name": name, "repo_id": repo_id, "revision": revision, "license": licence, "episodes": len(episodes),
            "files": len(files), "cameras": [c for c, _, _ in cams], "fps": fps, "configs": out}


def _episode_lengths(info: dict, local: Path) -> np.ndarray:
    if info["codebase_version"].startswith("v2"):
        return np.array([json.loads(l)["length"] for l in (local / "meta" / "episodes.jsonl").read_text().splitlines() if l])
    em = pd.concat([pd.read_parquet(p, columns=["length"]) for p in sorted((local / "meta" / "episodes").rglob("*.parquet"))])
    return em["length"].to_numpy()
