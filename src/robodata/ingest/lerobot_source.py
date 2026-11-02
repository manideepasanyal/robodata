"""Reader for open-source LeRobot datasets (codebase v2.1 per-episode files and v3.0 aggregated files).

Deliberately independent of the ``lerobot`` package: it only needs parquet + mp4, so it
also works on datasets whose codebase version the installed lerobot can't load.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import io

import av
import numpy as np
import pandas as pd
from PIL import Image


@dataclass
class SourceEpisode:
    index: int
    task: str
    fps: float
    frames: pd.DataFrame  # one row per source frame
    video_files: dict[str, tuple[Path, float, float]]  # key -> (path, from_ts, to_ts)

    def __len__(self) -> int:
        return len(self.frames)

    def column(self, key: str) -> np.ndarray:
        return np.stack(self.frames[key].to_numpy()).astype(np.float32)

    def iter_video(self, key: str):
        """Yield RGB frames of this episode for one camera (handles v3 multi-episode files and image features)."""
        if key not in self.video_files:  # 'image' dtype: PNG/JPEG bytes stored in the parquet rows
            for cell in self.frames[key]:
                raw = cell["bytes"] if isinstance(cell, dict) else cell
                yield np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
            return
        path, t0, t1 = self.video_files[key]
        n = len(self)
        half = 0.5 / self.fps
        with av.open(str(path)) as c:
            s = c.streams.video[0]
            s.thread_type = "AUTO"
            if t0 > 0:
                c.seek(int(max(0.0, t0 - 1.0) / s.time_base), stream=s, backward=True, any_frame=False)
            k = 0
            for fr in c.decode(s):
                ts = float(fr.pts * s.time_base) if fr.pts is not None else None
                if ts is not None and ts < t0 - half:
                    continue
                if k >= n or (ts is not None and ts > t1 + half):
                    break
                yield fr.to_ndarray(format="rgb24")
                k += 1


class LeRobotSource:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        self.version = self.info["codebase_version"]
        self.fps = float(self.info["fps"])
        self.video_keys = [k for k, v in self.info["features"].items() if v["dtype"] == "video"]
        if self.version.startswith("v2"):
            eps = [json.loads(l) for l in (self.root / "meta" / "episodes.jsonl").read_text().splitlines() if l]
            cs = self.info.get("chunks_size", 1000)
            # partial downloads: keep episodes whose parquet is present locally
            self._episodes = [e for e in eps if (self.root / self.info["data_path"].format(
                episode_chunk=e["episode_index"] // cs, episode_index=e["episode_index"])).exists()]
        elif self.version.startswith("v3"):
            files = sorted((self.root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
            df = pd.concat([pd.read_parquet(f) for f in files]).sort_values("episode_index").reset_index(drop=True)
            have = df.apply(lambda r: (self.root / self.info["data_path"].format(
                chunk_index=int(r["data/chunk_index"]), file_index=int(r["data/file_index"]))).exists(), axis=1)
            self._ep_df = df[have].reset_index(drop=True)
            self._episodes = self._ep_df[["episode_index", "length"]].to_dict("records")
            self._data_cache: dict[tuple[int, int], pd.DataFrame] = {}
        else:
            raise ValueError(f"unsupported LeRobot codebase version {self.version}")
        self._pos = {int(e["episode_index"]): i for i, e in enumerate(self._episodes)}

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    @property
    def episode_indices(self) -> list[int]:
        return [int(e["episode_index"]) for e in self._episodes]

    def by_index(self, episode_index: int) -> SourceEpisode:
        return self.episode(self._pos[int(episode_index)])

    def episode(self, i: int) -> SourceEpisode:
        if self.version.startswith("v2"):
            return self._episode_v2(i)
        return self._episode_v3(i)

    def _episode_v2(self, i: int) -> SourceEpisode:
        ep = self._episodes[i]
        idx = ep["episode_index"]
        chunk = idx // self.info.get("chunks_size", 1000)
        df = pd.read_parquet(self.root / self.info["data_path"].format(episode_chunk=chunk, episode_index=idx))
        vids = {}
        for k in self.video_keys:
            p = self.root / self.info["video_path"].format(episode_chunk=chunk, video_key=k, episode_index=idx)
            vids[k] = (p, 0.0, float(len(df) - 1) / self.fps)
        return SourceEpisode(idx, ep["tasks"][0], self.fps, df, vids)

    def _episode_v3(self, i: int) -> SourceEpisode:
        row = self._ep_df.iloc[i]
        key = (int(row["data/chunk_index"]), int(row["data/file_index"]))
        if key not in self._data_cache:
            self._data_cache = {key: pd.read_parquet(
                self.root / self.info["data_path"].format(chunk_index=key[0], file_index=key[1]))}
        data = self._data_cache[key]
        df = data[data["episode_index"] == row["episode_index"]].sort_values("frame_index").reset_index(drop=True)
        vids = {}
        for k in self.video_keys:
            p = self.root / self.info["video_path"].format(
                video_key=k, chunk_index=int(row[f"videos/{k}/chunk_index"]), file_index=int(row[f"videos/{k}/file_index"]))
            vids[k] = (p, float(row[f"videos/{k}/from_timestamp"]), float(row[f"videos/{k}/to_timestamp"]))
        tasks = row["tasks"]
        task = str(tasks[0]) if len(tasks) else ""
        if not task and "task_index" in df and (self.root / "meta" / "tasks.parquet").exists():
            t = pd.read_parquet(self.root / "meta" / "tasks.parquet")
            task = str(t.index[t.task_index == int(df.task_index.iloc[0])][0]) if "task_index" in t else ""
        return SourceEpisode(int(row["episode_index"]), task or "unspecified", self.fps, df, vids)
