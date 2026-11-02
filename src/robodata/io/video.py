"""Thin PyAV wrappers for encoding/decoding RGB episode videos."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import av
import numpy as np


class VideoWriter:
    """Streaming RGB encoder. Frame i gets pts=i; real timestamps live in the sidecar."""

    def __init__(self, path: str | Path, fps: float, width: int, height: int, codec: str = "libx264",
                 crf: int | None = 18, preset: str | int | None = "veryfast", gop: int | None = 30,
                 pix_fmt: str = "yuv420p", extra: dict | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(self.path), mode="w")
        rate = int(round(fps)) if abs(fps - round(fps)) < 1e-6 else fps
        self.stream = self.container.add_stream(codec, rate=rate)
        self.stream.width, self.stream.height, self.stream.pix_fmt = width, height, pix_fmt
        opts = dict(extra or {})
        if crf is not None:
            opts["crf"] = str(crf)
        if preset is not None:
            opts["preset"] = str(preset)
        if gop is not None:
            self.stream.codec_context.gop_size = int(gop)
            if codec == "libsvtav1":
                opts.setdefault("g", str(gop))
        self.stream.options = opts
        self.n = 0

    def write(self, rgb: np.ndarray) -> None:
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb, dtype=np.uint8), format="rgb24")
        frame.pts = self.n
        for pkt in self.stream.encode(frame):
            self.container.mux(pkt)
        self.n += 1

    def close(self) -> None:
        for pkt in self.stream.encode():
            self.container.mux(pkt)
        self.container.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def iter_frames(path: str | Path, fmt: str = "rgb24") -> Iterator[np.ndarray]:
    """Decode every frame sequentially. Decode errors propagate (health checks catch them)."""
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in c.decode(stream):
            yield frame.to_ndarray(format=fmt)


def read_frames(path: str | Path, indices: list[int] | np.ndarray | None = None) -> np.ndarray:
    """Decode frames (all, or a sorted subset of indices) into an (N, H, W, 3) array."""
    if indices is None:
        return np.stack(list(iter_frames(path)))
    wanted = np.asarray(indices)
    order = np.argsort(wanted, kind="stable")
    out: list[np.ndarray | None] = [None] * len(wanted)
    j = 0
    for i, fr in enumerate(iter_frames(path)):
        while j < len(order) and wanted[order[j]] == i:
            out[order[j]] = fr
            j += 1
        if j >= len(order):
            break
    missing = [int(wanted[k]) for k in range(len(wanted)) if out[k] is None]
    if missing:
        raise IndexError(f"{path}: frames {missing[:5]}... not decodable")
    return np.stack(out)


def probe(path: str | Path) -> dict:
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        return {
            "codec": s.codec_context.name,
            "width": s.codec_context.width,
            "height": s.codec_context.height,
            "pix_fmt": s.codec_context.pix_fmt,
            "frames_header": s.frames,
            "rate": float(s.average_rate) if s.average_rate else None,
            "bytes": Path(path).stat().st_size,
        }
