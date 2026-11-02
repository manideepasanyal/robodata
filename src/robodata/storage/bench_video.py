"""Video encoding benchmark: codec / CRF / keyframe interval vs size, quality, speed, random access, integrity."""

from __future__ import annotations

import time
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

from robodata.io.raw_episode import RawEpisode, list_episodes
from robodata.io.video import iter_frames

RGB_CONFIGS = [
    # name, codec, pix_fmt, options, gop
    ("h264_crf18_g30", "libx264", "yuv420p", {"crf": "18", "preset": "veryfast"}, 30),
    ("h264_crf23_g30", "libx264", "yuv420p", {"crf": "23", "preset": "veryfast"}, 30),
    ("h264_crf28_g30", "libx264", "yuv420p", {"crf": "28", "preset": "veryfast"}, 30),
    ("h264_crf23_g2", "libx264", "yuv420p", {"crf": "23", "preset": "veryfast"}, 2),
    ("hevc_crf28_g30", "libx265", "yuv420p", {"crf": "28", "preset": "fast", "x265-params": "log-level=error"}, 30),
    ("av1_crf30_g2 (LeRobot default)", "libsvtav1", "yuv420p", {"crf": "30", "preset": "8", "g": "2"}, 2),
    ("av1_crf30_g30", "libsvtav1", "yuv420p", {"crf": "30", "preset": "8", "g": "30"}, 30),
    ("av1_crf40_g30", "libsvtav1", "yuv420p", {"crf": "40", "preset": "8", "g": "30"}, 30),
    ("ffv1_lossless", "ffv1", "yuv444p", {}, 1),
]
DEPTH_CONFIGS = [
    ("ffv1_gray16_lossless", "ffv1", "gray16le", {}, 1),
    ("hevc_gray12_lossless (LeRobot)", "libx265", "gray12le", {"x265-params": "lossless=1:log-level=error"}, 2),
    ("hevc_gray12_crf20", "libx265", "gray12le", {"crf": "20", "x265-params": "log-level=error"}, 30),
]


def _encode(frames, path: Path, codec, pix_fmt, opts, gop, fps=30, fmt="rgb24") -> float:
    t0 = time.perf_counter()
    with av.open(str(path), "w") as c:
        s = c.add_stream(codec, rate=fps)
        s.width, s.height, s.pix_fmt = frames[0].shape[1], frames[0].shape[0], pix_fmt
        s.codec_context.gop_size = gop
        s.options = opts
        for i, f in enumerate(frames):
            vf = av.VideoFrame.from_ndarray(f, format=fmt)
            vf.pts = i
            for p in s.encode(vf):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return time.perf_counter() - t0


def _decode_all(path: Path, fmt: str):
    t0 = time.perf_counter()
    out = list(iter_frames(path, fmt))
    return out, time.perf_counter() - t0


def _random_access_ms(path: Path, n: int, k: int = 15, seed: int = 0) -> float:
    """Seek-and-decode latency for random single frames (what a shuffled data loader does)."""
    rng = np.random.default_rng(seed)
    times = []
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        for target in rng.integers(0, n, k):
            t0 = time.perf_counter()
            c.seek(int(target), stream=s, backward=True, any_frame=False)
            for fr in c.decode(s):
                if fr.pts >= target:
                    break
            times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1000)


def _psnr(a, b, peak=255.0) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(peak**2 / mse))


def _ssim(a, b) -> float:
    a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = cv2.GaussianBlur(a, (11, 11), 1.5), cv2.GaussianBlur(b, (11, 11), 1.5)
    sa = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a**2
    sb = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b**2
    sab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    return float((((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a**2 + mu_b**2 + C1) * (sa + sb + C2))).mean())


def run_video_bench(raw_root, out_dir, n_episodes: int = 3, max_frames: int = 300) -> str:
    out_dir = Path(out_dir)
    tmp = out_dir / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    rows = []
    eps = []
    for robot in ("ur5_catheter", "ur5_sim"):
        eps += [p for p in list_episodes(raw_root, robot) if "__fault-" not in p.name][: max(1, n_episodes // 2 + 1)]
    for ep_path in eps[:n_episodes]:
        ep = RawEpisode(ep_path)
        src = list(iter_frames(ep.video_path("wrist")))[:max_frames]  # raw archive (crf 18) as the reference
        raw_bytes = src[0].nbytes * len(src)
        for name, codec, pix, opts, gop in RGB_CONFIGS:
            p = tmp / f"{ep.id}_{name.split()[0]}.mkv"
            enc_s = _encode(src, p, codec, pix, opts, gop)
            dec, dec_s = _decode_all(p, "rgb24")
            n_ok = len(dec) == len(src)
            idx = np.linspace(0, min(len(dec), len(src)) - 1, 12).astype(int)
            rows.append({
                "modality": "rgb", "episode": ep.id, "is_sim": ep.meta.is_sim, "config": name,
                "resolution": f"{src[0].shape[1]}x{src[0].shape[0]}", "frames": len(src),
                "kB_per_frame": p.stat().st_size / len(src) / 1e3, "compression_x": raw_bytes / p.stat().st_size,
                "psnr_db": float(np.mean([_psnr(src[i], dec[i]) for i in idx])),
                "ssim": float(np.mean([_ssim(src[i], dec[i]) for i in idx])),
                "encode_fps": len(src) / enc_s, "decode_fps": len(dec) / dec_s,
                "random_access_ms": _random_access_ms(p, len(src)), "frame_count_ok": n_ok,
            })
        # depth
        d = np.asarray(ep.root["front_depth"]["data"][: max_frames])
        d = d[:, : d.shape[1] // 2 * 2, : d.shape[2] // 2 * 2]
        zarr_bytes = sum(f.stat().st_size for f in (ep.path / "streams.zarr" / "front_depth" / "data").rglob("*") if f.is_file())
        zarr_bytes *= len(d) / max(1, ep.root["front_depth"]["data"].shape[0])
        rows.append({"modality": "depth", "episode": ep.id, "is_sim": ep.meta.is_sim, "config": "zarr_zstd_uint16 (raw store)",
                     "resolution": f"{d.shape[2]}x{d.shape[1]}", "frames": len(d), "kB_per_frame": zarr_bytes / len(d) / 1e3,
                     "compression_x": d.nbytes / zarr_bytes, "depth_mae_mm": 0.0, "frame_count_ok": True})
        for name, codec, pix, opts, gop in DEPTH_CONFIGS:
            p = tmp / f"{ep.id}_{name.split()[0]}.mkv"
            src_d = d if "16" in pix else np.minimum(d, 4095)  # 12-bit codecs: clamp to 4.095 m (mm units)
            fmt = "gray16le" if pix == "gray16le" else "gray12le"
            try:
                enc_s = _encode([x.astype(np.uint16) for x in src_d], p, codec, pix, opts, gop, fmt=fmt)
                dec, dec_s = _decode_all(p, fmt)
                dec = np.stack(dec)
                n = min(len(dec), len(src_d))
                rows.append({"modality": "depth", "episode": ep.id, "is_sim": ep.meta.is_sim, "config": name,
                             "resolution": f"{d.shape[2]}x{d.shape[1]}", "frames": len(d),
                             "kB_per_frame": p.stat().st_size / len(d) / 1e3, "compression_x": d.nbytes / p.stat().st_size,
                             "depth_mae_mm": float(np.abs(dec[:n].astype(np.int32) - src_d[:n].astype(np.int32)).mean()),
                             "encode_fps": len(d) / enc_s, "decode_fps": n / dec_s,
                             "random_access_ms": _random_access_ms(p, len(d)), "frame_count_ok": len(dec) == len(d)})
            except Exception as exc:
                rows.append({"modality": "depth", "episode": ep.id, "config": name, "error": str(exc)})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "video_bench.csv", index=False)
    agg = df.groupby(["modality", "config"], sort=False).mean(numeric_only=True).drop(columns=["is_sim"], errors="ignore")
    agg["frame_count_ok"] = df.groupby(["modality", "config"], sort=False).frame_count_ok.all()
    md = ["# Video encoding benchmark", "",
          f"{len(eps[:n_episodes])} episodes (real + sim), wrist RGB and front depth, {max_frames} frames max each. "
          "Reference = raw-archive frames; PSNR/SSIM on 12 evenly spaced frames; random access = median seek+decode "
          "latency of a single random frame (what a shuffled training loader pays).", "",
          agg.round(3).to_markdown()]
    (out_dir / "video_bench.md").write_text("\n".join(md) + "\n")
    for p in tmp.glob("*"):
        p.unlink()
    tmp.rmdir()
    return str(out_dir / "video_bench.md")
