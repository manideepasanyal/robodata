"""Duplicate detection.

* exact:  identical bytes (all low-dim data + the reference camera video)
* near:   trajectory shape (DTW on z-scored EE/joint path) AND visual sequence (DTW over per-frame
          z-normalised thumbnails) are both close.
* video_reuse: the same footage attached to a different trajectory (mislabelled / swapped upload). DTW on both makes it robust to re-encoding, small
          sensor noise, trimming and time-shifts; requiring both avoids two failure modes
          (same motion in another scene; same scene with another motion).

A vectorised Euclidean prefilter keeps it near-linear for hundreds of episodes per group.
"""

from __future__ import annotations

import hashlib

import numpy as np

from robodata.config import RobotConfig
from robodata.io.raw_episode import RawEpisode
from robodata.io.video import iter_frames

N_POINTS = 64
N_HASHES = 32
SIG_RATE_HZ = 5.0  # fixed-rate trajectory for DTW: a trimmed copy is then a true subsequence
MAX_SIG_LEN = 600
SEQ_POINTS = 128  # DTW sequences: >= 128 samples for a median-length episode (small interpolation-phase error)


def _trajectory(ep: RawEpisode, robot: RobotConfig):
    for kind in ("pose", "joint_state", "joint_cmd"):
        for s, cfg in robot.streams.items():
            if cfg.kind == kind and s in ep.streams and ep.has(s, "data"):
                x = ep.data(s)[:, :3] if kind == "pose" else ep.data(s)
                t = ep.times(s)
                if len(x) >= 4 and np.isfinite(x).all():
                    return t, x
    return None


def trajectory_signature(ep: RawEpisode, robot: RobotConfig) -> np.ndarray | None:
    """Fixed number of points over the episode's own duration (equal lengths -> vectorised prefilter)."""
    tx = _trajectory(ep, robot)
    if tx is None:
        return None
    t, x = tx
    g = np.linspace(t[0], t[-1], N_POINTS)
    return np.stack([np.interp(g, t, x[:, k]) for k in range(x.shape[1])], 1).astype(np.float32)


def trajectory_samples(ep: RawEpisode, robot: RobotConfig, max_rate_hz: float = 60.0):
    """(t_seconds_from_start, x) downsampled to <= max_rate_hz; the group-level resampling happens later."""
    tx = _trajectory(ep, robot)
    if tx is None:
        return None
    t, x = tx
    ts = (t - t[0]) / 1e9
    g = np.arange(0.0, ts[-1] + 1e-9, 1.0 / max_rate_hz)
    return g.astype(np.float32), np.stack([np.interp(g, ts, x[:, k]) for k in range(x.shape[1])], 1).astype(np.float32)


def group_rate(durations: list[float]) -> float:
    """One sampling rate per robot group (a trimmed copy must stay a subsequence of its original):
    >= SEQ_POINTS for a median episode, <= MAX_SIG_LEN for the longest, within [1, 30] Hz."""
    med, longest = float(np.median(durations)), float(max(durations))
    return float(np.clip(min(SEQ_POINTS / max(med, 1e-3), MAX_SIG_LEN / max(longest, 1e-3)), 1.0, 30.0))


def trajectory_sequence(samples, rate_hz: float) -> np.ndarray | None:
    if samples is None:
        return None
    t, x = samples
    g = np.arange(0.0, float(t[-1]) + 1e-9, 1.0 / rate_hz)
    if len(g) < 4:
        g = np.linspace(0.0, float(t[-1]), 4)
    return np.stack([np.interp(g, t, x[:, k]) for k in range(x.shape[1])], 1).astype(np.float32)


def robust_scale(trajs: list[np.ndarray]) -> np.ndarray:
    """Per-dimension std with a floor (10 % of the median dim std): near-constant dims must not explode."""
    sd = np.concatenate(trajs).std(0)
    return np.maximum(sd, 0.1 * float(np.median(sd)) + 1e-6)


def visual_signature(ep: RawEpisode, robot: RobotConfig, k: int = N_HASHES, with_frames: bool = False):
    """k evenly spaced frames of the reference camera as z-normalised 16x12 grey thumbnails.

    Thumbnails survive re-encoding and sensor noise far better than 64-bit perceptual hashes
    (measured: pHash flipped 17/64 bits on a crf-30 re-encode of small noisy frames).
    """
    import cv2

    cam = robot.sync.reference_stream
    if robot.streams[cam].kind != "rgb" or not ep.video_path(cam).exists():
        return None
    n = len(ep.times(cam))
    want = np.linspace(0, n - 1, k).astype(int)
    out, grey = [], []
    try:
        for i, fr in enumerate(iter_frames(ep.video_path(cam))):
            while len(out) < k and want[len(out)] == i:
                g64 = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY), (64, 48), interpolation=cv2.INTER_AREA)
                g = cv2.resize(g64, (16, 12), interpolation=cv2.INTER_AREA).astype(np.float32)
                out.append((g - g.mean()) / (g.std() + 1e-3))
                grey.append(g64)
            if len(out) >= k:
                break
    except Exception:
        return (None, None) if with_frames else None
    if len(out) != k:
        return (None, None) if with_frames else None
    thumbs = np.stack(out).astype(np.float16)
    return (thumbs, np.stack(grey)) if with_frames else thumbs


def same_footage(fa: np.ndarray | None, fb: np.ndarray | None, max_p90_grey: float = 1.5) -> bool:
    """Frame-level confirmation: 90th percentile of per-frame mean |difference| (64x48 grey levels).

    Measured: swapped (identical) footage 0.0; different demos in static sim scenes >= 2.95.
    """
    if fa is None or fb is None or fa.shape != fb.shape:
        return False
    d = np.abs(fa.astype(np.float32) - fb.astype(np.float32)).mean((1, 2))
    return float(np.percentile(d, 90)) <= max_p90_grey


def content_hash(ep: RawEpisode, robot: RobotConfig) -> str:
    """Byte-level identity: all low-dim data + the reference camera's encoded video."""
    h = hashlib.sha1()
    for s in sorted(ep.streams):
        if ep.has(s, "data") and robot.streams.get(s) is not None and robot.streams[s].kind not in ("rgb", "depth"):
            h.update(np.ascontiguousarray(ep.data(s)).tobytes())
    vp = ep.video_path(robot.sync.reference_stream)
    if vp.exists():
        h.update(vp.read_bytes())
    return h.hexdigest()


def dtw(a: np.ndarray, b: np.ndarray, window: int = 10, cost=None, slack: float = 0.15) -> float:
    """Banded open-begin/open-end DTW, normalised by the longer sequence.

    ``slack`` lets either sequence skip up to that fraction at its start / end, so a copy with the first
    or last seconds trimmed still aligns at ~zero cost; identical sequences give 0. Each row is a
    min-plus scan, D[i, j] = S[j] + min(D[i, 0], min_{k<=j}(T[k] - S[k])) with S = cumsum(cost row),
    which numpy evaluates with minimum.accumulate (no Python loop over columns).
    """
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    n, m = len(a), len(b)
    sa, sb = int(slack * n), int(slack * m)
    w = max(window, abs(n - m)) + max(sa, sb)
    big = np.float64(1e9)
    jj = np.arange(1, m + 1)
    prev = np.full(m + 1, big)
    prev[: sb + 1] = 0.0
    last_rows = []
    for i in range(1, n + 1):
        if cost is None:  # RMS over dimensions: thresholds then mean the same for a 2-D or a 54-D state
            c = (np.linalg.norm(b - a[i - 1], axis=1) / np.sqrt(a.shape[1])).astype(np.float64)
        else:
            c = cost(a[i - 1], b)
        c = np.where(np.abs(jj - i) <= w, c, big)
        tmp = c + np.minimum(prev[1:], prev[:-1])
        S = np.cumsum(c)
        d0 = 0.0 if i <= sa else big
        row = S + np.minimum(d0, np.minimum.accumulate(tmp - S))
        prev = np.concatenate([[d0], row])
        if i >= n - sa:
            last_rows.append(row[-1])
    end = min(prev[m - sb:].min(), min(last_rows))
    return float(min(end, big) / max(n, m))


def shift_align_distance(sa, sb, max_shift_frac: float = 0.25, min_overlap: float = 0.7) -> float:
    """Phase-free distance for time-shifted copies (trims): RMS per dimension after the best time offset.

    A trimmed copy is the original shifted by tau. Interpolating the original at the copy's own
    sample times + tau removes the sampling-phase error that independent resampling leaves (which
    on fast 50 Hz joints is ~0.05-0.09 per dim - as large as the dedup threshold).
    """
    if sa is None or sb is None:
        return np.inf
    (ta, xa), (tb, xb) = sa, sb
    if len(ta) < 4 or len(tb) < 4:
        return np.inf
    dur = float(max(ta[-1], tb[-1]))
    shorter = float(min(ta[-1], tb[-1]))
    g = np.arange(0.0, dur + 1e-9, 0.1)  # coarse search on a 10 Hz grid
    A = np.stack([np.interp(g, ta, xa[:, k], left=np.nan, right=np.nan) for k in range(xa.shape[1])], 1)
    B = np.stack([np.interp(g, tb, xb[:, k], left=np.nan, right=np.nan) for k in range(xb.shape[1])], 1)
    best_lag, best = 0, np.inf
    for lag in range(-int(max_shift_frac * len(g)), int(max_shift_frac * len(g)) + 1):
        if lag >= 0:
            d = A[lag:] - B[: len(g) - lag]
        else:
            d = A[: lag] - B[-lag:]
        ok = ~np.isnan(d).any(1)
        if ok.sum() * 0.1 < min_overlap * shorter:
            continue
        v = float(np.sqrt(np.mean(d[ok] ** 2)))
        if v < best:
            best, best_lag = v, lag
    if not np.isfinite(best):
        return np.inf
    for tau in best_lag * 0.1 + np.linspace(-0.1, 0.1, 41):  # refine at the copy's native sample times
        t = tb + tau
        ok = (t >= ta[0]) & (t <= ta[-1])
        if ok.sum() < 4 or (t[ok][-1] - t[ok][0]) < min_overlap * shorter:
            continue
        ai = np.stack([np.interp(t[ok], ta, xa[:, k]) for k in range(xa.shape[1])], 1)
        best = min(best, float(np.sqrt(np.mean((ai - xb[ok]) ** 2))))
    return best


def _thumb_cost(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """RMS between one thumbnail and each thumbnail of a sequence (vectorised)."""
    d = Y.astype(np.float32) - x.astype(np.float32)
    return np.sqrt((d.reshape(len(Y), -1) ** 2).mean(1)).astype(np.float64)


VIDEO_REUSE_THR = 0.15  # DTW-aligned RMS between z-normalised thumbnails: essentially the same footage
MAX_DURATION_RATIO = 1.25  # a copy (incl. an 8 % trim) keeps its length; the same path at 1.6x speed is a new demo
MAX_FRAMECOUNT_RATIO = 1.10  # reused footage IS the same recording: same number of frames (+/- drops)


def _ratio(x, y) -> float:
    return max(x, y) / max(min(x, y), 1e-9) if x and y else 1.0


def _thumb_matrix(vis: list[np.ndarray]) -> np.ndarray:
    """Mean per-position RMS distance between thumbnail sequences (vectorised prefilter)."""
    X = np.stack(vis).astype(np.float32).reshape(len(vis), len(vis[0]), -1)
    out = np.zeros((len(X), len(X)), np.float32)
    for a in range(len(X)):
        out[a] = np.sqrt(((X - X[a]) ** 2).mean(2)).mean(1)
    return out


def _shift_dist(T: np.ndarray, a: int, max_shift: int = 8) -> np.ndarray:
    """Min over relative shifts of the mean pointwise distance on the overlap (trim/offset tolerant)."""
    best = np.full(len(T), np.inf, np.float32)
    for sh in range(-max_shift, max_shift + 1):
        if sh >= 0:
            d = np.linalg.norm(T[:, sh:] - T[a, : T.shape[1] - sh], axis=2).mean(1)
        else:
            d = np.linalg.norm(T[:, :sh] - T[a, -sh:], axis=2).mean(1)
        d = d / np.sqrt(T.shape[2])
        best = np.minimum(best, d)
    return best


def find_duplicates(sigs: dict[str, dict], traj_thr: float, vis_thr: float, n_candidates: int = 10
                    ) -> tuple[list[dict], dict[str, dict]]:
    """sigs: id -> {"traj": (64, D) scaled, "seq": (n, D) scaled fixed-rate, "vis": thumbnails,
    "hash": str, "group": key}. Groups are per robot (a copy re-labelled with another task is still a copy).

    Returns (pairs, nearest-neighbour report per episode).
    """
    found: dict[tuple[str, str], dict] = {}
    nn: dict[str, dict] = {}

    def add(a, b, **kw):
        key = (min(a, b), max(a, b))
        if key not in found or (found[key]["kind"] != "exact" and kw["kind"] == "exact"):
            found[key] = {"a": key[0], "b": key[1], **kw}

    by_group: dict[str, list[str]] = {}
    for k, s in sigs.items():
        by_group.setdefault(s["group"], []).append(k)
    for ids in by_group.values():
        ids = sorted(ids)
        seen: dict[str, str] = {}
        for i in ids:  # byte-identical copies
            h = sigs[i]["hash"]
            if h in seen:
                add(seen[h], i, kind="exact", traj_dist=0.0, vis_dist=0.0)
            else:
                seen[h] = i
        with_traj = [i for i in ids if sigs[i]["traj"] is not None]
        if len(with_traj) >= 2:
            T = np.stack([sigs[i]["traj"] for i in with_traj])
            for a in range(len(T)):
                ia = with_traj[a]
                d = _shift_dist(T, a)
                d[a] = np.inf
                cand = sorted((set(np.argsort(d)[:n_candidates].tolist()) | set(np.where(d < 3 * traj_thr)[0].tolist())) - {a})
                best = (np.inf, np.inf, None)
                for b in cand:
                    ib = with_traj[b]
                    dt = dtw(sigs[ia]["seq"], sigs[ib]["seq"])
                    if dt > traj_thr and dt < 4 * traj_thr:  # possibly a trimmed / shifted copy
                        dt = min(dt, shift_align_distance(sigs[ia].get("samples"), sigs[ib].get("samples")))
                    va, vb = sigs[ia]["vis"], sigs[ib]["vis"]
                    dv = dtw(va, vb, window=4, cost=_thumb_cost) if va is not None and vb is not None else 0.0
                    if dt < best[0]:
                        best = (dt, dv, ib)
                    if dt <= traj_thr and dv <= vis_thr and _ratio(sigs[ia].get("duration"), sigs[ib].get("duration")) <= MAX_DURATION_RATIO:
                        add(ia, ib, kind="near", traj_dist=dt, vis_dist=dv)  # registered from either side
                nn[ia] = {"nn": best[2], "nn_traj_dist": best[0], "nn_vis_dist": best[1]}
        # same footage attached to a different trajectory = mislabelled / swapped upload. "Same" is judged
        # relative to the dataset: much closer (< 1/2) than distinct demos of this group typically are,
        # which adapts to static scenes where every episode looks alike.
        vis_ids = [i for i in ids if sigs[i]["vis"] is not None]
        if len(vis_ids) >= 3:
            V = _thumb_matrix([sigs[i]["vis"] for i in vis_ids])
            typical = float(np.median(V[np.triu_indices(len(vis_ids), 1)]))
            thr = min(VIDEO_REUSE_THR, 0.5 * typical)
            for a, b in zip(*np.where(np.triu(V <= 2 * thr, 1))):
                ia, ib = vis_ids[a], vis_ids[b]
                if (min(ia, ib), max(ia, ib)) in found:
                    continue
                dv = dtw(sigs[ia]["vis"], sigs[ib]["vis"], window=4, cost=_thumb_cost)
                sa, sb = sigs[ia]["seq"], sigs[ib]["seq"]
                dt = dtw(sa, sb) if sa is not None and sb is not None else np.inf
                if (dv <= thr and dt > traj_thr and _ratio(sigs[ia].get("n_frames"), sigs[ib].get("n_frames")) <= MAX_FRAMECOUNT_RATIO
                        and same_footage(sigs[ia].get("frames"), sigs[ib].get("frames"))):
                    add(ia, ib, kind="video_reuse", traj_dist=dt, vis_dist=dv, vis_thr=thr)
    return list(found.values()), nn


def clusters(ids: list[str], pairs: list[dict]) -> dict[str, str]:
    """Union-find -> episode_id: cluster root id."""
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in pairs:
        ra, rb = find(p["a"]), find(p["b"])
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {i: find(i) for i in ids}
