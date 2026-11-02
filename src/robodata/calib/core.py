"""Camera calibration: intrinsics (chessboard), static extrinsics (PnP), hand-eye (eye-in-hand).

Works on any list of RGB images + robot poses. ``robodata.calib.sim`` drives it with
images rendered in the MuJoCo twin so every estimate can be scored against the exact
ground truth, and the same functions run on real checkerboard captures.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

from robodata.config import CONFIG_DIR

CALIB_DIR = CONFIG_DIR / "calibration"
_SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)


def _outer_square_dark(gray: np.ndarray, c: np.ndarray, cols: int) -> bool | None:
    """Is the square diagonally outside corner 0 darker than its edge-neighbour? (board convention: black)"""
    p0, p1, pc = c[0], c[1], c[cols]
    h, w = gray.shape

    def sample(p):
        x, y = int(round(p[0])), int(round(p[1]))
        return float(gray[y, x]) if 0 <= x < w and 0 <= y < h else None

    diag = sample(p0 + 0.5 * (p0 - p1) + 0.5 * (p0 - pc))
    side = sample(p0 + 0.5 * (p0 - p1) - 0.5 * (p0 - pc))
    return None if diag is None or side is None else diag < side


def detect_corners(rgb: np.ndarray, pattern: tuple[int, int]) -> np.ndarray | None:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ok, corners = cv2.findChessboardCornersSB(gray, pattern, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    if not ok:
        ok, corners = cv2.findChessboardCorners(gray, pattern)
        if not ok:
            return None
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), _SUBPIX)
    return corners.reshape(-1, 2).astype(np.float32)


def calibrate_intrinsics(images: list[np.ndarray], pattern: tuple[int, int], obj_pts: np.ndarray) -> dict:
    img_pts, used = [], []
    for i, im in enumerate(images):
        c = detect_corners(im, pattern)
        if c is not None:
            img_pts.append(c)
            used.append(i)
    if len(img_pts) < 5:
        raise RuntimeError(f"only {len(img_pts)} usable chessboard views")
    h, w = images[0].shape[:2]
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera([obj_pts] * len(img_pts), img_pts, (w, h), None, None,
                                                     flags=cv2.CALIB_FIX_K3)
    per_view = []
    for o, ip, rv, tv in zip([obj_pts] * len(img_pts), img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, rv, tv, K, dist)
        per_view.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - ip) ** 2, 1)))))
    return {"K": K, "dist": dist.ravel(), "rms_px": float(rms), "views_used": used, "per_view_rms_px": per_view,
            "size": [w, h]}


def board_pose(rgb: np.ndarray, pattern, obj_pts, K, dist) -> np.ndarray | None:
    """T_cam_board (4x4) from one image, or None if the board isn't found.

    Detectors may return the grid starting at any corner (and mirrored). We try the four
    orderings and keep the one whose board normal faces the camera and whose first outer
    square is black - otherwise poses flip by 180 degrees between views.
    """
    c = detect_corners(rgb, pattern)
    if c is None:
        return None
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cols, rows = pattern
    grid = c.reshape(rows, cols, 2)
    best, best_score = None, None
    for cand in (grid, grid[:, ::-1], grid[::-1], grid[::-1, ::-1]):
        pts = np.ascontiguousarray(cand.reshape(-1, 2))
        ok, rvec, tvec = cv2.solvePnP(obj_pts, pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(rvec)[0]
        T[:3, 3] = tvec.ravel()
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
        err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - pts, axis=1)))
        facing = T[2, 2] < 0
        dark = _outer_square_dark(gray, pts, cols)
        score = (facing, bool(dark), -err)
        if best_score is None or score > best_score:
            best, best_score = T, score
    return best


def hand_eye(T_base_flange: list[np.ndarray], T_cam_board: list[np.ndarray],
             method: int = cv2.CALIB_HAND_EYE_PARK) -> np.ndarray:
    """Eye-in-hand: solve T_flange_cam from robot poses and board observations (board fixed in base)."""
    R_g2b = [T[:3, :3] for T in T_base_flange]
    t_g2b = [T[:3, 3] for T in T_base_flange]
    R_t2c = [T[:3, :3] for T in T_cam_board]
    t_t2c = [T[:3, 3] for T in T_cam_board]
    R, t = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=method)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t.ravel()
    return T


def pose_error(T_est: np.ndarray, T_gt: np.ndarray) -> dict:
    dR = T_est[:3, :3].T @ T_gt[:3, :3]
    ang = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    return {"trans_mm": float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]) * 1000), "rot_deg": ang}


def intrinsics_error(K: np.ndarray, K_gt: np.ndarray) -> dict:
    return {"fx_pct": float(abs(K[0, 0] / K_gt[0, 0] - 1) * 100), "fy_pct": float(abs(K[1, 1] / K_gt[1, 1] - 1) * 100),
            "cx_px": float(abs(K[0, 2] - K_gt[0, 2])), "cy_px": float(abs(K[1, 2] - K_gt[1, 2]))}


def _clean(o):
    if isinstance(o, np.ndarray):
        return [_clean(x) for x in o.tolist()] if o.ndim else o.item()
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(x) for x in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def save_calibration(robot_id: str, name: str, cameras: dict, method: dict, notes: str = "") -> Path:
    """Write configs/calibration/<robot_id>/<name>.yaml; version id '<robot_id>/<name>' + content hash."""
    body = {"robot_id": robot_id, "method": method, "cameras": _clean(cameras), "notes": notes}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]
    doc = {"calibration_version": f"{robot_id}/{name}", "content_hash": digest,
           "created": datetime.now(timezone.utc).isoformat(), **body}
    path = CALIB_DIR / robot_id / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False, width=120))
    return path


def load_calibration(version: str) -> dict | None:
    path = CALIB_DIR / f"{version}.yaml"
    return yaml.safe_load(path.read_text()) if path.exists() else None
