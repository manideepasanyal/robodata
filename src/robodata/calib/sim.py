"""Calibrate the MuJoCo twin's cameras from rendered chessboard views and score against ground truth."""

from __future__ import annotations

import numpy as np

from robodata.calib.core import (board_pose, calibrate_intrinsics, hand_eye, intrinsics_error, pose_error,
                                 save_calibration)
from robodata.sim.generate import dls_ik_step, quat_down
from robodata.sim.scene import BOARD_COLS, BOARD_POS, BOARD_ROWS, BOARD_SURFACE_Z, HOME_Q, SCENES, board_object_points, build_mjcf

PATTERN = (BOARD_COLS - 1, BOARD_ROWS - 1)
_MJ2CV = np.diag([1.0, -1.0, -1.0])  # MuJoCo camera looks down -z with +y up; OpenCV looks down +z with +y down


def _T(R, p) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def gt_K(model, cam: str, w: int, h: int) -> np.ndarray:
    fovy = np.radians(model.cam_fovy[model.camera(cam).id])
    f = (h / 2) / np.tan(fovy / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def cam_pose_world(data, model, cam: str) -> np.ndarray:
    cid = model.camera(cam).id
    return _T(data.cam_xmat[cid].reshape(3, 3) @ _MJ2CV, data.cam_xpos[cid])


def body_pose(data, model, body: str) -> np.ndarray:
    b = model.body(body).id
    return _T(data.xmat[b].reshape(3, 3), data.xpos[b])


def board_surface(data, model) -> np.ndarray:
    """Board frame used by the corner coordinates: body frame lifted to the printed surface."""
    T = body_pose(data, model, "board")
    T[:3, 3] += T[:3, :3] @ np.array([0, 0, BOARD_SURFACE_Z])
    return T


def _random_board_quat(rng) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    q = R.from_euler("xyz", [rng.uniform(-0.35, 0.35), rng.uniform(-0.35, 0.35), rng.uniform(-0.5, 0.5)]).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def run_sim_calibration(scene_id: str = "sim_scene_00", n_views: int = 40, width: int = 320, height: int = 240,
                        seed: int = 0, save_as: str | None = "v1") -> dict:
    import mujoco

    rng = np.random.default_rng(seed)
    scene = SCENES[scene_id]
    model = mujoco.MjModel.from_xml_string(build_mjcf(scene, scene.block_xy, 0.0, np.random.default_rng(7), width,
                                                      height, with_board=True))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height, width)
    # hide the task object just under the table top, below the board: the fixed cameras track it (targetbody)
    model.body_pos[model.body("block").id] = [BOARD_POS[0], BOARD_POS[1], -0.1]
    board_id = model.body("board").id
    pos0, quat0 = model.body_pos[board_id].copy(), model.body_quat[board_id].copy()
    obj = board_object_points()
    result: dict = {"cameras": {}}

    def render(cam):
        renderer.update_scene(data, camera=cam)
        return renderer.render().copy()

    # --- fixed cameras: intrinsics from a moving board, extrinsics from the board at its nominal pose ----------
    data.qpos[:6] = HOME_Q
    for cam in ("front", "base"):
        imgs = []
        for _ in range(n_views):
            model.body_pos[board_id] = pos0 + np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.08, 0.08), rng.uniform(0.08, 0.2)])
            model.body_quat[board_id] = _random_board_quat(rng)
            mujoco.mj_forward(model, data)
            imgs.append(render(cam))
        intr = calibrate_intrinsics(imgs, PATTERN, obj)
        model.body_pos[board_id], model.body_quat[board_id] = pos0, quat0
        mujoco.mj_forward(model, data)
        T_cam_board = board_pose(render(cam), PATTERN, obj, intr["K"], intr["dist"])
        T_base_cam = board_surface(data, model) @ np.linalg.inv(T_cam_board) if T_cam_board is not None else None
        K_gt = gt_K(model, cam, width, height)
        entry = {"width": width, "height": height, "K": intr["K"], "dist": intr["dist"], "rms_px": intr["rms_px"],
                 "views_used": len(intr["views_used"]), "gt_error": {"intrinsics": intrinsics_error(intr["K"], K_gt)}}
        if T_base_cam is not None:
            entry["T_base_cam"] = T_base_cam
            entry["gt_error"]["extrinsics"] = pose_error(T_base_cam, cam_pose_world(data, model, cam))
        result["cameras"][cam] = entry

    # --- wrist camera: eye-in-hand. Board fixed, arm moves through random poses above it ---------------------
    model.body_pos[board_id], model.body_quat[board_id] = pos0, quat0
    tcp = model.site("tcp").id
    imgs, flange_poses, gt_flange_cam = [], [], None
    q = np.array(HOME_Q)
    from scipy.spatial.transform import Rotation as R

    for _ in range(n_views * 2):
        target = np.array(BOARD_POS) + np.array([rng.uniform(-0.08, 0.08), rng.uniform(-0.2, -0.05), rng.uniform(0.25, 0.4)])
        tilt = R.from_euler("xyz", [rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25), 0])
        qd = quat_down(rng.uniform(-0.5, 0.5))
        qd = (tilt * R.from_quat(qd[[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]]
        for _ in range(150):
            data.qpos[:6] = q
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            q = dls_ik_step(model, data, tcp, target, qd, q, damping=0.05, max_step=0.1)
        data.qpos[:6] = q
        mujoco.mj_forward(model, data)
        imgs.append(render("wrist"))
        flange_poses.append(body_pose(data, model, "tool"))
        if gt_flange_cam is None:
            gt_flange_cam = np.linalg.inv(flange_poses[-1]) @ cam_pose_world(data, model, "wrist")
    intr = calibrate_intrinsics(imgs, PATTERN, obj)
    T_cb, T_bf = [], []
    for im, Tf in zip(imgs, flange_poses):
        Tc = board_pose(im, PATTERN, obj, intr["K"], intr["dist"])
        if Tc is not None:
            T_cb.append(Tc)
            T_bf.append(Tf)
    T_flange_cam = hand_eye(T_bf, T_cb)
    result["cameras"]["wrist"] = {
        "width": width, "height": height, "K": intr["K"], "dist": intr["dist"], "rms_px": intr["rms_px"],
        "views_used": len(intr["views_used"]), "hand_eye_views": len(T_cb), "T_flange_cam": T_flange_cam,
        "gt_error": {"intrinsics": intrinsics_error(intr["K"], gt_K(model, "wrist", width, height)),
                     "hand_eye": pose_error(T_flange_cam, gt_flange_cam)},
    }
    renderer.close()
    for cam in ("front_depth", "base_depth", "wrist_depth"):  # depth is rendered from the colour camera pose
        result["cameras"][cam] = {"aligned_to": cam.replace("_depth", "")}
    if save_as:
        result["path"] = str(save_calibration(
            "ur5_sim", save_as, result["cameras"],
            {"intrinsics": "opencv chessboard (findChessboardCornersSB), k3 fixed", "extrinsics": "solvePnP on fixed board",
             "hand_eye": "cv2.calibrateHandEye PARK (eye-in-hand)", "pattern": list(PATTERN), "square_m": float(obj[1, 0] - obj[0, 0])},
            notes=f"rendered views from {scene_id}; gt_error = deviation from the MuJoCo model's exact camera parameters"))
    return result
