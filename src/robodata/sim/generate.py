"""Generate simulated teleoperated insertion demonstrations in MuJoCo.

A scripted *operator* produces SpaceMouse-like Cartesian target poses at 30 Hz
(min-jerk segments + Ornstein-Uhlenbeck hand tremor + reaction lag + occasional lateral
misalignment), a damped-least-squares IK turns them into joint position commands at
125 Hz, and physics runs at 500 Hz. Every sensor is sampled at its native rate with
noise and written to a *native sim log* (HDF5, robomimic/ACT-like layout). The
sim-to-dataset converter (robodata.sim.convert) maps that log onto the real-robot schema.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from robodata.sim.scene import (BLOCK_HEIGHT, HOLE_HALF, HOME_Q, PEG_RADIUS, SCENES, SceneParams, build_mjcf)

PHYS_HZ, CTRL_HZ, TELEOP_HZ, CAM_HZ = 500, 125, 30, 30
PHASES = ["approach", "align", "insert", "hold"]
TASK = "Insert the peg into the hole of the block and push it down until it reaches the bottom."


@dataclass
class OperatorStyle:
    name: str
    speed: float  # m/s nominal approach speed
    tremor_mm: float
    misalign_mm: float  # std of lateral bias at insertion
    hesitation_s: float


OPERATORS = {
    "sim_op_careful": OperatorStyle("sim_op_careful", 0.08, 0.4, 1.5, 0.6),
    "sim_op_fast": OperatorStyle("sim_op_fast", 0.14, 0.8, 2.6, 0.2),
}


def min_jerk(s: np.ndarray) -> np.ndarray:
    s = np.clip(s, 0, 1)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def quat_down(yaw: float) -> np.ndarray:
    """Tool z pointing down (world -z), rotated by yaw. MuJoCo quaternion order (w, x, y, z)."""
    from scipy.spatial.transform import Rotation as R

    q = (R.from_euler("z", yaw) * R.from_euler("x", np.pi)).as_quat()  # x y z w
    return np.array([q[3], q[0], q[1], q[2]])


class Operator:
    """Builds the time-parameterised Cartesian target trajectory with human-like imperfections."""

    def __init__(self, style: OperatorStyle, start: np.ndarray, hole_top: np.ndarray, yaw: float,
                 rng: np.random.Generator):
        self.style, self.rng = style, rng
        bias = rng.normal(0, style.misalign_mm / 1000.0, 2)
        above = hole_top + np.array([*bias * 0.3, 0.07])
        align = hole_top + np.array([*bias, 0.012])
        bottom = hole_top + np.array([*bias, -(BLOCK_HEIGHT + 0.002)])  # press 2 mm into the hole bottom
        d1 = np.linalg.norm(above - start) / style.speed + rng.uniform(0.3, 0.8)
        d2 = 0.9 + rng.uniform(0, 0.6) + style.hesitation_s
        d3 = (BLOCK_HEIGHT + 0.012) / (style.speed * 0.25) + rng.uniform(0, 0.5)
        d4 = 0.8 + rng.uniform(0, 0.4)
        self.wp = [(start, above, d1), (above, align, d2), (align, bottom, d3), (bottom, bottom, d4)]
        self.t_end = sum(w[2] for w in self.wp)
        self.q_target = quat_down(yaw)
        self.q_start = None  # set by caller (current tool orientation); blended into q_target during approach
        self.bias_mm = float(np.linalg.norm(bias) * 1000)
        self._ou = np.zeros(3)

    def target(self, t: float) -> tuple[np.ndarray, int]:
        acc = 0.0
        for k, (a, b, d) in enumerate(self.wp):
            if t <= acc + d or k == len(self.wp) - 1:
                s = min_jerk(np.array((t - acc) / d))
                pos = a + (b - a) * s
                break
            acc += d
        # OU tremor (mean reverting), smaller near contact like a careful human
        sigma = self.style.tremor_mm / 1000.0 * (0.4 if k >= 2 else 1.0)
        self._ou += -self._ou * (1 / TELEOP_HZ) / 0.3 + self.rng.normal(0, sigma * 0.35, 3)
        return pos + self._ou, k

    def orientation(self, t: float) -> np.ndarray:
        from scipy.spatial.transform import Rotation as R, Slerp

        if self.q_start is None:
            return self.q_target
        s = float(min_jerk(np.array(t / self.wp[0][2])))
        wxyz = np.stack([self.q_start, self.q_target])
        rots = R.from_quat(wxyz[:, [1, 2, 3, 0]])
        q = Slerp([0, 1], rots)(s).as_quat()
        return np.array([q[3], q[0], q[1], q[2]])


def dls_ik_step(model, data, site_id: int, pos_t: np.ndarray, quat_t: np.ndarray, q: np.ndarray,
                damping: float = 0.05, max_step: float = 0.05) -> np.ndarray:
    import mujoco

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    J = np.vstack([jacp[:, :6], jacr[:, :6]])
    err_p = pos_t - data.site_xpos[site_id]
    cur = np.zeros(4)
    mujoco.mju_mat2Quat(cur, data.site_xmat[site_id])
    neg = np.zeros(4)
    mujoco.mju_negQuat(neg, cur)
    dq_rot = np.zeros(4)
    mujoco.mju_mulQuat(dq_rot, quat_t, neg)
    err_r = np.zeros(3)
    mujoco.mju_quat2Vel(err_r, dq_rot, 1.0)
    e = np.concatenate([err_p, 0.5 * err_r])
    dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), e)
    return q + np.clip(dq, -max_step, max_step)


def run_episode(scene: SceneParams, operator: str, seed: int, out_path: Path, width: int = 320, height: int = 240,
                renderer_cache: dict | None = None) -> dict:
    import mujoco

    rng = np.random.default_rng(seed)
    style = OPERATORS[operator]
    cache = renderer_cache if renderer_cache is not None else {}
    if cache.get("scene") != scene.scene_id:
        if "renderer" in cache:
            cache["renderer"].close()
        model = mujoco.MjModel.from_xml_string(build_mjcf(scene, scene.block_xy, 0.0, np.random.default_rng(7),
                                                          width, height))
        cache.update(scene=scene.scene_id, model=model, renderer=mujoco.Renderer(model, height, width),
                     body_pos0=model.body_pos.copy())
    model, renderer = cache["model"], cache["renderer"]
    model.body_pos[:] = cache["body_pos0"]
    bid = model.body("block").id
    xy = np.array(scene.block_xy) + rng.uniform(-0.03, 0.03, 2)
    yaw = rng.uniform(-0.4, 0.4)
    model.body_pos[bid, :2] = xy
    model.body_quat[bid] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

    data = mujoco.MjData(model)
    ik = mujoco.MjData(model)  # kinematic copy: IK runs on the commanded configuration, not the measured one
    q0 = np.array(HOME_Q) + rng.normal(0, 0.03, 6)
    data.qpos[:6] = q0
    data.ctrl[:] = q0
    mujoco.mj_forward(model, data)
    for _ in range(int(0.5 * PHYS_HZ)):  # settle before recording starts
        mujoco.mj_step(model, data)
    data.time = 0.0
    tcp = model.site("tcp").id
    hole_top = np.array([*xy, BLOCK_HEIGHT])
    op = Operator(style, data.site_xpos[tcp].copy(), hole_top, yaw + rng.normal(0, 0.05), rng)
    op.q_start = np.zeros(4)
    mujoco.mju_mat2Quat(op.q_start, data.site_xmat[tcp])

    cams = ["front", "wrist", "base"]
    log: dict[str, list] = {k: [] for k in ["t_q", "qpos", "qvel", "t_ee", "tcp_pose", "t_ft", "ft", "t_cmd", "ctrl",
                                            "t_tel", "teleop", "phase"]}
    imgs = {c: {"t": [], "rgb": [], "depth": []} for c in cams}
    q_cmd = q0.copy()
    q_prev = q0.copy()
    target_pos, phase = op.target(0.0)
    target_q = op.orientation(0.0)
    filt_pos = target_pos.copy()
    alpha = 1 - np.exp(-(1 / CTRL_HZ) / 0.06)  # 60 ms command smoothing (servoL-style lookahead)
    z_latch = None
    f_contact = 0.0  # low-pass filtered contact force the "operator" feels
    ft_rest = None
    sub = PHYS_HZ // CTRL_HZ
    n_steps = int((op.t_end + 0.3) * PHYS_HZ)
    ft_bias = rng.normal(0, [0.3, 0.3, 0.5, 0.01, 0.01, 0.005])  # sensor zero offset
    next_cam = 0.0
    max_force = 0.0
    for step in range(n_steps):
        t = step / PHYS_HZ
        if step % (PHYS_HZ // CTRL_HZ) == 0:
            if t + 1e-9 >= len(log["t_tel"]) / TELEOP_HZ:
                lag = 0.08  # operator reaction / device latency
                target_pos, phase = op.target(max(0.0, t - lag))
                target_q = op.orientation(max(0.0, t - lag))
                if f_contact > 10.0 and z_latch is None:  # operator feels axial resistance: stop going deeper
                    z_latch = data.site_xpos[tcp][2] - 0.001
                if z_latch is not None:
                    target_pos[2] = max(target_pos[2], z_latch)
                log["t_tel"].append(t)
                qt = target_q
                log["teleop"].append(np.concatenate([target_pos, [qt[1], qt[2], qt[3], qt[0]]]))  # xyzw
                log["phase"].append(phase)
            filt_pos += alpha * (target_pos - filt_pos)
            for _ in range(2):
                ik.qpos[:6] = q_cmd
                mujoco.mj_kinematics(model, ik)
                mujoco.mj_comPos(model, ik)
                q_cmd = dls_ik_step(model, ik, tcp, filt_pos, target_q, q_cmd, max_step=0.02)
            q_prev = data.ctrl[:6].copy()
            log["t_cmd"].append(t)
            log["ctrl"].append(np.concatenate([q_cmd, [1.0]]))
            log["t_q"].append(t)
            log["qpos"].append(np.concatenate([data.qpos[:6] + rng.normal(0, 1e-5, 6), [1.0]]))
            log["qvel"].append(data.qvel[:6].copy())
            log["t_ee"].append(t)
            xq = np.zeros(4)
            mujoco.mju_mat2Quat(xq, data.site_xmat[tcp])
            log["tcp_pose"].append(np.concatenate([data.site_xpos[tcp], xq]))  # wxyz (MuJoCo order)
        data.ctrl[:] = q_prev + (q_cmd - q_prev) * ((step % sub) + 1) / sub  # controller interpolates commands
        ft = data.sensordata[:6].copy()
        if step == 25:
            ft_rest = ft.copy()
        if ft_rest is not None:
            f_contact += 0.05 * (abs(float(ft[2] - ft_rest[2])) - f_contact)
        log["t_ft"].append(t)
        log["ft"].append(ft + ft_bias + rng.normal(0, [0.05, 0.05, 0.08, 0.002, 0.002, 0.001]))
        if step < 50:
            ft0 = ft.copy()
        max_force = max(max_force, float(np.linalg.norm(ft[:3] - ft0[:3])))
        if step == round(next_cam * PHYS_HZ / CAM_HZ):  # frame k on the physics step closest to k / CAM_HZ
            for c in cams:
                renderer.disable_depth_rendering()
                renderer.update_scene(data, camera=c)
                imgs[c]["rgb"].append(renderer.render().copy())  # ideal render; sensor noise is applied by the converter
                renderer.enable_depth_rendering()
                renderer.update_scene(data, camera=c)
                imgs[c]["depth"].append(renderer.render().astype(np.float32))
                imgs[c]["t"].append(t)
            renderer.disable_depth_rendering()
            next_cam += 1
        mujoco.mj_step(model, data)

    depth_reached = float(hole_top[2] - data.site_xpos[tcp][2])
    lateral = float(np.linalg.norm(data.site_xpos[tcp][:2] - xy))
    success = bool(depth_reached > BLOCK_HEIGHT * 0.6 and lateral < HOLE_HALF + PEG_RADIUS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        a = f.attrs
        a.update(sim="mujoco", mujoco_version=mujoco.__version__, seed=seed, scene_id=scene.scene_id,
                 operator=operator, task=TASK, success=success, physics_hz=PHYS_HZ, control_hz=CTRL_HZ,
                 depth_reached_m=depth_reached, lateral_err_m=lateral, misalign_bias_mm=op.bias_mm,
                 max_contact_force_n=max_force, phase_names=",".join(PHASES))
        a["model_xml"] = build_mjcf(scene, scene.block_xy, 0.0, np.random.default_rng(7), width, height)
        g = f.create_group("sensors")
        for name, tk, vk in [("qpos", "t_q", "qpos"), ("qvel", "t_q", "qvel"), ("tcp_pose", "t_ee", "tcp_pose"),
                             ("ft", "t_ft", "ft"), ("ctrl", "t_cmd", "ctrl"), ("teleop_target", "t_tel", "teleop"),
                             ("phase", "t_tel", "phase")]:
            s = g.create_group(name)
            s.create_dataset("t", data=np.asarray(log[tk], np.float64))
            s.create_dataset("data", data=np.asarray(log[vk]))
        cg = f.create_group("cameras")
        for c in cams:
            s = cg.create_group(f"cam_{c}")
            s.create_dataset("t", data=np.asarray(imgs[c]["t"]))
            s.create_dataset("rgb", data=np.stack(imgs[c]["rgb"]), chunks=(1, height, width, 3), compression="gzip",
                             compression_opts=3, shuffle=True)
            s.create_dataset("depth_m", data=np.stack(imgs[c]["depth"]), chunks=(1, height, width), compression="gzip",
                             compression_opts=3, shuffle=True)
            s.attrs["fovy_deg"] = float(model.cam_fovy[model.camera(c).id])
    return {"path": str(out_path), "success": success, "depth": depth_reached, "lateral_mm": lateral * 1000,
            "bias_mm": op.bias_mm, "duration_s": op.t_end, "scene": scene.scene_id, "operator": operator}


def _worker(jobs: list[tuple[str, str, int, str]]) -> list[dict]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    cache: dict = {}
    out = [run_episode(SCENES[s], op, seed, Path(p), renderer_cache=cache) for s, op, seed, p in jobs]
    if "renderer" in cache:
        cache["renderer"].close()
    return out


def generate(out_dir: str | Path, episodes_per_scene: int = 8, scenes: list[str] | None = None, workers: int = 6,
             base_seed: int = 1000) -> list[dict]:
    """Episodes are grouped by scene so each worker keeps one compiled model + renderer per scene."""
    out_dir = Path(out_dir)
    scenes = scenes or sorted(SCENES)
    jobs_by_scene = {}
    k = 0
    for s in scenes:
        jobs = []
        for i in range(episodes_per_scene):
            op = sorted(OPERATORS)[i % len(OPERATORS)]
            jobs.append((s, op, base_seed + k, str(out_dir / s / f"sim_{s}_{i:04d}.hdf5")))
            k += 1
        jobs_by_scene[s] = jobs
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_worker, jobs) for jobs in jobs_by_scene.values()]
        for f in as_completed(futs):
            for r in f.result():
                results.append(r)
                print(f"[sim] {Path(r['path']).name} success={r['success']} lateral={r['lateral_mm']:.1f}mm "
                      f"depth={r['depth'] * 1000:.0f}mm", flush=True)
    return sorted(results, key=lambda r: r["path"])
