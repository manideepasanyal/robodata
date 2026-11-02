"""MJCF builder for the UR5 insertion cell (digital twin of the catheter-insertion rig).

UR5 link offsets follow the MuJoCo Menagerie UR5e model (BSD-3); geometry is simplified
to capsules so the scene is self-contained. Domain randomisation (lighting, table colour,
block pose, camera jitter) is grouped into discrete *scenes* so that train/val/eval splits
can be made by scene without leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PEG_RADIUS = 0.004
PEG_LENGTH = 0.09
HOLE_HALF = 0.0058  # 1.8 mm radial clearance
BLOCK_HALF = 0.06
BLOCK_HEIGHT = 0.05
TABLE_Z = 0.0
HOME_Q = [-3.30, -1.18, 1.39, -1.75, -1.60, 1.20]  # same start pose as the real rig
JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]


@dataclass
class SceneParams:
    scene_id: str
    table_rgb: tuple
    light_dir: tuple
    light_diffuse: float
    block_xy: tuple  # nominal block centre, world frame
    cam_jitter: float


SCENES = {
    f"sim_scene_{i:02d}": SceneParams(
        scene_id=f"sim_scene_{i:02d}",
        table_rgb=tuple(np.round(np.array(c), 3)),
        light_dir=d,
        light_diffuse=l,
        block_xy=xy,
        cam_jitter=j,
    )
    for i, (c, d, l, xy, j) in enumerate([
        ((0.86, 0.85, 0.82), (0.3, 0.2, -1), 0.8, (-0.52, 0.10), 0.00),
        ((0.78, 0.80, 0.84), (-0.3, 0.4, -1), 0.7, (-0.50, 0.05), 0.01),
        ((0.90, 0.88, 0.80), (0.0, -0.4, -1), 0.9, (-0.55, 0.12), 0.01),
        ((0.70, 0.72, 0.70), (0.5, 0.0, -1), 0.6, (-0.48, 0.08), 0.02),
        ((0.92, 0.92, 0.92), (-0.2, -0.2, -1), 1.0, (-0.53, 0.02), 0.02),
        ((0.80, 0.76, 0.70), (0.2, 0.5, -1), 0.75, (-0.57, 0.07), 0.015),
    ])
}


def _block_geoms() -> str:
    """Square block with a through hole, built from 4 boxes (MuJoCo primitives cannot be hollow)."""
    w = (BLOCK_HALF - HOLE_HALF) / 2
    off = HOLE_HALF + w
    h = BLOCK_HEIGHT / 2
    g = []
    for name, (px, py, sx, sy) in {
        "n": (0, off, BLOCK_HALF, w), "s": (0, -off, BLOCK_HALF, w),
        "e": (off, 0, w, HOLE_HALF), "w": (-off, 0, w, HOLE_HALF),
    }.items():
        g.append(f'<geom name="block_{name}" type="box" pos="{px} {py} {h}" size="{sx} {sy} {h}" material="alu" '
                 f'friction="0.4 0.005 0.0001" solref="0.02 1"/>')
    # decorative (non-colliding) dummy holes on the top face, like the real hole plate
    for k, (dx, dy) in enumerate([(0.03, 0.03), (-0.03, 0.03), (0.03, -0.03), (-0.03, -0.03), (0.0, 0.035), (0.035, 0.0)]):
        g.append(f'<geom name="deco_{k}" type="cylinder" pos="{dx} {dy} {BLOCK_HEIGHT + 0.0002}" size="0.006 0.0003" '
                 f'rgba="0.15 0.15 0.17 1" contype="0" conaffinity="0"/>')
    g.append(f'<site name="hole_top" pos="0 0 {BLOCK_HEIGHT}" size="0.003" rgba="1 0 0 0"/>')
    return "\n        ".join(g)


def build_mjcf(p: SceneParams, block_xy: tuple, block_yaw: float, rng: np.random.Generator | None = None,
               width: int = 320, height: int = 240, with_board: bool = False) -> str:
    rng = rng or np.random.default_rng(0)
    j = p.cam_jitter
    front_pos = np.array([-0.55, 0.65, 0.35]) + rng.normal(0, j, 3)
    base_pos = np.array([-0.15, 0.55, 0.40]) + rng.normal(0, j, 3)
    tr, tg, tb = p.table_rgb
    lx, ly, lz = p.light_dir
    bx, by = block_xy
    board = _board_xml() if with_board else ""
    return f"""
<mujoco model="ur5_insertion">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="5"/>
  <visual>
    <global offwidth="{max(width, 640)}" offheight="{max(height, 480)}" fovy="45"/>
    <quality shadowsize="2048"/>
    <headlight ambient="0.15 0.15 0.15" diffuse="0.25 0.25 0.25" specular="0 0 0"/>
    <map znear="0.01" zfar="5"/>
  </visual>
  <asset>
    <texture name="tabletex" type="2d" builtin="flat" rgb1="{tr} {tg} {tb}" mark="random" markrgb="{tr*0.9:.3f} {tg*0.9:.3f} {tb*0.9:.3f}" random="0.02" width="512" height="512"/>
    <material name="table" texture="tabletex" texrepeat="2 2" specular="0.1"/>
    <material name="alu" rgba="0.62 0.64 0.68 1" specular="0.6" shininess="0.6"/>
    <material name="ur_grey" rgba="0.72 0.74 0.76 1" specular="0.4"/>
    <material name="ur_blue" rgba="0.36 0.52 0.68 1"/>
    <material name="dark" rgba="0.1 0.1 0.11 1"/>
    <material name="cable" rgba="0.55 0.55 0.58 1"/>
  </asset>
  <default>
    <joint damping="5" armature="0.3" range="-6.2832 6.2832"/>
    <geom contype="1" conaffinity="1" solref="0.004 1" solimp="0.95 0.99 0.001"/>
    <default class="link"><geom type="capsule" material="ur_grey" contype="0" conaffinity="0"/></default>
    <default class="jointcap"><geom type="cylinder" material="ur_blue" contype="0" conaffinity="0"/></default>
  </default>
  <worldbody>
    <light name="key" directional="true" dir="{lx} {ly} {lz}" diffuse="{p.light_diffuse*0.55:.3f} {p.light_diffuse*0.55:.3f} {p.light_diffuse*0.55:.3f}" castshadow="true"/>
    <light name="fill" pos="-0.5 0.5 1.5" dir="0 0 -1" diffuse="0.15 0.15 0.15" castshadow="false"/>
    <geom name="floor" type="plane" size="3 3 0.1" pos="0 0 -0.75" rgba="0.12 0.12 0.13 1" contype="0" conaffinity="0"/>
    <geom name="table" type="box" pos="-0.45 0.1 -0.02" size="0.45 0.55 0.02" material="table"/>
    <camera name="front" pos="{front_pos[0]:.4f} {front_pos[1]:.4f} {front_pos[2]:.4f}" mode="targetbody" target="block" fovy="58"/>
    <camera name="base" pos="{base_pos[0]:.4f} {base_pos[1]:.4f} {base_pos[2]:.4f}" mode="targetbody" target="block" fovy="58"/>
    <body name="block" pos="{bx:.4f} {by:.4f} {TABLE_Z}" euler="0 0 {block_yaw:.4f}">
        {_block_geoms()}
    </body>
    {board}
    <body name="base" pos="0 0 0">
      <geom class="jointcap" size="0.075 0.04" pos="0 0 0.04"/>
      <body name="shoulder_link" pos="0 0 0.163" gravcomp="1">
        <inertial mass="3.7" pos="0 0 0" diaginertia="0.0103 0.0103 0.0067"/>
        <joint name="shoulder_pan_joint" axis="0 0 1"/>
        <geom class="jointcap" size="0.06 0.07"/>
        <body name="upper_arm_link" pos="0 0.138 0" quat="1 0 1 0" gravcomp="1">
          <inertial mass="8.393" pos="0 0 0.2125" diaginertia="0.134 0.134 0.0151"/>
          <joint name="shoulder_lift_joint" axis="0 1 0"/>
          <geom class="jointcap" size="0.06 0.07" quat="1 1 0 0"/>
          <geom class="link" fromto="0 0 0 0 0 0.425" size="0.05"/>
          <body name="forearm_link" pos="0 -0.131 0.425" gravcomp="1">
            <inertial mass="2.275" pos="0 0 0.196" diaginertia="0.0312 0.0312 0.004"/>
            <joint name="elbow_joint" axis="0 1 0" range="-3.1415 3.1415"/>
            <geom class="jointcap" size="0.05 0.06" quat="1 1 0 0"/>
            <geom class="link" fromto="0 0 0 0 0 0.392" size="0.038"/>
            <body name="wrist_1_link" pos="0 0 0.392" quat="1 0 1 0" gravcomp="1">
              <inertial mass="1.219" pos="0 0.127 0" diaginertia="0.0025599 0.0025599 0.0021942"/>
              <joint name="wrist_1_joint" axis="0 1 0"/>
              <geom class="jointcap" size="0.04 0.05" quat="1 1 0 0"/>
              <body name="wrist_2_link" pos="0 0.127 0" gravcomp="1">
                <inertial mass="1.219" pos="0 0 0.1" diaginertia="0.0025599 0.0025599 0.0021942"/>
                <joint name="wrist_2_joint" axis="0 0 1"/>
                <geom class="jointcap" size="0.04 0.05"/>
                <body name="wrist_3_link" pos="0 0 0.1" gravcomp="1">
                  <inertial mass="0.1889" pos="0 0.0771 0" diaginertia="0.000132 0.000132 0.0001"/>
                  <joint name="wrist_3_joint" axis="0 1 0"/>
                  <geom class="jointcap" size="0.04 0.04" quat="1 1 0 0" material="dark"/>
                  <body name="tool" pos="0 0.1 0" quat="-1 1 0 0" gravcomp="1">
                    <inertial mass="0.9" pos="0 0 0.05" diaginertia="0.002 0.002 0.001"/>
                    <site name="ft_site" pos="0 0 0" size="0.005"/>
                    <geom name="ft_body" type="cylinder" size="0.038 0.015" pos="0 0 0.015" material="alu" contype="0" conaffinity="0"/>
                    <geom name="gripper_body" type="box" size="0.03 0.045 0.035" pos="0 0 0.065" material="dark" contype="0" conaffinity="0"/>
                    <geom name="finger_l" type="box" size="0.008 0.012 0.025" pos="0 0.012 0.125" material="dark" contype="0" conaffinity="0"/>
                    <geom name="finger_r" type="box" size="0.008 0.012 0.025" pos="0 -0.012 0.125" material="dark" contype="0" conaffinity="0"/>
                    <camera name="wrist" pos="0 -0.065 0.115" xyaxes="1 0 0 0 -0.966 0.259" fovy="75"/>
                    <geom name="peg" type="capsule" fromto="0 0 0.12 0 0 {0.12 + PEG_LENGTH}" size="{PEG_RADIUS}" material="cable" friction="0.3 0.005 0.0001" condim="3" solref="0.02 1"/>
                    <site name="tcp" pos="0 0 {0.12 + PEG_LENGTH + PEG_RADIUS}" size="0.002"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a0" joint="shoulder_pan_joint" kp="6000" dampratio="1" forcerange="-150 150"/>
    <position name="a1" joint="shoulder_lift_joint" kp="6000" dampratio="1" forcerange="-150 150"/>
    <position name="a2" joint="elbow_joint" kp="4000" dampratio="1" forcerange="-150 150"/>
    <position name="a3" joint="wrist_1_joint" kp="1500" dampratio="1" forcerange="-28 28"/>
    <position name="a4" joint="wrist_2_joint" kp="1500" dampratio="1" forcerange="-28 28"/>
    <position name="a5" joint="wrist_3_joint" kp="1500" dampratio="1" forcerange="-28 28"/>
  </actuator>
  <sensor>
    <force name="ft_force" site="ft_site"/>
    <torque name="ft_torque" site="ft_site"/>
  </sensor>
</mujoco>
"""


# --- calibration target ----------------------------------------------------------------
BOARD_COLS, BOARD_ROWS, BOARD_SQ = 9, 6, 0.035  # odd x even: no 180-degree ambiguity; inner corners (8, 5)
BOARD_POS = (-0.45, 0.10, 0.001)
BOARD_SURFACE_Z = 0.0007  # printed squares sit this far above the board body origin


def _board_xml() -> str:
    g = [f'<geom type="box" size="{(BOARD_COLS + 2) * BOARD_SQ / 2} {(BOARD_ROWS + 2) * BOARD_SQ / 2} 0.0005" '
         f'rgba="1 1 1 1" contype="0" conaffinity="0"/>']
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if (r + c) % 2 == 0:
                x = (c - BOARD_COLS / 2 + 0.5) * BOARD_SQ
                y = (r - BOARD_ROWS / 2 + 0.5) * BOARD_SQ
                g.append(f'<geom type="box" pos="{x:.4f} {y:.4f} 0.0006" size="{BOARD_SQ / 2} {BOARD_SQ / 2} 0.0001" '
                         f'rgba="0 0 0 1" contype="0" conaffinity="0"/>')
    bx, by, bz = BOARD_POS
    return f'<body name="board" pos="{bx} {by} {bz}">' + "".join(g) + "</body>"


def board_object_points() -> np.ndarray:
    """Inner-corner coordinates in the board *surface* frame (OpenCV ordering: row-major)."""
    pts = []
    for r in range(1, BOARD_ROWS):
        for c in range(1, BOARD_COLS):
            pts.append([(c - BOARD_COLS / 2) * BOARD_SQ, (r - BOARD_ROWS / 2) * BOARD_SQ, 0.0])  # planar (OpenCV)
    return np.array(pts, np.float32)
