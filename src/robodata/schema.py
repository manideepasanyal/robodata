"""Naming conventions shared by every stage (raw ids, LeRobot feature keys)."""

from __future__ import annotations

import re

# LeRobot-style feature keys. Cameras: observation.images.<cam>, depth: observation.images.<cam>_depth
IMAGE_PREFIX = "observation.images."
DEPTH_SUFFIX = "_depth"
STATE = "observation.state"
EE_POSE = "observation.ee_pose"
FORCE_TORQUE = "observation.force_torque"
TELEOP = "teleop.target_pose"
ACTION = "action"
IS_SIM = "is_sim"
PHASE = "phase_index"
SYNC_RESIDUAL = "observation.sync_residual_ms"

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe(s: str) -> str:
    return _SAFE.sub("-", s)


def episode_id(robot_id: str, session_id: str, index: int) -> str:
    """Globally unique, sortable, filesystem-safe episode id."""
    return f"{safe(robot_id)}__{safe(session_id)}__ep{index:06d}"


def image_key(cam: str) -> str:
    return f"{IMAGE_PREFIX}{cam}"


def depth_key(cam: str) -> str:
    return f"{IMAGE_PREFIX}{cam}{DEPTH_SUFFIX}"
