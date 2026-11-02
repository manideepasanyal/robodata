"""ROS message <-> numpy conversions (no cv_bridge dependency)."""

from __future__ import annotations

import numpy as np

_IMG_DTYPES = {"rgb8": (np.uint8, 3), "bgr8": (np.uint8, 3), "rgba8": (np.uint8, 4), "bgra8": (np.uint8, 4),
               "mono8": (np.uint8, 1), "16UC1": (np.uint16, 1), "mono16": (np.uint16, 1), "32FC1": (np.float32, 1)}


def stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def set_stamp(header, t_ns: int) -> None:
    header.stamp.sec = int(t_ns // 1_000_000_000)
    header.stamp.nanosec = int(t_ns % 1_000_000_000)


def image_to_numpy(msg, kind: str) -> np.ndarray:
    if hasattr(msg, "format"):  # CompressedImage
        import cv2

        buf = np.frombuffer(bytes(msg.data), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        return img[..., ::-1].copy() if kind == "rgb" and img.ndim == 3 else img
    dtype, ch = _IMG_DTYPES[msg.encoding]
    row = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)
    px = row[:, : msg.width * ch * np.dtype(dtype).itemsize].view(dtype)
    img = px.reshape(msg.height, msg.width, ch) if ch > 1 else px.reshape(msg.height, msg.width)
    if msg.encoding.startswith("bgr"):
        img = img[..., [2, 1, 0]]
    if kind == "rgb":
        return np.ascontiguousarray(img[..., :3])
    if msg.encoding == "32FC1":  # metres -> uint16 millimetres (0 = invalid)
        mm = np.nan_to_num(img * 1000.0, nan=0.0, posinf=0.0)
        return np.clip(mm, 0, 65535).astype(np.uint16)
    return img.astype(np.uint16)


def numpy_to_image(arr: np.ndarray, kind: str, msg_cls):
    msg = msg_cls()
    msg.height, msg.width = arr.shape[:2]
    if kind == "rgb":
        msg.encoding, msg.step = "rgb8", arr.shape[1] * 3
        msg.data = np.ascontiguousarray(arr, np.uint8).tobytes()
    else:
        msg.encoding, msg.step = "16UC1", arr.shape[1] * 2
        msg.data = np.ascontiguousarray(arr, np.uint16).tobytes()
    return msg


def _quat_to_euler(q):
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat(q).as_euler("xyz")


def msg_to_numpy(msg, kind: str, shape: list[int], names: list[str] | None = None) -> np.ndarray:
    if kind in ("rgb", "depth"):
        return image_to_numpy(msg, kind)
    if kind in ("joint_state", "joint_cmd", "twist_cmd") and hasattr(msg, "position"):
        pos = np.asarray(msg.position, np.float32)
        if names and list(msg.name):
            lut = dict(zip(msg.name, pos))
            return np.array([lut.get(n, np.nan) for n in names], np.float32)
        return pos[: shape[0]]
    if kind in ("pose", "pose_cmd"):
        p, o = msg.pose.position, msg.pose.orientation
        q = [o.x, o.y, o.z, o.w]
        if shape[0] == 6:
            return np.array([p.x, p.y, p.z, *_quat_to_euler(q)], np.float32)
        return np.array([p.x, p.y, p.z, *q], np.float32)
    if kind == "wrench":
        f, t = msg.wrench.force, msg.wrench.torque
        return np.array([f.x, f.y, f.z, t.x, t.y, t.z], np.float32)
    if kind == "twist_cmd":
        l, a = msg.twist.linear, msg.twist.angular
        return np.array([l.x, l.y, l.z, a.x, a.y, a.z], np.float32)
    raise ValueError(f"no conversion for kind={kind} msg={type(msg).__name__}")


def numpy_to_msg(x: np.ndarray, kind: str, msg_cls, names: list[str] | None = None):
    msg = msg_cls()
    if kind in ("rgb", "depth"):
        return numpy_to_image(x, kind, msg_cls)
    if hasattr(msg, "position"):  # JointState carries joint states and command vectors
        msg.name = list(names or [])
        msg.position = [float(v) for v in x]
        return msg
    if kind in ("pose", "pose_cmd"):
        p, o = msg.pose.position, msg.pose.orientation
        p.x, p.y, p.z = map(float, x[:3])
        if len(x) == 6:
            from scipy.spatial.transform import Rotation

            q = Rotation.from_euler("xyz", x[3:6]).as_quat()
        else:
            q = x[3:7]
        o.x, o.y, o.z, o.w = map(float, q)
        return msg
    if kind == "wrench":
        f, t = msg.wrench.force, msg.wrench.torque
        f.x, f.y, f.z, t.x, t.y, t.z = map(float, x[:6])
        return msg
    raise ValueError(kind)
