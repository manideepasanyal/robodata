"""Reader for the C++ stamp_monitor binary logs (``*.rdts``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

STAMP_DTYPE = np.dtype([("t_device_ns", "<i8"), ("t_host_ns", "<i8"), ("bytes", "<u4")])


def read_stamp_log(path: str | Path) -> np.ndarray:
    raw = Path(path).read_bytes()
    if not raw.startswith(b"RDTS1\n"):
        raise ValueError(f"{path}: not an RDTS1 stamp log")
    body = raw[6:]
    n = len(body) // STAMP_DTYPE.itemsize  # tolerate a torn final record after a crash
    return np.frombuffer(body[: n * STAMP_DTYPE.itemsize], dtype=STAMP_DTYPE)
