"""ctypes bridge for the Mojo interval-query kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import threading

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "dist", "libmojo-intervaltree.so")
SRC = os.path.join(ROOT, "src", "intervaltree.mojo")

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mit_build_index": ([I] * 11, I),
    "mit_query_one": ([I] * 8 + [F, F] + [I, I], I),
    "mit_count_many": ([I] * 14, I),
    "mit_fill_many": ([I] * 14, I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if (
        not force
        and os.path.exists(LIB)
        and os.path.getmtime(LIB) >= os.path.getmtime(SRC)
    ):
        return LIB
    mojo = shutil.which("mojo")
    if mojo is None:
        raise BuildError("mojo not found; run inside the Pixi environment")
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None
_lock = threading.Lock()


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        with _lock:
            if _library is None:
                loaded = ctypes.CDLL(build())
                for name, (argtypes, restype) in _SIGNATURES.items():
                    fn = getattr(loaded, name)
                    fn.argtypes = argtypes
                    fn.restype = restype
                _library = loaded
    return _library


def addr(array: np.ndarray, dtype, *, writable: bool = False) -> int:
    """Return an address only for a non-empty ABI-compatible NumPy buffer."""
    if not isinstance(array, np.ndarray):
        raise TypeError("native buffers must be NumPy arrays")
    if array.dtype != np.dtype(dtype):
        raise TypeError(f"native buffer must have dtype {np.dtype(dtype)}")
    if array.ndim != 1 or not array.flags.c_contiguous or not array.flags.aligned:
        raise ValueError("native buffers must be aligned, contiguous 1D arrays")
    if array.size == 0:
        raise ValueError("empty arrays do not have a native buffer")
    if writable and not array.flags.writeable:
        raise ValueError("native output buffers must be writable")
    address = int(array.ctypes.data)
    if address == 0:
        raise ValueError("native buffer has a null address")
    return address
