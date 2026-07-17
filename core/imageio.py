"""Unicode-safe image file IO.

On Windows, OpenCV's ``cv2.imread`` / ``cv2.imwrite`` use the ANSI C
file API and silently fail on paths containing Chinese (or any
non-ASCII) characters — ``imread`` returns None, ``imwrite`` returns
False.  These wrappers route the bytes through NumPy's
``fromfile`` / ``tofile`` (which use Python's Unicode-aware file
handling) and OpenCV's in-memory ``imdecode`` / ``imencode``, so
every path works on every platform.

Use these everywhere instead of cv2.imread / cv2.imwrite.
"""

from __future__ import annotations

import os

import cv2
import numpy as np


def imread(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Drop-in unicode-safe replacement for cv2.imread."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite(path: str, image: np.ndarray, params=None) -> bool:
    """Drop-in unicode-safe replacement for cv2.imwrite."""
    ext = os.path.splitext(path)[1] or ".png"
    try:
        ok, buf = cv2.imencode(ext, image, params or [])
    except cv2.error:
        return False
    if not ok:
        return False
    try:
        buf.tofile(path)
    except OSError:
        return False
    return True
