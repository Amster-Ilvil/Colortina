"""Detect pages that are already in color.

Idea borrowed from Manga-Colorization-FJ's "skip already colorized
images" feature: before running the (expensive) colorizer on a page,
check whether the page already contains meaningful chroma.  Scanned
color pages, publisher color inserts, and previously-colorized output
all get skipped instead of being re-colorized (which usually makes
them worse).

Detection is done in Lab space: a pixel counts as "colored" when its
chroma distance sqrt((a-128)^2 + (b-128)^2) exceeds `chroma_thresh`.
A page is "already colored" when the fraction of colored pixels
exceeds `frac_thresh`.  Thresholds are deliberately conservative —
JPEG chroma noise and slightly yellowed paper stay below them.
"""

from __future__ import annotations

import cv2
import numpy as np

# Max analysis size — detection doesn't need full resolution.
_ANALYZE_MAX_SIDE = 512


def color_fraction(image_bgr: np.ndarray, chroma_thresh: float = 14.0) -> float:
    """Return the fraction of pixels with Lab chroma above `chroma_thresh`."""
    if image_bgr is None or image_bgr.size == 0:
        return 0.0
    if image_bgr.ndim == 2 or image_bgr.shape[2] == 1:
        return 0.0

    h, w = image_bgr.shape[:2]
    scale = _ANALYZE_MAX_SIDE / max(h, w)
    if scale < 1.0:
        image_bgr = cv2.resize(image_bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                               interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0
    chroma = np.sqrt(a * a + b * b)
    return float(np.mean(chroma > chroma_thresh))


def is_already_colored(image_bgr: np.ndarray,
                       chroma_thresh: float = 14.0,
                       frac_thresh: float = 0.03) -> bool:
    """True if the page already contains significant color content."""
    return color_fraction(image_bgr, chroma_thresh) >= frac_thresh
