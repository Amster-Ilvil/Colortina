"""Utilities for precise reference-to-target point colour transfer."""

from __future__ import annotations

import cv2
import numpy as np


def sample_reference_rgb(image_bgr: np.ndarray, x_norm: float, y_norm: float,
                         radius_px: int = 12) -> tuple[int, int, int]:
    """Sample a robust identity colour around a point on a colour reference.

    Neutral paper, black ink and blown highlights are ignored when possible;
    the median of the remaining pixels is stable on gradients and JPEG noise.
    """
    h, w = image_bgr.shape[:2]
    x = min(w - 1, max(0, int(round(float(x_norm) * (w - 1)))))
    y = min(h - 1, max(0, int(round(float(y_norm) * (h - 1)))))
    radius_px = max(2, int(radius_px))
    x1, x2 = max(0, x - radius_px), min(w, x + radius_px + 1)
    y1, y2 = max(0, y - radius_px), min(h, y + radius_px + 1)
    crop = image_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    valid = ((hsv[..., 1] >= 18) & (hsv[..., 2] >= 25) &
             (hsv[..., 2] <= 245))
    pixels = crop[valid] if np.any(valid) else crop.reshape(-1, 3)
    b, g, r = np.median(pixels.astype(np.float32), axis=0)
    return int(r), int(g), int(b)
