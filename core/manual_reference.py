"""Manual character-reference helpers.

Automatic anime-face detection is useful for ordinary upright pages, but dense
covers, rotated characters and occlusion still need an explicit correction
path.  These helpers turn a user-selected head rectangle and sampled identity
colours into stable, dependency-free matching features.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.anime_face_detector import lineart_descriptor
from core.reference_points import sample_reference_rgb


def clip_bbox(bbox: tuple[int, int, int, int],
              shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    x, y, bw, bh = (int(round(v)) for v in bbox)
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    bw = max(1, min(w - x, bw))
    bh = max(1, min(h - y, bh))
    return x, y, bw, bh


def rotate_crop(crop: np.ndarray, rotation: int) -> np.ndarray:
    rotation %= 360
    if rotation == 90:
        return cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(crop, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return crop


def sample_hex_at(image_bgr: np.ndarray, x: int, y: int,
                  radius_px: int = 9) -> str:
    """Robustly sample one colour while rejecting paper/ink outliers."""
    h, w = image_bgr.shape[:2]
    rgb = sample_reference_rgb(
        image_bgr,
        float(np.clip(x, 0, w - 1)) / max(1, w - 1),
        float(np.clip(y, 0, h - 1)) / max(1, h - 1),
        radius_px=max(2, int(radius_px)))
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def manual_head_features(image_bgr: np.ndarray,
                         head_bbox: tuple[int, int, int, int],
                         rotation: int = 0
                         ) -> tuple[float, list[float], float, float, list[float]]:
    """Return tone/hist/shape/lineart features for a selected head crop."""
    h, w = image_bgr.shape[:2]
    x, y, bw, bh = clip_bbox(head_bbox, (h, w))
    crop = rotate_crop(image_bgr[y:y + bh, x:x + bw], rotation)
    if crop.size == 0:
        raise ValueError("empty manual reference crop")
    gray = (crop if crop.ndim == 2
            else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))

    ch, cw = gray.shape[:2]
    yy, xx = np.mgrid[0:ch, 0:cw]
    # A head selection normally includes hair around a central face.  Use the
    # top and side ring for tone features and fall back to the whole crop.
    central = (((xx - cw * 0.50) / max(1.0, cw * 0.31)) ** 2 +
               ((yy - ch * 0.55) / max(1.0, ch * 0.36)) ** 2) <= 1.0
    hair_zone = ~central & ((yy < ch * 0.72) |
                            (xx < cw * 0.23) | (xx > cw * 0.77))
    pixels = gray[hair_zone]
    pixels = pixels[(pixels > 8) & (pixels < 250)]
    if len(pixels) < 32:
        pixels = gray.reshape(-1)
    hist, _ = np.histogram(pixels, bins=12, range=(0, 256))
    hist = hist.astype(np.float32)
    hist /= max(1.0, float(hist.sum()))
    tone = float(np.median(pixels))
    aspect = float(cw / max(1.0, ch))
    area_frac = float(bw * bh / max(1.0, h * w))

    # lineart_descriptor operates on an image+bbox.  The already-normalized
    # crop avoids orientation-dependent descriptors.
    descriptor = lineart_descriptor(
        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        (0, 0, cw, ch), rotation=0)
    return tone, hist.astype(float).tolist(), aspect, area_frac, descriptor
