"""Lightweight manga face-part heuristics for eye/skin consistency.

This is not a face detector.  It supplements CLIP inside an already matched
head box so tiny iris regions are not lost merely because zero-shot CLIP cannot
classify a few pixels reliably.
"""

from __future__ import annotations

import cv2
import numpy as np


def _full_region_mask(segmentation, region, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    low = (segmentation.labels == int(region.label_id)).astype(np.uint8)
    return cv2.resize(low, (w, h), interpolation=cv2.INTER_NEAREST) > 0


def _hex_from_bgr_pixels(pixels: np.ndarray) -> str | None:
    if pixels.size == 0:
        return None
    b, g, r = np.median(pixels.astype(np.float32), axis=0)
    return f"#{int(np.clip(r, 0, 255)):02x}{int(np.clip(g, 0, 255)):02x}{int(np.clip(b, 0, 255)):02x}"


def sample_face_palette(color_bgr: np.ndarray, segmentation, hair_region,
                        head_bbox: tuple[int, int, int, int]) -> dict[str, str]:
    """Sample canonical skin and iris colours from a reference head crop."""
    H, W = color_bgr.shape[:2]
    x, y, w, h = head_bbox
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return {}
    crop = color_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
    hair_mask = _full_region_mask(segmentation, hair_region, (H, W))[y1:y2, x1:x2]

    # Broad anime-skin gate in YCrCb; exclude paper/highlights, ink and hair.
    Y, Cr, Cb = cv2.split(ycrcb)
    skin = ((Y > 45) & (Y < 242) & (Cr > 120) & (Cr < 183) &
            (Cb > 72) & (Cb < 145) & ~hair_mask)
    # Focus on central/lower head where face/neck are more likely than hair.
    yy, xx = np.mgrid[0:crop.shape[0], 0:crop.shape[1]]
    skin &= (xx > crop.shape[1] * 0.18) & (xx < crop.shape[1] * 0.82)
    skin &= (yy > crop.shape[0] * 0.22) & (yy < crop.shape[0] * 0.95)

    out: dict[str, str] = {}
    if int(skin.sum()) >= 24:
        useful = crop[skin]
        # Remove extreme saturation that is more likely eye/mouth decoration.
        useful_hsv = cv2.cvtColor(useful.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        useful = useful[useful_hsv[:, 1] < 150]
        value = _hex_from_bgr_pixels(useful)
        if value:
            out["skin"] = value

    # Iris candidates: chromatic, non-skin, non-hair pixels in the eye band.
    eye_band = ((yy > crop.shape[0] * 0.28) & (yy < crop.shape[0] * 0.68) &
                (xx > crop.shape[1] * 0.08) & (xx < crop.shape[1] * 0.92))
    iris = (eye_band & ~hair_mask & ~skin &
            (hsv[..., 1] > 45) & (hsv[..., 2] > 35) & (hsv[..., 2] < 235))
    n, comps, stats, centroids = cv2.connectedComponentsWithStats(
        iris.astype(np.uint8), connectivity=8)
    candidates = []
    crop_area = crop.shape[0] * crop.shape[1]
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 2 or area > max(80, crop_area * 0.035):
            continue
        cx, cy = centroids[i]
        candidates.append((i, area, float(cx), float(cy)))
    # Prefer a plausible left/right pair at similar height; otherwise the best
    # single compact candidate is still useful for a close-up.
    chosen: list[int] = []
    best_pair = None
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a, b = candidates[i], candidates[j]
            horizontal = abs(a[2] - b[2])
            vertical = abs(a[3] - b[3])
            if horizontal < crop.shape[1] * 0.12 or vertical > crop.shape[0] * 0.12:
                continue
            balance = abs(a[1] - b[1]) / max(a[1], b[1])
            score = vertical / max(1, crop.shape[0]) + balance * 0.35
            if best_pair is None or score < best_pair[0]:
                best_pair = (score, a[0], b[0])
    if best_pair is not None:
        chosen = [best_pair[1], best_pair[2]]
    elif candidates:
        chosen = [max(candidates, key=lambda item: item[1])[0]]
    if chosen:
        pixels = crop[np.isin(comps, chosen)]
        value = _hex_from_bgr_pixels(pixels)
        if value:
            out["eyes"] = value
    return out


def find_eye_region_ids(segmentation, hair_region,
                        page_shape: tuple[int, int], head_bbox,
                        existing_ids: set[int] | None = None) -> list[int]:
    """Find a plausible pair of tiny eye regions inside a head box."""
    existing_ids = existing_ids or set()
    x, y, w, h = head_bbox
    candidates = []
    for region in segmentation.regions:
        if region.label_id == hair_region.label_id or region.label_id in existing_ids:
            continue
        rx, ry, rw, rh = region.bbox
        cx, cy = rx + rw / 2, ry + rh / 2
        if not (x + 0.08 * w <= cx <= x + 0.92 * w and
                y + 0.25 * h <= cy <= y + 0.70 * h):
            continue
        if region.frac > 0.006 or region.area < 3:
            continue
        aspect = rw / max(1, rh)
        if not 0.25 <= aspect <= 4.0:
            continue
        if not 25 <= region.mean_gray <= 225:
            continue
        candidates.append(region)
    if len(candidates) < 2:
        return []
    best = None
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a, b = candidates[i], candidates[j]
            acx = a.bbox[0] + a.bbox[2] / 2
            bcx = b.bbox[0] + b.bbox[2] / 2
            acy = a.bbox[1] + a.bbox[3] / 2
            bcy = b.bbox[1] + b.bbox[3] / 2
            horizontal = abs(acx - bcx)
            vertical = abs(acy - bcy)
            if horizontal < w * 0.12 or vertical > h * 0.12:
                continue
            size_balance = abs(a.area - b.area) / max(a.area, b.area)
            score = vertical / max(1, h) + size_balance * 0.25
            if best is None or score < best[0]:
                best = (score, a.label_id, b.label_id)
    return [int(best[1]), int(best[2])] if best is not None else []
