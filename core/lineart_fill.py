"""Lineart-bounded region tools - 完整优化版（解决头发等复杂区域）"""

from __future__ import annotations

import cv2
import numpy as np

from core.paint_bucket import hex_to_lab


# ─────────────────────────────────────────────────────────────────────────
# Shared
# ─────────────────────────────────────────────────────────────────────────

def _fillable_mask(gray_or_bgr: np.ndarray, line_low: int = 75, gap_close: int = 10) -> np.ndarray:
    """强力封口版本"""
    gray = cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY) if gray_or_bgr.ndim == 3 else gray_or_bgr
    ink = (gray < line_low).astype(np.uint8) * 255
    if gap_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap_close, gap_close))
        ink = cv2.dilate(ink, k, iterations=2)
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, k)
    return np.where(ink > 0, 0, 255).astype(np.uint8)


def label_regions(gray_or_bgr: np.ndarray, *, line_low: int = 75,
                  gap_close: int = 10) -> np.ndarray:
    fillable = _fillable_mask(gray_or_bgr, line_low, gap_close)
    _, labels = cv2.connectedComponents(fillable, connectivity=4)
    return labels.astype(np.int32)


def clip_stamp_to_region(labels: np.ndarray, px: int, py: int, radius_px: int):
    """Brush hint clipping - 原功能保留"""
    h, w = labels.shape
    if not (0 <= px < w and 0 <= py < h):
        return None
    label = labels[py, px]
    if label == 0:
        return None

    x1 = max(0, px - radius_px)
    y1 = max(0, py - radius_px)
    x2 = min(w, px + radius_px + 1)
    y2 = min(h, py + radius_px + 1)
    local_labels = labels[y1:y2, x1:x2]

    circle = np.zeros_like(local_labels, dtype=np.uint8)
    cv2.circle(circle, (px - x1, py - y1), radius_px, 255, -1)

    stamp = np.where((local_labels == label) & (circle > 0), 255, 0).astype(np.uint8)
    return stamp, x1, y1


def clip_hint_radii(original_bgr: np.ndarray, hint_points: list, *,
                    line_low: int = 75, gap_close: int = 10) -> list:
    """Fallback"""
    if not hint_points:
        return hint_points
    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY) if original_bgr.ndim == 3 else original_bgr
    h, w = gray.shape[:2]
    fillable = _fillable_mask(gray, line_low, gap_close)
    dist = cv2.distanceTransform(fillable, cv2.DIST_L2, 3)

    clipped = []
    for point in hint_points:
        if len(point) == 4:
            xn, yn, rgb, radius_norm = point
        else:
            xn, yn, rgb = point
            radius_norm = 0.006
        px = min(w - 1, max(0, int(xn * w)))
        py = min(h - 1, max(0, int(yn * h)))
        safe_px = float(dist[py, px])
        radius_px = radius_norm * w
        new_radius_px = max(2.0, min(radius_px, safe_px))
        clipped.append((xn, yn, rgb, new_radius_px / w))
    return clipped


# ─────────────────────────────────────────────────────────────────────────
# 区域上色 - 核心（重点优化）
# ─────────────────────────────────────────────────────────────────────────

def lineart_mask_at_point(gray_or_bgr: np.ndarray, x: int, y: int, *,
                          line_low: int = 75, gap_close: int = 10) -> np.ndarray:
    """增强版：更好处理头发、复杂区域"""
    h, w = gray_or_bgr.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return np.zeros((h, w), dtype=np.uint8)

    fillable = _fillable_mask(gray_or_bgr, line_low, gap_close)

    if fillable[y, x] == 0:
        ys, xs = np.where(fillable > 0)
        if len(xs) > 0:
            d2 = (xs - x) ** 2 + (ys - y) ** 2
            i = int(np.argmin(d2))
            x, y = int(xs[i]), int(ys[i])

    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY
    cv2.floodFill(fillable, mask, (int(x), int(y)), 0, loDiff=0, upDiff=0, flags=flags)

    # 如果区域太小，尝试降低阈值再填充一次
    filled_area = np.sum(mask[1:-1, 1:-1] > 0)
    if filled_area < 300:
        fillable2 = _fillable_mask(gray_or_bgr, line_low=95, gap_close=gap_close//2)
        cv2.floodFill(fillable2, mask, (int(x), int(y)), 0, loDiff=0, upDiff=0, flags=flags)

    return mask[1:-1, 1:-1]


def lineart_region_recolor(original_bw: np.ndarray, result_bgr: np.ndarray,
                           x: int, y: int, hex_color: str, *,
                           line_low: int = 75, gap_close: int = 10,
                           mode: str = "shift", feather: int = 5):
    """完整区域上色"""
    mask = lineart_mask_at_point(original_bw, x, y, line_low=line_low, gap_close=gap_close)
    if not mask.any():
        return result_bgr, mask

    h, w = mask.shape
    bx, by, bw_, bh_ = cv2.boundingRect(mask)
    margin = feather * 2 + 2
    x1 = max(0, bx - margin)
    y1 = max(0, by - margin)
    x2 = min(w, bx + bw_ + margin)
    y2 = min(h, by + bh_ + margin)

    roi_mask = mask[y1:y2, x1:x2]
    soft = cv2.GaussianBlur(roi_mask.astype(np.float32),
                            (feather * 2 + 1, feather * 2 + 1), 0) / 255.0
    np.clip(soft, 0.0, 1.0, out=soft)

    L_target, a_target, b_target = hex_to_lab(hex_color)
    roi = result_bgr[y1:y2, x1:x2]
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)

    if mode == "shift":
        seed_lab = cv2.cvtColor(result_bgr[max(0, y):y+1, max(0, x):x+1], 
                               cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
        da = a_target - seed_lab[1]
        db = b_target - seed_lab[2]
        lab[:, :, 1] = np.clip(lab[:, :, 1] + da * soft, 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + db * soft, 0, 255)
    else:
        lab[:, :, 1] = lab[:, :, 1] * (1.0 - soft) + a_target * soft
        lab[:, :, 2] = lab[:, :, 2] * (1.0 - soft) + b_target * soft
        if mode == "flat":
            lab[:, :, 0] = lab[:, :, 0] * (1.0 - soft) + L_target * soft
        np.clip(lab, 0, 255, out=lab)

    result_bgr[y1:y2, x1:x2] = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return result_bgr, mask