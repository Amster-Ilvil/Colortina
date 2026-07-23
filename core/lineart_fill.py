"""Line-art bounded region tools shared by brush hints and bucket edits."""

from __future__ import annotations

import cv2
import numpy as np

from core.paint_bucket import hex_to_lab
from core.region_map import RegionMap, build_region_map


def _nearest_valid_seed(binary: np.ndarray, x: int, y: int) -> tuple[int, int] | None:
    h, w = binary.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return None
    if binary[y, x]:
        return int(x), int(y)
    for radius in (1, 2, 3, 5, 8, 12):
        x1, x2 = max(0, x - radius), min(w, x + radius + 1)
        y1, y2 = max(0, y - radius), min(h, y + radius + 1)
        ys, xs = np.nonzero(binary[y1:y2, x1:x2] > 0)
        if ys.size:
            d2 = (xs - (x - x1)) ** 2 + (ys - (y - y1)) ** 2
            idx = int(np.argmin(d2))
            return int(xs[idx] + x1), int(ys[idx] + y1)
    return None


def _connected_component(binary: np.ndarray, x: int, y: int) -> np.ndarray:
    seed = _nearest_valid_seed(binary, x, y)
    if seed is None:
        return np.zeros_like(binary, dtype=np.uint8)
    sx, sy = seed
    num, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=4)
    if num <= 1:
        return np.zeros_like(binary, dtype=np.uint8)
    label = int(labels[sy, sx])
    if label <= 0:
        return np.zeros_like(binary, dtype=np.uint8)
    return np.where(labels == label, 255, 0).astype(np.uint8)


def _refine_region_mask(original_bw: np.ndarray, result_bgr: np.ndarray,
                        mask: np.ndarray, x: int, y: int,
                        region_map: RegionMap | None = None) -> np.ndarray:
    """Split suspiciously leaky regions using local tone/chroma continuity.

    If a line gap leaves two semantic areas connected, the raw connected
    component can become far too large.  This refinement keeps only the seed's
    locally consistent sub-component, guided by source tone, current colour and
    a boundary safety band.
    """
    if mask is None or not np.any(mask):
        return mask
    binary = mask > 0
    area = int(np.count_nonzero(binary))
    if area < 48:
        return mask

    h, w = binary.shape[:2]
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    seed = _nearest_valid_seed(binary.astype(np.uint8), x, y)
    if seed is None:
        return mask
    sx, sy = seed

    gray = original_bw if original_bw.ndim == 2 else cv2.cvtColor(original_bw, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)
    lab = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    region_id = region_map.region_at(sx, sy) if region_map is not None else 0
    if region_map is not None and region_id > 0:
        safety = region_map.safe_interior(region_id, margin_px=2.4)
    else:
        safety = binary.astype(np.float32)

    x1, x2 = max(0, sx - 1), min(w, sx + 2)
    y1, y2 = max(0, sy - 1), min(h, sy + 2)
    seed_window = binary[y1:y2, x1:x2]
    if np.any(seed_window):
        sg = float(np.median(gray[y1:y2, x1:x2][seed_window]))
        sab = np.median(lab[y1:y2, x1:x2, 1:3][seed_window], axis=0).astype(np.float32)
    else:
        sg = float(gray[sy, sx])
        sab = lab[sy, sx, 1:3].astype(np.float32)

    gray_tol = float(np.clip(16.0 + np.sqrt(float(area)) * 0.10, 16.0, 42.0))
    chroma_tol = float(np.clip(18.0 + np.sqrt(float(area)) * 0.08, 18.0, 34.0))
    candidate = binary.copy()
    candidate &= safety > 0.16
    candidate &= gray > 18.0
    candidate &= np.abs(gray - sg) <= gray_tol
    delta_ab = np.linalg.norm(lab[..., 1:3] - sab[None, None, :], axis=2)
    candidate &= ((delta_ab <= chroma_tol) | (gray >= 235.0))

    component = _connected_component(candidate.astype(np.uint8), sx, sy)
    if not np.any(component):
        return mask

    comp_area = int(np.count_nonzero(component))
    # Adopt the refined component when it materially reduces an over-large or
    # leaky region, but never collapse a normal fill down to a tiny fragment.
    min_keep = min(max(36, area // 200), max(96, area // 20))
    if comp_area >= min_keep and comp_area <= int(area * 0.97):
        return component
    if comp_area >= int(area * 0.97):
        return component
    return mask


def _fillable_mask(gray_or_bgr: np.ndarray, line_low: int = 75,
                   gap_close: int = 4) -> np.ndarray:
    return build_region_map(gray_or_bgr, line_low=line_low,
                            gap_close=gap_close).fillable


def label_regions(gray_or_bgr: np.ndarray, *, line_low: int = 75,
                  gap_close: int = 4) -> np.ndarray:
    return build_region_map(gray_or_bgr, line_low=line_low,
                            gap_close=gap_close).labels


def clip_stamp_to_region(labels: np.ndarray, px: int, py: int, radius_px: int):
    """Clip a circular hint stamp to the connected region under its centre."""
    h, w = labels.shape
    if not (0 <= px < w and 0 <= py < h):
        return None
    label = int(labels[py, px])
    if label == 0:
        return None

    radius_px = max(1, int(radius_px))
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
                    line_low: int = 75, gap_close: int = 4) -> list:
    """Shrink legacy circular hints to the distance from the nearest line."""
    if not hint_points:
        return hint_points
    region_map = build_region_map(original_bgr, line_low=line_low,
                                  gap_close=gap_close)
    h, w = region_map.shape
    dist = cv2.distanceTransform(region_map.fillable, cv2.DIST_L2, 3)

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


def lineart_mask_at_point(gray_or_bgr: np.ndarray, x: int, y: int, *,
                          line_low: int = 75, gap_close: int = 4,
                          region_map: RegionMap | None = None) -> np.ndarray:
    """Return the exact connected region at a click, snapping off ink locally."""
    supplied_map = region_map is not None
    region_map = region_map or build_region_map(
        gray_or_bgr, line_low=line_low, gap_close=gap_close)
    search_radius = max(10, int(gap_close) * 2 + 8)
    region_id = region_map.region_at(int(x), int(y), search_radius=search_radius)
    mask = region_map.mask(region_id)

    # If the click lands on anti-aliased border pixels or embedded text, probe a
    # small neighbourhood and prefer the smallest reasonable nearby component.
    area = int(np.count_nonzero(mask))
    if area == 0 or area > int(mask.size * 0.20):
        h, w = region_map.shape
        x1, x2 = max(0, int(x) - 4), min(w, int(x) + 5)
        y1, y2 = max(0, int(y) - 4), min(h, int(y) + 5)
        local = region_map.labels[y1:y2, x1:x2]
        candidates = [int(v) for v in np.unique(local) if int(v) > 0]
        if candidates:
            areas = np.bincount(region_map.labels.ravel())
            page_cap = max(96, int(mask.size * 0.16))
            viable = [rid for rid in candidates if rid < len(areas) and 0 < int(areas[rid]) <= page_cap]
            if viable:
                region_id = min(viable, key=lambda rid: (int(areas[rid]), rid))
                mask = region_map.mask(region_id)
                area = int(np.count_nonzero(mask))

    # For extremely tiny regions retry with a slightly more permissive line
    # threshold, but never combine the two masks (which previously leaked).
    if 0 < area < 48 and not supplied_map:
        retry = build_region_map(gray_or_bgr, line_low=min(110, line_low + 20),
                                 gap_close=gap_close)
        retry_id = retry.region_at(int(x), int(y), search_radius=search_radius)
        retry_mask = retry.mask(retry_id)
        base_area = area
        retry_area = int(np.count_nonzero(retry_mask))
        page_area = int(mask.size)
        safe_cap = min(int(page_area * 0.10), max(512, base_area * 16))
        if base_area < retry_area <= safe_cap:
            mask = retry_mask
    return mask


def _inside_feather(mask: np.ndarray, feather: int) -> np.ndarray:
    """Feather inward only; output is guaranteed zero outside ``mask``."""
    binary = (mask > 0).astype(np.uint8)
    if feather <= 0:
        return binary.astype(np.float32)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    edge_soft = np.clip(distance / max(1.0, float(feather)), 0.0, 1.0)
    # The feather remains strictly inside the region, but every selected pixel
    # gets at least 78% of the requested colour in one operation. Thin regions
    # therefore no longer need repeated clicks to accumulate visible colour.
    soft = binary.astype(np.float32) * (0.78 + 0.22 * edge_soft)
    return np.clip(soft, 0.0, 1.0).astype(np.float32)


def _representative_lab(result_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Robust mid-tone colour for hue shifting, excluding ink/highlights."""
    lab = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    valid = mask > 0
    if not np.any(valid):
        return np.array([128.0, 128.0, 128.0], dtype=np.float32)
    L = lab[..., 0]
    preferred = valid & (L >= 55) & (L <= 220)
    pixels = lab[preferred] if np.any(preferred) else lab[valid]
    return np.median(pixels, axis=0).astype(np.float32)


def lineart_region_recolor(original_bw: np.ndarray, result_bgr: np.ndarray,
                           x: int, y: int, hex_color: str, *,
                           line_low: int = 75, gap_close: int = 4,
                           mode: str = "shift", feather: int = 5,
                           region_map: RegionMap | None = None):
    """Recolour one enclosed area while preserving existing shading and ink."""
    region_map = region_map or build_region_map(
        original_bw, line_low=line_low, gap_close=gap_close)
    mask = lineart_mask_at_point(original_bw, x, y, line_low=line_low,
                                 gap_close=gap_close, region_map=region_map)
    if not mask.any():
        return result_bgr, mask
    mask = _refine_region_mask(original_bw, result_bgr, mask, x, y, region_map=region_map)

    h, w = mask.shape
    bx, by, bw_, bh_ = cv2.boundingRect(mask)
    margin = max(2, feather + 2)
    x1 = max(0, bx - margin)
    y1 = max(0, by - margin)
    x2 = min(w, bx + bw_ + margin)
    y2 = min(h, by + bh_ + margin)

    roi_mask = mask[y1:y2, x1:x2]
    soft = _inside_feather(roi_mask, feather)

    L_target, a_target, b_target = hex_to_lab(hex_color)
    # One-pass chroma compensation: preserving existing luminance otherwise
    # makes a selected colour appear too weak, especially on pale manga tones.
    a_target = float(np.clip(128.0 + (a_target - 128.0) * 1.10, 0.0, 255.0))
    b_target = float(np.clip(128.0 + (b_target - 128.0) * 1.10, 0.0, 255.0))
    roi = result_bgr[y1:y2, x1:x2]
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)

    if mode == "shift":
        seed_lab = _representative_lab(result_bgr, mask)
        residual = lab[:, :, 1:3] - seed_lab[None, None, 1:3]
        desired_ab = np.dstack([
            np.full_like(lab[:, :, 0], float(a_target)),
            np.full_like(lab[:, :, 0], float(b_target)),
        ]).astype(np.float32) + residual * 0.06
        lab[:, :, 1:3] = np.clip(lab[:, :, 1:3] * (1.0 - soft[:, :, None]) + desired_ab * soft[:, :, None], 0, 255)
        l_soft = np.clip(soft * 0.18, 0.0, 1.0)
        lab[:, :, 0] = lab[:, :, 0] * (1.0 - l_soft) + float(L_target) * l_soft
    else:
        lab[:, :, 1] = lab[:, :, 1] * (1.0 - soft) + float(a_target) * soft
        lab[:, :, 2] = lab[:, :, 2] * (1.0 - soft) + float(b_target) * soft
        if mode == "flat":
            lab[:, :, 0] = lab[:, :, 0] * (1.0 - soft) + float(L_target) * soft
        elif mode == "shading":
            l_soft = np.clip(soft * 0.10, 0.0, 1.0)
            lab[:, :, 0] = lab[:, :, 0] * (1.0 - l_soft) + float(L_target) * l_soft
        np.clip(lab, 0, 255, out=lab)

    recolored = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    # Defensive guarantee: even rounding/ROI code may never touch pixels
    # outside the connected component.
    binary = (roi_mask > 0)[:, :, None]
    roi[:] = np.where(binary, recolored, roi)
    result_bgr[y1:y2, x1:x2] = roi
    return result_bgr, mask
