"""Strictly local post-colour brush editing.

The model-hint brush used in older Colortina versions attached every dab to the
whole connected line-art region. On a face, one small blush dab could therefore
change the entire skin region after regeneration. This module performs a local
LAB chroma edit instead:

- the brush footprint is always bounded by its visible radius;
- pixels are clipped to the connected region under the cursor;
- the visible brush edits a *local patch* rather than only the exact seed dot;
- ink and high-confidence line pixels are protected;
- source/result lightness and texture are preserved;
- no model re-run is required.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.region_map import RegionMap, build_region_map
from core.perceptual_recolor import normalize_recolor_mode, recolor_with_mode
from core.pupil_boundary import constrain_pupil_alpha_to_lineart



def _smooth_radial(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    distance = np.sqrt(xx * xx + yy * yy).astype(np.float32)
    # Full strength through the central half, then a smooth cosine-like falloff.
    t = np.clip((radius - distance) / max(1.0, radius * 0.48), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)




def _connected_seed_component(binary: np.ndarray, seed_x: int, seed_y: int) -> np.ndarray:
    h, w = binary.shape[:2]
    if not (0 <= seed_x < w and 0 <= seed_y < h):
        return np.zeros_like(binary, dtype=np.uint8)
    if binary[seed_y, seed_x] == 0:
        found = False
        for radius in (1, 2, 3):
            x1, x2 = max(0, seed_x - radius), min(w, seed_x + radius + 1)
            y1, y2 = max(0, seed_y - radius), min(h, seed_y + radius + 1)
            ys, xs = np.nonzero(binary[y1:y2, x1:x2] > 0)
            if ys.size:
                d2 = (xs - (seed_x - x1)) ** 2 + (ys - (seed_y - y1)) ** 2
                idx = int(np.argmin(d2))
                seed_x = int(xs[idx] + x1)
                seed_y = int(ys[idx] + y1)
                found = True
                break
        if not found:
            return np.zeros_like(binary, dtype=np.uint8)
    num, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=4)
    if num <= 1:
        return np.zeros_like(binary, dtype=np.uint8)
    label = int(labels[seed_y, seed_x])
    if label <= 0:
        return np.zeros_like(binary, dtype=np.uint8)
    return np.where(labels == label, 255, 0).astype(np.uint8)


def _local_patch_mask(
    source_bw_bgr: np.ndarray,
    result_bgr: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    x: int, y: int,
    radial: np.ndarray,
    region_slice: np.ndarray | None,
    line_conf_slice: np.ndarray | None,
    safe_slice: np.ndarray | None,
) -> np.ndarray:
    """Grow a local patch inside the visible brush circle.

    The patch is still hard-bounded by the brush radius, but instead of only
    tinting the centre pixels it captures the connected local area around the
    seed when nearby pixels have similar tone / chroma.
    """
    gray = (source_bw_bgr if source_bw_bgr.ndim == 2 else
            cv2.cvtColor(source_bw_bgr, cv2.COLOR_BGR2GRAY)).astype(np.float32)
    gray_local = gray[y1:y2, x1:x2]
    crop = result_bgr[y1:y2, x1:x2]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    cx = int(np.clip(x - x1, 0, x2 - x1 - 1))
    cy = int(np.clip(y - y1, 0, y2 - y1 - 1))

    circle = (radial > 0.04).astype(np.uint8)
    if not np.any(circle):
        return radial.astype(np.float32)

    seed_gray = float(np.median(gray_local[max(0, cy - 1):min(gray_local.shape[0], cy + 2),
                                         max(0, cx - 1):min(gray_local.shape[1], cx + 2)]))
    seed_ab = np.median(lab[max(0, cy - 1):min(lab.shape[0], cy + 2),
                             max(0, cx - 1):min(lab.shape[1], cx + 2), 1:3], axis=(0, 1))

    candidate = circle > 0
    if region_slice is not None and np.any(region_slice):
        candidate &= region_slice.astype(bool)
    if safe_slice is not None:
        candidate &= safe_slice.astype(np.float32) > 0.16
    candidate &= gray_local >= 18.0
    if line_conf_slice is not None:
        candidate &= line_conf_slice.astype(np.float32) < 0.34

    gray_tol = max(9.0, min(30.0, 11.0 + radial.shape[0] * 0.07))
    candidate &= np.abs(gray_local - seed_gray) <= gray_tol

    chroma = np.linalg.norm(lab[..., 1:3] - seed_ab[None, None, :], axis=2)
    # Skin / flat areas should spread locally; highly dissimilar colours should not.
    candidate &= ((chroma <= 22.0) | (gray_local >= 234.0))

    component = _connected_seed_component(candidate.astype(np.uint8), cx, cy)
    if not np.any(component):
        fallback = candidate.astype(np.float32) * np.maximum(radial.astype(np.float32), 0.32)
        return fallback.astype(np.float32)

    distance = cv2.distanceTransform(component, cv2.DIST_L2, 3)
    patch = np.clip(distance / 1.6, 0.0, 1.0).astype(np.float32)
    # Keep the edge within the visible brush soft while letting the interior act
    # as a filled local patch instead of a single colour dot.
    patch = np.maximum(patch * 0.92, radial * 0.28)
    patch *= circle.astype(np.float32)
    return patch.astype(np.float32)



def restore_local_brush_from_reference(
    source_bw_bgr: np.ndarray,
    current_bgr: np.ndarray,
    restore_bgr: np.ndarray,
    x: int,
    y: int,
    radius_px: int,
    *,
    opacity: float = 0.88,
    region_map: RegionMap | None = None,
    gap_close: int = 4,
    snap_to_lineart: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Restore the brush footprint from a reference image (typically the
    stored AI result).

    Reuses the exact same strict-locality mask as ``apply_local_brush_recolor``
    so restoring a mistake cannot spill beyond the visible brush area.
    Returns ``(edited_bgr, alpha_mask)``.
    """
    if current_bgr is None or restore_bgr is None:
        return current_bgr, np.zeros(source_bw_bgr.shape[:2], np.float32)
    sentinel = (255, 64, 64)
    _tmp, alpha = apply_local_brush_recolor(
        source_bw_bgr, current_bgr, x, y, radius_px, sentinel,
        opacity=opacity, region_map=region_map, gap_close=gap_close,
        snap_to_lineart=snap_to_lineart)
    if alpha is None or float(alpha.max()) <= 1e-5:
        return current_bgr.copy(), np.zeros(current_bgr.shape[:2], np.float32)
    edited = current_bgr.copy()
    if restore_bgr.shape[:2] != current_bgr.shape[:2]:
        restore_bgr = cv2.resize(restore_bgr, (current_bgr.shape[1], current_bgr.shape[0]),
                                 interpolation=cv2.INTER_AREA)
    a3 = np.clip(alpha[..., None], 0.0, 1.0).astype(np.float32)
    blended = current_bgr.astype(np.float32) * (1.0 - a3) + restore_bgr.astype(np.float32) * a3
    edited[alpha > 1e-5] = np.clip(blended, 0, 255).astype(np.uint8)[alpha > 1e-5]
    return edited, alpha

def apply_local_brush_recolor(
    source_bw_bgr: np.ndarray,
    result_bgr: np.ndarray,
    x: int,
    y: int,
    radius_px: int,
    rgb: tuple[int, int, int],
    *,
    opacity: float = 0.95,
    region_map: RegionMap | None = None,
    gap_close: int = 4,
    mode: str = "shift",
    snap_to_lineart: bool = True,
    pupil_blend: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Recolour only the visible brush footprint.

    Returns ``(edited_bgr, alpha_mask)``.  ``alpha_mask`` is full-resolution and
    is exactly zero outside the changed local area, making the locality contract
    straightforward to test.
    """
    if result_bgr is None or result_bgr.size == 0:
        return result_bgr, np.zeros(source_bw_bgr.shape[:2], np.float32)

    h, w = result_bgr.shape[:2]
    if source_bw_bgr.shape[:2] != (h, w):
        source_bw_bgr = cv2.resize(source_bw_bgr, (w, h), interpolation=cv2.INTER_AREA)
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    radius = max(1, int(radius_px))
    opacity = float(np.clip(opacity, 0.0, 1.0))
    if opacity <= 0.0:
        return result_bgr.copy(), np.zeros((h, w), np.float32)

    region_id = 0
    if snap_to_lineart:
        if region_map is None or region_map.shape != (h, w):
            region_map = build_region_map(source_bw_bgr, gap_close=max(0, int(gap_close)))
        region_id = region_map.region_at(x, y, search_radius=max(3, min(radius, 18)))
    # Crop exactly to the visible brush footprint.  Region clipping and line
    # confidence are evaluated inside this hard locality bound.
    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = min(w, x + radius + 1)
    y2 = min(h, y + radius + 1)
    if x1 >= x2 or y1 >= y2:
        return result_bgr.copy(), np.zeros((h, w), np.float32)

    radial_full = _smooth_radial(radius)
    rx1 = x1 - (x - radius)
    ry1 = y1 - (y - radius)
    rx2 = rx1 + (x2 - x1)
    ry2 = ry1 + (y2 - y1)
    radial = radial_full[ry1:ry2, rx1:rx2].astype(np.float32)

    region_slice = None
    safe_slice = None
    if region_id > 0:
        region_slice = (region_map.labels[y1:y2, x1:x2] == int(region_id)).astype(np.uint8)
        if np.any(region_slice):
            # Fade before the connected-region edge. Even when the region is
            # accidentally huge, the radial footprint still provides a hard
            # locality bound.
            distance = cv2.distanceTransform(region_slice, cv2.DIST_L2, 3)
            region_weight = np.clip(distance / 1.35, 0.0, 1.0)
            radial *= region_weight.astype(np.float32)
            safe_full = region_map.safe_interior(region_id, margin_px=max(2.0, radius * 0.22))
            safe_slice = safe_full[y1:y2, x1:x2].astype(np.float32)
            radial *= np.clip(safe_slice / 0.75, 0.0, 1.0)
        else:
            radial[:] = 0.0

    gray = (source_bw_bgr if source_bw_bgr.ndim == 2 else
            cv2.cvtColor(source_bw_bgr, cv2.COLOR_BGR2GRAY))
    gray_local = gray[y1:y2, x1:x2].astype(np.float32)
    # Preserve solid ink. White/near-white face interiors remain editable.
    ink_gate = np.clip((gray_local - 16.0) / 28.0, 0.0, 1.0)
    radial *= ink_gate

    confidence = None
    if region_map is not None and region_map.line_confidence is not None:
        confidence = region_map.line_confidence[y1:y2, x1:x2].astype(np.float32)
        radial *= np.clip(1.0 - confidence * 1.00, 0.0, 1.0)

    patch = _local_patch_mask(
        source_bw_bgr, result_bgr, x1, y1, x2, y2, x, y,
        radial, region_slice, confidence, safe_slice)
    alpha_local = np.clip(patch * opacity, 0.0, 1.0)
    # One-pass response: strengthen every already-valid brush pixel without
    # expanding the footprint.  This keeps the same hard locality/line guards
    # but avoids requiring several strokes to approach the selected colour.
    active_alpha = alpha_local > 1e-5
    alpha_local[active_alpha] = np.power(alpha_local[active_alpha], 0.58)
    # Resilient fallback: if the strict local patch becomes too weak to be
    # visible, fall back to the conservative safe radial footprint instead of
    # behaving like a dead brush.
    if float(alpha_local.max()) < 0.06 or int(np.count_nonzero(alpha_local > 0.05)) < 12:
        fallback = radial.copy()
        if safe_slice is not None:
            fallback *= np.clip(safe_slice / 0.72, 0.0, 1.0)
        if confidence is not None:
            fallback *= np.clip(1.0 - confidence * 0.95, 0.0, 1.0)
        alpha_local = np.maximum(alpha_local, np.clip(fallback * opacity, 0.0, 1.0))
    if float(alpha_local.max()) <= 1e-5:
        return result_bgr.copy(), np.zeros((h, w), np.float32)

    # Uniform-hue and pure-colour modes must have a genuinely authoritative
    # brush core.  At 100% opacity the central valid footprint is exact, while
    # the visible outer edge remains softly blended and line-bounded.
    canonical_mode = normalize_recolor_mode(mode)
    if opacity >= 0.995 and canonical_mode in {"shading", "flat"}:
        strong_core = (radial >= 0.72) & (alpha_local >= 0.48)
        alpha_local[strong_core] = 1.0

    if pupil_blend and canonical_mode != "flat":
        alpha_local = constrain_pupil_alpha_to_lineart(
            source_bw_bgr[y1:y2, x1:x2], alpha_local,
            seed=(x - x1, y - y1))
        if float(alpha_local.max()) <= 1e-5:
            return result_bgr.copy(), np.zeros((h, w), np.float32)

    edited = result_bgr.copy()
    crop = edited[y1:y2, x1:x2]
    crop_out = recolor_with_mode(
        crop, rgb, alpha_local,
        active=alpha_local > 0.02, mode=mode,
        pupil_blend=pupil_blend)
    crop[alpha_local > 1e-5] = crop_out[alpha_local > 1e-5]

    alpha = np.zeros((h, w), np.float32)
    alpha[y1:y2, x1:x2] = alpha_local
    return edited, alpha
