"""Atomic manual-edit operations shared by brush and region fill.

A manual edit must update both the visible result and the stable pre-filter base.
Otherwise a later filter refresh can silently restore the old base and make the
edit appear to have failed.  This module keeps both layers synchronized and
provides a conservative fallback when line/region guards reject every pixel.
"""
from __future__ import annotations

import cv2
import numpy as np

from core.local_brush import apply_local_brush_recolor
from core.lineart_fill import lineart_region_recolor, refined_lineart_mask_at_point
from core.perceptual_recolor import normalize_recolor_mode, recolor_with_mode
from core.pupil_boundary import (
    constrain_pupil_alpha_to_lineart, constrain_pupil_mask_to_lineart,
)
from core.structural_line_detector import closed_regions_from_selection


def _image_changed(before: np.ndarray, after: np.ndarray, *, threshold: float = 0.05) -> bool:
    if before is None or after is None or before.shape != after.shape:
        return before is not after
    diff = np.abs(after.astype(np.int16) - before.astype(np.int16))
    return bool(float(diff.mean()) > threshold or int(diff.max()) >= 2)


def _fallback_local_recolor(source_bw_bgr: np.ndarray,
                            result_bgr: np.ndarray,
                            x: int, y: int, radius_px: int,
                            rgb: tuple[int, int, int],
                            opacity: float,
                            mode: str = "shift",
                            pupil_blend: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Conservative visible-radius fallback for a dead brush dab.

    It never crosses outside the displayed brush circle and protects solid ink.
    Only chroma is strongly redirected; luminance is retained.
    """
    h, w = result_bgr.shape[:2]
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    radius = max(1, int(radius_px))
    x1, y1 = max(0, x-radius), max(0, y-radius)
    x2, y2 = min(w, x+radius+1), min(h, y+radius+1)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    dist = np.sqrt((xx-x)**2 + (yy-y)**2).astype(np.float32)
    radial = np.clip(1.0 - dist / max(1.0, float(radius)), 0.0, 1.0)
    radial = radial * radial * (3.0 - 2.0 * radial)

    source = source_bw_bgr
    if source.shape[:2] != (h, w):
        source = cv2.resize(source, (w, h), interpolation=cv2.INTER_AREA)
    gray = source if source.ndim == 2 else cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    ink_gate = np.clip((gray[y1:y2, x1:x2].astype(np.float32) - 18.0) / 34.0, 0.0, 1.0)
    opacity = float(np.clip(opacity, 0.0, 1.0))
    alpha = np.clip(radial * ink_gate * opacity * 0.92, 0.0, 1.0)
    canonical_mode = normalize_recolor_mode(mode)
    if opacity >= 0.995 and canonical_mode in {"shading", "flat"}:
        alpha[(radial >= 0.72) & (ink_gate >= 0.88)] = 1.0
    if pupil_blend and canonical_mode != "flat":
        alpha = constrain_pupil_alpha_to_lineart(
            source[y1:y2, x1:x2], alpha, seed=(x - x1, y - y1))
    if float(alpha.max()) <= 1e-5:
        return result_bgr.copy(), np.zeros((h, w), np.float32)

    edited = result_bgr.copy()
    roi = edited[y1:y2, x1:x2]
    roi_out = recolor_with_mode(
        roi, rgb, alpha, active=alpha > 1e-5, mode=mode,
        pupil_blend=pupil_blend)
    roi[alpha > 1e-5] = roi_out[alpha > 1e-5]
    full = np.zeros((h, w), np.float32)
    full[y1:y2, x1:x2] = alpha
    return edited, full


def apply_brush_edit(source_bw_bgr: np.ndarray,
                     result_bgr: np.ndarray,
                     filter_base_bgr: np.ndarray | None,
                     x: int, y: int, radius_px: int,
                     rgb: tuple[int, int, int], *,
                     opacity: float = 1.0,
                     region_map=None,
                     gap_close: int = 4,
                     mode: str = "shift",
                     snap_to_lineart: bool = True,
                     pupil_blend: bool = False):
    """Apply one brush dab atomically to visible and filter-base layers."""
    before = result_bgr.copy()
    edited, mask = apply_local_brush_recolor(
        source_bw_bgr, result_bgr, x, y, radius_px, rgb,
        opacity=opacity, region_map=region_map, gap_close=gap_close, mode=mode,
        snap_to_lineart=snap_to_lineart, pupil_blend=pupil_blend)
    if not _image_changed(before, edited) or float(mask.max()) <= 1e-5:
        edited, mask = _fallback_local_recolor(
            source_bw_bgr, result_bgr, x, y, radius_px, rgb, opacity, mode,
            pupil_blend)

    base = filter_base_bgr
    if base is None or base.shape != result_bgr.shape:
        base = before.copy()
    else:
        base = base.copy()
    base_edited, base_mask = apply_local_brush_recolor(
        source_bw_bgr, base, x, y, radius_px, rgb,
        opacity=opacity, region_map=region_map, gap_close=gap_close, mode=mode,
        snap_to_lineart=snap_to_lineart, pupil_blend=pupil_blend)
    if not _image_changed(base, base_edited) or float(base_mask.max()) <= 1e-5:
        base_edited, _ = _fallback_local_recolor(
            source_bw_bgr, base, x, y, radius_px, rgb, opacity, mode,
            pupil_blend)
    return edited, base_edited, mask, _image_changed(before, edited)


def _connected_component(mask: np.ndarray, x: int, y: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    h, w = binary.shape[:2]
    if not (0 <= x < w and 0 <= y < h) or binary[y, x] == 0:
        return np.zeros_like(binary, dtype=np.uint8)
    count, labels = cv2.connectedComponents(binary, connectivity=4)
    if count <= 1:
        return np.zeros_like(binary, dtype=np.uint8)
    label = int(labels[y, x])
    return np.where(labels == label, 255, 0).astype(np.uint8)


def _fallback_region_mask(source_bw_bgr: np.ndarray,
                          result_bgr: np.ndarray,
                          x: int, y: int) -> np.ndarray:
    """Build a safe region when the strict line-art map finds nothing.

    The fallback combines source-line protection with a connected color/tone
    similarity mask.  If that component is implausibly huge, it is clipped to a
    local circular patch instead of silently doing nothing or flooding the page.
    """
    h, w = result_bgr.shape[:2]
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    source = source_bw_bgr
    if source.shape[:2] != (h, w):
        source = cv2.resize(source, (w, h), interpolation=cv2.INTER_AREA)
    gray = source if source.ndim == 2 else cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    safe = gray >= 28

    smooth = cv2.bilateralFilter(result_bgr, 5, 24, 5)
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB).astype(np.float32)
    seed = lab[y, x]
    d_l = np.abs(lab[..., 0] - seed[0])
    d_ab = np.linalg.norm(lab[..., 1:3] - seed[None, None, 1:3], axis=2)
    candidate = safe & (d_l <= 30.0) & (d_ab <= 34.0)
    component = _connected_component(candidate.astype(np.uint8), x, y)
    area = int(np.count_nonzero(component))
    page_area = h * w

    if area < 12 or area > int(page_area * 0.42):
        radius = max(12, min(72, int(round(min(h, w) * 0.075))))
        yy, xx = np.ogrid[:h, :w]
        circle = ((xx - x) ** 2 + (yy - y) ** 2) <= radius * radius
        local = safe & circle
        component = _connected_component(local.astype(np.uint8), x, y)
        if not np.any(component):
            component = np.where(local, 255, 0).astype(np.uint8)
    return component.astype(np.uint8)


def _recolor_mask(result_bgr: np.ndarray, mask: np.ndarray,
                  hex_color: str, *, feather: int = 2,
                  mode: str = "shading",
                  pupil_blend: bool = False) -> np.ndarray:
    if not np.any(mask):
        return result_bgr.copy()
    value = hex_color.lstrip('#')
    if len(value) != 6:
        return result_bgr.copy()
    r, g, b = int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)

    binary = (mask > 0).astype(np.uint8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    if feather > 0:
        # Feather strictly inward: pixels outside the selection remain exactly
        # unchanged, while the edge ramps smoothly from 0 to full strength.
        alpha = binary.astype(np.float32) * np.clip(
            distance / max(1.0, float(feather)), 0.0, 1.0)
    else:
        alpha = binary.astype(np.float32)
    return recolor_with_mode(
        result_bgr, (r, g, b), alpha, active=binary > 0, mode=mode,
        pupil_blend=pupil_blend)


def build_region_edit_mask(source_bw_bgr: np.ndarray,
                           result_bgr: np.ndarray,
                           x: int, y: int, *,
                           gap_close: int = 4, region_map=None) -> np.ndarray:
    """Return one closed line-art block for bucket fill.

    Region fill must target a single closed line block. Unlike brush/selection
    tools it must never fall back to a loose image-space mask, because that can
    spill through open contours and effectively recolor the whole page.
    """
    mask = refined_lineart_mask_at_point(
        source_bw_bgr, result_bgr, x, y, gap_close=gap_close,
        region_map=region_map)
    if mask is None or mask.size == 0 or not np.any(mask):
        return np.zeros(result_bgr.shape[:2], np.uint8)

    binary = mask > 0
    h, w = binary.shape
    area = int(np.count_nonzero(binary))
    if area <= 0:
        return np.zeros((h, w), np.uint8)

    # Bucket fill is for enclosed line blocks only. If the component leaks to
    # the page border, or becomes implausibly page-wide, reject it instead of
    # using a permissive fallback that can recolor the whole page.
    touches_border = bool(np.any(binary[0, :]) or np.any(binary[-1, :])
                          or np.any(binary[:, 0]) or np.any(binary[:, -1]))
    if touches_border:
        return np.zeros((h, w), np.uint8)

    page_area = int(h * w)
    if area > int(page_area * 0.35):
        return np.zeros((h, w), np.uint8)

    # Keep only the clicked connected component in case local refinement left
    # tiny detached islands.
    component = _connected_component(binary.astype(np.uint8), x, y)
    return np.where(component > 0, 255, 0).astype(np.uint8)


def apply_region_edit(source_bw_bgr: np.ndarray,
                      result_bgr: np.ndarray,
                      filter_base_bgr: np.ndarray | None,
                      x: int, y: int, hex_color: str, *,
                      gap_close: int = 4,
                      mode: str = "shift",
                      feather: int = 2,
                      region_map=None):
    """Apply one region recolor atomically to visible and filter-base layers.

    Strict line-art selection is preferred. When it cannot identify a region,
    a bounded line/color-aware fallback guarantees that the tool still produces
    a visible, safe edit instead of acting like a dead button.
    """
    before = result_bgr.copy()
    mask = build_region_edit_mask(
        source_bw_bgr, result_bgr, x, y, gap_close=gap_close, region_map=region_map)
    if not mask.any():
        base = filter_base_bgr.copy() if filter_base_bgr is not None else before.copy()
        return result_bgr.copy(), base, mask, False
    edited = _recolor_mask(result_bgr, mask, hex_color, feather=feather, mode=mode)

    base = filter_base_bgr
    if base is None or base.shape != result_bgr.shape:
        base = before.copy()
    else:
        base = base.copy()
    base_edited = _recolor_mask(base, mask, hex_color, feather=feather, mode=mode)
    return edited, base_edited, mask, _image_changed(before, edited)


def build_polygon_selection_mask(shape: tuple[int, int], points: list[tuple[int, int]]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if not points or len(points) < 3:
        return mask
    pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def build_rect_selection_mask(shape: tuple[int, int], x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, x2 = sorted((int(np.clip(x1, 0, w - 1)), int(np.clip(x2, 0, w - 1))))
    y1, y2 = sorted((int(np.clip(y1, 0, h - 1)), int(np.clip(y2, 0, h - 1))))
    if x2 <= x1 or y2 <= y1:
        return mask
    mask[y1:y2 + 1, x1:x2 + 1] = 255
    return mask


def combine_selection_masks(existing_mask: np.ndarray | None,
                            incoming_mask: np.ndarray | None,
                            mode: str = 'replace') -> np.ndarray | None:
    """Combine pending selection masks using explicit replace / add / subtract logic.

    This helper keeps the selection algebra in one place so lasso and rectangle
    selection use exactly the same behaviour. ``mode`` accepts ``replace``,
    ``add`` and ``subtract``.
    """
    if incoming_mask is None or incoming_mask.size == 0 or not np.any(incoming_mask):
        return None if existing_mask is None or not np.any(existing_mask) else np.where(existing_mask > 0, 255, 0).astype(np.uint8)

    incoming = np.where(incoming_mask > 0, 255, 0).astype(np.uint8)
    existing = None
    if existing_mask is not None and existing_mask.size != 0 and np.any(existing_mask):
        if existing_mask.shape != incoming.shape:
            raise ValueError('selection masks must have the same shape to combine')
        existing = np.where(existing_mask > 0, 255, 0).astype(np.uint8)

    mode = str(mode or 'replace').strip().lower()
    if mode == 'add':
        if existing is None:
            combined = incoming
        else:
            combined = np.where((existing > 0) | (incoming > 0), 255, 0).astype(np.uint8)
    elif mode == 'subtract':
        if existing is None:
            combined = np.zeros_like(incoming, dtype=np.uint8)
        else:
            combined = np.where((existing > 0) & ~(incoming > 0), 255, 0).astype(np.uint8)
    else:
        combined = incoming
    return combined if np.any(combined) else None


def _selection_safe_mask(source_bw_bgr: np.ndarray, selection_mask: np.ndarray) -> np.ndarray:
    h, w = selection_mask.shape[:2]
    source = source_bw_bgr
    if source.shape[:2] != (h, w):
        source = cv2.resize(source, (w, h), interpolation=cv2.INTER_AREA)
    gray = source if source.ndim == 2 else cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    safe = (selection_mask > 0) & (gray >= 18)
    return np.where(safe, 255, 0).astype(np.uint8)




def build_closed_region_selection_mask(source_bw_bgr: np.ndarray,
                                       selection_mask: np.ndarray, *,
                                       reject_dominant: bool = False,
                                       extra_probability: np.ndarray | None = None,
                                       expand_px: int = 0,
                                       min_area: int = 6,
                                       min_thickness: int = 0) -> np.ndarray:
    """Return only structural-line-enclosed regions inside a selection.

    This delegates to the selection-scoped structural detector instead of a
    fixed grayscale threshold. Pale antialiased contours and short scan gaps are
    recovered, edge-connected space is removed, and a suspicious dominant
    panel/background component is rejected rather than filling the whole frame.
    """
    return closed_regions_from_selection(
        source_bw_bgr, selection_mask, gap_close=4,
        min_area=max(0, int(min_area)),
        min_thickness=max(0, int(min_thickness)),
        max_component_ratio=0.72,
        reject_dominant=reject_dominant,
        extra_probability=extra_probability,
        expand_px=expand_px)

def build_selection_edit_mask(source_bw_bgr: np.ndarray,
                              selection_mask: np.ndarray, *,
                              closed_only: bool = False,
                              reject_dominant: bool = False,
                              extra_probability: np.ndarray | None = None,
                              expand_px: int = 0,
                              min_area: int = 6,
                              min_thickness: int = 0) -> np.ndarray:
    """Return the hard authoritative selection mask used by every fill mode."""
    if selection_mask is None or selection_mask.size == 0:
        shape = source_bw_bgr.shape[:2]
        return np.zeros(shape, np.uint8)
    return (build_closed_region_selection_mask(
                source_bw_bgr, selection_mask,
                reject_dominant=reject_dominant,
                extra_probability=extra_probability,
                expand_px=expand_px,
                min_area=min_area,
                min_thickness=min_thickness)
            if closed_only else _selection_safe_mask(source_bw_bgr, selection_mask))


def apply_selection_edit(source_bw_bgr: np.ndarray,
                         result_bgr: np.ndarray,
                         filter_base_bgr: np.ndarray | None,
                         selection_mask: np.ndarray,
                         hex_color: str, *,
                         feather: int = 2,
                         closed_only: bool = False,
                         mode: str = "shading",
                         pupil_blend: bool = False):
    """Apply a hard-limited selection recolor; it never paints outside the mask."""
    if selection_mask is None or selection_mask.size == 0:
        base = filter_base_bgr.copy() if filter_base_bgr is not None else result_bgr.copy()
        return result_bgr.copy(), base, np.zeros(result_bgr.shape[:2], np.uint8), False
    safe_mask = build_selection_edit_mask(
        source_bw_bgr, selection_mask, closed_only=closed_only)
    if pupil_blend and normalize_recolor_mode(mode) != "flat":
        safe_mask = constrain_pupil_mask_to_lineart(source_bw_bgr, safe_mask)
    if not np.any(safe_mask):
        base = filter_base_bgr.copy() if filter_base_bgr is not None else result_bgr.copy()
        return result_bgr.copy(), base, safe_mask, False
    before = result_bgr.copy()
    edited = _recolor_mask(
        result_bgr, safe_mask, hex_color, feather=feather, mode=mode,
        pupil_blend=pupil_blend)
    base = filter_base_bgr.copy() if filter_base_bgr is not None else before.copy()
    base_edited = _recolor_mask(
        base, safe_mask, hex_color, feather=feather, mode=mode,
        pupil_blend=pupil_blend)
    return edited, base_edited, safe_mask, _image_changed(before, edited)
