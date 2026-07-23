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
from core.lineart_fill import lineart_region_recolor
from core.perceptual_recolor import perceptual_target_ab


def _image_changed(before: np.ndarray, after: np.ndarray, *, threshold: float = 0.05) -> bool:
    if before is None or after is None or before.shape != after.shape:
        return before is not after
    diff = np.abs(after.astype(np.int16) - before.astype(np.int16))
    return bool(float(diff.mean()) > threshold or int(diff.max()) >= 2)


def _fallback_local_recolor(source_bw_bgr: np.ndarray,
                            result_bgr: np.ndarray,
                            x: int, y: int, radius_px: int,
                            rgb: tuple[int, int, int],
                            opacity: float) -> tuple[np.ndarray, np.ndarray]:
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
    alpha = np.clip(radial * ink_gate * float(np.clip(opacity, 0.0, 1.0)) * 0.92, 0.0, 1.0)
    if float(alpha.max()) <= 1e-5:
        return result_bgr.copy(), np.zeros((h, w), np.float32)

    r, g, b = [int(np.clip(v, 0, 255)) for v in rgb]
    target = np.array([[[b, g, r]]], dtype=np.uint8)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    edited = result_bgr.copy()
    roi = edited[y1:y2, x1:x2]
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_ab = 128.0 + (target_lab[1:3] - 128.0) * 1.12
    a3 = alpha[..., None]
    lab[..., 1:3] = lab[..., 1:3] * (1.0-a3) + target_ab[None, None, :] * a3
    # Preserve AI-authored lightness; only a tiny cue from the selected color.
    la = alpha * 0.035
    lab[..., 0] = lab[..., 0] * (1.0-la) + target_lab[0] * la
    roi_out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
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
                     gap_close: int = 4):
    """Apply one brush dab atomically to visible and filter-base layers."""
    before = result_bgr.copy()
    edited, mask = apply_local_brush_recolor(
        source_bw_bgr, result_bgr, x, y, radius_px, rgb,
        opacity=opacity, region_map=region_map, gap_close=gap_close)
    if not _image_changed(before, edited) or float(mask.max()) <= 1e-5:
        edited, mask = _fallback_local_recolor(
            source_bw_bgr, result_bgr, x, y, radius_px, rgb, opacity)

    base = filter_base_bgr
    if base is None or base.shape != result_bgr.shape:
        base = before.copy()
    else:
        base = base.copy()
    base_edited, base_mask = apply_local_brush_recolor(
        source_bw_bgr, base, x, y, radius_px, rgb,
        opacity=opacity, region_map=region_map, gap_close=gap_close)
    if not _image_changed(base, base_edited) or float(base_mask.max()) <= 1e-5:
        base_edited, _ = _fallback_local_recolor(
            source_bw_bgr, base, x, y, radius_px, rgb, opacity)
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
                  hex_color: str, *, feather: int = 2) -> np.ndarray:
    if not np.any(mask):
        return result_bgr.copy()
    value = hex_color.lstrip('#')
    if len(value) != 6:
        return result_bgr.copy()
    r, g, b = int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
    target_bgr = np.array([[[b, g, r]]], dtype=np.uint8)
    target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)

    binary = (mask > 0).astype(np.uint8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    if feather > 0:
        alpha = binary.astype(np.float32) * (0.82 + 0.18 * np.clip(distance / max(1.0, float(feather)), 0.0, 1.0))
    else:
        alpha = binary.astype(np.float32)
    lab = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    active = binary > 0
    target_ab = 128.0 + (target_lab[1:3] - 128.0) * 1.08
    desired_ab = perceptual_target_ab(
        lab, active, target_ab, texture_retention=0.62, chroma_retention=0.82)
    lab[..., 1:3] = lab[..., 1:3] * (1.0 - alpha[..., None]) + desired_ab * alpha[..., None]
    # Preserve the existing shading; only a tiny amount of target luminance is mixed.
    l_alpha = alpha * 0.045
    lab[..., 0] = lab[..., 0] * (1.0 - l_alpha) + target_lab[0] * l_alpha
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    result = result_bgr.copy()
    result[active] = out[active]
    return result


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
    edited, mask = lineart_region_recolor(
        source_bw_bgr, result_bgr.copy(), x, y, hex_color,
        gap_close=gap_close, mode=mode, feather=feather, region_map=region_map)
    used_fallback = False
    selected_area = int(np.count_nonzero(mask))
    suspicious_flood = selected_area > int(mask.size * 0.68)
    if not mask.any() or not _image_changed(before, edited) or suspicious_flood:
        mask = _fallback_region_mask(source_bw_bgr, result_bgr, x, y)
        if not mask.any():
            base = filter_base_bgr.copy() if filter_base_bgr is not None else before.copy()
            return result_bgr.copy(), base, mask, False
        edited = _recolor_mask(result_bgr, mask, hex_color, feather=feather)
        used_fallback = True

    base = filter_base_bgr
    if base is None or base.shape != result_bgr.shape:
        base = before.copy()
    else:
        base = base.copy()
    if used_fallback:
        base_edited = _recolor_mask(base, mask, hex_color, feather=feather)
    else:
        base_edited, base_mask = lineart_region_recolor(
            source_bw_bgr, base, x, y, hex_color,
            gap_close=gap_close, mode=mode, feather=feather, region_map=region_map)
        if not base_mask.any() or not _image_changed(base, base_edited):
            base_edited = _recolor_mask(base, mask, hex_color, feather=feather)
    return edited, base_edited, mask, _image_changed(before, edited)
