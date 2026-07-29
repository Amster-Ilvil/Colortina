"""Strict line-art boundary guard for natural iris recolouring.

The natural iris option adjusts tonal weighting, but it must never enlarge the
caller's brush/selection mask or cross the iris contour.  This module therefore
selects one paintable line-art component under the brush core and intersects the
original alpha with that component.  It is deliberately independent from brush
snapping and can be used when snapping is disabled.
"""
from __future__ import annotations

import cv2
import numpy as np


def _as_gray(source_bgr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    source = source_bgr
    if source.shape[:2] != (h, w):
        source = cv2.resize(source, (w, h), interpolation=cv2.INTER_AREA)
    if source.ndim == 2:
        return source.astype(np.uint8)
    return cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)


def _nearest_active_paintable(
    paintable_active: np.ndarray,
    seed_x: int,
    seed_y: int,
) -> tuple[int, int] | None:
    ys, xs = np.nonzero(paintable_active)
    if ys.size == 0:
        return None
    d2 = (xs.astype(np.int64) - int(seed_x)) ** 2 + (
        ys.astype(np.int64) - int(seed_y)
    ) ** 2
    index = int(np.argmin(d2))
    return int(xs[index]), int(ys[index])


def constrain_pupil_alpha_to_lineart(
    source_bw_bgr: np.ndarray,
    alpha: np.ndarray,
    *,
    seed: tuple[int, int] | None = None,
    ink_threshold: int = 108,
) -> np.ndarray:
    """Clip an iris alpha field to one enclosed line-art component.

    The returned alpha is always a subset of the input alpha.  Ink pixels are
    removed, a one-pixel inward boundary guard is applied, and only the component
    nearest the brush core is retained.  If no safe component can be identified,
    an empty alpha is returned instead of allowing a pupil edit to leak outside.
    """
    alpha_f = np.asarray(alpha, dtype=np.float32)
    if alpha_f.ndim != 2 or source_bw_bgr is None or source_bw_bgr.size == 0:
        return np.zeros_like(alpha_f, dtype=np.float32)
    alpha_f = np.clip(alpha_f, 0.0, 1.0)
    active = alpha_f > 1e-6
    if int(np.count_nonzero(active)) < 4:
        return np.zeros_like(alpha_f, dtype=np.float32)

    h, w = alpha_f.shape
    gray = _as_gray(source_bw_bgr, (h, w))

    # Treat anti-aliased dark-grey contour pixels as ink as well.  A tiny close
    # repairs one-pixel scan/model gaps without growing into nearby eye details.
    ink = (gray <= int(np.clip(ink_threshold, 48, 176))).astype(np.uint8)
    ink = cv2.morphologyEx(
        ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1
    )
    paintable = ink == 0
    candidate = active & paintable
    if not np.any(candidate):
        return np.zeros_like(alpha_f, dtype=np.float32)

    if seed is None:
        weights = np.maximum(alpha_f, 1e-6)
        total = float(weights[active].sum())
        if total <= 1e-6:
            return np.zeros_like(alpha_f, dtype=np.float32)
        yy, xx = np.indices((h, w), dtype=np.float32)
        seed_x = int(round(float((xx * weights).sum() / weights.sum())))
        seed_y = int(round(float((yy * weights).sum() / weights.sum())))
    else:
        seed_x = int(np.clip(seed[0], 0, w - 1))
        seed_y = int(np.clip(seed[1], 0, h - 1))

    nearest = _nearest_active_paintable(candidate, seed_x, seed_y)
    if nearest is None:
        return np.zeros_like(alpha_f, dtype=np.float32)
    seed_x, seed_y = nearest

    count, labels = cv2.connectedComponents(paintable.astype(np.uint8), connectivity=4)
    if count <= 1:
        return np.zeros_like(alpha_f, dtype=np.float32)
    label = int(labels[seed_y, seed_x])
    if label <= 0:
        return np.zeros_like(alpha_f, dtype=np.float32)

    component = labels == label
    constrained = active & component
    if not np.any(constrained):
        return np.zeros_like(alpha_f, dtype=np.float32)

    # Keep blending strictly inward.  This protects the iris outline even when
    # the source line is thin or anti-aliased.  No blur is performed here.
    distance = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 3)
    boundary_gate = np.clip(distance / 1.35, 0.0, 1.0).astype(np.float32)
    out = alpha_f * constrained.astype(np.float32) * boundary_gate
    return np.where(active, out, 0.0).astype(np.float32)


def constrain_pupil_mask_to_lineart(
    source_bw_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    seed: tuple[int, int] | None = None,
) -> np.ndarray:
    """Uint8 convenience wrapper used by selection/finished-stroke paths."""
    alpha = constrain_pupil_alpha_to_lineart(
        source_bw_bgr, (np.asarray(mask) > 0).astype(np.float32), seed=seed
    )
    return np.where(alpha > 1e-5, 255, 0).astype(np.uint8)
