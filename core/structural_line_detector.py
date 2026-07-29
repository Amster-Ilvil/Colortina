"""Selection-scoped structural manga line detection.

The rectangle/lasso closed-fill tool needs a *topological barrier map*, not a
pretty sketch preview.  A fixed grayscale threshold misses antialiased, scanned
or compressed contours and may treat an entire panel as one closed region.

This module fuses the official MangaLineExtraction_PyTorch structural-line
probability with conservative native ink evidence and short-gap repair.  When an
AI map is supplied it becomes the primary barrier source, so screentone and
colour texture rejected by the model are not reintroduced by noisy Canny edges.
The native path remains available for tests and explicit fallback calls.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.line_boundary import (
    as_gray,
    build_line_confidence,
    repair_short_gaps,
)


@dataclass(frozen=True)
class StructuralLineAnalysis:
    confidence: np.ndarray  # float32 [0, 1]
    barrier: np.ndarray     # uint8 0/255
    repaired: np.ndarray    # uint8 0/255


def _normalise(values: np.ndarray, percentile: float = 97.5) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    scale = float(np.percentile(values, percentile)) if values.size else 0.0
    if scale <= 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(values / scale, 0.0, 1.0).astype(np.float32)


def _selection_bbox(selection_mask: np.ndarray, pad: int = 8) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(selection_mask > 0)
    if xs.size == 0:
        return 0, 0, selection_mask.shape[1], selection_mask.shape[0]
    h, w = selection_mask.shape[:2]
    x1 = max(0, int(xs.min()) - int(pad))
    y1 = max(0, int(ys.min()) - int(pad))
    x2 = min(w, int(xs.max()) + int(pad) + 1)
    y2 = min(h, int(ys.max()) + int(pad) + 1)
    return x1, y1, x2, y2


def _adaptive_block_size(height: int, width: int) -> int:
    # Keep a useful local window on both small eyes and full-page selections.
    size = int(round(min(height, width) * 0.09))
    size = max(15, min(61, size))
    return size if size % 2 else size + 1


def _remove_texture_dots(mask: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """Drop screentone/print dots while preserving real contour strokes.

    The previous implementation kept virtually every component with three
    pixels.  On manga pages that turns halftone and colour texture into hundreds
    of false barriers, so a large clothing/skin region is reduced to a few tiny
    islands.  Real ink is normally elongated, sizeable, or supported by very
    high line confidence; compact low-confidence dots are not.
    """
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return binary * 255
    keep = np.zeros(count, dtype=bool)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        span = max(width, height)
        short = max(1, min(width, height))
        aspect = span / float(short)
        pixels = labels == label
        mean_conf = float(confidence[pixels].mean()) if np.any(pixels) else 0.0
        elongated = span >= 5 and aspect >= 1.55
        substantial = area >= 14 or span >= 9
        very_strong_line = (
            mean_conf >= 0.84 and span >= 4 and aspect >= 1.25)
        keep[label] = bool(elongated or substantial or very_strong_line)
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def detect_structural_lines(
    source_bgr: np.ndarray,
    selection_mask: np.ndarray | None = None,
    *,
    gap_close: int = 4,
    extra_probability: np.ndarray | None = None,
) -> StructuralLineAnalysis:
    """Return a strong topological line barrier for a selection.

    ``extra_probability`` is a full-resolution 0..1 map produced by the
    integrated MangaLineExtraction_PyTorch model.  When present it is the
    primary topology cue; native evidence only restores unquestionably dark ink.
    """
    gray_full = as_gray(source_bgr)
    h, w = gray_full.shape[:2]
    if selection_mask is None:
        selection = np.ones((h, w), np.uint8)
    else:
        if selection_mask.shape[:2] != (h, w):
            selection = cv2.resize(
                selection_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            selection = selection_mask
        selection = (selection > 0).astype(np.uint8)
    if not np.any(selection):
        zeros = np.zeros((h, w), np.uint8)
        return StructuralLineAnalysis(zeros.astype(np.float32), zeros, zeros)

    x1, y1, x2, y2 = _selection_bbox(selection, pad=max(6, int(gap_close) + 3))
    gray = gray_full[y1:y2, x1:x2]
    local_sel = selection[y1:y2, x1:x2]

    # Existing project detector: darkness + black-hat ridges + Scharr edges.
    native = build_line_confidence(gray, line_low=88)

    # Contrast-normalised pass catches pale scan/antialias contours without
    # changing the source pixels used for the actual recolour.
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(6, 6)).apply(gray)
    contrast = build_line_confidence(clahe, line_low=102)

    g = gray.astype(np.float32)
    local_background = cv2.GaussianBlur(g, (0, 0), 2.3)
    dark_ridge = _normalise(np.maximum(local_background - g, 0.0), 98.0)

    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    gradient = _normalise(cv2.magnitude(gx, gy), 97.5)
    tone_gate = np.clip((250.0 - np.minimum(g, local_background)) / 92.0, 0.0, 1.0)
    structural_edge = gradient * tone_gate

    block = _adaptive_block_size(*gray.shape[:2])
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block, 7)
    adaptive_score = (adaptive.astype(np.float32) / 255.0) * np.maximum(
        dark_ridge, structural_edge * 0.82)

    # Canny is used only as a structural contour cue.  It is especially useful
    # for pale anti-aliased outlines whose centre pixels are almost white and
    # therefore weak in an absolute-darkness detector.  Small Canny components
    # are removed below, so ordinary screentone dots do not become fill walls.
    canny_input = cv2.GaussianBlur(gray, (3, 3), 0)
    canny = cv2.Canny(canny_input, 18, 54, L2gradient=True)

    native_confidence = np.maximum.reduce([
        native,
        contrast * 0.82,
        dark_ridge * 0.90,
        structural_edge * 0.78,
        adaptive_score * 0.86,
    ]).astype(np.float32)

    extra = None
    if extra_probability is not None:
        extra = np.asarray(extra_probability, dtype=np.float32)
        if extra.shape != (h, w):
            extra = cv2.resize(extra, (w, h), interpolation=cv2.INTER_LINEAR)
        extra = np.clip(extra[y1:y2, x1:x2], 0.0, 1.0)

    if extra is not None and np.any(extra > 0.02):
        # AI-primary fusion: the manga model was trained specifically to retain
        # structural strokes while suppressing screentone and shade texture.
        # Re-adding every weak Canny/adaptive edge would undo that advantage.
        confidence = np.maximum.reduce([
            extra,
            native * 0.58,
            dark_ridge * 0.46,
            structural_edge * 0.36,
        ]).astype(np.float32)
        base = ((extra >= 0.20)
                | (native >= 0.76)
                | (gray <= 72)).astype(np.uint8) * 255
    else:
        confidence = native_confidence
        # Lightweight fallback retained for environments/tests without weights.
        base = ((confidence >= 0.28)
                | (gray <= 92)
                | (canny > 0)).astype(np.uint8) * 255
    base[local_sel == 0] = 0
    base = _remove_texture_dots(base, confidence)

    repaired = repair_short_gaps(base, gray, max_gap=max(1, int(gap_close)))
    barrier = np.where((base > 0) | (repaired > 0), 255, 0).astype(np.uint8)

    # Seal one-pixel antialias holes without globally swallowing narrow regions.
    # The old two-iteration square dilation expanded every texture mark by up to
    # four pixels and was the main reason only tiny blue preview specks survived.
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    barrier = cv2.morphologyEx(
        barrier, cv2.MORPH_CLOSE, cross, iterations=1)
    barrier = cv2.dilate(barrier, cross, iterations=1)
    barrier[local_sel == 0] = 0

    full_conf = np.zeros((h, w), np.float32)
    full_barrier = np.zeros((h, w), np.uint8)
    full_repaired = np.zeros((h, w), np.uint8)
    full_conf[y1:y2, x1:x2] = confidence
    full_barrier[y1:y2, x1:x2] = barrier
    full_repaired[y1:y2, x1:x2] = repaired
    return StructuralLineAnalysis(full_conf, full_barrier, full_repaired)


def _component_shape_metrics(labels: np.ndarray,
                             stats: np.ndarray,
                             label: int) -> tuple[float, float, float, float]:
    """Return robust geometry for one paintable connected component.

    The values are ``(aspect_ratio, effective_thickness, compactness,
    rotated_fill_ratio)``.  ``effective_thickness`` combines the rotated short
    side, area/major-axis width, and hydraulic width.  This catches long thin
    or winding cavities whose total area is large enough to bypass an area-only
    filter, while compact eyes/buttons/small round details remain untouched.
    """
    x = int(stats[label, cv2.CC_STAT_LEFT])
    y = int(stats[label, cv2.CC_STAT_TOP])
    width = int(stats[label, cv2.CC_STAT_WIDTH])
    height = int(stats[label, cv2.CC_STAT_HEIGHT])
    area = float(stats[label, cv2.CC_STAT_AREA])
    if area <= 0.0 or width <= 0 or height <= 0:
        return 1.0, 0.0, 1.0, 1.0

    roi = np.where(labels[y:y + height, x:x + width] == label, 255, 0).astype(np.uint8)
    points = cv2.findNonZero(roi)
    if points is None or len(points) < 3:
        short = float(max(1, min(width, height)))
        long = float(max(width, height))
        return long / short, min(short, area / max(long, 1.0)), 0.0, 1.0

    (_, _), (rect_w, rect_h), _ = cv2.minAreaRect(points)
    major = max(float(rect_w), float(rect_h), 1.0)
    minor = max(min(float(rect_w), float(rect_h)), 1.0)
    aspect = major / minor

    contours, _ = cv2.findContours(
        roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = float(sum(cv2.arcLength(c, True) for c in contours))
    compactness = (4.0 * np.pi * area / (perimeter * perimeter)
                   if perimeter > 1e-6 else 1.0)
    rotated_fill = float(np.clip(
        area / max(major * minor, 1.0), 0.0, 1.0))

    axis_width = area / major
    hydraulic_width = (2.0 * area / perimeter
                       if perimeter > 1e-6 else axis_width)
    effective_thickness = min(minor, axis_width, hydraulic_width)
    return (float(aspect), float(effective_thickness),
            float(compactness), float(rotated_fill))


def _is_slender_noise_component(labels: np.ndarray,
                                stats: np.ndarray,
                                label: int,
                                min_thickness: int) -> bool:
    """Decide whether a closed component is an adjustable thin stray region.

    A width threshold alone would incorrectly remove compact small details.
    Therefore a component must be both thinner than the requested value and
    demonstrably elongated/string-like by at least one independent geometry
    cue.  Setting ``min_thickness`` to zero fully disables this filter.
    """
    threshold = max(0.0, float(min_thickness))
    if threshold <= 0.0:
        return False
    aspect, thickness, compactness, fill_ratio = _component_shape_metrics(
        labels, stats, label)
    elongated = aspect >= 2.40
    narrow_corridor = aspect >= 1.75 and compactness <= 0.36
    winding_or_fragmented = compactness <= 0.20 and fill_ratio <= 0.64
    is_slender = elongated or narrow_corridor or winding_or_fragmented
    return bool(is_slender and thickness < threshold)


def _expand_closed_regions_within_paintable(closed: np.ndarray,
                                          allowed_mask: np.ndarray,
                                          expand_px: int) -> np.ndarray:
    """Uniformly grow recognised closed regions outward without crossing lines.

    Expansion is geodesic: the mask may only grow into ``allowed_mask`` pixels,
    so it can never leave the user selection.  The caller typically passes a
    *relaxed* interior mask that reclaims a thin fringe near detected lines
    while still excluding the strongest structural-line core.
    """
    steps = max(0, int(expand_px))
    if steps <= 0 or closed.size == 0 or not np.any(closed):
        return closed.astype(np.uint8)
    grown = (closed > 0).astype(np.uint8)
    # A seed can legitimately overlap dark shading or antialiased ink evidence
    # in the original page even though it was accepted by the AI topology map.
    # Geodesic growth must never delete that authoritative seed.  The previous
    # implementation intersected the whole dilated mask with ``allowed`` on
    # every iteration, so a small component could shrink or become completely
    # frozen as soon as the source-ink guard touched it.
    allowed = np.where((allowed_mask > 0) | (grown > 0), 1, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(steps):
        frontier = cv2.dilate(grown, kernel, iterations=1)
        # Monotonic geodesic dilation: retain every accepted pixel and add only
        # newly reachable paintable pixels.  This guarantees that increasing
        # the slider can never make a closed region smaller.
        nxt = np.where(
            (grown > 0) | ((frontier > 0) & (allowed > 0)),
            1, 0).astype(np.uint8)
        if np.array_equal(nxt, grown):
            break
        grown = nxt
    return (grown * 255).astype(np.uint8)


def closed_regions_from_selection(
    source_bgr: np.ndarray,
    selection_mask: np.ndarray,
    *,
    gap_close: int = 4,
    min_area: int = 6,
    min_thickness: int = 0,
    max_component_ratio: float = 0.96,
    reject_dominant: bool = False,
    extra_probability: np.ndarray | None = None,
    expand_px: int = 0,
) -> np.ndarray:
    """Keep only genuine line-enclosed components inside a selection.

    Components connected to the selection edge are open and removed. Small
    components can be filtered by area, while long narrow/string-like cavities
    can be filtered independently by effective thickness. A dominant component
    covering most of the rectangle is also rejected: that is normally
    panel/background space caused by missed internal contours, and returning an
    empty mask is safer than recolouring the whole rectangle.
    """
    if selection_mask is None or selection_mask.size == 0:
        return np.zeros(source_bgr.shape[:2], np.uint8)
    h, w = selection_mask.shape[:2]
    source = source_bgr
    if source.shape[:2] != (h, w):
        source = cv2.resize(source, (w, h), interpolation=cv2.INTER_AREA)
    selection = (selection_mask > 0).astype(np.uint8)
    if not np.any(selection):
        return np.zeros((h, w), np.uint8)

    analysis = detect_structural_lines(
        source, selection, gap_close=gap_close,
        extra_probability=extra_probability)
    barrier = analysis.barrier > 0
    paintable = ((selection > 0) & ~barrier).astype(np.uint8)
    if not np.any(paintable):
        return np.zeros((h, w), np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        paintable, connectivity=4)
    if count <= 1:
        return np.zeros((h, w), np.uint8)

    # The one-pixel selection perimeter is the escape boundary.
    eroded = cv2.erode(selection, np.ones((3, 3), np.uint8), iterations=1)
    perimeter = (selection > 0) & (eroded == 0)
    open_labels = {
        int(value)
        for value in np.unique(labels[perimeter & (paintable > 0)])
        if int(value) > 0
    }

    selection_area = max(1, int(np.count_nonzero(selection)))
    candidates: list[tuple[int, int, float, float]] = []
    # label, area, selection ratio, bbox occupancy
    ys, xs = np.nonzero(selection)
    sel_w = max(1, int(xs.max() - xs.min() + 1))
    sel_h = max(1, int(ys.max() - ys.min() + 1))
    for label in range(1, count):
        if label in open_labels:
            continue
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(0, int(min_area)):
            continue
        if _is_slender_noise_component(
                labels, stats, label, min_thickness=min_thickness):
            continue
        ratio = area / float(selection_area)
        bbox_ratio = (
            int(stats[label, cv2.CC_STAT_WIDTH]) / float(sel_w)
            * int(stats[label, cv2.CC_STAT_HEIGHT]) / float(sel_h)
        )
        # Never return a near-whole-frame component. This is the exact failure
        # mode reported by users when pale/internal lines were missed.
        component_limit = float(np.clip(max_component_ratio, 0.20, 0.99))
        if reject_dominant:
            component_limit = min(component_limit, 0.62)
        if ratio > component_limit:
            continue
        candidates.append((label, area, ratio, bbox_ratio))

    if not candidates:
        return np.zeros((h, w), np.uint8)

    # When there are several closed components, discard a surrounding dominant
    # background/panel cavity while retaining the smaller semantic regions.
    if reject_dominant and len(candidates) > 1:
        largest = max(candidates, key=lambda item: item[1])
        others_area = sum(item[1] for item in candidates if item[0] != largest[0])
        if (largest[2] > 0.38 and largest[3] > 0.50
                and largest[1] > max(1, int(others_area * 1.5))):
            candidates = [item for item in candidates if item[0] != largest[0]]

    closed = np.zeros((h, w), np.uint8)
    for label, _area, _ratio, _bbox_ratio in candidates:
        closed[labels == label] = 255

    # A failed topology map can split one giant near-full-frame cavity into a
    # few labels. The per-component dominant check above would miss that case,
    # so apply a total-area guard as well. Returning empty is safer than painting
    # almost the entire rectangle.
    if reject_dominant:
        closed_ratio = float(np.count_nonzero(closed) / max(1, selection_area))
        if closed_ratio > 0.68:
            return np.zeros((h, w), np.uint8)

    if int(expand_px) > 0:
        # Expansion compensates for an AI contour that is shifted slightly
        # inward.  Therefore the AI barrier itself must not be the hard stop.
        # Instead, grow toward the actual ink evidence in the original B/W
        # source.  This makes every pixel value on the slider meaningful while
        # still protecting the real manga line and the user's selection edge.
        gray = as_gray(source)
        local_bg = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2.0)
        ridge = _normalise(np.maximum(local_bg - gray.astype(np.float32), 0.0), 98.5)
        native_ink = build_line_confidence(gray, line_low=96)
        ink_confidence = np.maximum(native_ink, ridge * 0.92).astype(np.float32)
        # Use the *core* of real ink as the stop wall.  A broad ``gray <= 112``
        # gate treated antialiasing, gray screentone and shaded small regions as
        # solid walls, which made their expansion stop at 0 px.  Strong black
        # strokes remain protected, while weak gray evidence is allowed to be
        # reclaimed until the true line core is reached.
        ink_guard = (((gray <= 72)
                      | ((gray <= 142) & (native_ink >= 0.82))
                      | ((gray <= 158) & (ridge >= 0.76))).astype(np.uint8) * 255)
        ink_guard[selection == 0] = 0
        ink_guard = _remove_texture_dots(ink_guard, ink_confidence)
        ink_guard = cv2.morphologyEx(
            ink_guard,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
            iterations=1)
        relaxed_allowed = ((selection > 0) & (ink_guard == 0)).astype(np.uint8)
    else:
        relaxed_allowed = paintable
    return _expand_closed_regions_within_paintable(
        closed, relaxed_allowed, expand_px=expand_px)
