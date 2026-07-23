"""Conservative manga line detection and local gap repair.

The old implementation interpreted ``gap_close`` as a morphology radius and
closed/dilated the *whole page*.  Increasing the slider therefore thickened all
ink, erased narrow regions and could make a bucket fill affect a very large
connected component.

This module treats the value as a **maximum candidate gap length** instead:

* build a multi-cue line-confidence map (dark ink + local dark ridges + edges),
* remove isolated screentone dots,
* propose only short horizontal/vertical/diagonal bridges,
* accept bridges only when they are thin and touch plausible line endpoints,
* never scale the global line thickness with the gap value.

The resulting barrier map is shared by bucket fills, hint clipping and region
segmentation so every part of the application agrees about boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BoundaryAnalysis:
    confidence: np.ndarray      # float32 [0, 1]
    base_ink: np.ndarray        # uint8 0/255
    repaired: np.ndarray        # uint8 0/255, bridge pixels only
    barrier: np.ndarray         # uint8 0/255, base + repaired


def as_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _normalize_percentile(values: np.ndarray, percentile: float = 97.0) -> np.ndarray:
    scale = float(np.percentile(values, percentile))
    if scale <= 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(values.astype(np.float32) / scale, 0.0, 1.0)


def _remove_isolated_dots(mask: np.ndarray, min_span: int = 5,
                          min_area: int = 8) -> np.ndarray:
    """Remove tiny round components while retaining thin, elongated strokes."""
    binary = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n <= 1:
        return binary * 255
    keep = np.zeros(n, dtype=bool)
    for idx in range(1, n):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        width = int(stats[idx, cv2.CC_STAT_WIDTH])
        height = int(stats[idx, cv2.CC_STAT_HEIGHT])
        span = max(width, height)
        elongated = span >= min_span and min(width, height) <= max(3, span // 3)
        keep[idx] = area >= min_area or elongated
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def build_line_confidence(image: np.ndarray, *, line_low: int = 75) -> np.ndarray:
    """Return a multi-scale probability-like map of manga boundary lines."""
    gray = as_gray(image)
    g = gray.astype(np.float32)

    # Absolute darkness remains the strongest signal for printed manga ink.
    dark = np.clip((float(line_low + 58) - g) / 82.0, 0.0, 1.0)

    # Black-hat responds to thin dark ridges even when scanning/anti-aliasing
    # makes them lighter than the fixed threshold.
    blackhat = cv2.morphologyEx(
        gray, cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    ridge = _normalize_percentile(blackhat, 98.0)

    # Scharr edges recover pale contours.  Gate them by local tone so white
    # paper texture does not become a barrier.
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    edge = _normalize_percentile(cv2.magnitude(gx, gy), 97.5)
    local_mean = cv2.GaussianBlur(g, (0, 0), 2.0)
    edge_gate = np.clip((246.0 - np.minimum(g, local_mean)) / 80.0, 0.0, 1.0)
    edge *= edge_gate

    confidence = np.maximum(dark, np.maximum(ridge * 0.82, edge * 0.66))
    # Strong black pixels must always remain barriers.
    confidence[gray <= int(line_low)] = 1.0
    return np.clip(confidence, 0.0, 1.0).astype(np.float32)


def _direction_kernels(length: int) -> list[np.ndarray]:
    length = max(3, int(length))
    if length % 2 == 0:
        length += 1
    horizontal = np.ones((1, length), np.uint8)
    vertical = np.ones((length, 1), np.uint8)
    diagonal = np.eye(length, dtype=np.uint8)
    anti = np.fliplr(diagonal).copy()
    return [horizontal, vertical, diagonal, anti]


def _candidate_components(candidate: np.ndarray):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (candidate > 0).astype(np.uint8), 8)
    for idx in range(1, n):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        local = labels[y:y + h, x:x + w] == idx
        yield x, y, w, h, area, local


def _repair_short_gaps_native(base_ink: np.ndarray, gray: np.ndarray,
                              max_gap: int) -> np.ndarray:
    """Return accepted local bridge pixels without thickening the whole page.

    Candidate lengths are evaluated incrementally so increasing ``max_gap``
    never loses bridges that were already valid at a smaller value.
    """
    max_gap = max(0, int(max_gap))
    if max_gap <= 0:
        return np.zeros_like(base_ink, dtype=np.uint8)

    base = (base_ink > 0).astype(np.uint8)
    accepted = np.zeros_like(base, dtype=np.uint8)
    _n, base_labels, base_stats, _centroids = cv2.connectedComponentsWithStats(base, 8)
    # A page-level topology budget prevents an extreme slider value from
    # constructing a dense web between screentones/text. Short bridges are
    # evaluated first, so the budget keeps the safest candidates.
    repair_ratio = min(0.015, 0.0015 * max_gap)
    repair_budget = max(32, int(base.size * repair_ratio))
    accepted_count = 0

    lengths = list(range(3, max_gap + 2, 2))
    if lengths[-1] < max_gap + 1:
        lengths.append(max_gap + 1 if (max_gap + 1) % 2 else max_gap)
    for length in sorted(set(max(3, int(v)) for v in lengths)):
        max_area = max(4, int(length * 2.8))
        max_span = length + 2
        for kernel in _direction_kernels(length):
            closed = cv2.morphologyEx(base, cv2.MORPH_CLOSE, kernel, iterations=1)
            proposed = ((closed > 0) & (base == 0) & (accepted == 0)).astype(np.uint8)
            for x, y, w, h, area, local in _candidate_components(proposed):
                if area > max_area or max(w, h) > max_span:
                    continue
                if min(w, h) > max(3, int(length * 0.45) + 1):
                    continue

                x1, y1 = max(0, x - 1), max(0, y - 1)
                x2 = min(base.shape[1], x + w + 1)
                y2 = min(base.shape[0], y + h + 1)
                padded = np.zeros((y2 - y1, x2 - x1), np.uint8)
                padded[(y - y1):(y - y1 + h), (x - x1):(x - x1 + w)] = local.astype(np.uint8)
                ring = cv2.dilate(padded, np.ones((3, 3), np.uint8)) > 0
                nearby_base = base[y1:y2, x1:x2] > 0
                contact = ring & nearby_base
                cy, cx = np.nonzero(contact)
                if len(cx) < 2:
                    continue
                contact_span = max(float(np.ptp(cx)) if len(cx) else 0.0,
                                   float(np.ptp(cy)) if len(cy) else 0.0)
                if contact_span < max(2.0, min(max(w, h), length) * 0.45):
                    continue
                touching_labels = np.unique(base_labels[y1:y2, x1:x2][contact])
                touching_labels = touching_labels[touching_labels > 0]
                if len(touching_labels) > 2:
                    continue
                # Do not bridge neighbouring screentone dots.  Separate endpoint
                # components must have line-like extent or substantial area.
                if len(touching_labels) == 2:
                    plausible = True
                    for label_id in touching_labels:
                        stats = base_stats[int(label_id)]
                        comp_area = int(stats[cv2.CC_STAT_AREA])
                        comp_w = int(stats[cv2.CC_STAT_WIDTH])
                        comp_h = int(stats[cv2.CC_STAT_HEIGHT])
                        span = max(comp_w, comp_h)
                        aspect = span / max(1.0, float(min(comp_w, comp_h)))
                        if span < 5 or (comp_area < 18 and aspect < 1.65):
                            plausible = False
                            break
                    if not plausible:
                        continue

                ys, xs = np.nonzero(local)
                sample = gray[y + ys, x + xs]
                if sample.size and float(np.percentile(sample, 35)) < 118.0:
                    continue
                target = accepted[y:y + h, x:x + w]
                newly_added = local & (target == 0)
                target[local] = 1
                accepted_count += int(np.count_nonzero(newly_added))
                if accepted_count >= repair_budget:
                    return accepted.astype(np.uint8) * 255

    return accepted.astype(np.uint8) * 255


def repair_short_gaps(base_ink: np.ndarray, gray: np.ndarray,
                      max_gap: int) -> np.ndarray:
    """Scale-aware wrapper for local bridge detection.

    Full-resolution confidence/ink is retained, but bridge proposals run on a
    bounded working copy for responsive bucket editing on 300-DPI pages.
    """
    h, w = gray.shape[:2]
    max_edge = 1200
    if max(h, w) <= max_edge:
        return _repair_short_gaps_native(base_ink, gray, max_gap)
    scale = max_edge / float(max(h, w))
    sw, sh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    small_base = cv2.resize(base_ink, (sw, sh), interpolation=cv2.INTER_NEAREST)
    small_gray = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
    small_gap = max(1, int(round(max_gap * scale))) if max_gap > 0 else 0
    small_repaired = _repair_short_gaps_native(small_base, small_gray, small_gap)
    repaired = cv2.resize(small_repaired, (w, h), interpolation=cv2.INTER_NEAREST)
    repaired[base_ink > 0] = 0
    return repaired.astype(np.uint8)


def analyze_boundaries(image: np.ndarray, *, line_low: int = 75,
                       gap_close: int = 4) -> BoundaryAnalysis:
    gray = as_gray(image)
    confidence = build_line_confidence(gray, line_low=line_low)
    # Conservative threshold plus hard-black guarantee.
    base = ((confidence >= 0.58) | (gray <= int(line_low))).astype(np.uint8) * 255
    base = _remove_isolated_dots(base, min_span=5, min_area=8)
    repaired = repair_short_gaps(base, gray, max_gap=gap_close)
    barrier = np.where((base > 0) | (repaired > 0), 255, 0).astype(np.uint8)
    return BoundaryAnalysis(confidence=confidence, base_ink=base,
                            repaired=repaired, barrier=barrier)


def boundary_safety_weight(image: np.ndarray, *, line_low: int = 75,
                           gap_close: int = 4,
                           margin_px: float = 2.0) -> np.ndarray:
    """Soft [0,1] interior weight that reaches zero on detected boundaries."""
    analysis = analyze_boundaries(image, line_low=line_low, gap_close=gap_close)
    fillable = np.where(analysis.barrier > 0, 0, 1).astype(np.uint8)
    distance = cv2.distanceTransform(fillable, cv2.DIST_L2, 5)
    weight = np.clip(distance / max(0.5, float(margin_px)), 0.0, 1.0)
    # Confidence supplies a second safety gate for pale anti-aliased contours.
    weight *= np.clip(1.0 - analysis.confidence * 0.92, 0.0, 1.0)
    return weight.astype(np.float32)
