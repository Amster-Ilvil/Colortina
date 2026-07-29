"""Post-lasso boundary snapping for manga line art.

The user's polygon is already a filled selection mask.  Snapping therefore
must move only its *boundary* inside a narrow band; it must never flood-grow a
whole RegionMap component or preserve a full rectangle by accident.

This implementation uses marker-controlled watershed:

* the eroded lasso interior is definite foreground;
* pixels outside a dilated lasso are definite background;
* only the narrow band between them may change;
* dark ink and image gradients form watershed ridges;
* results with no nearby line support or unreasonable area drift fall back to
  the exact user mask.

The result can both expand and contract, which is what a real boundary snap
needs.  A disabled/zero-distance call remains byte-for-byte unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.region_map import RegionMap, build_region_map


@dataclass(frozen=True)
class SelectionSnapDiagnostics:
    raw_area: int
    snapped_area: int
    selected_regions: tuple[int, ...]
    max_distance: int
    used_fallback: bool


def _normalise_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    if mask is None or mask.size == 0:
        return np.zeros((h, w), np.uint8)
    out = mask
    if out.shape[:2] != (h, w):
        out = cv2.resize(out.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return np.where(out > 0, 255, 0).astype(np.uint8)


def _ellipse(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))


def _boundary(binary: np.ndarray) -> np.ndarray:
    src = (binary > 0).astype(np.uint8)
    return cv2.morphologyEx(src, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0


def _largest_seed(mask: np.ndarray, radius: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    core = cv2.erode(binary, _ellipse(max(1, radius)), iterations=1)
    if np.any(core):
        return core
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    if float(distance.max()) <= 0.0:
        return np.zeros_like(binary)
    y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    seed = np.zeros_like(binary)
    cv2.circle(seed, (int(x), int(y)), max(1, min(3, radius)), 1, -1)
    return seed


def _keep_component_touching_seed(candidate: np.ndarray, seed: np.ndarray) -> np.ndarray:
    binary = (candidate > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return binary
    overlaps = np.bincount(labels[seed > 0].ravel(), minlength=count)
    ids = [idx for idx in range(1, count) if int(overlaps[idx]) > 0]
    if not ids:
        ids = [int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1]
    keep = max(ids, key=lambda idx: (int(overlaps[idx]), int(stats[idx, cv2.CC_STAT_AREA])))
    return np.where(labels == keep, 1, 0).astype(np.uint8)


def _line_distance(region_map: RegionMap) -> np.ndarray:
    barrier = getattr(region_map, "barrier", None)
    if barrier is None:
        barrier = np.where(region_map.fillable > 0, 0, 255).astype(np.uint8)
    line = (barrier > 0).astype(np.uint8)
    # distanceTransform returns distance to a zero pixel, so invert line pixels.
    return cv2.distanceTransform(np.where(line > 0, 0, 1).astype(np.uint8), cv2.DIST_L2, 5)


def _boundary_metrics(mask: np.ndarray, line_distance: np.ndarray,
                      max_distance: int) -> tuple[float, float]:
    edge = _boundary(mask)
    values = line_distance[edge]
    if values.size == 0:
        return float(max_distance + 2), 0.0
    clipped = np.minimum(values.astype(np.float32), float(max_distance + 2))
    mean_distance = float(np.mean(clipped))
    near_fraction = float(np.mean(values <= 2.25))
    return mean_distance, near_fraction


def _watershed_elevation(source_bw_bgr: np.ndarray,
                         region_map: RegionMap) -> np.ndarray:
    if source_bw_bgr.ndim == 2:
        gray = source_bw_bgr.astype(np.uint8)
    else:
        gray = cv2.cvtColor(source_bw_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    if float(magnitude.max()) > 1e-6:
        magnitude = magnitude * (255.0 / float(magnitude.max()))
    gradient = np.clip(magnitude, 0, 255).astype(np.uint8)

    barrier = getattr(region_map, "barrier", None)
    if barrier is None:
        barrier_u8 = np.where(region_map.fillable > 0, 0, 255).astype(np.uint8)
    else:
        barrier_u8 = np.where(barrier > 0, 255, 0).astype(np.uint8)
    confidence = getattr(region_map, "line_confidence", None)
    if confidence is not None:
        confidence_u8 = np.clip(np.asarray(confidence, np.float32) * 255.0, 0, 255).astype(np.uint8)
        barrier_u8 = np.maximum(barrier_u8, confidence_u8)

    elevation = np.maximum(gradient, barrier_u8)
    return cv2.GaussianBlur(elevation, (3, 3), 0)


def snap_selection_mask_to_lineart(
    source_bw_bgr: np.ndarray,
    raw_mask: np.ndarray,
    *,
    region_map: RegionMap | None = None,
    gap_close: int = 4,
    max_distance: int = 8,
    max_growth_ratio: float = 1.55,
) -> tuple[np.ndarray, SelectionSnapDiagnostics]:
    """Move a completed lasso boundary toward nearby manga ink lines.

    Only pixels within ``max_distance`` of the original boundary may change.
    With no reliable nearby line, the original mask is returned unchanged.
    """
    if source_bw_bgr is None or source_bw_bgr.size == 0:
        shape = raw_mask.shape[:2] if raw_mask is not None else (1, 1)
        empty = np.zeros(shape, np.uint8)
        return empty, SelectionSnapDiagnostics(0, 0, (), 0, True)

    h, w = source_bw_bgr.shape[:2]
    raw = _normalise_mask(raw_mask, (h, w))
    raw_area = int(np.count_nonzero(raw))
    distance = max(0, int(max_distance))
    if raw_area <= 0 or distance <= 0:
        return raw, SelectionSnapDiagnostics(raw_area, raw_area, (), distance, False)

    if region_map is None or region_map.shape != (h, w):
        region_map = build_region_map(
            source_bw_bgr, gap_close=max(0, int(gap_close)))

    binary = (raw > 0).astype(np.uint8)
    outer = cv2.dilate(binary, _ellipse(distance), iterations=1)
    seed_radius = max(1, min(8, int(round(distance * 0.55))))
    core = _largest_seed(binary, seed_radius)
    if not np.any(core):
        return raw, SelectionSnapDiagnostics(raw_area, raw_area, (), distance, True)

    # If no ink/structural line lies inside the editable boundary band, there
    # is nothing meaningful to snap to.  Preserve the exact lasso.
    inner = cv2.erode(binary, _ellipse(distance), iterations=1)
    band = (outer > 0) & (inner == 0)
    barrier = getattr(region_map, "barrier", None)
    if barrier is None:
        line = region_map.fillable == 0
    else:
        line = np.asarray(barrier) > 0
    if int(np.count_nonzero(line & band)) < max(3, distance // 2):
        return raw, SelectionSnapDiagnostics(raw_area, raw_area, (), distance, True)

    markers = np.zeros((h, w), np.int32)
    markers[outer == 0] = 1       # definite background
    markers[core > 0] = 2        # definite foreground
    elevation = _watershed_elevation(source_bw_bgr, region_map)
    cv2.watershed(cv2.cvtColor(elevation, cv2.COLOR_GRAY2BGR), markers)

    candidate = ((markers == 2) & (outer > 0)).astype(np.uint8)
    # Do not paint the detected ink itself; the editable result stops just under
    # the line shoulder and therefore visually reads as snapped to the contour.
    candidate &= (region_map.fillable > 0).astype(np.uint8)
    candidate = _keep_component_touching_seed(candidate, core)
    out = np.where(candidate > 0, 255, 0).astype(np.uint8)
    snapped_area = int(np.count_nonzero(out))

    if snapped_area <= 0:
        return raw, SelectionSnapDiagnostics(raw_area, raw_area, (), distance, True)

    area_ratio = snapped_area / float(max(1, raw_area))
    max_ratio = max(1.05, float(max_growth_ratio))
    # A rough lasso may be drawn well outside a closed ink contour, so
    # legitimate snapping can contract substantially.  Expansion remains much
    # stricter because it is the dangerous whole-area failure mode.
    min_ratio = max(0.28, 1.0 / (max_ratio * 1.55))
    if not (min_ratio <= area_ratio <= max_ratio):
        return raw, SelectionSnapDiagnostics(raw_area, raw_area, (), distance, True)

    line_distance = _line_distance(region_map)
    raw_mean, raw_near = _boundary_metrics(raw, line_distance, distance)
    out_mean, out_near = _boundary_metrics(out, line_distance, distance)

    # Accept only a visible improvement toward ink.  This prevents a flat white
    # page from producing an arbitrary watershed contour while still allowing
    # both inward and outward movement when a real line is nearby.
    improved = (out_mean <= raw_mean - 0.20
                or out_near >= raw_near + 0.035
                or (out_near >= 0.30 and out_mean <= raw_mean + 0.10))
    changed = int(np.count_nonzero((out > 0) != (raw > 0)))
    if not improved or changed < max(6, int(round(np.sqrt(raw_area) * 0.35))):
        return raw, SelectionSnapDiagnostics(raw_area, raw_area, (), distance, True)

    labels = region_map.labels
    selected_regions = tuple(sorted(
        int(v) for v in np.unique(labels[out > 0]) if int(v) > 0))
    return out, SelectionSnapDiagnostics(
        raw_area=raw_area,
        snapped_area=snapped_area,
        selected_regions=selected_regions,
        max_distance=distance,
        used_fallback=False,
    )
