"""Shared conservative line-art region maps.

All manual tools, model hints and semantic segmentation use the same barrier
map.  ``gap_close`` means maximum local bridge length; it no longer controls a
whole-page morphology radius.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.line_boundary import analyze_boundaries


@dataclass(frozen=True)
class RegionMap:
    labels: np.ndarray
    fillable: np.ndarray
    line_low: int
    gap_close: int
    barrier: np.ndarray | None = None
    line_confidence: np.ndarray | None = None
    repaired: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.labels.shape[:2]

    def region_at(self, x: int, y: int, *, search_radius: int = 24) -> int:
        """Return the region id at a pixel, snapping off an ink line locally."""
        h, w = self.shape
        if not (0 <= x < w and 0 <= y < h):
            return 0
        region_id = int(self.labels[y, x])
        if region_id:
            return region_id

        radius = max(1, int(search_radius))
        x1, x2 = max(0, x - radius), min(w, x + radius + 1)
        y1, y2 = max(0, y - radius), min(h, y + radius + 1)
        local = self.labels[y1:y2, x1:x2]
        ys, xs = np.nonzero(local > 0)
        if ys.size == 0:
            return 0
        d2 = (xs - (x - x1)) ** 2 + (ys - (y - y1)) ** 2
        candidate_labels = local[ys, xs].astype(np.int32)
        best_distance: dict[int, float] = {}
        for label, distance in zip(candidate_labels.tolist(), d2.tolist()):
            if label <= 0:
                continue
            best_distance[label] = min(float(distance), best_distance.get(label, float("inf")))
        if not best_distance:
            return 0
        nearest = min(best_distance.values())
        # At a line boundary, both sides may be equally close. Prefer the
        # smaller enclosed component instead of accidentally snapping to the
        # page-wide background.
        near_labels = [label for label, distance in best_distance.items()
                       if distance <= nearest + 4.0]
        areas = np.bincount(self.labels.ravel())
        return int(min(near_labels, key=lambda label: (
            int(areas[label]) if label < len(areas) else 2**31, label)))

    def region_at_norm(self, x_norm: float, y_norm: float) -> int:
        h, w = self.shape
        x = min(w - 1, max(0, int(round(float(x_norm) * (w - 1)))))
        y = min(h - 1, max(0, int(round(float(y_norm) * (h - 1)))))
        return self.region_at(x, y)

    def mask(self, region_id: int) -> np.ndarray:
        if region_id <= 0:
            return np.zeros(self.shape, dtype=np.uint8)
        return np.where(self.labels == int(region_id), 255, 0).astype(np.uint8)

    def region_area(self, region_id: int) -> int:
        if region_id <= 0:
            return 0
        return int(np.count_nonzero(self.labels == int(region_id)))

    def touches_border(self, region_id: int) -> bool:
        if region_id <= 0:
            return False
        rid = int(region_id)
        return bool(
            np.any(self.labels[0, :] == rid)
            or np.any(self.labels[-1, :] == rid)
            or np.any(self.labels[:, 0] == rid)
            or np.any(self.labels[:, -1] == rid)
        )

    def is_background_region(self, region_id: int, *,
                             max_area_ratio: float = 0.35) -> bool:
        """Return True for page background or an implausibly large leak."""
        if region_id <= 0:
            return True
        area = self.region_area(region_id)
        page_area = max(1, int(self.labels.size))
        ratio = area / float(page_area)
        return bool(self.touches_border(region_id)
                    or ratio > float(np.clip(max_area_ratio, 0.01, 1.0)))

    def safe_interior(self, region_id: int, margin_px: float = 2.0) -> np.ndarray:
        """Soft interior mask with a boundary safety band."""
        binary = (self.labels == int(region_id)).astype(np.uint8)
        if not np.any(binary):
            return binary.astype(np.float32)
        distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        weight = np.clip(distance / max(0.5, float(margin_px)), 0.0, 1.0)
        if self.line_confidence is not None:
            weight *= np.clip(1.0 - self.line_confidence * 0.92, 0.0, 1.0)
        return weight.astype(np.float32)


def build_region_map(gray_or_bgr: np.ndarray, *, line_low: int = 75,
                     gap_close: int = 4) -> RegionMap:
    analysis = analyze_boundaries(
        gray_or_bgr, line_low=int(line_low), gap_close=max(0, int(gap_close)))
    fillable = np.where(analysis.barrier > 0, 0, 255).astype(np.uint8)
    _, labels = cv2.connectedComponents(fillable, connectivity=4)
    return RegionMap(
        labels=labels.astype(np.int32), fillable=fillable,
        line_low=int(line_low), gap_close=max(0, int(gap_close)),
        barrier=analysis.barrier, line_confidence=analysis.confidence,
        repaired=analysis.repaired)
