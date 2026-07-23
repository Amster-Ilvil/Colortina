"""Lineart-bounded region segmentation using the shared conservative barrier."""

from dataclasses import dataclass

import cv2
import numpy as np

from core.region_map import build_region_map

_SEG_MAX_EDGE = 1400


@dataclass
class Region:
    label_id: int
    area: int
    bbox: tuple[int, int, int, int]
    interior_point: tuple[int, int]
    mean_gray: float
    frac: float


class Segmentation:
    def __init__(self, regions: list[Region], labels: np.ndarray, scale: float):
        self.regions = regions
        self.labels = labels
        self.scale = scale

    def interior_points(self, region: Region, spacing: int = 110,
                        max_points: int = 12) -> list[tuple[int, int]]:
        m = (self.labels == region.label_id).astype(np.uint8)
        dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        step = max(24, int(spacing * self.scale))
        pts: list[tuple[int, int]] = []
        h, w = dist.shape
        for yy in range(step // 2, h, step):
            for xx in range(step // 2, w, step):
                if dist[yy, xx] > 6:
                    pts.append((int(xx / self.scale), int(yy / self.scale)))
                    if len(pts) >= max_points:
                        return pts
        return pts


def segment_regions(gray: np.ndarray, *, line_low: int = 100, gap_close: int = 4,
                    min_area_frac: float = 0.00012,
                    max_regions: int = 96) -> Segmentation:
    """Segment a page using the same local-gap barrier as editing tools."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    scale = 1.0
    g = gray
    if max(h, w) > _SEG_MAX_EDGE:
        scale = _SEG_MAX_EDGE / max(h, w)
        g = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    gh, gw = g.shape[:2]

    # Convert full-resolution intent into working-resolution gap length.
    working_gap = max(0, int(round(float(gap_close) * scale)))
    region_map = build_region_map(g, line_low=line_low, gap_close=working_gap)
    fillable = (region_map.fillable > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fillable, connectivity=4)
    if n <= 1:
        return Segmentation([], labels.astype(np.int32), scale)

    page_area = gh * gw
    min_area = max(64, int(page_area * min_area_frac))
    inv = 1.0 / scale

    regions: list[Region] = []
    order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1
    for i in order:
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or len(regions) >= max_regions:
            break
        x, y, ww, hh = (int(stats[i, cv2.CC_STAT_LEFT]),
                        int(stats[i, cv2.CC_STAT_TOP]),
                        int(stats[i, cv2.CC_STAT_WIDTH]),
                        int(stats[i, cv2.CC_STAT_HEIGHT]))
        crop_mask = (labels[y:y + hh, x:x + ww] == i).astype(np.uint8)
        dist = cv2.distanceTransform(crop_mask, cv2.DIST_L2, 3)
        iy, ix = np.unravel_index(int(np.argmax(dist)), dist.shape)
        pixels = g[y:y + hh, x:x + ww][crop_mask.astype(bool)]
        mean_gray = float(pixels.mean()) if pixels.size else 255.0
        regions.append(Region(
            label_id=int(i), area=int(area * inv * inv),
            bbox=(int(x * inv), int(y * inv), int(ww * inv), int(hh * inv)),
            interior_point=(int((x + ix) * inv), int((y + iy) * inv)),
            mean_gray=mean_gray, frac=area / page_area))

    return Segmentation(regions, labels.astype(np.int32), scale)
