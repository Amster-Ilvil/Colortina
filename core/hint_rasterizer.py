"""Soft, region-clipped rasterization for mc-v2 colour hints."""

from __future__ import annotations

import cv2
import numpy as np

from core.hint_spec import HintSpec
from core.lineart_fill import clip_stamp_to_region

_SOURCE_RADIUS_PX = {
    "style_only": (1, 1),
    "scene_palette": (1, 2),
    "auto_instance": (1, 2),
    "character_identity": (2, 3),
    "manual": (2, 32),
    "manual_region": (2, 64),
}


def model_geometry(h: int, w: int, size: int) -> tuple[int, int, int, int]:
    if h < w:
        ratio = h / (size * 1.5)
        rh = int(size * 1.5)
        rw = int(np.ceil(w / ratio))
        return rh, rw + ((-rw) % 32), rh, rw
    ratio = w / size
    rw = size
    rh = int(np.ceil(h / ratio))
    return rh + ((-rh) % 32), rw, rh, rw


def _soft_kernel(radius: int, strength: float) -> np.ndarray:
    radius = max(1, int(radius))
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    sigma = max(0.75, radius * 0.48)
    kernel = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma)).astype(np.float32)
    kernel *= float(np.clip(strength, 0.0, 1.0))
    return kernel


def _resized_labels(label_map, page_gray, rw: int, rh: int, original_w: int):
    source_labels = label_map
    line_low, gap_close = 75, 4
    try:
        from core.region_map import RegionMap
        if isinstance(label_map, RegionMap):
            source_labels = label_map.labels
            line_low = label_map.line_low
            gap_close = max(0, int(round(label_map.gap_close * rw / max(1, original_w))))
    except Exception:
        pass
    if page_gray is not None:
        from core.lineart_fill import label_regions
        small_gray = cv2.resize(page_gray, (rw, rh), interpolation=cv2.INTER_AREA)
        return label_regions(small_gray, line_low=line_low, gap_close=gap_close)
    if source_labels is not None:
        return cv2.resize(source_labels.astype(np.int32), (rw, rh), interpolation=cv2.INTER_NEAREST)
    return None


def rasterize_hint_specs(h: int, w: int, size: int, hint_specs,
                         label_map=None, page_gray=None) -> tuple[np.ndarray, np.ndarray]:
    ph, pw, rh, rw = model_geometry(h, w, size)
    hint = np.full((ph, pw, 3), 128, dtype=np.float32)
    alpha = np.zeros((ph, pw), dtype=np.float32)
    rgb_acc = np.zeros((ph, pw, 3), dtype=np.float32)
    weight_acc = np.zeros((ph, pw), dtype=np.float32)
    labels = _resized_labels(label_map, page_gray, rw, rh, w)

    specs = [HintSpec.from_any(v) for v in (hint_specs or [])]
    # Lower priority first, so stronger/manual points can dominate overlap.
    specs.sort(key=lambda s: (s.priority or 0, s.effective_strength))

    for spec in specs:
        strength = spec.effective_strength
        if strength <= 0.01 or spec.source == "style_only":
            continue
        px = min(rw - 1, max(0, int(round(spec.x_norm * rw))))
        py = min(rh - 1, max(0, int(round(spec.y_norm * rh))))
        min_px, max_px = _SOURCE_RADIUS_PX.get(spec.source, (1, 3))
        radius = int(round(spec.radius_norm * rw))
        radius = int(np.clip(radius, min_px, max_px))

        if labels is not None and labels[py, px] == 0:
            search = max(2, radius + 2)
            y1, y2 = max(0, py - search), min(rh, py + search + 1)
            x1, x2 = max(0, px - search), min(rw, px + search + 1)
            ys, xs = np.nonzero(labels[y1:y2, x1:x2] > 0)
            if ys.size:
                d2 = (ys - (py - y1)) ** 2 + (xs - (px - x1)) ** 2
                k = int(np.argmin(d2))
                py, px = y1 + int(ys[k]), x1 + int(xs[k])

        kernel = _soft_kernel(radius, strength)
        kh, kw = kernel.shape
        x1, y1 = px - radius, py - radius
        x2, y2 = x1 + kw, y1 + kh
        sx1, sy1 = max(0, -x1), max(0, -y1)
        sx2, sy2 = kw - max(0, x2 - rw), kh - max(0, y2 - rh)
        dx1, dy1 = max(0, x1), max(0, y1)
        dx2, dy2 = min(rw, x2), min(rh, y2)
        if dx1 >= dx2 or dy1 >= dy2:
            continue
        local = kernel[sy1:sy2, sx1:sx2].copy()

        if labels is not None:
            region_id = int(labels[py, px])
            if region_id <= 0:
                continue
            region = labels[dy1:dy2, dx1:dx2] == region_id
            if not np.any(region):
                continue
            # Boundary safety band: colour strength decays to zero before the
            # connected-region edge.  This is more robust than a binary one-pixel
            # erosion on pale/anti-aliased lines and still preserves thin regions.
            region_u8 = region.astype(np.uint8)
            distance = cv2.distanceTransform(region_u8, cv2.DIST_L2, 3)
            margin = 1.0 if spec.source in ("manual", "manual_region") else 2.2
            interior = np.clip(distance / margin, 0.0, 1.0)
            if spec.source not in ("manual", "manual_region") and np.any(distance >= 1.5):
                interior[distance < 1.25] = 0.0
            local *= interior.astype(np.float32)

        target_w = weight_acc[dy1:dy2, dx1:dx2]
        target_rgb = rgb_acc[dy1:dy2, dx1:dx2]
        target_w += local
        target_rgb += local[..., None] * np.asarray(spec.rgb, dtype=np.float32)
        alpha[dy1:dy2, dx1:dx2] = np.maximum(alpha[dy1:dy2, dx1:dx2], local)

    used = weight_acc > 1e-6
    if np.any(used):
        hint[used] = rgb_acc[used] / weight_acc[used, None]
    return np.clip(hint, 0, 255).astype(np.uint8), np.clip(alpha, 0.0, 1.0).astype(np.float32)
