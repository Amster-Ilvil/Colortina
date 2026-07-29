"""Independent custom-colour-bias brush.

This module deliberately does not use the normal manual brush, model hints,
selection tools, or mc-v2.  A stroke prepares one colour-biased candidate from
the current edited layer, then composites that candidate only through the
accumulated soft brush mask.  Pixels outside the visible stroke stay byte-exact.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.custom_color_bias import apply_global_color_bias


@dataclass(frozen=True)
class BiasBrushConfig:
    rgb: tuple[int, int, int] = (120, 160, 255)
    strength: float = 0.80
    tone_range: str = "all"
    protect_skin: bool = True
    protect_lineart: bool = True
    protect_saturated: bool = True

    @classmethod
    def from_dict(cls, data: dict | None) -> "BiasBrushConfig":
        value = dict(data or {})
        rgb = tuple(int(np.clip(v, 0, 255)) for v in
                    tuple(value.get("rgb") or cls.rgb))
        if len(rgb) != 3:
            rgb = cls.rgb
        strength = value.get("strength", cls.strength)
        # UI stores 0..200, while the core bias routine expects 0..2.
        strength = float(strength)
        if strength > 2.0:
            strength /= 100.0
        return cls(
            rgb=rgb,
            strength=float(np.clip(strength, 0.0, 2.0)),
            tone_range=str(value.get("tone_range", cls.tone_range) or "all"),
            protect_skin=bool(value.get("protect_skin", cls.protect_skin)),
            protect_lineart=bool(value.get("protect_lineart", cls.protect_lineart)),
            protect_saturated=bool(value.get("protect_saturated", cls.protect_saturated)),
        )


def empty_stroke_alpha(shape: tuple[int, int]) -> np.ndarray:
    return np.zeros(tuple(map(int, shape)), dtype=np.float32)


def _clamped_dab_bounds(shape: tuple[int, int], ix: int, iy: int,
                        radius_px: int) -> tuple[int, int, int, int] | None:
    h, w = map(int, shape)
    radius = max(1, int(radius_px))
    if h <= 0 or w <= 0:
        return None
    ix = int(np.clip(ix, 0, max(0, w - 1)))
    iy = int(np.clip(iy, 0, max(0, h - 1)))
    x0, x1 = max(0, ix - radius), min(w, ix + radius + 1)
    y0, y1 = max(0, iy - radius), min(h, iy + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    return x0, y0, x1, y1


def add_soft_round_dab_inplace(alpha: np.ndarray, ix: int, iy: int,
                               radius_px: int,
                               *, edge_softness: float = 0.22
                               ) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    """In-place version of :func:`add_soft_round_dab`.

    Returns the modified alpha and the affected ROI as ``(x0, y0, x1, y1)``.
    The ROI can be fed directly into ROI-only compositing so dragging the bias
    brush does not have to recompute the whole page on every mouse move.
    """
    if alpha is None or np.asarray(alpha).ndim != 2:
        raise ValueError("stroke alpha must be a 2-D array")
    out = np.asarray(alpha, dtype=np.float32)
    bounds = _clamped_dab_bounds(out.shape, ix, iy, radius_px)
    if bounds is None:
        return out, None
    x0, y0, x1, y1 = bounds
    radius = max(1, int(radius_px))
    ix = int(np.clip(ix, 0, max(0, out.shape[1] - 1)))
    iy = int(np.clip(iy, 0, max(0, out.shape[0] - 1)))
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    distance = np.sqrt((xx - float(ix)) ** 2 + (yy - float(iy)) ** 2)
    softness = float(np.clip(edge_softness, 0.0, 0.95))
    inner = float(radius) * (1.0 - softness)
    if softness <= 1e-6:
        dab = (distance <= float(radius)).astype(np.float32)
    else:
        dab = np.clip((float(radius) - distance) /
                      max(1e-6, float(radius) - inner), 0.0, 1.0)
    crop = out[y0:y1, x0:x1]
    out[y0:y1, x0:x1] = 1.0 - (1.0 - crop) * (1.0 - dab)
    np.clip(out, 0.0, 1.0, out=out)
    return out, bounds


def add_soft_round_dab(alpha: np.ndarray, ix: int, iy: int, radius_px: int,
                       *, edge_softness: float = 0.22) -> np.ndarray:
    """Accumulate one round dab into a 0..1 stroke alpha mask.

    The interior is fully active; only the outer edge is softened.  Combining
    dabs uses opacity union, avoiding darker seams where circles overlap.
    """
    out = np.asarray(alpha, dtype=np.float32).copy()
    add_soft_round_dab_inplace(out, ix, iy, radius_px, edge_softness=edge_softness)
    return out


def prepare_bias_candidate(result_bgr: np.ndarray,
                           source_bw_bgr: np.ndarray | None,
                           config: BiasBrushConfig | dict | None) -> np.ndarray:
    """Create one full-resolution candidate for a single brush stroke.

    Scope is intentionally fixed to ``page``: the brush mask itself defines the
    affected scope.  This also keeps the tool independent from global
    character/background segmentation and its settings.
    """
    if result_bgr is None:
        raise ValueError("a colorized result is required")
    cfg = config if isinstance(config, BiasBrushConfig) else BiasBrushConfig.from_dict(config)
    return apply_global_color_bias(
        np.asarray(result_bgr), source_bw_bgr, cfg.rgb, cfg.strength,
        scope="page", tone_range=cfg.tone_range,
        protect_skin=cfg.protect_skin,
        protect_lineart=cfg.protect_lineart,
        protect_saturated=cfg.protect_saturated,
    )


def composite_bias_candidate(base_bgr: np.ndarray, candidate_bgr: np.ndarray,
                             alpha: np.ndarray) -> tuple[np.ndarray, int]:
    """Composite candidate only under the stroke alpha.

    Returns the edited image and the number of pixels that changed.  Outside
    the stroke, base pixels are explicitly restored byte-for-byte.
    """
    if base_bgr is None or candidate_bgr is None:
        raise ValueError("base and candidate images are required")
    base = np.asarray(base_bgr)
    candidate = np.asarray(candidate_bgr)
    if candidate.shape[:2] != base.shape[:2]:
        candidate = cv2.resize(candidate, (base.shape[1], base.shape[0]),
                               interpolation=cv2.INTER_AREA)
    work_alpha = np.asarray(alpha, dtype=np.float32)
    if work_alpha.shape != base.shape[:2]:
        work_alpha = cv2.resize(work_alpha, (base.shape[1], base.shape[0]),
                                interpolation=cv2.INTER_LINEAR)
    work_alpha = np.clip(work_alpha, 0.0, 1.0)
    a3 = work_alpha[..., None]
    out = np.rint(base.astype(np.float32) * (1.0 - a3) +
                  candidate.astype(np.float32) * a3).clip(0, 255).astype(np.uint8)
    inactive = work_alpha <= 0.0
    out[inactive] = base[inactive]
    changed = int(np.count_nonzero(np.any(out != base, axis=2)))
    return out, changed


def composite_bias_candidate_roi(base_bgr: np.ndarray, candidate_bgr: np.ndarray,
                                 alpha: np.ndarray,
                                 roi: tuple[int, int, int, int] | None,
                                 *, dst: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """ROI-only version of :func:`composite_bias_candidate`.

    ``roi`` is ``(x0, y0, x1, y1)`` in image pixels.  Only this rectangle is
    recomputed; pixels outside it are left untouched.  When ``dst`` is passed,
    the result is written in place, which keeps dragging responsive on large
    pages.
    """
    if base_bgr is None or candidate_bgr is None:
        raise ValueError("base and candidate images are required")
    base = np.asarray(base_bgr)
    candidate = np.asarray(candidate_bgr)
    if candidate.shape[:2] != base.shape[:2]:
        candidate = cv2.resize(candidate, (base.shape[1], base.shape[0]),
                               interpolation=cv2.INTER_AREA)
    work_alpha = np.asarray(alpha, dtype=np.float32)
    if work_alpha.shape != base.shape[:2]:
        work_alpha = cv2.resize(work_alpha, (base.shape[1], base.shape[0]),
                                interpolation=cv2.INTER_LINEAR)
    if roi is None:
        return composite_bias_candidate(base, candidate, work_alpha)
    x0, y0, x1, y1 = [int(v) for v in roi]
    x0 = max(0, min(x0, base.shape[1]))
    x1 = max(x0, min(x1, base.shape[1]))
    y0 = max(0, min(y0, base.shape[0]))
    y1 = max(y0, min(y1, base.shape[0]))
    if x0 >= x1 or y0 >= y1:
        out = base.copy() if dst is None else dst
        return out, 0
    out = base.copy() if dst is None else dst
    base_roi = base[y0:y1, x0:x1]
    cand_roi = candidate[y0:y1, x0:x1]
    alpha_roi = np.clip(work_alpha[y0:y1, x0:x1], 0.0, 1.0)
    a3 = alpha_roi[..., None]
    roi_out = np.rint(base_roi.astype(np.float32) * (1.0 - a3) +
                      cand_roi.astype(np.float32) * a3).clip(0, 255).astype(np.uint8)
    inactive = alpha_roi <= 0.0
    roi_out[inactive] = base_roi[inactive]
    out[y0:y1, x0:x1] = roi_out
    changed = int(np.count_nonzero(np.any(roi_out != base_roi, axis=2)))
    return out, changed



def _soft_fill_from_binary(mask_u8: np.ndarray, feather: int = 5,
                           *, floor: float = 0.82) -> np.ndarray:
    binary = (np.asarray(mask_u8) > 0).astype(np.uint8)
    if not np.any(binary):
        return binary.astype(np.float32)
    if feather <= 0:
        return binary.astype(np.float32)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    edge_soft = np.clip(distance / max(1.0, float(feather)), 0.0, 1.0)
    soft = binary.astype(np.float32) * (float(np.clip(floor, 0.0, 1.0)) +
                                        (1.0 - float(np.clip(floor, 0.0, 1.0))) * edge_soft)
    return np.clip(soft, 0.0, 1.0).astype(np.float32)


def _local_texture_fill_mask(source_bw_bgr: np.ndarray | None,
                             result_bgr: np.ndarray,
                             seed_binary: np.ndarray,
                             x: int, y: int, radius_px: int) -> np.ndarray:
    """Expand a seed into a cohesive local paintable block.

    This is a fallback for screentone / broken-line regions where an exact
    closed component does not exist.  It grows only within a local window,
    avoids solid line art, follows approximate tone similarity, then closes
    tiny gaps so the final fill looks like one coherent painted block rather
    than many isolated fragments.
    """
    h, w = result_bgr.shape[:2]
    if source_bw_bgr is None:
        gray_page = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY)
    else:
        src = source_bw_bgr
        if src.shape[:2] != (h, w):
            src = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)
        gray_page = src if src.ndim == 2 else cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    x = int(np.clip(x, 0, max(0, w - 1)))
    y = int(np.clip(y, 0, max(0, h - 1)))
    radius = max(4, int(radius_px))
    reach = max(radius * 3, 22)
    x0, x1 = max(0, x - reach), min(w, x + reach + 1)
    y0, y1 = max(0, y - reach), min(h, y + reach + 1)
    if x0 >= x1 or y0 >= y1:
        return np.zeros((h, w), np.uint8)
    roi_seed = (seed_binary[y0:y1, x0:x1] > 0).astype(np.uint8)
    if not np.any(roi_seed):
        cv2.circle(roi_seed, (x - x0, y - y0), max(2, radius // 2), 1, -1)
    work_gray = cv2.GaussianBlur(cv2.cvtColor(result_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (0, 0), 1.2)
    roi_gray_page = gray_page[y0:y1, x0:x1]
    paintable = (roi_gray_page >= 28).astype(np.uint8)
    if not np.any(paintable):
        return np.zeros((h, w), np.uint8)

    seed_ys, seed_xs = np.nonzero(roi_seed > 0)
    sy = int(np.median(seed_ys)) if seed_ys.size else int(np.clip(y - y0, 0, max(0, y1 - y0 - 1)))
    sx = int(np.median(seed_xs)) if seed_xs.size else int(np.clip(x - x0, 0, max(0, x1 - x0 - 1)))
    mask_ff = np.zeros((work_gray.shape[0] + 2, work_gray.shape[1] + 2), np.uint8)
    flood_img = work_gray.copy()
    tol = int(np.clip(18 + radius * 0.65, 16, 40))
    flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
    cv2.floodFill(flood_img, mask_ff, (sx, sy), 0,
                  loDiff=tol, upDiff=tol, flags=flags)
    flood = (mask_ff[1:-1, 1:-1] > 0).astype(np.uint8)

    # Keep growth local to the stroke neighbourhood and inside paintable areas.
    distance = cv2.distanceTransform((roi_seed == 0).astype(np.uint8), cv2.DIST_L2, 3)
    local_band = (distance <= float(max(radius * 2.1, 12))).astype(np.uint8)
    flood &= local_band
    flood &= paintable

    # Repair tiny holes / cracks created by screentone, anti-aliased gaps or
    # broken contours, but never cross strong line art.
    close_k = max(3, int(radius * 0.42))
    if close_k % 2 == 0:
        close_k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    flood = cv2.morphologyEx(flood, cv2.MORPH_CLOSE, kernel, iterations=1)
    flood = cv2.dilate(flood, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                        (max(3, radius // 2 * 2 + 1),
                                                         max(3, radius // 2 * 2 + 1))),
                       iterations=1)
    flood &= paintable
    flood |= roi_seed

    out = np.zeros((h, w), np.uint8)
    out[y0:y1, x0:x1] = np.where(flood > 0, 255, 0).astype(np.uint8)
    return out


def _outer_contour_fill_mask(source_bw_bgr: np.ndarray | None,
                             result_bgr: np.ndarray,
                             seed_binary: np.ndarray,
                             x: int, y: int, radius_px: int) -> np.ndarray:
    """Fill inside a locally simplified outer contour.

    Thin internal texture / hatch lines are stripped by morphological opening,
    while stronger outer contours remain as barriers.  Flooding the remaining
    fillable space from the stroke seed yields a much more solid block on
    fabrics, hair masses and broken-line regions.
    """
    h, w = result_bgr.shape[:2]
    if source_bw_bgr is None:
        gray_page = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY)
    else:
        src = source_bw_bgr
        if src.shape[:2] != (h, w):
            src = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)
        gray_page = src if src.ndim == 2 else cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    x = int(np.clip(x, 0, max(0, w - 1)))
    y = int(np.clip(y, 0, max(0, h - 1)))
    radius = max(4, int(radius_px))
    reach = max(radius * 4, 24)
    x0, x1 = max(0, x - reach), min(w, x + reach + 1)
    y0, y1 = max(0, y - reach), min(h, y + reach + 1)
    if x0 >= x1 or y0 >= y1:
        return np.zeros((h, w), np.uint8)
    roi_gray = gray_page[y0:y1, x0:x1]
    seed_roi = (seed_binary[y0:y1, x0:x1] > 0).astype(np.uint8)
    if not np.any(seed_roi):
        cv2.circle(seed_roi, (x - x0, y - y0), max(2, radius // 2), 1, -1)

    # Simplify the line map so thick outer contours survive while many inner
    # hatch / screentone strokes disappear.
    dark_thr = int(np.clip(np.percentile(roi_gray, 42) + 18, 150, 195))
    dark = (roi_gray < dark_thr).astype(np.uint8) * 255
    open_k = max(3, int(radius * 0.30))
    if open_k % 2 == 0:
        open_k += 1
    strong = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)),
                              iterations=1)
    strong = cv2.dilate(strong, np.ones((3, 3), np.uint8), iterations=1)
    fillable = (strong == 0).astype(np.uint8)
    num, labels = cv2.connectedComponents(fillable, connectivity=4)
    sx = int(np.clip(x - x0, 0, max(0, fillable.shape[1] - 1)))
    sy = int(np.clip(y - y0, 0, max(0, fillable.shape[0] - 1)))
    rid = int(labels[sy, sx]) if labels.size else 0
    if rid <= 0:
        ys, xs = np.nonzero(seed_roi > 0)
        if ys.size:
            rid = int(labels[int(ys[0]), int(xs[0])])
    if rid <= 0:
        return np.zeros((h, w), np.uint8)
    comp = (labels == rid).astype(np.uint8) * 255
    # Limit the contour fill to a local stroke neighbourhood so it never grows
    # to an unrelated distant area.
    distance = cv2.distanceTransform((seed_roi == 0).astype(np.uint8), cv2.DIST_L2, 3)
    comp[distance > float(max(radius * 3.1, 16))] = 0

    # Inside the simplified outer contour, keep only broadly seed-similar tone
    # zones. This removes distant but still contour-connected empty background
    # while retaining the main textured fabric / hair mass as one cohesive block.
    blur = cv2.GaussianBlur(roi_gray, (0, 0), 2.0)
    seed_vals = blur[seed_roi > 0]
    seed_tone = float(np.median(seed_vals)) if seed_vals.size else float(blur[int(np.clip(y - y0, 0, blur.shape[0] - 1)), int(np.clip(x - x0, 0, blur.shape[1] - 1))])
    tol = int(np.clip(34.0 + float(np.std(seed_vals)) * 0.45 + radius * 0.40, 34, 50))
    sim = (np.abs(blur.astype(np.float32) - seed_tone) <= float(tol)).astype(np.uint8) * 255
    comp &= sim

    comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                      (lambda k: (k, k))(max(5, int(radius * 0.60)) + (0 if max(5, int(radius * 0.60)) % 2 == 1 else 1))),
                            iterations=1)
    out = np.zeros((h, w), np.uint8)
    out[y0:y1, x0:x1] = np.where(comp > 0, 255, 0).astype(np.uint8)
    return out


def _local_tone_blob_mask(source_bw_bgr: np.ndarray | None,
                          result_bgr: np.ndarray,
                          seed_binary: np.ndarray,
                          x: int, y: int, radius_px: int) -> np.ndarray:
    """Return a contiguous local blob guided by seed tone similarity.

    This intentionally ignores many interior texture strokes.  It is designed
    for clothing/hair regions with screentone, hatching or partially open
    contours where exact line-bounded segmentation becomes a patchwork of tiny
    fragments.
    """
    h, w = result_bgr.shape[:2]
    if source_bw_bgr is None:
        gray_page = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY)
    else:
        src = source_bw_bgr
        if src.shape[:2] != (h, w):
            src = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)
        gray_page = src if src.ndim == 2 else cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    x = int(np.clip(x, 0, max(0, w - 1)))
    y = int(np.clip(y, 0, max(0, h - 1)))
    radius = max(4, int(radius_px))
    reach = max(radius * 4, 24)
    x0, x1 = max(0, x - reach), min(w, x + reach + 1)
    y0, y1 = max(0, y - reach), min(h, y + reach + 1)
    if x0 >= x1 or y0 >= y1:
        return np.zeros((h, w), np.uint8)
    roi_seed = (seed_binary[y0:y1, x0:x1] > 0).astype(np.uint8)
    if not np.any(roi_seed):
        cv2.circle(roi_seed, (x - x0, y - y0), max(2, radius // 2), 1, -1)
    roi_gray = gray_page[y0:y1, x0:x1]
    blur = cv2.GaussianBlur(roi_gray, (0, 0), 2.0)
    paintable = (roi_gray >= 28)
    if not np.any(paintable):
        return np.zeros((h, w), np.uint8)

    seed_vals = blur[roi_seed > 0]
    if seed_vals.size == 0:
        seed_vals = blur[max(0, y - y0):min(blur.shape[0], y - y0 + 1),
                         max(0, x - x0):min(blur.shape[1], x - x0 + 1)]
    seed_tone = float(np.median(seed_vals)) if seed_vals.size else float(blur[int(np.clip(y - y0, 0, blur.shape[0] - 1)), int(np.clip(x - x0, 0, blur.shape[1] - 1))])
    local_std = float(np.std(seed_vals)) if seed_vals.size else 0.0
    tol = int(np.clip(28.0 + local_std * 0.55 + radius * 0.65, 28, 62))

    dist = cv2.distanceTransform((roi_seed == 0).astype(np.uint8), cv2.DIST_L2, 3)
    local_band = dist <= float(max(radius * 3.0, 16))
    sim = np.abs(blur.astype(np.float32) - seed_tone) <= float(tol)
    allowed = sim & local_band & paintable
    close_k = max(5, int(radius * 0.75))
    if close_k % 2 == 0:
        close_k += 1
    allowed_u8 = (allowed.astype(np.uint8) * 255)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    allowed_u8 = cv2.morphologyEx(allowed_u8, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    allowed_u8 = cv2.morphologyEx(allowed_u8, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    allowed = allowed_u8 > 0
    num, labels = cv2.connectedComponents(allowed.astype(np.uint8), connectivity=4)
    sx = int(np.clip(x - x0, 0, max(0, allowed.shape[1] - 1)))
    sy = int(np.clip(y - y0, 0, max(0, allowed.shape[0] - 1)))
    rid = int(labels[sy, sx]) if labels.size else 0
    if rid <= 0:
        ys, xs = np.nonzero(roi_seed > 0)
        if ys.size:
            rid = int(labels[int(ys[0]), int(xs[0])])
    if rid <= 0:
        return np.zeros((h, w), np.uint8)
    comp = (labels == rid).astype(np.uint8) * 255
    comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    comp = cv2.dilate(comp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                      (max(3, radius // 2 * 2 + 1),
                                                       max(3, radius // 2 * 2 + 1))), iterations=1)
    comp &= (paintable.astype(np.uint8) * 255)
    out = np.zeros((h, w), np.uint8)
    out[y0:y1, x0:x1] = np.where(comp > 0, 255, 0).astype(np.uint8)
    return out


def build_cohesive_stroke_alpha(stroke_alpha: np.ndarray,
                                source_bw_bgr: np.ndarray | None,
                                result_bgr: np.ndarray,
                                *,
                                brush_radius_px: int = 18,
                                line_low: int = 75,
                                gap_close: int = 4,
                                max_region_ratio: float = 0.18) -> np.ndarray:
    """Turn a sparse bias-brush stroke into a more cohesive fill mask.

    For closed regions, this expands each touched seed to the whole refined
    line-art-bounded region.  For broken / heavily textured regions, it falls
    back to a conservative local flood+close fill.  The returned alpha keeps
    the original stroke but upgrades it from many isolated fragments to a
    visually solid painted block.
    """
    alpha = np.asarray(stroke_alpha, dtype=np.float32)
    if alpha.ndim != 2 or alpha.size == 0:
        return alpha
    seed = (alpha > 0.10).astype(np.uint8)
    if not np.any(seed):
        return alpha.copy()
    h, w = seed.shape
    expanded = np.zeros((h, w), np.uint8)
    try:
        from core.region_map import build_region_map
        from core.lineart_fill import refined_lineart_mask_at_point
        region_map = build_region_map(source_bw_bgr if source_bw_bgr is not None else result_bgr,
                                      line_low=int(line_low), gap_close=max(0, int(gap_close)))
    except Exception:
        region_map = None
        refined_lineart_mask_at_point = None  # type: ignore

    count, labels, stats, cents = cv2.connectedComponentsWithStats(seed, connectivity=8)
    local_reach = max(int(brush_radius_px) * 4, 24)
    local_window_cap = int(((local_reach * 2 + 1) ** 2) * 0.60)
    page_cap = int(min(
        max(96, seed.size * float(np.clip(max_region_ratio, 0.02, 0.8))),
        max(512, seed.size),
    ))
    candidate_cap = int(max(page_cap, local_window_cap))
    for cid in range(1, int(count)):
        area = int(stats[cid, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        cx = int(np.clip(round(float(cents[cid][0])), 0, max(0, w - 1)))
        cy = int(np.clip(round(float(cents[cid][1])), 0, max(0, h - 1)))
        component_seed = np.where(labels == cid, 255, 0).astype(np.uint8)
        region_mask = np.zeros((h, w), np.uint8)
        if region_map is not None and refined_lineart_mask_at_point is not None:
            try:
                region_mask = refined_lineart_mask_at_point(
                    source_bw_bgr if source_bw_bgr is not None else result_bgr,
                    result_bgr, cx, cy,
                    line_low=int(line_low), gap_close=max(0, int(gap_close)),
                    region_map=region_map)
            except Exception:
                region_mask = np.zeros((h, w), np.uint8)
        texture_mask = _local_texture_fill_mask(
            source_bw_bgr, result_bgr, component_seed, cx, cy,
            max(4, int(brush_radius_px)))
        contour_mask = _outer_contour_fill_mask(
            source_bw_bgr, result_bgr, component_seed, cx, cy,
            max(4, int(brush_radius_px)))
        tone_blob_mask = _local_tone_blob_mask(
            source_bw_bgr, result_bgr, component_seed, cx, cy,
            max(4, int(brush_radius_px)))

        candidate_masks = []
        for candidate in (region_mask, texture_mask, contour_mask, tone_blob_mask):
            area_candidate = int(np.count_nonzero(candidate))
            if 24 <= area_candidate <= candidate_cap:
                candidate_masks.append((area_candidate, candidate))
        if candidate_masks:
            # Prefer the most cohesive reasonable area rather than tiny exact
            # fragments: on textured or unclosed regions the larger candidate is
            # usually the one that looks like a properly filled painted block.
            candidate_masks.sort(key=lambda item: item[0])
            expanded |= np.where(candidate_masks[-1][1] > 0, 255, 0).astype(np.uint8)
        else:
            expanded |= component_seed

    expanded |= np.where(seed > 0, 255, 0).astype(np.uint8)
    fill_alpha = _soft_fill_from_binary(expanded, feather=max(2, int(round(brush_radius_px * 0.35))),
                                        floor=0.84)
    return np.maximum(alpha, fill_alpha).astype(np.float32)
