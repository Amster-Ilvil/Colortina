"""Seeded editable selection derived from mc-v2's own colour grouping.

The model often colours one semantic object coherently even when it chooses the
wrong hue.  A click inside that object can therefore recover a much more
complete mask than flood-filling the monochrome scan.  A pre-existing rough
lasso/rectangle can be supplied as ``limit_mask``; this is the recommended
workflow for hair touching similarly coloured backgrounds.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class SemanticSelectionSettings:
    max_work_edge: int = 800
    hue_tolerance: float = 18.0
    relaxed_hue_tolerance: float = 29.0
    strict_saturation_floor: int = 165
    relaxed_saturation_floor: int = 138
    lab_tolerance: float = 105.0
    relaxed_lab_tolerance: float = 145.0
    close_px: int = 9
    grow_px: int = 3
    grow_iterations: int = 6
    line_threshold: int = 38
    thick_line_radius: float = 2.6
    max_fraction: float = 0.36
    max_hole_area: int = 900
    protect_skin_like: bool = True
    skin_saturation_ceiling: int = 155
    skin_dilate_px: int = 2
    edge_expand_px: int = 2
    # Final full-resolution collar pass.  The low-resolution semantic mask can
    # otherwise miss antialiased warm pixels around thin bangs and long hair.
    # Growth is strictly local and candidate-gated, so it cannot jump back to
    # a rejected panel rail or flood the face.
    edge_refine_iterations: int = 8
    edge_refine_radius: int = 2
    edge_hue_tolerance: float = 34.0
    edge_saturation_floor: int = 96
    lineart_edge_threshold: int = 105
    # Full-object completion closes narrow highlight cuts and reconnects thin
    # bangs/tips, but only through locally plausible hair/object support.
    complete_visible_shape: bool = True
    completion_gap_px: int = 18
    completion_hole_area: int = 2600
    completion_highlight_gray_min: int = 150
    # Lettering/caption regions are logical occluders, not terminal barriers.
    # The final renderer restores them above the recoloured layer.
    complete_text_occlusions: bool = True
    occlusion_dilate_px: int = 3
    occlusion_entry_px: int = 4
    occlusion_bridge_px: int = 72
    occlusion_max_added_ratio: float = 1.35
    reject_panel_rules: bool = True
    panel_rule_aspect: float = 5.0
    # Hair-object mode rebuilds the selection from the dominant mc-v2 hair
    # colour, a page-local person/skin scope, trapped-ball-like line regions and
    # text occlusion components.  The resulting object mask, not the original
    # visible colour fragment, becomes the direct black-hair control mask.
    complete_hair_object: bool = False
    hair_object_scope_px: int = 92
    hair_object_gap_px: int = 96
    hair_object_line_regions: bool = True
    hair_object_max_ratio: float = 2.15


def _component_from_seed(binary: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    binary = (binary > 0).astype(np.uint8)
    n, labels, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.zeros_like(binary)
    x, y = seed
    x = int(np.clip(x, 0, binary.shape[1] - 1))
    y = int(np.clip(y, 0, binary.shape[0] - 1))
    label = int(labels[y, x])
    if label == 0:
        for radius in (2, 4, 8, 16, 32):
            y0, y1 = max(0, y-radius), min(binary.shape[0], y+radius+1)
            x0, x1 = max(0, x-radius), min(binary.shape[1], x+radius+1)
            yy, xx = np.where(binary[y0:y1, x0:x1] > 0)
            if yy.size:
                d = (xx + x0 - x) ** 2 + (yy + y0 - y) ** 2
                k = int(np.argmin(d))
                label = int(labels[yy[k] + y0, xx[k] + x0])
                break
    return (labels == label).astype(np.uint8) if label else np.zeros_like(binary)


def _fill_small_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    work = (mask > 0).astype(np.uint8)
    inv = (work == 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    h, w = work.shape
    for label in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[label]]
        touches_edge = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        if not touches_edge and area <= max_area:
            work[labels == label] = 1
    return work


def _normalise_mask(mask: np.ndarray | None,
                    shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.zeros(shape, np.uint8)
    work = np.asarray(mask)
    if work.ndim == 3:
        work = np.max(work, axis=2)
    if work.shape != shape:
        work = cv2.resize(work.astype(np.uint8), (shape[1], shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    return np.where(work > 0, 1, 0).astype(np.uint8)


def _directional_close(mask: np.ndarray, gap_px: int) -> np.ndarray:
    """Close short horizontal/vertical/diagonal breaks without a round blob."""
    binary = (mask > 0).astype(np.uint8)
    gap = max(3, int(gap_px)) | 1
    thin = max(3, int(round(gap * 0.18))) | 1
    kernels = [
        np.ones((thin, gap), np.uint8),
        np.ones((gap, thin), np.uint8),
    ]
    diag = np.zeros((gap, gap), np.uint8)
    band = max(1, thin // 2)
    for offset in range(-band, band + 1):
        yy = np.arange(gap)
        xx = yy + offset
        valid = (xx >= 0) & (xx < gap)
        diag[yy[valid], xx[valid]] = 1
    kernels.extend([diag, np.fliplr(diag).copy()])
    closed = binary.copy()
    for kernel in kernels:
        closed |= cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return closed


def _complete_visible_shape(mask: np.ndarray, candidate: np.ndarray,
                            gray: np.ndarray, skin_mask: np.ndarray,
                            limit_mask: np.ndarray,
                            cfg: SemanticSelectionSettings):
    """Recover highlight-separated strands, fine tips and enclosed hair gaps."""
    visible = (mask > 0)
    if not cfg.complete_visible_shape or not np.any(visible):
        return visible.astype(np.uint8), 0, 0
    h, w = visible.shape
    px_scale = max(0.75, max(h, w) / 1600.0)
    gap = max(5, int(round(cfg.completion_gap_px * px_scale)))
    closed = _directional_close(visible.astype(np.uint8), gap) > 0
    near = cv2.dilate(
        visible.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (gap * 2 + 1, gap * 2 + 1)),
        iterations=1) > 0
    highlight = (gray >= int(cfg.completion_highlight_gray_min)) & near
    support = ((candidate > 0) | highlight) & (skin_mask == 0) & (limit_mask > 0)
    before = int(np.count_nonzero(visible))
    visible |= closed & support

    max_hole = max(32, int(round(cfg.completion_hole_area * px_scale * px_scale)))
    filled = _fill_small_holes(visible.astype(np.uint8), max_hole) > 0
    holes_added = filled & ~visible & (skin_mask == 0) & (limit_mask > 0)
    visible |= holes_added
    added = int(np.count_nonzero(visible) - before)
    return visible.astype(np.uint8), added, int(np.count_nonzero(holes_added))


def _bridge_text_occlusions(mask: np.ndarray, candidate: np.ndarray,
                            occlusion_mask: np.ndarray, skin_mask: np.ndarray,
                            limit_mask: np.ndarray,
                            cfg: SemanticSelectionSettings):
    """Geodesically continue a semantic object through local text occluders."""
    visible = (mask > 0)
    occ = (occlusion_mask > 0) & (skin_mask == 0) & (limit_mask > 0)
    if (not cfg.complete_text_occlusions or not np.any(visible)
            or not np.any(occ)):
        return visible.astype(np.uint8), 0, 0
    h, w = visible.shape
    px_scale = max(0.75, max(h, w) / 1600.0)
    entry = max(1, int(round(cfg.occlusion_entry_px * px_scale)))
    max_steps = max(entry + 1, int(round(cfg.occlusion_bridge_px * px_scale)))
    distance_from_visible = cv2.distanceTransform(
        (~visible).astype(np.uint8), cv2.DIST_L2, 5)
    near_limit = distance_from_visible <= float(max_steps)
    occ &= near_limit
    entry_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (entry * 2 + 1, entry * 2 + 1))
    path = (cv2.dilate(visible.astype(np.uint8), entry_kernel) > 0) & occ
    if not np.any(path):
        return visible.astype(np.uint8), 0, 0

    allowed = ((candidate > 0) | occ) & near_limit & (skin_mask == 0) & (limit_mask > 0)
    count, labels = cv2.connectedComponents(allowed.astype(np.uint8), connectivity=8)
    if count <= 1:
        return visible.astype(np.uint8), 0, 0
    touched = np.unique(labels[path])
    touched = touched[touched > 0]
    if touched.size == 0:
        return visible.astype(np.uint8), 0, 0
    path = np.isin(labels, touched) & allowed

    additions = path & ~visible
    before = max(1, int(np.count_nonzero(visible)))
    max_added = max(64, int(round(before * float(cfg.occlusion_max_added_ratio))))
    if int(np.count_nonzero(additions)) > max_added:
        # Keep only the logical lettering pixels when the candidate side would
        # otherwise balloon into a nearby same-colour background.
        additions &= occ
    visible |= additions
    return (visible.astype(np.uint8),
            int(np.count_nonzero(additions)),
            int(np.count_nonzero(additions & occ)))


def _prune_full_resolution_panel_branches(mask: np.ndarray,
                                          seed_xy: tuple[int, int],
                                          enabled: bool = True):
    """Remove thin horizontal panel-rail branches still attached to hair.

    A large character can overlap a neighbouring panel border.  At work scale,
    the border and hair may become one colour component.  Detect only extremely
    thin, long horizontal skeletons well above the clicked seed, remove a short
    downward band around them, then retake the seed component.  Irregular bangs
    are much thicker and do not satisfy this gate.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if not enabled or not np.any(binary):
        return binary, 0
    h, w = binary.shape
    sx = int(np.clip(seed_xy[0], 0, w - 1))
    sy = int(np.clip(seed_xy[1], 0, h - 1))
    kernel_w = max(80, int(round(w * 0.085)))
    horizontal = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, np.ones((3, kernel_w), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        horizontal, connectivity=8)
    remove = np.zeros_like(binary)
    minimum_above = max(30, int(round(h * 0.020)))
    downward = max(42, int(round(h * 0.025)))
    side_pad = max(20, int(round(w * 0.022)))
    for label in range(1, count):
        x, y, bw, bh, pixels = [int(v) for v in stats[label]]
        aspect = bw / max(1.0, float(bh))
        if (bw < kernel_w or pixels < kernel_w or aspect < 12.0 or
                bh > max(6, int(round(h * 0.008))) or
                y + bh >= sy - minimum_above):
            continue
        x0 = max(0, x - side_pad)
        x1 = min(w, x + bw + side_pad)
        y0 = max(0, y - 5)
        y1 = min(sy - minimum_above + 1, y + downward)
        if y1 > y0:
            remove[y0:y1, x0:x1] = 1
    if not np.any(remove):
        return binary, 0
    before = int(np.count_nonzero(binary))
    binary[remove > 0] = 0
    binary = _component_from_seed(binary, (sx, sy))
    return binary, max(0, before - int(np.count_nonzero(binary)))



def _dominant_hair_hsv(colour_bgr: np.ndarray, seed_mask: np.ndarray,
                       seed_xy: tuple[int, int]) -> tuple[float, float, float]:
    hsv = cv2.cvtColor(colour_bgr, cv2.COLOR_BGR2HSV)
    active = (seed_mask > 0) & (hsv[..., 1] >= 70) & (hsv[..., 2] > 24)
    if np.count_nonzero(active) >= 16:
        hues = hsv[..., 0][active].astype(np.int32)
        hue = float(np.bincount(hues, minlength=180).argmax())
        return hue, float(np.median(hsv[..., 1][active])), float(np.median(hsv[..., 2][active]))
    x = int(np.clip(seed_xy[0], 0, colour_bgr.shape[1] - 1))
    y = int(np.clip(seed_xy[1], 0, colour_bgr.shape[0] - 1))
    px = hsv[y, x]
    return float(px[0]), float(px[1]), float(px[2])


def _full_skin_exclusion(original_bgr: np.ndarray, colour_bgr: np.ndarray,
                         limit_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return broad protected skin and a conservative normal-skin core.

    This mirrors the selector's existing negative skin gate, while preserving a
    separate normal-skin component map used to estimate the clicked character's
    local instance extent.  It is intentionally colour-result based: mc-v2's
    skin prior is more stable than the source manga tone alone.
    """
    h, w = colour_bgr.shape[:2]
    scale = min(1.0, 800.0 / max(h, w))
    ww, wh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    col = cv2.resize(colour_bgr, (ww, wh), interpolation=cv2.INTER_AREA)
    src = cv2.resize(original_bgr, (ww, wh), interpolation=cv2.INTER_AREA)
    limit = cv2.resize((limit_mask > 0).astype(np.uint8), (ww, wh),
                       interpolation=cv2.INTER_NEAREST)
    hsv = cv2.cvtColor(col, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1]
    yy, cr, cb = cv2.split(cv2.cvtColor(col, cv2.COLOR_BGR2YCrCb))
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    dark = (gray < 38).astype(np.uint8)
    thick_dark = cv2.distanceTransform(dark, cv2.DIST_L2, 5) > 2.6

    normal = ((yy > 160) & (yy < 252) &
              (cr > 124) & (cr < 184) &
              (cb > 62) & (cb < 134) &
              (sat >= 12) & (sat < 190) &
              (limit > 0) & (~thick_dark)).astype(np.uint8)
    normal = cv2.morphologyEx(
        normal, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    muted = ((yy > 80) & (yy < 248) &
             (cr > 118) & (cr < 190) &
             (cb > 55) & (cb < 155) &
             (sat >= 18) & (sat < 155))
    vivid = ((yy > 145) & (yy < 252) &
             (cr > 122) & (cr < 190) &
             (cb > 48) & (cb < 150) &
             (sat >= 45))
    protected = ((muted | vivid) & (limit > 0) & (~thick_dark)).astype(np.uint8)
    protected = cv2.morphologyEx(
        protected, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    protected = cv2.dilate(protected, np.ones((5, 5), np.uint8), iterations=1)
    return (
        cv2.resize(protected, (w, h), interpolation=cv2.INTER_NEAREST),
        cv2.resize(normal, (w, h), interpolation=cv2.INTER_NEAREST),
    )


def _source_panel_top_guard(original_bgr: np.ndarray, seed_xy: tuple[int, int],
                            skin_top: int, default_top: int) -> int:
    """Move an instance top below a long manga panel rule when one is present."""
    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    sx, sy = seed_xy
    edges = cv2.Canny(gray, 40, 120)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0,
        threshold=max(30, int(w * 0.055)),
        minLineLength=max(60, int(w * 0.18)), maxLineGap=12)
    best = None
    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            if abs(int(y2) - int(y1)) > max(3, int(round(h * 0.0025))):
                continue
            xa, xb = sorted((int(x1), int(x2)))
            length = xb - xa
            yy = int(round((int(y1) + int(y2)) * 0.5))
            # A page can contain several unrelated horizontal rules at similar
            # heights. Only a rule crossing, or lying close to, the clicked
            # hair instance may define its top boundary.
            x_near = (xa <= sx <= xb or
                      min(abs(sx - xa), abs(sx - xb)) <= int(w * 0.08))
            if (not x_near or length < int(w * 0.18)
                    or yy >= min(sy, skin_top)):
                continue
            if yy < skin_top - int(h * 0.16):
                continue
            if best is None or yy > best:
                best = yy
    if best is None:
        return int(default_top)
    return max(int(default_top), int(best + max(6, round(h * 0.010))))


def _hair_instance_scope(original_bgr: np.ndarray, colour_bgr: np.ndarray,
                         core_mask: np.ndarray, limit_mask: np.ndarray,
                         seed_xy: tuple[int, int], normal_skin: np.ndarray):
    """Build a page-local person scope, optionally refined by anime-segmentation."""
    h, w = core_mask.shape
    scope = (limit_mask > 0).astype(np.uint8)
    used_character_model = False

    # SkyTNT anime-segmentation is already an optional Colortina dependency.
    # Its foreground mask is used only as a scope, never as the hair classifier.
    try:
        from core.character_scope import character_likelihood
        likelihood = character_likelihood(colour_bgr)
        if likelihood is not None and likelihood.shape == (h, w):
            foreground = (likelihood >= 0.10).astype(np.uint8)
            foreground = cv2.dilate(
                foreground, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
            touched = cv2.dilate((core_mask > 0).astype(np.uint8),
                                 np.ones((15, 15), np.uint8)) > 0
            count, labels = cv2.connectedComponents(foreground, connectivity=8)
            ids = np.unique(labels[touched])
            ids = ids[ids > 0]
            if ids.size:
                foreground = np.isin(labels, ids).astype(np.uint8)
                refined_scope = scope & foreground
                if np.any(refined_scope):
                    scope = refined_scope
                    used_character_model = True
    except Exception:
        used_character_model = False

    # Heuristic instance fallback: choose the substantial normal-skin component
    # nearest the clicked hair. Its upper edge is a robust proxy for face/head
    # location and prevents same-colour panel rails above the character joining.
    count, labels, stats, centres = cv2.connectedComponentsWithStats(
        (normal_skin > 0).astype(np.uint8), connectivity=8)
    sx, sy = seed_xy
    near = cv2.dilate((core_mask > 0).astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (121, 121))) > 0
    best_id = 0
    best_score = -1e30
    page_area = max(1, h * w)
    for label in range(1, count):
        x, y, bw, bh, area = [int(v) for v in stats[label]]
        if area < max(80, int(page_area * 0.0002)) or area > int(page_area * 0.22):
            continue
        component = labels == label
        contact = int(np.count_nonzero(component & near))
        cx, cy = centres[label]
        distance = float((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5
        score = float(area) + contact * 24.0 - distance * 28.0
        if score > best_score:
            best_score = score
            best_id = label
    top_guard = 0
    skin_bbox = None
    if best_id > 0:
        x, y, bw, bh, _area = [int(v) for v in stats[best_id]]
        skin_bbox = (x, y, bw, bh)
        # Leave enough room above the detected face for bangs/crown. The old
        # 0.15-width margin cut off the upper 70-100 px of large manga hair.
        top_guard = max(0, int(round(y - max(40.0, bw * 0.22))))
        top_guard = _source_panel_top_guard(
            original_bgr, seed_xy, y, top_guard)
        scope[:top_guard, :] = 0

    return scope.astype(np.uint8), {
        "character_model_used": bool(used_character_model),
        "skin_instance_bbox": skin_bbox,
        "instance_top_guard": int(top_guard),
    }


def _absorb_lineart_hair_regions(original_bgr: np.ndarray,
                                  colour_bgr: np.ndarray,
                                  current_mask: np.ndarray,
                                  strong_hair: np.ndarray,
                                  skin_mask: np.ndarray,
                                  scope_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Conservatively absorb enclosed line regions belonging to the hair.

    This follows trapped-ball/region-graph ideas used by line-art filling tools:
    closed regions are classified as units instead of growing a round pixel
    blob.  It fills white highlight islands and anti-aliased strand fragments
    while rejecting page-wide/background regions.
    """
    h, w = current_mask.shape
    scale = min(1.0, 1050.0 / max(h, w))
    ww, wh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    src = cv2.resize(original_bgr, (ww, wh), interpolation=cv2.INTER_AREA)
    colour = cv2.resize(colour_bgr, (ww, wh), interpolation=cv2.INTER_AREA)
    mask = cv2.resize((current_mask > 0).astype(np.uint8), (ww, wh),
                      interpolation=cv2.INTER_NEAREST)
    strong = cv2.resize((strong_hair > 0).astype(np.uint8), (ww, wh),
                        interpolation=cv2.INTER_NEAREST)
    skin = cv2.resize((skin_mask > 0).astype(np.uint8), (ww, wh),
                      interpolation=cv2.INTER_NEAREST)
    scope = cv2.resize((scope_mask > 0).astype(np.uint8), (ww, wh),
                       interpolation=cv2.INTER_NEAREST)
    try:
        from core.region_map import build_region_map
        region_map = build_region_map(src, line_low=75, gap_close=4)
    except Exception:
        return current_mask, 0
    labels = region_map.labels
    n = int(labels.max()) + 1
    area = np.bincount(labels.ravel(), minlength=n).astype(np.float64)
    overlap = np.bincount(
        labels.ravel(), weights=(mask > 0).ravel().astype(np.float64), minlength=n)
    warm = np.bincount(
        labels.ravel(), weights=(strong > 0).ravel().astype(np.float64), minlength=n)
    skin_count = np.bincount(
        labels.ravel(), weights=(skin > 0).ravel().astype(np.float64), minlength=n)
    scope_count = np.bincount(
        labels.ravel(), weights=(scope > 0).ravel().astype(np.float64), minlength=n)
    near = cv2.dilate(mask, np.ones((29, 29), np.uint8), iterations=1) > 0
    near_count = np.bincount(
        labels.ravel(), weights=near.ravel().astype(np.float64), minlength=n)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    highlights = gray >= 165
    highlight_count = np.bincount(
        labels.ravel(), weights=highlights.ravel().astype(np.float64), minlength=n)
    denom = np.maximum(area, 1.0)
    overlap_ratio = overlap / denom
    warm_ratio = warm / denom
    skin_ratio = skin_count / denom
    scope_ratio = scope_count / denom
    near_ratio = near_count / denom
    highlight_ratio = highlight_count / denom
    page = float(labels.size)
    eligible_size = (area >= 6) & (area <= page * 0.025)
    safe = eligible_size & (scope_ratio >= 0.82) & (skin_ratio < 0.48)
    keep = safe & (
        ((overlap >= 8) & (overlap_ratio >= 0.16)) |
        ((warm_ratio >= 0.58) & (near_ratio >= 0.22)) |
        ((highlight_ratio >= 0.72) & (near_ratio >= 0.62) &
         (area <= page * 0.006)))
    keep[0] = False
    regions = keep[labels]
    candidate = ((mask > 0) | regions).astype(np.uint8)
    # Only retain components linked to the existing object.
    count, components = cv2.connectedComponents(candidate, connectivity=8)
    touched = np.unique(components[mask > 0])
    touched = touched[touched > 0]
    candidate = np.isin(components, touched).astype(np.uint8)
    full = cv2.resize(candidate * 255, (w, h), interpolation=cv2.INTER_NEAREST)
    full[(scope_mask <= 0) | (skin_mask > 0)] = 0
    full[current_mask > 0] = 255
    return full, max(0, int(np.count_nonzero(full) - np.count_nonzero(current_mask)))




def _filter_hair_occlusion_components(occlusion: np.ndarray,
                                      support: np.ndarray) -> np.ndarray:
    """Keep lettering components actually embedded in the hair object.

    A nearby caption/gutter touching hair on only one side is not an occluder of
    that hair. Real text drawn over hair normally has colour/line support on two
    opposing sides (left/right or top/bottom). This component-level test keeps
    those bridges while rejecting adjacent blank caption paper.
    """
    occ = (np.asarray(occlusion) > 0).astype(np.uint8)
    support = (np.asarray(support) > 0).astype(np.uint8)
    if not np.any(occ) or not np.any(support):
        return np.zeros_like(occ)
    h, w = occ.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        occ, connectivity=8)
    out = np.zeros_like(occ)
    scale = max(1.0, max(h, w) / 1000.0)
    ring_px = max(3, int(round(7 * scale)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ring_px * 2 + 1, ring_px * 2 + 1))
    for label in range(1, count):
        x, y, bw, bh, area = [int(v) for v in stats[label]]
        if area <= 0:
            continue
        component = (labels == label).astype(np.uint8)
        ring = (cv2.dilate(component, kernel, iterations=1) > 0) & (component == 0)
        contact = ring & (support > 0)
        yy, xx = np.where(contact)
        if xx.size < max(4, int(round(area * 0.008))):
            continue
        margin_x = max(2, int(round(bw * 0.28)))
        margin_y = max(2, int(round(bh * 0.28)))
        left = bool(np.any(xx <= x + margin_x))
        right = bool(np.any(xx >= x + bw - 1 - margin_x))
        top = bool(np.any(yy <= y + margin_y))
        bottom = bool(np.any(yy >= y + bh - 1 - margin_y))
        if (left and right) or (top and bottom):
            out[component > 0] = 1
    return out


def build_complete_hair_object_mask(original_bgr: np.ndarray,
                                    colour_bgr: np.ndarray,
                                    visible_mask: np.ndarray,
                                    seed_xy: tuple[int, int] | None = None,
                                    *, limit_mask: np.ndarray | None = None,
                                    occlusion_mask: np.ndarray | None = None,
                                    use_line_regions: bool = True,
                                    max_ratio: float = 2.15):
    """Rebuild a complete page-local hair object from a visible seed mask.

    The final mask includes colour-coherent strands, enclosed/highlight regions
    and local text occluders.  Face/hand skin, other character instances and
    horizontal panel structures remain negative evidence.  This mask is safe to
    use directly for in-generator black-hair control and final compositing.
    """
    original = np.asarray(original_bgr)
    colour = np.asarray(colour_bgr)
    if original.ndim == 2:
        original = cv2.cvtColor(original.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if colour.ndim == 2:
        colour = cv2.cvtColor(colour.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if colour.shape[:2] != original.shape[:2]:
        colour = cv2.resize(colour, (original.shape[1], original.shape[0]),
                            interpolation=cv2.INTER_AREA)
    h, w = original.shape[:2]
    visible = _normalise_mask(visible_mask, (h, w))
    if not np.any(visible):
        return visible.astype(np.uint8) * 255, {"enabled": True, "added_pixels": 0}
    if seed_xy is None:
        dist = cv2.distanceTransform(visible.astype(np.uint8), cv2.DIST_L2, 5)
        iy, ix = np.unravel_index(int(np.argmax(dist)), dist.shape)
        seed_xy = (int(ix), int(iy))
    sx = int(np.clip(seed_xy[0], 0, w - 1))
    sy = int(np.clip(seed_xy[1], 0, h - 1))
    limit = (np.ones((h, w), np.uint8) if limit_mask is None else
             _normalise_mask(limit_mask, (h, w)))
    protected_skin, normal_skin = _full_skin_exclusion(original, colour, limit)
    hue, target_sat, target_val = _dominant_hair_hsv(colour, visible, (sx, sy))
    hsv = cv2.cvtColor(colour, cv2.COLOR_BGR2HSV)
    hh, ss, vv = cv2.split(hsv)
    hue_delta = np.minimum(
        np.abs(hh.astype(np.float32) - hue),
        180.0 - np.abs(hh.astype(np.float32) - hue))
    _yy, cr, _cb = cv2.split(cv2.cvtColor(colour, cv2.COLOR_BGR2YCrCb))
    saturation_floor = max(78.0, min(150.0, target_sat * 0.48))
    strong = ((hue_delta <= 19.0) & (ss >= saturation_floor) & (vv > 28) &
              (limit > 0) &
              ((protected_skin == 0) | (cr >= 195) |
               ((ss >= 205) & (hue_delta <= 10.0))))

    # Remove long horizontal colour structures unless supported by a vertical
    # object run. This rejects wooden rails/panel bars that share the hair hue.
    horizontal = cv2.morphologyEx(
        strong.astype(np.uint8), cv2.MORPH_OPEN,
        np.ones((5, max(45, int(round(w * 0.055)))), np.uint8)) > 0
    vertical = cv2.morphologyEx(
        strong.astype(np.uint8), cv2.MORPH_OPEN,
        np.ones((max(25, int(round(h * 0.018))), 5), np.uint8)) > 0
    vertical_near = cv2.dilate(
        vertical.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    strong &= ~(horizontal & ~vertical_near)

    core = _component_from_seed(strong.astype(np.uint8), (sx, sy)) > 0
    if np.count_nonzero(core) < 64:
        core = (visible > 0) & strong
    if np.count_nonzero(core) < 64:
        core = visible > 0

    scope, scope_diag = _hair_instance_scope(
        original, colour, core.astype(np.uint8), limit, (sx, sy), normal_skin)
    core &= scope > 0
    if not np.any(core):
        core = (visible > 0) & (scope > 0)

    px_scale = max(0.75, max(h, w) / 1600.0)
    gap = max(17, int(round(96 * px_scale))) | 1
    radius = max(16, int(round(92 * px_scale)))
    directional = _directional_close(core.astype(np.uint8), gap) > 0
    near = cv2.dilate(
        core.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (radius * 2 + 1, radius * 2 + 1))) > 0
    envelope = (directional | near) & (scope > 0)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    # Bright regions are accepted only when directional object closure says
    # they lie between hair supports. A simple proximity gate also selected
    # blank gutters/caption paper sitting just above the hair crown.
    highlight = ((gray >= 145) & directional & near &
                 (protected_skin == 0))

    if occlusion_mask is None:
        try:
            from core.text_guard import text_occlusion_mask
            logical_occ = text_occlusion_mask(
                original, dilate_px=3,
                vicinity_mask=(core.astype(np.uint8) * 255), use_fallback=True)
        except Exception:
            logical_occ = None
    else:
        logical_occ = occlusion_mask
    logical_occ = (_normalise_mask(logical_occ, (h, w)) > 0)
    logical_occ &= envelope & (scope > 0) & (protected_skin == 0)
    logical_occ = _filter_hair_occlusion_components(
        logical_occ, strong | core) > 0

    allowed = (strong | highlight | logical_occ) & envelope & (scope > 0)
    count, labels = cv2.connectedComponents(allowed.astype(np.uint8), connectivity=8)
    touched = cv2.dilate(core.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    ids = np.unique(labels[touched])
    ids = ids[ids > 0]
    result = core | (np.isin(labels, ids) & allowed)

    # Recover dark strand boundaries locally without opening a path through
    # page-wide panel lines.
    line_candidate = ((gray < 115) & envelope & (scope > 0) &
                      (protected_skin == 0))
    for _ in range(max(3, int(round(8 * px_scale)))):
        adjacent = cv2.dilate(result.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        result |= adjacent & line_candidate

    before_regions = int(np.count_nonzero(result))
    line_region_added = 0
    if use_line_regions:
        region_result, line_region_added = _absorb_lineart_hair_regions(
            original, colour, result.astype(np.uint8) * 255,
            strong.astype(np.uint8), protected_skin, scope)
        result = region_result > 0

    max_hole = max(96, int(round(12000 * px_scale * px_scale)))
    result = _fill_small_holes(result.astype(np.uint8), max_hole) > 0
    result &= (scope > 0)
    result &= ((protected_skin == 0) | strong)
    result |= core
    result, panel_removed = _prune_full_resolution_panel_branches(
        result.astype(np.uint8) * 255, (sx, sy), enabled=True)
    result = result > 0

    visible_count = max(1, int(np.count_nonzero(visible)))
    result_count = int(np.count_nonzero(result))
    if result_count > int(round(visible_count * max(1.10, float(max_ratio)))):
        # Fall back to the rebuilt colour core, not the original visible mask:
        # the latter may contain the very panel rail / neighbouring object this
        # complete-object pass was asked to remove.
        result = core & (scope > 0)
        result_count = int(np.count_nonzero(result))
        line_region_added = 0
    output = np.where(result, 255, 0).astype(np.uint8)
    diagnostics = {
        "enabled": True,
        "target_hue": float(hue),
        "target_saturation": float(target_sat),
        "target_value": float(target_val),
        "visible_pixels": int(visible_count),
        "selected_pixels": int(result_count),
        "added_pixels": int(np.count_nonzero(result & ~(visible > 0))),
        "removed_visible_pixels": int(np.count_nonzero((visible > 0) & ~result)),
        "strong_hair_pixels": int(np.count_nonzero(strong)),
        "text_occlusion_pixels": int(np.count_nonzero(logical_occ)),
        "line_region_added_pixels": int(line_region_added),
        "pre_line_region_pixels": int(before_regions),
        "panel_removed_pixels": int(panel_removed),
        **scope_diag,
    }
    return output, diagnostics

def select_semantic_region(original_bgr: np.ndarray,
                           colorized_bgr: np.ndarray,
                           seed_xy: tuple[int, int],
                           settings: SemanticSelectionSettings | None = None,
                           limit_mask: np.ndarray | None = None,
                           occlusion_mask: np.ndarray | None = None,
                           complete_hair_object: bool | None = None):
    """Return a binary full-resolution selection and diagnostics.

    ``limit_mask`` is a rough user lasso/rectangle. It does not need to follow
    hair edges; it merely prevents a same-coloured wall/panel from joining the
    clicked object.  Colour affinity and original line-art barriers determine
    the final detailed boundary inside it.
    """
    cfg = settings or SemanticSelectionSettings()
    original = np.asarray(original_bgr)
    colour = np.asarray(colorized_bgr)
    if original.ndim == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    if colour.ndim == 2:
        colour = cv2.cvtColor(colour, cv2.COLOR_GRAY2BGR)
    if colour.shape[:2] != original.shape[:2]:
        colour = cv2.resize(colour, (original.shape[1], original.shape[0]),
                            interpolation=cv2.INTER_AREA)

    h, w = original.shape[:2]
    scale = min(1.0, float(cfg.max_work_edge) / max(h, w))
    ww, wh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    src = cv2.resize(original, (ww, wh), interpolation=cv2.INTER_AREA)
    col = cv2.resize(colour, (ww, wh), interpolation=cv2.INTER_AREA)
    sx = int(np.clip(round(seed_xy[0] * scale), 0, ww - 1))
    sy = int(np.clip(round(seed_xy[1] * scale), 0, wh - 1))

    if limit_mask is None:
        limit = np.ones((wh, ww), np.uint8)
    else:
        raw = np.asarray(limit_mask)
        if raw.ndim == 3:
            raw = np.max(raw, axis=2)
        limit = cv2.resize((raw > 0).astype(np.uint8), (ww, wh),
                           interpolation=cv2.INTER_NEAREST)
        if not limit[sy, sx]:
            limit[sy, sx] = 1

    hsv = cv2.cvtColor(col, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(col, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    radius = max(2, int(round(7 * scale)))
    y0, y1 = max(0, sy-radius), min(wh, sy+radius+1)
    x0, x1 = max(0, sx-radius), min(ww, sx+radius+1)
    patch_hsv = hsv[y0:y1, x0:x1]
    patch_lab = lab[y0:y1, x0:x1]
    valid = (patch_hsv[..., 1] > 80) & (gray[y0:y1, x0:x1] > cfg.line_threshold)
    if np.any(valid):
        target_hsv = np.median(patch_hsv[valid], axis=0)
        target_lab = np.median(patch_lab[valid], axis=0)
    else:
        target_hsv = hsv[sy, sx].astype(np.float32)
        target_lab = lab[sy, sx]

    hue = hsv[..., 0].astype(np.float32)
    sat = hsv[..., 1].astype(np.float32)
    val = hsv[..., 2].astype(np.float32)
    hue_delta = np.minimum(np.abs(hue - target_hsv[0]),
                           180.0 - np.abs(hue - target_hsv[0]))
    delta = lab - target_lab[None, None, :]
    lab_distance = np.sqrt((delta[..., 0] * 0.30) ** 2 +
                           delta[..., 1] ** 2 + delta[..., 2] ** 2)

    # Only thick black masses are hard barriers. Thin hair-strand lines are
    # intentionally bridgeable, while panel frames and clothing blocks stop
    # propagation.
    dark = (gray < cfg.line_threshold).astype(np.uint8)
    thick_dark = cv2.distanceTransform(dark, cv2.DIST_L2, 5) > cfg.thick_line_radius

    target_sat = float(target_hsv[1])
    strict_floor = max(float(cfg.strict_saturation_floor), target_sat * 0.66)
    relaxed_floor = max(float(cfg.relaxed_saturation_floor), target_sat * 0.54)
    strict = ((hue_delta <= cfg.hue_tolerance) &
              (sat >= strict_floor) & (val > 25) &
              (lab_distance <= cfg.lab_tolerance) &
              (limit > 0) & (~thick_dark))
    relaxed = ((hue_delta <= cfg.relaxed_hue_tolerance) &
               (sat >= relaxed_floor) & (val > 18) &
               (lab_distance <= cfg.relaxed_lab_tolerance) &
               (limit > 0) & (~thick_dark))

    # Anime faces often contain saturated blush/hatching whose hue happens to
    # match red/orange hair. Build a broad low-saturation skin core from the
    # model result and expand it slightly before semantic propagation. This is
    # not used to *find* hair; it is only a negative guard, so blue/green hair
    # and ordinary backgrounds are unaffected. The clicked seed is restored
    # afterwards in case it lies directly on an antialiased boundary.
    skin_exclusion = np.zeros_like(limit, dtype=bool)
    if cfg.protect_skin_like:
        ycrcb = cv2.cvtColor(col, cv2.COLOR_BGR2YCrCb)
        yy, cr, cb = cv2.split(ycrcb)
        # mc-v2 often renders skin with very high saturation (especially
        # blush/hatching), so a saturation ceiling alone misses the exact
        # pixels that most easily join warm red/orange hair.  In its output,
        # skin is primarily separated from warm hair by luminance and Cr:
        # faces/hands are light with moderate Cr, while orange/red hair keeps a
        # much stronger Cr response.  Keep the older low-saturation gate as a
        # fallback for muted skin, but add this high-saturation light-skin gate.
        muted_skin = ((yy > 80) & (yy < 248) &
                      (cr > 118) & (cr < 190) &
                      (cb > 55) & (cb < 155) &
                      (sat >= 18) &
                      (sat < float(cfg.skin_saturation_ceiling)))
        vivid_skin = ((yy > 145) & (yy < 252) &
                      (cr > 122) & (cr < 190) &
                      (cb > 48) & (cb < 150) &
                      (sat >= 45))
        skin_core = ((muted_skin | vivid_skin) &
                     (limit > 0) & (~thick_dark)).astype(np.uint8)
        skin_core = cv2.morphologyEx(
            skin_core, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        radius = max(1, int(round(cfg.skin_dilate_px * scale)))
        skin_exclusion = cv2.dilate(
            skin_core, np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8),
            iterations=1) > 0
        strict &= ~skin_exclusion
        relaxed &= ~skin_exclusion
    strict[sy, sx] = True
    relaxed[sy, sx] = True

    close = max(3, int(round(cfg.close_px * scale))) | 1
    strict = cv2.morphologyEx(strict.astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones((close, close), np.uint8))
    core = _component_from_seed(strict, (sx, sy))
    grown = core.copy()
    grow = max(1, int(round(cfg.grow_px * scale)))
    kernel = np.ones((grow * 2 + 1, grow * 2 + 1), np.uint8)
    for _ in range(max(1, int(cfg.grow_iterations))):
        neighbourhood = cv2.dilate(grown, kernel)
        grown = (((neighbourhood > 0) & relaxed) | (core > 0)).astype(np.uint8)
    grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE,
                             np.ones((close, close), np.uint8))
    grown = _fill_small_holes(grown, max(16, int(cfg.max_hole_area * scale * scale)))

    # mc-v2 can assign the same warm colour to a wooden panel rail touching
    # the character panel. First cut along long straight horizontal *ink* lines
    # from the original manga. This is more reliable than colour geometry when
    # the rail and hair have merged into one orange component.
    rejected_panel_pixels = 0
    rejected_panel_line_pixels = 0
    if cfg.reject_panel_rules and np.any(grown):
        edges = cv2.Canny(gray, 40, 120)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180.0, threshold=max(20, int(ww * 0.055)),
            minLineLength=max(24, int(ww * 0.12)), maxLineGap=8)
        line_cut = np.zeros_like(grown, np.uint8)
        if lines is not None:
            y_limit = sy + max(4, int(round(wh * 0.03)))
            # OpenCV wheels do not agree on the Python output shape here:
            # common builds return (N, 1, 4), while some return (N, 4).
            # Normalise both forms before iteration; using ``lines[:, 0]``
            # turns the latter into numpy.int32 scalars and raises
            # "object is not iterable" on Windows/macOS combinations.
            line_rows = np.asarray(lines, dtype=np.int32).reshape(-1, 4)
            for raw_line in line_rows:
                x1, y1, x2, y2 = [int(v) for v in raw_line]
                if abs(y2 - y1) > max(2, int(round(wh * 0.004))):
                    continue
                yy = int(round((y1 + y2) * 0.5))
                if yy > y_limit:
                    continue
                xa, xb = sorted((x1, x2))
                if xb - xa < max(24, int(ww * 0.12)):
                    continue
                # A panel rule crossing the clicked seed's x coordinate is
                # often the actual top border behind the character and cutting
                # it would slice straight through bangs. Reject only rails that
                # sit clearly to one side of the semantic seed.
                if xa <= sx <= xb:
                    continue
                band_y0, band_y1 = max(0, yy - 3), min(wh, yy + 4)
                band_x0, band_x1 = max(0, xa - 5), min(ww, xb + 6)
                selected_overlap = int(np.count_nonzero(
                    grown[band_y0:band_y1, band_x0:band_x1]))
                if selected_overlap < max(10, int((xb - xa) * 0.12)):
                    continue
                line_cut[band_y0:band_y1, band_x0:band_x1] = 1
        if np.any(line_cut):
            rejected_panel_line_pixels = int(np.count_nonzero(grown & line_cut))
            grown[line_cut > 0] = 0
            grown = _component_from_seed(grown, (sx, sy))

        # A filled wooden rail may survive after its black centre lines are cut.
        # Detect a long run that sits wholly to one side of the semantic seed
        # and occupies only a shallow band above it, then remove that side band.
        # Real bangs around the clicked point normally span the seed x position
        # and therefore fail this gate.
        min_run = max(32, int(round(ww * 0.20)))
        side_gap = max(5, int(round(ww * 0.010)))
        scan_top = max(0, sy - int(round(wh * 0.16)))
        runs_by_side = {"left": [], "right": []}
        for yy in range(scan_top, min(wh, sy + 1)):
            row_x = np.flatnonzero(grown[yy] > 0)
            if row_x.size == 0:
                continue
            breaks = np.flatnonzero(np.diff(row_x) > 1)
            starts = np.r_[0, breaks + 1]
            ends = np.r_[breaks, row_x.size - 1]
            for start_i, end_i in zip(starts, ends):
                xa = int(row_x[start_i]); xb = int(row_x[end_i])
                if xb - xa + 1 < min_run:
                    continue
                if xa > sx + side_gap:
                    runs_by_side["right"].append((yy, xa, xb))
                elif xb < sx - side_gap:
                    runs_by_side["left"].append((yy, xa, xb))

        for side, side_runs in runs_by_side.items():
            if len(side_runs) < 3:
                continue
            # Split into consecutive row groups and reject only shallow groups.
            groups = []
            current = [side_runs[0]]
            for item in side_runs[1:]:
                if item[0] <= current[-1][0] + 2:
                    current.append(item)
                else:
                    groups.append(current); current = [item]
            groups.append(current)
            for group in groups:
                y0 = min(v[0] for v in group)
                y1 = max(v[0] for v in group)
                if y1 - y0 > max(8, int(round(wh * 0.055))):
                    continue
                xa = min(v[1] for v in group)
                xb = max(v[2] for v in group)
                branch = np.zeros_like(grown, np.uint8)
                by0 = max(0, y0 - 3); by1 = min(wh, y1 + 6)
                if side == "right":
                    bx0 = max(sx + side_gap, xa - 8); bx1 = min(ww, xb + 9)
                else:
                    bx0 = max(0, xa - 8); bx1 = min(sx - side_gap + 1, xb + 9)
                if bx1 > bx0:
                    branch[by0:by1, bx0:bx1] = 1
                    removed = int(np.count_nonzero(grown & branch))
                    if removed:
                        rejected_panel_line_pixels += removed
                        grown[branch > 0] = 0
                        grown = _component_from_seed(grown, (sx, sy))

    # Fallback: reject long, shallow colour branches above the clicked seed,
    # then re-take the seed component. Irregular bangs and long vertical hair
    # do not satisfy this geometry gate.
    if cfg.reject_panel_rules and np.any(grown):
        kernel_w = max(25, int(round(ww * 0.085)))
        horizontal = cv2.morphologyEx(
            grown.astype(np.uint8), cv2.MORPH_OPEN,
            np.ones((3, kernel_w), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(horizontal, connectivity=8)
        reject = np.zeros_like(grown, np.uint8)
        y_limit = sy + max(4, int(round(wh * 0.03)))
        for label in range(1, count):
            x, y, bw, bh, pixels = [int(v) for v in stats[label]]
            aspect = bw / max(1.0, float(bh))
            if (pixels >= kernel_w and aspect >= float(cfg.panel_rule_aspect)
                    and y + bh <= y_limit):
                reject[labels == label] = 1
        if np.any(reject):
            # Expand mainly in the vertical direction so the complete shallow
            # panel rail is removed, not just its one-pixel horizontal core.
            # A narrow horizontal kernel avoids eating into irregular bangs at
            # the point where the rail touches the character silhouette.
            vertical = max(9, int(round(wh * 0.035))) | 1
            reject = cv2.dilate(
                reject, np.ones((vertical, 7), np.uint8), iterations=1)
            rejected_panel_pixels = int(np.count_nonzero(grown & reject))
            grown[reject > 0] = 0
            grown = _component_from_seed(grown, (sx, sy))

    area = int(np.count_nonzero(grown))
    permitted = int(np.count_nonzero(limit))
    max_area = int((permitted if limit_mask is not None else wh * ww) *
                   (0.92 if limit_mask is not None else cfg.max_fraction))
    if area == 0 or area > max_area:
        grown = core

    full = cv2.resize(grown * 255, (w, h), interpolation=cv2.INTER_NEAREST)
    if cfg.edge_expand_px > 0:
        radius = max(1, int(cfg.edge_expand_px))
        full = cv2.dilate(
            full, np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8),
            iterations=1)
    skin_full = np.zeros((h, w), np.uint8)
    if np.any(skin_exclusion):
        skin_full = cv2.resize(
            skin_exclusion.astype(np.uint8), (w, h),
            interpolation=cv2.INTER_NEAREST)
        full[skin_full > 0] = 0
    limit_full = np.ones((h, w), np.uint8)
    if limit_mask is not None:
        limit_full = _normalise_mask(limit_mask, (h, w))
        full[limit_full <= 0] = 0

    # Full-resolution semantic support is reused by edge recovery, highlight
    # completion and text-occlusion continuation.
    full_hsv = cv2.cvtColor(colour, cv2.COLOR_BGR2HSV)
    full_ycrcb = cv2.cvtColor(colour, cv2.COLOR_BGR2YCrCb)
    full_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    fh = full_hsv[..., 0].astype(np.float32)
    fs = full_hsv[..., 1].astype(np.float32)
    fv = full_hsv[..., 2].astype(np.float32)
    hue_delta_full = np.minimum(
        np.abs(fh - float(target_hsv[0])),
        180.0 - np.abs(fh - float(target_hsv[0])))
    fy, fcr, fcb = cv2.split(full_ycrcb)
    non_skin_colour = ((skin_full == 0) &
                       ((fy < 150) | (fcr >= 202) |
                        (fcb < 48) | (fcb >= 150)))
    colour_candidate = (
        (hue_delta_full <= float(cfg.edge_hue_tolerance)) &
        (fs >= float(cfg.edge_saturation_floor)) & (fv > 18) &
        non_skin_colour)
    line_candidate = (
        (full_gray < int(cfg.lineart_edge_threshold)) &
        ((fv < 190) | ((hue_delta_full <= float(cfg.edge_hue_tolerance)) &
                       (fs >= 60))))
    candidate = ((colour_candidate | line_candidate) &
                 (limit_full > 0) & (skin_full == 0)).astype(np.uint8)

    edge_refined_pixels = 0
    if cfg.edge_refine_iterations > 0 and np.any(full):
        refined = full > 0
        radius = max(1, int(cfg.edge_refine_radius))
        kernel = np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)
        before = int(np.count_nonzero(refined))
        for _ in range(max(0, int(cfg.edge_refine_iterations))):
            nearby = cv2.dilate(refined.astype(np.uint8), kernel,
                                iterations=1) > 0
            refined |= nearby & (candidate > 0)
        full = np.where(refined, 255, 0).astype(np.uint8)
        edge_refined_pixels = int(np.count_nonzero(full) - before)

    completed, completed_pixels, completed_hole_pixels = _complete_visible_shape(
        full, candidate, full_gray, skin_full, limit_full, cfg)
    full = np.where(completed > 0, 255, 0).astype(np.uint8)

    detected_occlusion = False
    heuristic_occlusion = False
    logical_occlusion = _normalise_mask(occlusion_mask, (h, w)) if occlusion_mask is not None else None
    if logical_occlusion is None and cfg.complete_text_occlusions and np.any(full):
        try:
            from core.text_guard import available as text_detector_available
            from core.text_guard import text_occlusion_mask
            detected_occlusion = bool(text_detector_available())
            logical_occlusion = text_occlusion_mask(
                original, dilate_px=max(0, int(cfg.occlusion_dilate_px)),
                vicinity_mask=full, use_fallback=True)
            heuristic_occlusion = bool(logical_occlusion is not None and
                                       np.any(logical_occlusion) and
                                       not detected_occlusion)
        except Exception:  # optional detector/heuristic must never break selection
            logical_occlusion = None
    if logical_occlusion is None:
        logical_occlusion = np.zeros((h, w), np.uint8)

    bridged, occlusion_added_pixels, occlusion_core_pixels = _bridge_text_occlusions(
        full, candidate, logical_occlusion, skin_full, limit_full, cfg)
    full = np.where(bridged > 0, 255, 0).astype(np.uint8)
    full[skin_full > 0] = 0
    full[limit_full <= 0] = 0
    seed_full = (int(np.clip(seed_xy[0], 0, w - 1)),
                 int(np.clip(seed_xy[1], 0, h - 1)))
    pruned_full, rejected_panel_pixels_full = _prune_full_resolution_panel_branches(
        full, seed_full, enabled=bool(cfg.reject_panel_rules))
    full = np.where(pruned_full > 0, 255, 0).astype(np.uint8)

    hair_object_diag = {"enabled": False, "added_pixels": 0}
    hair_mode = (bool(cfg.complete_hair_object) if complete_hair_object is None
                 else bool(complete_hair_object))
    if hair_mode and np.any(full):
        full, hair_object_diag = build_complete_hair_object_mask(
            original, colour, full, seed_full,
            limit_mask=limit_full, occlusion_mask=logical_occlusion,
            use_line_regions=bool(cfg.hair_object_line_regions),
            max_ratio=float(cfg.hair_object_max_ratio))

    # Completion is allowed to recover missing strands, but it may not turn a
    # local semantic request into a page-sized selection.
    full_area = int(np.count_nonzero(full))
    full_permitted = int(np.count_nonzero(limit_full))
    full_limit = int((full_permitted if limit_mask is not None else h * w) *
                     (1.0 if limit_mask is not None else min(0.46, cfg.max_fraction * 1.28)))
    if full_area > full_limit:
        full = cv2.resize(grown * 255, (w, h), interpolation=cv2.INTER_NEAREST)
        full[skin_full > 0] = 0
        full[limit_full <= 0] = 0
        completed_pixels = 0
        completed_hole_pixels = 0
        occlusion_added_pixels = 0
        occlusion_core_pixels = 0

    full = np.where(full > 0, 255, 0).astype(np.uint8)
    diagnostics = {
        "work_size": (ww, wh),
        "seed_work": (sx, sy),
        "target_hsv": [float(v) for v in target_hsv],
        "target_lab": [float(v) for v in target_lab],
        "selected_pixels": int(np.count_nonzero(full)),
        "selected_fraction": float(np.count_nonzero(full) / max(1, h*w)),
        "limited_by_rough_selection": bool(limit_mask is not None),
        "skin_protection_pixels_work": int(np.count_nonzero(skin_exclusion)),
        "rejected_panel_pixels_work": int(rejected_panel_pixels),
        "rejected_panel_line_pixels_work": int(rejected_panel_line_pixels),
        "rejected_panel_pixels_full": int(rejected_panel_pixels_full),
        "edge_refined_pixels_full": int(edge_refined_pixels),
        "visible_completion_pixels_full": int(completed_pixels),
        "visible_completion_hole_pixels_full": int(completed_hole_pixels),
        "text_occlusion_mask_pixels_full": int(np.count_nonzero(logical_occlusion)),
        "text_occlusion_added_pixels_full": int(occlusion_added_pixels),
        "text_occlusion_core_pixels_full": int(occlusion_core_pixels),
        "text_occlusion_detector_used": bool(detected_occlusion),
        "text_occlusion_heuristic_used": bool(heuristic_occlusion),
        "text_occlusion_override_used": bool(occlusion_mask is not None),
        "complete_hair_object_enabled": bool(hair_mode),
        "complete_hair_object": dict(hair_object_diag),
    }
    return full, diagnostics
