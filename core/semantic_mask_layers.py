"""Deterministic semantic-mask colour overrides for Colortina.

This module deliberately sits *after* mc-v2 generation.  mc-v2 remains the
base automatic colourist; semantic layers only take authority over explicitly
enabled regions (hair, iris and background).

The compositor is replace/overwrite based.  It never alpha-composites a colour
layer over the base image.  A replacement image may preserve the base image's
luminance/shading internally, but once rendered, pixels selected by the final
binary mask are copied byte-for-byte into the output. Pixels outside the mask
remain byte-identical to the input.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import threading

import cv2
import numpy as np


_CLIP_CLASSIFIER_LOCK = threading.Lock()
_CLIP_CLASSIFIERS: dict[bool, Any] = {}


def _get_region_classifier(allow_download: bool):
    """Return one lazy CLIP classifier per network-policy mode.

    This mirrors the bounded-resource pattern used by robust userscript image
    workers: expensive shared state is created once and reused across pages.
    No browser/server functionality is included.
    """
    key = bool(allow_download)
    with _CLIP_CLASSIFIER_LOCK:
        if key in _CLIP_CLASSIFIERS:
            return _CLIP_CLASSIFIERS[key]
        try:
            from core.region_classifier import RegionClassifier
            candidate = RegionClassifier(allow_download=key)
            # Cache unavailable classifiers too: do not retry a missing local
            # model on every page in an offline batch.
            _ = candidate.available
            _CLIP_CLASSIFIERS[key] = candidate
            return candidate
        except Exception:
            _CLIP_CLASSIFIERS[key] = None
            return None


@dataclass
class SemanticMaskBundle:
    hair_mask: np.ndarray | None = None
    eye_mask: np.ndarray | None = None
    iris_mask: np.ndarray | None = None
    skin_mask: np.ndarray | None = None
    cloth_mask: np.ndarray | None = None
    glasses_mask: np.ndarray | None = None
    face_shadow_mask: np.ndarray | None = None
    background_mask: np.ndarray | None = None
    foreground_mask: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _binary(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.shape != shape:
        arr = cv2.resize(arr.astype(np.uint8), (shape[1], shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    return np.where(arr > 0, 255, 0).astype(np.uint8)


def _source_gray(source_bgr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    source = np.asarray(source_bgr)
    if source.shape[:2] != (h, w):
        source = cv2.resize(source, (w, h), interpolation=cv2.INTER_AREA)
    if source.ndim == 2:
        return source.astype(np.uint8)
    return cv2.cvtColor(source.astype(np.uint8), cv2.COLOR_BGR2GRAY)


def _paintable(mask: np.ndarray | None, source_bgr: np.ndarray,
               *, ink_threshold: int = 76, erode_px: int = 0,
               preserve_dark_fill: bool = False,
               ink_shell_px: float = 2.25) -> np.ndarray | None:
    """Return a binary interior mask while protecting structural black ink.

    Hair in manga is often a large solid-black fill. Excluding every dark pixel
    would make that hair impossible to recolour, so the default protects only
    the thin boundary/stroke shell of dark components and keeps thick black-fill
    interiors editable. Iris masks opt into ``preserve_dark_fill`` so a black
    pupil core remains untouched.
    """
    if mask is None:
        return None
    shape = source_bgr.shape[:2]
    out = _binary(mask, shape)
    if out is None or not np.any(out):
        return out
    gray = _source_gray(source_bgr, shape)
    dark = gray <= int(np.clip(ink_threshold, 32, 160))
    if preserve_dark_fill:
        protected = dark
    else:
        # Distance inside dark components separates thin contours from broad
        # solid ink. Typical 1-4 px manga strokes are fully protected, while
        # the interior of a filled black hairstyle remains recolourable.
        distance = cv2.distanceTransform(dark.astype(np.uint8), cv2.DIST_L2, 5)
        protected = dark & (distance <= float(max(0.75, ink_shell_px)))
    out[protected] = 0
    if erode_px > 0:
        k = erode_px * 2 + 1
        out = cv2.erode(out, np.ones((k, k), np.uint8), iterations=1)
    return out


def snap_semantic_to_line_regions(mask: np.ndarray | None, source_bgr: np.ndarray,
                                  *, min_overlap_ratio: float = 0.34,
                                  max_region_area_ratio: float = 0.16,
                                  max_total_growth: float = 1.55) -> tuple[np.ndarray | None, dict]:
    """Conservatively expand a semantic prior to enclosed line-art regions.

    The semantic model decides *what* the object is; the line map decides its
    exact paint boundary. Only regions substantially covered by the semantic
    prior are admitted, so an open hair/face boundary cannot flood the face.
    """
    if mask is None or not np.any(mask):
        return mask, {"reason": "empty", "added_pixels": 0}
    shape = source_bgr.shape[:2]
    prior = _binary(mask, shape)
    before = int(np.count_nonzero(prior))
    try:
        from core.region_map import build_region_map
        region_map = build_region_map(source_bgr, line_low=78, gap_close=4)
    except Exception as exc:  # noqa: BLE001
        return prior, {"reason": f"region_map_error:{type(exc).__name__}",
                       "added_pixels": 0}

    labels = region_map.labels
    active_labels = np.unique(labels[prior > 0])
    page_area = max(1, labels.size)
    additions = np.zeros_like(prior)
    accepted = 0
    for rid in active_labels.tolist():
        rid = int(rid)
        if rid <= 0:
            continue
        region = labels == rid
        area = int(np.count_nonzero(region))
        if area <= 0 or area > page_area * float(max_region_area_ratio):
            continue
        overlap = int(np.count_nonzero(region & (prior > 0)))
        ratio = overlap / float(area)
        # Small enclosed regions may be largely line/antialias boundary; allow
        # a slightly lower overlap only when the absolute support is meaningful.
        gate = float(min_overlap_ratio) if area >= 80 else max(0.24, float(min_overlap_ratio) - 0.08)
        if ratio < gate or overlap < max(5, int(area * 0.20)):
            continue
        additions[region] = 255
        accepted += 1

    snapped = cv2.bitwise_or(prior, additions)
    max_pixels = int(max(before + 16, before * float(max_total_growth)))
    if int(np.count_nonzero(snapped)) > max_pixels:
        # Growth exceeded the safety budget. Keep the semantic prior rather
        # than partially accepting an unpredictable leak.
        return prior, {"reason": "growth_rejected", "added_pixels": 0,
                       "accepted_regions": accepted}
    return snapped, {
        "reason": "snapped",
        "accepted_regions": accepted,
        "added_pixels": max(0, int(np.count_nonzero(snapped)) - before),
    }


def _component_rows(mask: np.ndarray, min_area: int = 6):
    count, labels, stats, centers = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), 8)
    rows = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        cx, cy = [float(v) for v in centers[label]]
        rows.append((label, area, (x, y, w, h), (cx, cy), labels))
    return rows


def _central_iris_fallback(eye_component: np.ndarray, source_bgr: np.ndarray,
                           base_bgr: np.ndarray) -> np.ndarray:
    """Conservative fallback when line-art cannot isolate the iris.

    It favours already-colourful mc-v2 pixels near the middle of an eye and
    only falls back to a compact central ellipse.  This keeps the fallback from
    painting an entire eye white/sclera region.
    """
    ys, xs = np.nonzero(eye_component)
    out = np.zeros_like(eye_component, dtype=np.uint8)
    if ys.size < 6:
        return out
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    if bw < 3 or bh < 3:
        return out

    hsv = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1]
    gray = _source_gray(source_bgr, eye_component.shape)
    cx, cy = float(xs.mean()), float(ys.mean())
    yy, xx = np.indices(eye_component.shape, dtype=np.float32)
    rx = max(1.5, bw * 0.31)
    ry = max(1.5, bh * 0.46)
    central = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0)

    # Coloured automatic output is an excellent hint for where the iris body
    # already is; line-art darkness supports monochrome/low-saturation results.
    colourful = sat >= max(26, int(np.percentile(sat[eye_component], 58)))
    mid_dark = (gray >= 82) & (gray <= 224)
    candidate = eye_component & central & (colourful | mid_dark)
    if int(np.count_nonzero(candidate)) < max(4, int(0.06 * ys.size)):
        candidate = eye_component & central

    candidate_u8 = candidate.astype(np.uint8) * 255
    candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_CLOSE,
                                     np.ones((3, 3), np.uint8), iterations=1)
    candidate_u8[gray <= 76] = 0
    return candidate_u8



def _seed_points_from_mask(mask: np.ndarray | None, max_points: int = 4) -> list[tuple[int, int]]:
    """Return deep interior points from positive scribble/mask components.

    Extrema are poor seeds for manga hair: they often land on text, glasses or
    panel rails.  Distance-transform interior points mirror the region sampling
    pattern used by robust line-art segmenters and are much more stable.
    """
    binary = _binary(mask, np.asarray(mask).shape[:2]) if mask is not None else None
    if binary is None or not np.any(binary):
        return []
    work = (binary > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
    rows = []
    for rid in range(1, n):
        area = int(stats[rid, cv2.CC_STAT_AREA])
        if area < 3:
            continue
        comp = (labels == rid).astype(np.uint8)
        dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
        rows.append((area, (int(x), int(y))))
    rows.sort(key=lambda row: row[0], reverse=True)
    return [pt for _area, pt in rows[:max(1, int(max_points))]]


def _seed_scope_mask(seed_mask: np.ndarray | None, shape: tuple[int, int],
                     negative_mask: np.ndarray | None = None) -> np.ndarray | None:
    """Local, downward-biased scope for a hair seed.

    Hair can be long, but a seed near the head must never authorize the whole
    page.  The lower margin is larger than the upper one to include ponytails.
    """
    seed = _binary(seed_mask, shape)
    if seed is None or not np.any(seed):
        return None
    h, w = shape
    ys, xs = np.where(seed > 0)
    if ys.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    mx = max(42, int(round(bw * 0.85)))
    up = max(42, int(round(bh * 0.65)))
    down = max(80, int(round(bh * 1.70)))
    scope = np.zeros((h, w), np.uint8)
    bx0, by0 = max(0, x0 - mx), max(0, y0 - up)
    bx1, by1 = min(w, x1 + mx + 1), min(h, y1 + down + 1)
    scope[by0:by1, bx0:bx1] = 255
    veto = _binary(negative_mask, shape)
    if veto is not None and np.any(veto):
        veto_pad = cv2.dilate(veto, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1)
        scope[veto_pad > 0] = 0
    return scope




def _fill_holes_limited(mask: np.ndarray, max_area: int) -> np.ndarray:
    work = (np.asarray(mask) > 0).astype(np.uint8)
    inv = (work == 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    h, w = work.shape
    for rid in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[rid]]
        touches = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        if not touches and area <= int(max_area):
            work[labels == rid] = 1
    return work * 255




def _mask_bbox(mask: np.ndarray | None) -> tuple[int, int, int, int] | None:
    binary = np.asarray(mask) if mask is not None else None
    if binary is None or binary.size == 0 or not np.any(binary):
        return None
    ys, xs = np.where(binary > 0)
    if ys.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, int(x1 - x0 + 1), int(y1 - y0 + 1)


def _min_gap_to_mask(component: np.ndarray, mask: np.ndarray, dilate_px: int = 15) -> int:
    comp = (np.asarray(component) > 0)
    base = (np.asarray(mask) > 0)
    if not np.any(comp):
        return 999999
    for r in range(1, max(1, int(dilate_px)) + 1):
        k = max(3, 2 * r + 1)
        grown = cv2.dilate(base.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), 1) > 0
        if np.any(comp & grown):
            return int(r)
    return int(dilate_px + 1)


def _prune_neck_residual_components(mask: np.ndarray | None, seed: np.ndarray | None) -> tuple[np.ndarray | None, dict]:
    """Remove small central neck remnants while preserving the dominant side lock.

    After the broad neck-shadow pass a tiny shadow can remain because it sits
    close to the real hanging hair.  Inside a narrow neck band we therefore
    keep the dominant tall side component and reject only much smaller
    secondary components displaced toward the centre of the neck.  This is
    deliberately symmetric: it works for a dominant left or right hanging lock.
    """
    active = _binary(mask, np.asarray(mask).shape[:2] if mask is not None else (0, 0))
    if active is None or not np.any(active):
        return active, {"enabled": False, "reason": "empty_mask"}
    bbox = _mask_bbox(active)
    if bbox is None:
        return active, {"enabled": False, "reason": "empty_bbox"}
    x0, y0, bw, bh = bbox
    h, w = active.shape
    nx0 = max(0, int(round(x0 + bw * 0.18)))
    nx1 = min(w - 1, int(round(x0 + bw * 0.72)))
    ny0 = max(0, int(round(y0 + bh * 0.42)))
    ny1 = min(h - 1, int(round(y0 + bh * 0.69)))
    if nx1 <= nx0 or ny1 <= ny0:
        return active, {"enabled": False, "reason": "invalid_band"}

    band = np.zeros_like(active, np.uint8)
    band[ny0:ny1 + 1, nx0:nx1 + 1] = (active[ny0:ny1 + 1, nx0:nx1 + 1] > 0).astype(np.uint8)
    n, labels, stats, centers = cv2.connectedComponentsWithStats(band, connectivity=8)
    rows = []
    for rid in range(1, n):
        xx, yy, ww, hh, area = [int(v) for v in stats[rid]]
        if area <= 0:
            continue
        rows.append((rid, area, xx, yy, ww, hh, float(centers[rid][0]), float(centers[rid][1])))
    if len(rows) <= 1:
        return active, {"enabled": True, "reason": "single_neck_component",
                        "neck_band": [nx0, ny0, nx1 - nx0 + 1, ny1 - ny0 + 1],
                        "removed_pixels": 0}

    # Main side lock is typically both large and vertically persistent.
    rows.sort(key=lambda r: (r[1] * (1.0 + min(2.0, r[5] / max(1.0, r[4])) * 0.18)), reverse=True)
    main = rows[0]
    _, main_area, _, _, main_w, main_h, main_cx, _ = main
    hair_mid = x0 + bw * 0.5
    side = 1.0 if main_cx >= hair_mid else -1.0
    seed_bin = _binary(seed, active.shape)
    seed_probe = None
    if seed_bin is not None and np.any(seed_bin):
        seed_probe = cv2.dilate(seed_bin.astype(np.uint8),
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), 1) > 0

    work = active.copy()
    removed = []
    total_pixels = int(np.count_nonzero(active))
    for row in rows[1:]:
        rid, area, xx, yy, ww, hh, cx, cy = row
        comp = labels == rid
        seed_overlap = (int(np.count_nonzero(comp & seed_probe))
                        if seed_probe is not None else 0)
        # Positive scribbles are guidance, not an unconditional keep mask.  A
        # rough stroke may cross the neck shadow; compact central remnants may
        # still be rejected even when touched by the positive stroke.
        # Secondary neck remnants are compact/short, much smaller than the
        # persistent side lock, and displaced toward the head centre.
        smaller = area <= max(900, int(main_area * 0.34), int(total_pixels * 0.012))
        short = hh <= max(56, int(bh * 0.13))
        toward_center = (side > 0 and cx < main_cx - max(8.0, bw * 0.035)) or \
                        (side < 0 and cx > main_cx + max(8.0, bw * 0.035))
        not_tall_lock = hh < main_h * 0.52 or ww > hh * 0.78
        if not (smaller and short and toward_center and not_tall_lock):
            continue
        pixels = int(np.count_nonzero(comp))
        work[comp] = 0
        removed.append({
            "bbox": [int(xx), int(yy), int(ww), int(hh)],
            "area": int(area), "pixels": int(pixels),
            "seed_overlap": int(seed_overlap),
            "centre": [round(cx, 2), round(cy, 2)],
            "reason": "reject_secondary_neck_component",
        })

    return work, {
        "enabled": True,
        "reason": "pruned" if removed else "no_secondary_reject",
        "neck_band": [nx0, ny0, nx1 - nx0 + 1, ny1 - ny0 + 1],
        "main_component": {"area": int(main_area), "bbox_wh": [int(main_w), int(main_h)],
                           "centre_x": round(main_cx, 2), "side": "right" if side > 0 else "left"},
        "removed_pixels": int(sum(r["pixels"] for r in removed)),
        "removed_components": removed,
    }


def _refine_completed_hair_mask(mask: np.ndarray | None, source_bgr: np.ndarray,
                                seed: np.ndarray | None, scope: np.ndarray | None,
                                veto: np.ndarray | None = None) -> tuple[np.ndarray | None, dict]:
    """Patch small internal gaps and restore nearby edge fragments conservatively.

    This pass exists for the post-neck-fix regressions where the mask became too
    conservative: tiny highlight/scan holes remained on the head, and narrow
    edge fragments on a long lock were left behind.  The pass is strictly local
    and bounded; it cannot introduce page-wide growth.
    """
    active = _binary(mask, source_bgr.shape[:2])
    if active is None or not np.any(active):
        return active, {"enabled": False, "reason": "empty_mask"}
    seed_bin = _binary(seed, active.shape)
    scope_bin = _binary(scope, active.shape)
    veto_bin = _binary(veto, active.shape)
    refined = active.copy()
    bbox = _mask_bbox(refined)
    if bbox is None:
        return refined, {"enabled": False, "reason": "empty_bbox"}
    x0, y0, bw, bh = bbox
    h, w = refined.shape
    area0 = int(np.count_nonzero(refined))

    # 1) Fill small interior holes only inside a lightly closed local shell.
    close_x = max(7, min(17, int(round(bw * 0.10)) | 1))
    close_y = max(7, min(17, int(round(bh * 0.08)) | 1))
    shell = cv2.morphologyEx(refined, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_x, close_y)), 1)
    if scope_bin is not None:
        shell[scope_bin == 0] = 0
    if veto_bin is not None and np.any(veto_bin):
        shell[veto_bin > 0] = 0
    max_hole = max(18, int(area0 * 0.020))
    shell_filled = _fill_holes_limited(shell, max_area=max_hole)
    hole_add = ((shell_filled > 0) & (refined == 0))
    if seed_bin is not None and np.any(seed_bin):
        # Keep the repair local to the head/lock neighbourhood instead of
        # blindly filling elsewhere in scope.
        local_support = cv2.dilate(refined, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)), 1) > 0
        hole_add &= local_support
    filled_holes = int(np.count_nonzero(hole_add))
    if filled_holes:
        refined[hole_add] = 255

    # 2) Reattach small nearby dark fragments that align with hair flow.
    gray = _source_gray(source_bgr, refined.shape)
    dark = ((gray <= 172) & (refined == 0)).astype(np.uint8)
    if scope_bin is not None:
        dark[scope_bin == 0] = 0
    if veto_bin is not None and np.any(veto_bin):
        dark[veto_bin > 0] = 0
    near = cv2.dilate(refined, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), 1) > 0
    dark = ((dark > 0) & near).astype(np.uint8)
    cn, cl, cs, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    added_pixels = 0
    accepted = []
    total_added_budget = max(48, int(area0 * 0.14))
    head_upper = y0 + bh * 0.54
    side_gate_x = x0 + bw * 0.46
    neck_x0 = x0 + bw * 0.24
    neck_x1 = x0 + bw * 0.70
    neck_y0 = y0 + bh * 0.40
    neck_y1 = y0 + bh * 0.66
    for rid in range(1, cn):
        xx, yy, ww, hh, area = [int(v) for v in cs[rid]]
        if area < 12 or area > max(900, int(area0 * 0.11)):
            continue
        comp = cl == rid
        gap = _min_gap_to_mask(comp.astype(np.uint8), refined, dilate_px=12)
        if gap > 8:
            continue
        cx = xx + ww * 0.5
        cy = yy + hh * 0.5
        slender = max(ww, hh) / float(max(1, min(ww, hh)))
        verticalish = hh >= ww * 0.72
        in_neck = (neck_x0 <= cx <= neck_x1 and neck_y0 <= cy <= neck_y1)
        head_piece = cy <= head_upper
        side_piece = (cx >= side_gate_x and cy <= y0 + bh * 0.98 and verticalish)
        if in_neck:
            continue
        if not (head_piece or side_piece):
            continue
        if slender < 1.1 and not head_piece:
            continue
        refined[comp] = 255
        added_pixels += int(area)
        accepted.append({"bbox": [int(xx), int(yy), int(ww), int(hh)], "area": int(area),
                         "gap": int(gap), "head_piece": bool(head_piece), "side_piece": bool(side_piece)})
        if added_pixels >= total_added_budget:
            break

    # 3) Final tiny residual cleanup for central neck/chin shadows.
    residual_removed = 0
    vertical = cv2.morphologyEx(
        (refined > 0).astype(np.uint8), cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 17)), 1)
    vn, vl, vs, _ = cv2.connectedComponentsWithStats(vertical, connectivity=8)
    if vn > 1:
        main_v = 1 + int(np.argmax(vs[1:, cv2.CC_STAT_AREA]))
        main_support = (vl == main_v).astype(np.uint8)
        main_support = cv2.dilate(main_support, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), 1) > 0
        neck_band = np.zeros_like(refined, dtype=bool)
        nx0 = int(round(x0 + bw * 0.24))
        nx1 = int(round(x0 + bw * 0.66))
        ny0 = int(round(y0 + bh * 0.42))
        ny1 = int(round(y0 + bh * 0.69))
        neck_band[max(0, ny0):min(h, ny1 + 1), max(0, nx0):min(w, nx1 + 1)] = True
        unsupported = (refined > 0) & neck_band & (~main_support)
        if seed_bin is not None and np.any(seed_bin):
            seed_keep = cv2.dilate(seed_bin.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), 1) > 0
            unsupported &= ~seed_keep
        residual_removed = int(np.count_nonzero(unsupported))
        if residual_removed:
            refined[unsupported] = 0

    secondary_pruned, secondary_diag = _prune_neck_residual_components(refined, seed_bin)
    if secondary_pruned is not None and np.any(secondary_pruned):
        refined = _binary(secondary_pruned, refined.shape)

    if scope_bin is not None:
        refined[scope_bin == 0] = 0
    if veto_bin is not None and np.any(veto_bin):
        refined[veto_bin > 0] = 0
    return refined, {
        "enabled": True,
        "reason": "refined",
        "filled_hole_pixels": int(filled_holes),
        "reattached_pixels": int(added_pixels),
        "residual_neck_pixels_removed": int(residual_removed),
        "secondary_neck_prune": secondary_diag,
        "accepted_fragments": accepted[:16],
        "result_pixels": int(np.count_nonzero(refined)),
    }

def _prune_lower_neck_shadow(core: np.ndarray | None, seed: np.ndarray,
                             source_bgr: np.ndarray,
                             veto: np.ndarray | None = None) -> tuple[np.ndarray | None, dict]:
    """Remove compact lower-face/neck shadows that hitchhike into black hair.

    On manga pages a dark neck shadow may touch long black hair via a narrow
    jaw bridge.  A light opening breaks that bridge; compact central-lower
    pieces that are not seed-touched are rejected as neck shadows instead of
    becoming part of HairMask.
    """
    mask = _binary(core, seed.shape)
    if mask is None or not np.any(mask):
        return core, {"enabled": False, "reason": "empty_core"}
    seed_bin = _binary(seed, mask.shape)
    if seed_bin is None or not np.any(seed_bin):
        return mask.astype(np.uint8) * 255, {"enabled": False, "reason": "empty_seed"}
    h, w = mask.shape
    sy, sx = np.where(seed_bin)
    x0, x1 = int(sx.min()), int(sx.max())
    y0, y1 = int(sy.min()), int(sy.max())
    sw, sh = x1 - x0 + 1, y1 - y0 + 1
    seed_cx = x0 + sw * 0.5
    seed_bottom = y1

    work = (mask > 0).astype(np.uint8)
    k = max(5, min(11, int(round(max(5.0, min(mask.shape) * 0.010))))) | 1

    # Directional support pass: real long hair is vertically persistent, while
    # chin/neck shadows are usually short horizontal/diagonal bridges.  Keep a
    # generous dilation around the dominant vertically-opened component, then
    # strip unsupported pixels only inside the central lower-face/neck band.
    ys_all, xs_all = np.where(work > 0)
    directional_removed = 0
    directional_band = None
    if ys_all.size:
        hx0, hx1 = int(xs_all.min()), int(xs_all.max())
        hy0, hy1 = int(ys_all.min()), int(ys_all.max())
        hw, hh = hx1 - hx0 + 1, hy1 - hy0 + 1
        vkx = max(5, min(9, (k - 2) | 1))
        vky = max(15, min(29, int(round(k * 2.3)) | 1))
        vertical = cv2.morphologyEx(
            work, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (vkx, vky)), 1)
        vn, vl, vs, _ = cv2.connectedComponentsWithStats(vertical, connectivity=8)
        if vn > 1:
            main_v = 1 + int(np.argmax(vs[1:, cv2.CC_STAT_AREA]))
            main_support = (vl == main_v).astype(np.uint8)
            pad = max(9, int(round(k * 1.4))) | 1
            main_support = cv2.dilate(
                main_support,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad, pad)), 1) > 0
            # Central neck band inferred from the hair object's own bbox.  The
            # rightmost quarter is intentionally preserved for a hanging lock.
            nx0 = int(round(hx0 + hw * 0.22))
            nx1 = int(round(hx0 + hw * 0.75))
            ny0 = int(round(hy0 + hh * 0.44))
            ny1 = int(round(hy0 + hh * 0.63))
            neck_band = np.zeros_like(work, dtype=bool)
            neck_band[max(0, ny0):min(h, ny1 + 1),
                      max(0, nx0):min(w, nx1 + 1)] = True
            unsupported = (work > 0) & neck_band & (~main_support)
            # Never erase pixels directly marked positive by the user.
            if seed_bin is not None:
                seed_keep = cv2.dilate(
                    seed_bin.astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), 1) > 0
                unsupported &= ~seed_keep
            directional_removed = int(np.count_nonzero(unsupported))
            if directional_removed:
                work[unsupported] = 0
            directional_band = [int(nx0), int(ny0), int(max(0, nx1 - nx0 + 1)),
                                int(max(0, ny1 - ny0 + 1))]

    opened = cv2.morphologyEx(work, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), 1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if n <= 2:
        return work * 255, {"enabled": True, "reason": "single_component", "open_kernel": int(k),
                            "directional_removed_pixels": int(directional_removed),
                            "directional_neck_band": directional_band,
                            "removed_components": []}

    seed_probe = cv2.dilate(seed_bin.astype(np.uint8),
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), 1) > 0
    kept = np.zeros_like(work)
    removed_components = []
    gray = _source_gray(source_bgr, mask.shape)
    for rid in range(1, n):
        comp = labels == rid
        area = int(stats[rid, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        xx = int(stats[rid, cv2.CC_STAT_LEFT])
        yy = int(stats[rid, cv2.CC_STAT_TOP])
        bw = int(stats[rid, cv2.CC_STAT_WIDTH])
        bh = int(stats[rid, cv2.CC_STAT_HEIGHT])
        cx = xx + bw * 0.5
        cy = yy + bh * 0.5
        overlap = int(np.count_nonzero(comp & seed_probe))
        dark_ratio = float(np.mean(gray[comp] <= 150)) if np.any(comp) else 0.0
        compactness = area / float(max(1, bw * bh))
        in_neck_zone = (
            cy >= seed_bottom + max(8.0, sh * 0.04) and
            cy <= y0 + sh * 0.92 and
            abs(cx - seed_cx) <= max(18.0, sw * 0.34) and
            bh <= max(18.0, sh * 0.34) and
            bw <= max(56.0, sw * 0.74)
        )
        likely_neck_shadow = (
            overlap < 4 and
            area <= max(1200, int(np.count_nonzero(mask) * 0.18)) and
            compactness >= 0.26 and
            dark_ratio >= 0.28 and
            in_neck_zone
        )
        if likely_neck_shadow:
            removed_components.append({
                "id": int(rid), "bbox": [int(xx), int(yy), int(bw), int(bh)],
                "area": int(area), "dark_ratio": round(dark_ratio, 4),
                "compactness": round(compactness, 4), "reason": "reject_neck_shadow"})
            continue
        kept[comp] = 1

    if not np.any(kept):
        return work * 255, {"enabled": True, "reason": "prune_would_empty_core", "open_kernel": int(k), "removed_components": removed_components}

    grown = cv2.dilate(kept, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), 1)
    final = ((grown > 0) & (work > 0)).astype(np.uint8) * 255
    if veto is not None and np.any(veto):
        final[_binary(veto, final.shape) > 0] = 0
    return final, {"enabled": True, "reason": "pruned", "open_kernel": int(k),
                   "directional_removed_pixels": int(directional_removed),
                   "directional_neck_band": directional_band,
                   "removed_components": removed_components,
                   "result_pixels": int(np.count_nonzero(final))}


def _black_hair_core_from_seed(source_bgr: np.ndarray, seed: np.ndarray,
                               scope: np.ndarray, veto: np.ndarray | None) -> tuple[np.ndarray | None, dict]:
    """Recover the dominant thick black-hair object from a few scribbles.

    A large morphological opening removes glasses, straps and panel rails before
    connected-component selection.  Only the largest substantial component
    touched by the positive seed becomes the core; smaller touched components
    are deliberately left for later generator validation instead of being
    promoted automatically.
    """
    gray = _source_gray(source_bgr, seed.shape)
    active_seed = seed > 0
    if not np.any(active_seed):
        return None, {"enabled": False, "reason": "empty_seed"}
    dark_ratio = float(np.mean(gray[active_seed] <= 170))
    if dark_ratio < 0.26:
        return None, {"enabled": False, "reason": "not_black_hair", "seed_dark_ratio": dark_ratio}

    scale = max(seed.shape) / 1924.0
    k = max(9, int(round(15 * max(0.75, min(1.6, scale))))) | 1
    dark = ((gray <= 165) & (scope > 0)).astype(np.uint8)
    opened = cv2.morphologyEx(
        dark, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=1)
    if veto is not None and np.any(veto):
        veto_pad = cv2.dilate(veto, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1)
        opened[veto_pad > 0] = 0

    n, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    probe = cv2.dilate(seed.astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), 1) > 0
    touched = []
    for rid in range(1, n):
        area = int(stats[rid, cv2.CC_STAT_AREA])
        overlap = int(np.count_nonzero((labels == rid) & probe))
        if overlap >= 5 and area >= 64:
            touched.append((area, overlap, rid))
    if not touched:
        return None, {"enabled": True, "reason": "no_touched_core", "seed_dark_ratio": dark_ratio,
                      "open_kernel": k}
    touched.sort(reverse=True)
    main_area, main_overlap, main_id = touched[0]
    core = (labels == int(main_id)).astype(np.uint8) * 255

    # Recover small highlight holes and antialiased dark edge pixels without
    # reconnecting thin straps removed by the opening step.
    close_k = max(5, (k // 2) | 1)
    core = cv2.morphologyEx(
        core, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)), 1)
    core = _fill_holes_limited(core, max_area=max(120, int(main_area * 0.025)))
    core[scope == 0] = 0
    if veto is not None:
        core[veto > 0] = 0
    pruned_core, prune_diag = _prune_lower_neck_shadow(core, seed, source_bgr, veto)
    if pruned_core is not None and np.any(pruned_core):
        core = pruned_core
    return core, {
        "enabled": True,
        "reason": "black_core_ready",
        "seed_dark_ratio": round(dark_ratio, 4),
        "open_kernel": int(k),
        "main_component": int(main_id),
        "main_area": int(main_area),
        "main_overlap": int(main_overlap),
        "touched_components": [[int(r), int(a), int(o)] for a, o, r in touched[:12]],
        "core_pixels": int(np.count_nonzero(core)),
        "neck_shadow_prune": prune_diag,
    }

def _safe_selection_candidate(selected: np.ndarray | None, seed: np.ndarray,
                              scope: np.ndarray, veto: np.ndarray | None,
                              *, max_growth_factor: float = 5.5) -> tuple[np.ndarray | None, dict]:
    """Validate mc-v2 colour-object completion before it may affect HairMask."""
    if selected is None:
        return None, {"accepted": False, "reason": "empty"}
    selected = _binary(selected, seed.shape)
    if selected is None or not np.any(selected):
        return None, {"accepted": False, "reason": "empty"}
    selected[scope == 0] = 0
    if veto is not None and np.any(veto):
        selected[veto > 0] = 0
    seed_area = max(1, int(np.count_nonzero(seed)))
    area = int(np.count_nonzero(selected))
    scope_area = max(1, int(np.count_nonzero(scope)))
    page_area = max(1, selected.size)
    budget = min(
        max(seed_area + 512, int(page_area * 0.105)),
        max(seed_area + 512, int(scope_area * 0.50)),
        max(seed_area + 512, int(seed_area * float(max_growth_factor))),
    )
    near = cv2.dilate(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)), 1)
    contact = int(np.count_nonzero((selected > 0) & (near > 0)))
    if contact < max(12, int(min(area, seed_area) * 0.02)):
        return None, {"accepted": False, "reason": "no_local_contact", "pixels": area}
    if area > budget:
        return None, {"accepted": False, "reason": "growth_budget", "pixels": area,
                      "max_pixels": int(budget)}
    return selected, {"accepted": True, "reason": "safe", "pixels": area,
                      "max_pixels": int(budget)}


def complete_manual_hair_mask(base_bgr: np.ndarray, source_bgr: np.ndarray,
                              seed_mask: np.ndarray | None,
                              *, negative_mask: np.ndarray | None = None,
                              generator_feature_provider=None,
                              use_trappedball: bool = True,
                              seed_is_scribble: bool = False,
                              prior_mask: np.ndarray | None = None) -> tuple[np.ndarray | None, dict]:
    """Complete a local hair seed without allowing page-wide propagation.

    Pipeline:
      positive seed -> local mc-v2 colour-object proposals -> bounded generator
      component expansion -> final area guard.  Positive scribbles are guidance
      only; when ``seed_is_scribble`` is true their literal brush pixels are not
      copied into the output mask.
    """
    shape = base_bgr.shape[:2]
    seed = _binary(seed_mask, shape)
    veto = _binary(negative_mask, shape)
    prior = _binary(prior_mask, shape)
    if seed is None or not np.any(seed):
        return prior, {"reason": "empty_seed", "added_pixels": 0}
    scope = _seed_scope_mask(seed, shape, negative_mask=veto)
    if scope is None:
        return prior, {"reason": "empty_scope", "added_pixels": 0}

    # The authoritative starting object is an existing semantic prior, or the
    # region mask itself for non-scribble/manual selection. A scribble alone is
    # never painted directly.
    combined = np.zeros(shape, np.uint8)
    if prior is not None and np.any(prior):
        combined = cv2.bitwise_or(combined, prior)
    if not seed_is_scribble:
        combined = cv2.bitwise_or(combined, seed)
    combined[scope == 0] = 0
    if veto is not None:
        combined[veto > 0] = 0

    black_core, black_core_diag = _black_hair_core_from_seed(
        source_bgr, seed, scope, veto)
    if black_core is not None and np.any(black_core):
        combined = cv2.bitwise_or(combined, black_core)

    diagnostics = {
        "reason": "manual_seed",
        "seed_is_scribble": bool(seed_is_scribble),
        "black_core": black_core_diag,
        "selection_runs": [],
        "generator_expand": None,
    }

    # mc-v2's own colour grouping is useful for visible object completion, but
    # each result is independently validated. It no longer gets blanket trust.
    try:
        from core.mcv2_semantic_selection import (
            SemanticSelectionSettings, select_semantic_region)
        settings = SemanticSelectionSettings(
            complete_hair_object=True,
            max_fraction=0.16,
            hair_object_max_ratio=1.85,
        )
        for seed_xy in _seed_points_from_mask(seed, max_points=4):
            try:
                selected, raw_diag = select_semantic_region(
                    source_bgr, base_bgr, seed_xy, settings=settings,
                    limit_mask=scope, occlusion_mask=None,
                    complete_hair_object=True)
                safe, safe_diag = _safe_selection_candidate(
                    selected, seed, scope, veto)
                if safe is not None and np.any(safe):
                    if black_core is not None and np.any(black_core):
                        near_core = cv2.dilate(
                            black_core,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)), 1)
                        if int(np.count_nonzero((safe > 0) & (near_core > 0))) < 8:
                            safe_diag = dict(safe_diag)
                            safe_diag["accepted"] = False
                            safe_diag["reason"] = "not_connected_to_black_core"
                            safe = None
                    if safe is not None and np.any(safe):
                        combined = cv2.bitwise_or(combined, safe)
                diagnostics["selection_runs"].append({
                    "seed_xy": [int(seed_xy[0]), int(seed_xy[1])],
                    "validation": safe_diag,
                    "complete_hair_object": dict((raw_diag or {}).get("complete_hair_object", {})),
                })
            except Exception as exc:  # noqa: BLE001
                diagnostics["selection_runs"].append({
                    "seed_xy": [int(seed_xy[0]), int(seed_xy[1])],
                    "error": f"{type(exc).__name__}: {exc}"})
    except Exception as exc:  # noqa: BLE001
        diagnostics["selection_error"] = f"{type(exc).__name__}: {exc}"

    # Generator features are computed only on a compact local component map.
    # Use the positive scribble itself as the anchor even when its literal pixels
    # are not paintable; use combined as an additional prior afterwards.
    if generator_feature_provider is not None:
        try:
            from core.semantic_region_fusion import expand_seed_regions_with_generator
            generator_seed = (black_core if black_core is not None and np.any(black_core)
                              else seed)
            expanded, exp_diag = expand_seed_regions_with_generator(
                source_bgr, generator_seed,
                base_bgr=base_bgr,
                negative_mask=veto,
                generator_feature_provider=generator_feature_provider,
                enable_trappedball=bool(use_trappedball),
                preserve_seed=bool(black_core is not None and np.any(black_core))
                              or not bool(seed_is_scribble))
            expanded = _binary(expanded, shape)
            if expanded is not None and np.any(expanded):
                combined = cv2.bitwise_or(combined, expanded)
            diagnostics["generator_expand"] = dict(exp_diag or {})
        except Exception as exc:  # noqa: BLE001
            diagnostics["generator_expand"] = {"error": f"{type(exc).__name__}: {exc}"}

    combined[scope == 0] = 0
    if veto is not None and np.any(veto):
        combined[veto > 0] = 0

    # Final neck-shadow pass must run on the *completed* mask as well.  A neck
    # shadow can remain attached to the initial black core and therefore evade
    # candidate-level rejection; after completion the directional structure is
    # clearer and the unsupported central bridge can be removed safely.
    final_neck_prune = None
    if black_core is not None and np.any(black_core) and np.any(combined):
        pruned_final, final_neck_prune = _prune_lower_neck_shadow(
            combined, seed, source_bgr, veto)
        if pruned_final is not None and np.any(pruned_final):
            combined = _binary(pruned_final, shape)
            combined[scope == 0] = 0
            if veto is not None and np.any(veto):
                combined[veto > 0] = 0
    diagnostics["final_neck_shadow_prune"] = final_neck_prune

    refined_completion, refine_diag = _refine_completed_hair_mask(
        combined, source_bgr, seed, scope, veto)
    if refined_completion is not None and np.any(refined_completion):
        combined = _binary(refined_completion, shape)
    diagnostics["final_refine"] = refine_diag

    # Final whole-mask circuit breaker.  No matter what an upstream heuristic
    # does, a local hair request cannot become a page-sized recolour.
    seed_area = max(1, int(np.count_nonzero(seed)))
    result_area = int(np.count_nonzero(combined))
    scope_area = max(1, int(np.count_nonzero(scope)))
    page_area = max(1, combined.size)
    max_pixels = min(
        max(seed_area + 768, int(page_area * 0.12)),
        max(seed_area + 768, int(scope_area * 0.58)),
        max(seed_area + 768, int(seed_area * 10.0)),
    )
    if result_area > max_pixels:
        fallback = np.zeros(shape, np.uint8)
        if prior is not None:
            fallback = cv2.bitwise_or(fallback, prior)
        if not seed_is_scribble:
            fallback = cv2.bitwise_or(fallback, seed)
        fallback[scope == 0] = 0
        if veto is not None:
            fallback[veto > 0] = 0
        diagnostics["final_guard"] = {
            "rejected": True, "proposal_pixels": result_area,
            "max_pixels": int(max_pixels)}
        combined = fallback
    else:
        diagnostics["final_guard"] = {
            "rejected": False, "proposal_pixels": result_area,
            "max_pixels": int(max_pixels)}

    before = int(np.count_nonzero(seed))
    after = int(np.count_nonzero(combined))
    diagnostics["reason"] = "manual_hair_completed"
    diagnostics["seed_pixels"] = before
    diagnostics["added_pixels"] = max(0, after - (0 if seed_is_scribble else before))
    diagnostics["result_pixels"] = after
    return combined, diagnostics


def refine_iris_mask(eye_mask: np.ndarray | None, source_bgr: np.ndarray,
                     base_bgr: np.ndarray) -> tuple[np.ndarray | None, dict]:
    """Turn a coarse semantic eye mask into a conservative iris-only mask."""
    if eye_mask is None:
        return None, {"reason": "no_eye_mask", "iris_pixels": 0}
    eye = _binary(eye_mask, source_bgr.shape[:2])
    if eye is None or not np.any(eye):
        return None, {"reason": "empty_eye_mask", "iris_pixels": 0}

    try:
        from core.pupil_boundary import constrain_pupil_mask_to_lineart
    except Exception:
        constrain_pupil_mask_to_lineart = None

    final = np.zeros_like(eye)
    component_diags = []
    page_area = max(1, eye.shape[0] * eye.shape[1])
    min_area = max(6, int(page_area * 0.000002))
    for _label, area, box, center, labels in _component_rows(eye, min_area=min_area):
        component = labels == _label
        if area > page_area * 0.035:
            # A semantic parser sometimes merges a face/large patch into eye.
            component_diags.append({"box": box, "reason": "too_large"})
            continue
        cx, cy = int(round(center[0])), int(round(center[1]))
        isolated = None
        if constrain_pupil_mask_to_lineart is not None:
            isolated = constrain_pupil_mask_to_lineart(
                source_bgr, component.astype(np.uint8) * 255, seed=(cx, cy))
        isolated_pixels = int(np.count_nonzero(isolated)) if isolated is not None else 0
        # If the line guard returns almost the whole semantic eye, it did not
        # actually find an iris contour. Use a compact central fallback.
        if (isolated is None or isolated_pixels < 4 or
                isolated_pixels > int(area * 0.72)):
            isolated = _central_iris_fallback(component, source_bgr, base_bgr)
            reason = "central_fallback"
        else:
            reason = "line_component"
        if isolated is not None:
            final[isolated > 0] = 255
        component_diags.append({
            "box": box, "eye_pixels": int(area),
            "iris_pixels": int(np.count_nonzero(isolated)) if isolated is not None else 0,
            "reason": reason,
        })

    final = _paintable(final, source_bgr, ink_threshold=76, erode_px=0,
                       preserve_dark_fill=True)
    pixels = int(np.count_nonzero(final)) if final is not None else 0
    if pixels == 0:
        return None, {"reason": "iris_refinement_empty", "iris_pixels": 0,
                      "components": component_diags}
    return final, {"reason": "iris_ready", "iris_pixels": pixels,
                   "components": component_diags}


def _mask_fingerprint(mask: np.ndarray | None) -> tuple:
    if mask is None:
        return (0, 0, 0)
    import zlib
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    sy = max(1, arr.shape[0] // 64)
    sx = max(1, arr.shape[1] // 64)
    sample = np.ascontiguousarray(arr[::sy, ::sx][:64, :64])
    return (arr.shape[0], arr.shape[1], int(zlib.adler32(sample.tobytes())))


def _image_fingerprint(image: np.ndarray) -> tuple:
    import zlib
    arr = np.asarray(image)
    h, w = arr.shape[:2]
    sy, sx = max(1, h // 64), max(1, w // 64)
    sample = np.ascontiguousarray(arr[::sy, ::sx][:64, :64])
    return (h, w, int(zlib.adler32(sample.tobytes())))


def build_semantic_masks(base_bgr: np.ndarray, source_bgr: np.ndarray,
                         *, parser_backend: str = "auto",
                         parser_allow_download: bool = False,
                         manual_masks: dict | None = None,
                         generator_feature_provider=None,
                         generator_semantic_provider=None,
                         use_region_fusion: bool = True,
                         use_trappedball: bool = True,
                         use_clip_regions: bool = True,
                         clip_allow_download: bool = False,
                         use_cache: bool = True) -> SemanticMaskBundle:
    """Build authoritative semantic masks from multiple independent evidences.

    Caller-provided exact ``hair``/``iris``/``background`` masks remain final.
    Optional rough scribbles may be supplied as ``hair_positive`` /
    ``hair_negative`` or ``eye_positive`` / ``eye_negative``; they expand only
    to the line-bounded regions selected by the same fusion map.
    """
    manual_masks = dict(manual_masks or {})
    h, w = base_bgr.shape[:2]
    if source_bgr.shape[:2] != (h, w):
        source_bgr = cv2.resize(source_bgr, (w, h), interpolation=cv2.INTER_AREA)

    key = (
        "semantic-mask-v5156",
        _image_fingerprint(source_bgr), _image_fingerprint(base_bgr),
        str(parser_backend), bool(parser_allow_download), bool(use_region_fusion),
        bool(use_trappedball), bool(use_clip_regions), bool(generator_feature_provider),
        bool(generator_semantic_provider),
        getattr(generator_semantic_provider, "cache_token", None),
        tuple(sorted((str(k), _mask_fingerprint(v)) for k, v in manual_masks.items()
                     if isinstance(v, np.ndarray))),
    )

    def _compute() -> SemanticMaskBundle:
        # Generator-native semantics are authoritative when a trained head is
        # present.  The old parser/rule chain remains a fail-closed fallback; it
        # is no longer allowed to expand a valid head HairMask merely because a
        # black line happens to look hair-like.
        head = {}
        head_diag = {"enabled": bool(generator_semantic_provider), "available": False}
        if generator_semantic_provider is not None:
            try:
                prediction = generator_semantic_provider(source_bgr)
                if prediction is not None:
                    from core.generator_semantic_runtime import prediction_to_parser_dict
                    head = prediction_to_parser_dict(prediction)
                    head_diag = dict(head.get("diagnostics") or {})
                    head_diag["enabled"] = True
                    head_diag["available"] = True
            except FileNotFoundError:
                head_diag = {"enabled": True, "available": False, "reason": "weights_missing"}
            except Exception as exc:  # semantic head must never break colorization
                head_diag = {"enabled": True, "available": False,
                             "reason": f"head_error:{type(exc).__name__}:{exc}"}

        head_available = bool(head_diag.get("available", False))
        parsed = None
        parser_diag = {"skipped": bool(head_available)}
        if not head_available:
            try:
                from core.ai_hair_parser import parse_semantic_regions
                parsed = parse_semantic_regions(
                    base_bgr, source_bgr=source_bgr,
                    preferred_backend=parser_backend,
                    allow_download=bool(parser_allow_download))
                parser_diag = dict(getattr(parsed, "diagnostics", {}) or {})
            except Exception as exc:  # noqa: BLE001 - masks are optional/fail closed
                parser_diag = {"reason": f"parser_error:{type(exc).__name__}:{exc}"}

        head_hair = _binary(head.get("hair_mask"), (h, w)) if head_available else None
        head_iris = _binary(head.get("iris_mask"), (h, w)) if head_available else None
        head_background = _binary(head.get("background_mask"), (h, w)) if head_available else None
        head_skin = _binary(head.get("skin_mask"), (h, w)) if head_available else None
        head_clothes = _binary(head.get("cloth_mask"), (h, w)) if head_available else None
        head_glasses = _binary(head.get("glasses_mask"), (h, w)) if head_available else None
        head_face_shadow = _binary(head.get("face_shadow_mask"), (h, w)) if head_available else None

        semantic_hair = (head_hair if head_available else
                         _binary(getattr(parsed, "hair_mask", None) if parsed is not None else None, (h, w)))
        semantic_eye = (_binary(getattr(parsed, "eye_mask", None), (h, w))
                        if parsed is not None else None)
        foreground = _binary(manual_masks.get("foreground"), (h, w))
        if foreground is None and head_available and head_background is not None:
            foreground = cv2.bitwise_not(head_background)
        if foreground is None and parsed is not None:
            foreground = _binary(getattr(parsed, "foreground_mask", None), (h, w))
        semantic_bg = (head_background if head_available else
                       (cv2.bitwise_not(foreground) if foreground is not None and np.any(foreground) else None))

        fusion = None
        fusion_diag = {"enabled": False}
        clip_classifier = None
        if use_region_fusion and use_clip_regions:
            candidate = _get_region_classifier(bool(clip_allow_download))
            if candidate is not None and candidate.available:
                clip_classifier = candidate
        eye_scribble_requested = any(
            manual_masks.get(name) is not None
            for name in ("eye_positive", "eye_negative"))
        evidence_present = any(m is not None and np.any(m) for m in
                               (semantic_hair, semantic_eye, foreground))
        # Hair scribbles have a dedicated bounded completion path below.  Do
        # not feed a page-wide line-region map into the generator merely because
        # a hair stroke exists; that was both slow (thousands of regions) and
        # the source of the old over-expansion failure.
        if (not head_available and use_region_fusion and
                (evidence_present or clip_classifier is not None or eye_scribble_requested)):
            try:
                from core.semantic_region_fusion import fuse_semantic_regions
                fusion = fuse_semantic_regions(
                    source_bgr,
                    hair_prior=semantic_hair,
                    eye_prior=semantic_eye,
                    foreground_prior=foreground,
                    background_prior=semantic_bg,
                    clip_classifier=clip_classifier,
                    generator_feature_provider=(generator_feature_provider if evidence_present else None),
                    enable_trappedball=bool(use_trappedball))
                fusion_diag = dict(fusion.diagnostics or {})
                fusion_diag["enabled"] = True
            except Exception as exc:  # fail closed to semantic priors
                fusion_diag = {"enabled": True,
                               "error": f"{type(exc).__name__}: {exc}"}

        exact_hair = _binary(manual_masks.get("hair"), (h, w))
        rough_hair_seed = _binary(manual_masks.get("hair_positive"), (h, w))
        negative_hair = _binary(manual_masks.get("hair_negative"), (h, w))
        if exact_hair is not None:
            # A manual region mask may be incomplete; keep it authoritative but
            # allow bounded mc-v2/generator completion around it.
            hair, hair_snap_diag = complete_manual_hair_mask(
                base_bgr, source_bgr, exact_hair,
                negative_mask=negative_hair,
                generator_feature_provider=generator_feature_provider,
                use_trappedball=bool(use_trappedball),
                seed_is_scribble=False)
        else:
            if head_available:
                hair = _binary(head_hair, (h, w))
                hair_snap_diag = {"reason": "generator_semantic_head", "authoritative": True}
            else:
                hair = (_binary(fusion.hair_mask, (h, w)) if fusion is not None
                        else semantic_hair)
                if fusion is None:
                    hair, hair_snap_diag = snap_semantic_to_line_regions(hair, source_bgr)
                else:
                    hair_snap_diag = {"reason": "region_fusion",
                                      "backend": (fusion_diag.get("segmentation") or {}).get("backend")}
            if rough_hair_seed is not None and np.any(rough_hair_seed):
                # Positive strokes are guidance only.  They must never become
                # literal paint pixels (the source of the previous blue-page
                # failure); the recovered local object is what gets painted.
                completed_hair, manual_complete_diag = complete_manual_hair_mask(
                    base_bgr, source_bgr, rough_hair_seed,
                    negative_mask=negative_hair,
                    generator_feature_provider=generator_feature_provider,
                    use_trappedball=bool(use_trappedball),
                    seed_is_scribble=True,
                    prior_mask=hair)
                if completed_hair is not None and np.any(completed_hair):
                    hair = completed_hair
                hair_snap_diag["manual_completion"] = manual_complete_diag
            elif negative_hair is not None and hair is not None:
                hair = _binary(hair, (h, w))
                hair[negative_hair > 0] = 0
        if head_available and exact_hair is None and rough_hair_seed is None:
            # A trained generator-native Head is already the semantic boundary.
            # Do not punch structural black strokes back out of its HairMask:
            # doing so was the reason correctly classified thin bangs remained
            # black after a requested recolour. Shading-preserving replacement
            # retains their luminance naturally without deleting them from the
            # authoritative mask.
            hair = _binary(hair, (h, w))
            hair_snap_diag["paintability"] = "semantic_head_full_mask"
        else:
            hair = _paintable(hair, source_bgr, ink_threshold=76, ink_shell_px=1.15)
        # Authoritative final neck-shadow pruning runs after thin protected ink
        # has been removed. At this stage a chin/neck shadow that was glued to
        # the black-hair core is usually disconnected from the persistent
        # vertical hair structure and can be rejected without damaging the
        # right-side hanging lock.
        final_seed_for_neck = rough_hair_seed if rough_hair_seed is not None else exact_hair
        if (not head_available and final_seed_for_neck is not None
                and np.any(final_seed_for_neck)
                and hair is not None and np.any(hair)):
            pruned_hair, post_paint_neck_diag = _prune_lower_neck_shadow(
                hair, final_seed_for_neck, source_bgr, negative_hair)
            if pruned_hair is not None and np.any(pruned_hair):
                hair = _binary(pruned_hair, (h, w))
            hair_snap_diag["post_paint_neck_shadow_prune"] = post_paint_neck_diag

        if (not head_available and final_seed_for_neck is not None
                and np.any(final_seed_for_neck) and hair is not None
                and np.any(hair)):
            refine_scope = _seed_scope_mask(final_seed_for_neck, (h, w), negative_mask=negative_hair)
            refined_hair, post_refine_diag = _refine_completed_hair_mask(
                hair, source_bgr, final_seed_for_neck, refine_scope, negative_hair)
            if refined_hair is not None and np.any(refined_hair):
                hair = _binary(refined_hair, (h, w))
            hair_snap_diag["post_paint_refine"] = post_refine_diag

        eye = _binary(manual_masks.get("eye"), (h, w))
        if eye is None:
            eye = (_binary(fusion.eye_mask, (h, w)) if fusion is not None
                   else semantic_eye)
        if fusion is not None and manual_masks.get("eye") is None and (
                manual_masks.get("eye_positive") is not None
                or manual_masks.get("eye_negative") is not None):
            try:
                from core.scribble_mask_refiner import refine_mask_with_scribbles
                eye, eye_scrib_diag = refine_mask_with_scribbles(
                    fusion.labels,
                    positive=manual_masks.get("eye_positive"),
                    negative=manual_masks.get("eye_negative"),
                    prior=eye, source_bgr=source_bgr,
                    neighbour_steps=0)
                fusion_diag["eye_scribble"] = eye_scrib_diag
            except Exception as exc:
                fusion_diag["eye_scribble_error"] = f"{type(exc).__name__}: {exc}"

        manual_iris = _binary(manual_masks.get("iris"), (h, w))
        if manual_iris is not None:
            iris = _paintable(manual_iris, source_bgr, ink_threshold=76,
                              preserve_dark_fill=True)
            iris_diag = {"reason": "manual_iris", "iris_pixels": int(np.count_nonzero(iris))}
        elif head_available:
            iris = _paintable(head_iris, source_bgr, ink_threshold=76,
                              preserve_dark_fill=True) if head_iris is not None else None
            iris_diag = {"reason": "generator_semantic_head",
                         "iris_pixels": int(np.count_nonzero(iris)) if iris is not None else 0}
        else:
            iris, iris_diag = refine_iris_mask(eye, source_bgr, base_bgr)

        background = _binary(manual_masks.get("background"), (h, w))
        if background is None and head_available:
            background = _binary(head_background, (h, w))
        if background is None and fusion is not None:
            background = _binary(fusion.background_mask, (h, w))
        if background is None:
            if foreground is not None and np.any(foreground):
                background = cv2.bitwise_not(foreground)
            elif parsed is not None:
                semantic = np.zeros((h, w), np.uint8)
                for name in ("hair_mask", "eye_mask", "skin_mask", "cloth_mask", "face_mask"):
                    candidate = _binary(getattr(parsed, name, None), (h, w))
                    if candidate is not None:
                        semantic = cv2.bitwise_or(semantic, candidate)
                if np.count_nonzero(semantic) >= max(64, int(h * w * 0.008)):
                    semantic = cv2.dilate(semantic, np.ones((7, 7), np.uint8), 1)
                    background = cv2.bitwise_not(semantic)

        skin = (head_skin if head_available else
                _binary(getattr(parsed, "skin_mask", None), (h, w)) if parsed is not None else None)
        manual_cloth = _binary(manual_masks.get("cloth"), (h, w))
        cloth = (head_clothes if head_available else
                 (manual_cloth if manual_cloth is not None else
                  (_binary(getattr(parsed, "cloth_mask", None), (h, w))
                   if parsed is not None else None)))
        glasses = (head_glasses if head_available else
                   _binary(getattr(parsed, "eyeglass_mask", None), (h, w)) if parsed is not None else None)
        face_shadow = head_face_shadow if head_available else None

        # A parser cloth mask is only a semantic proposal.  On monochrome
        # multi-panel manga it frequently labels screentone/background as
        # clothing.  Keep a trained generator head authoritative, but make the
        # legacy/parser path fail closed unless the proposal lands in a small,
        # enclosed line-art region and agrees with the foreground prior.
        garment_safety = {"reason": "generator_semantic_head" if head_available else "not_run"}
        if not head_available and cloth is not None and np.any(cloth):
            try:
                from core.garment_uniform import build_safe_garment_mask
                safe_garment = build_safe_garment_mask(
                    source_bgr, cloth,
                    foreground_mask=foreground,
                    negative_masks={
                        "hair": hair, "skin": skin, "eyes": eye,
                        "iris": iris, "glasses": glasses,
                        "face_shadow": face_shadow, "background": background,
                    },
                )
                cloth = safe_garment.mask if np.any(safe_garment.mask) else None
                garment_safety = dict(safe_garment.diagnostics)
            except Exception as exc:
                cloth = None
                garment_safety = {
                    "reason": f"safety_error:{type(exc).__name__}",
                    "accepted_pixels": 0,
                }

        return SemanticMaskBundle(
            hair_mask=hair,
            eye_mask=eye,
            iris_mask=iris,
            skin_mask=skin,
            cloth_mask=cloth,
            glasses_mask=glasses,
            face_shadow_mask=face_shadow,
            background_mask=background,
            foreground_mask=foreground,
            diagnostics={
                "generator_semantic_head": head_diag,
                "parser": parser_diag,
                "region_fusion": fusion_diag,
                "hair_snap": hair_snap_diag,
                "iris": iris_diag,
                "garment_safety": garment_safety,
                "hair_pixels": int(np.count_nonzero(hair)) if hair is not None else 0,
                "eye_pixels": int(np.count_nonzero(eye)) if eye is not None else 0,
                "skin_pixels": int(np.count_nonzero(skin)) if skin is not None else 0,
                "cloth_pixels": int(np.count_nonzero(cloth)) if cloth is not None else 0,
                "glasses_pixels": int(np.count_nonzero(glasses)) if glasses is not None else 0,
                "face_shadow_pixels": int(np.count_nonzero(face_shadow)) if face_shadow is not None else 0,
                "background_pixels": int(np.count_nonzero(background)) if background is not None else 0,
            },
        )

    if not use_cache:
        return _compute()
    try:
        from core.semantic_job_manager import semantic_jobs
        return semantic_jobs.get_or_compute(key, _compute)
    except Exception:
        return _compute()


def resize_semantic_masks(bundle: SemanticMaskBundle, shape: tuple[int, int]) -> SemanticMaskBundle:
    """Resize a cached mask bundle with nearest-neighbour semantics."""
    h, w = shape
    return SemanticMaskBundle(
        hair_mask=_binary(bundle.hair_mask, (h, w)),
        eye_mask=_binary(bundle.eye_mask, (h, w)),
        iris_mask=_binary(bundle.iris_mask, (h, w)),
        skin_mask=_binary(bundle.skin_mask, (h, w)),
        cloth_mask=_binary(bundle.cloth_mask, (h, w)),
        glasses_mask=_binary(bundle.glasses_mask, (h, w)),
        face_shadow_mask=_binary(bundle.face_shadow_mask, (h, w)),
        background_mask=_binary(bundle.background_mask, (h, w)),
        foreground_mask=_binary(bundle.foreground_mask, (h, w)),
        diagnostics=dict(bundle.diagnostics or {}),
    )

def _hard_replace(base_bgr: np.ndarray, replacement_bgr: np.ndarray,
                  mask: np.ndarray | None) -> tuple[np.ndarray, int]:
    """Overwrite selected pixels; never alpha blend with the base image."""
    if mask is None:
        return base_bgr.copy(), 0
    active = np.asarray(mask) > 0
    if not np.any(active):
        return base_bgr.copy(), 0
    if replacement_bgr.shape != base_bgr.shape:
        raise ValueError("replacement shape must match base image")
    out = base_bgr.copy()
    out[active] = replacement_bgr[active]
    return out, int(np.count_nonzero(active))


def render_recolor_replacement(base_bgr: np.ndarray, mask: np.ndarray | None,
                               target_rgb: tuple[int, int, int],
                               *, mode: str = "shading") -> np.ndarray:
    """Render a complete replacement image for one mask.

    ``mode=shading`` is the default: chosen hue/saturation are authoritative
    while mc-v2 luminance modelling is retained. The returned image is *not*
    composited; the caller later performs a binary overwrite.
    """
    if mask is None or not np.any(mask):
        return base_bgr.copy()
    from core.perceptual_recolor import recolor_with_mode
    alpha = (np.asarray(mask) > 0).astype(np.float32)
    return recolor_with_mode(
        base_bgr, tuple(int(v) for v in target_rgb), alpha,
        active=alpha > 0, mode=mode, pupil_blend=False)


def render_background_replacement(base_bgr: np.ndarray, source_bgr: np.ndarray,
                                  *, mode: str = "grayscale") -> np.ndarray:
    mode = str(mode or "keep").strip().lower()
    if mode in {"keep", "off", "none"}:
        return base_bgr.copy()
    if source_bgr.shape[:2] != base_bgr.shape[:2]:
        source_bgr = cv2.resize(source_bgr, (base_bgr.shape[1], base_bgr.shape[0]),
                                interpolation=cv2.INTER_AREA)
    if mode in {"source_bw", "restore_bw", "monochrome_source"}:
        gray = _source_gray(source_bgr, base_bgr.shape[:2])
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if mode in {"grayscale", "gray", "mono"}:
        gray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if mode in {"desaturate", "low_saturation"}:
        hsv = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1].astype(np.float32) * 0.22, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    raise ValueError(f"unsupported background replacement mode: {mode}")



def build_character_semantic_overrides(page_context, character_library,
                                       masks: SemanticMaskBundle | None,
                                       shape: tuple[int, int]) -> list[dict]:
    """Build per-character Hair/Iris overwrite layers from identity assignments.

    CharacterLibrary already resolves page-local regions to stable book-level
    character ids and canonical colours.  This adapter intersects those region
    assignments with the authoritative semantic masks, so identity information
    can choose *which colour* a matched character receives without gaining any
    authority over mask boundaries.

    The returned rows are intentionally renderer-agnostic dictionaries.  They
    are consumed by :func:`apply_semantic_mask_overrides` after the global Hair
    layer and before/after the global Iris layer according to semantic priority.
    """
    if page_context is None or character_library is None or masks is None:
        return []
    assignments = dict(getattr(page_context, "identity_assignments", {}) or {})
    segmentation = getattr(page_context, "segmentation", None)
    labels = getattr(segmentation, "labels", None) if segmentation is not None else None
    if not assignments or labels is None:
        return []

    h, w = int(shape[0]), int(shape[1])
    label_map = np.asarray(labels, dtype=np.int32)
    if label_map.shape[:2] != (h, w):
        label_map = cv2.resize(label_map.astype(np.float32), (w, h),
                               interpolation=cv2.INTER_NEAREST).astype(np.int32)

    hair_support = _binary(masks.hair_mask, (h, w))
    iris_support = _binary(masks.iris_mask, (h, w))
    grouped: dict[tuple[int, str], dict] = {}

    for region_id, info in assignments.items():
        if not isinstance(info, dict):
            continue
        attr = str(info.get("attribute") or "").lower()
        if attr not in {"hair", "eyes"}:
            continue
        if not bool(info.get("lock_allowed", False) or info.get("forced", False)):
            continue
        char_id = info.get("char_id")
        if char_id is None:
            continue
        try:
            char_id = int(char_id)
            rid = int(region_id)
        except Exception:
            continue
        rgb = info.get("rgb")
        if rgb is None:
            profile = next((c for c in getattr(character_library, "characters", [])
                            if int(getattr(c, "char_id", -1)) == char_id), None)
            key = "eyes" if attr == "eyes" else "hair"
            value = (getattr(profile, "colors", {}) or {}).get(key) if profile is not None else None
            if isinstance(value, str) and len(value) == 7 and value.startswith("#"):
                try:
                    rgb = tuple(int(value[i:i+2], 16) for i in (1, 3, 5))
                except Exception:
                    rgb = None
        if rgb is None:
            continue
        try:
            rgb = tuple(int(np.clip(v, 0, 255)) for v in rgb[:3])
        except Exception:
            continue
        key = (char_id, attr)
        row = grouped.setdefault(key, {
            "char_id": char_id, "attribute": attr, "target_rgb": rgb,
            "region_ids": [],
        })
        row["region_ids"].append(rid)
        # Prefer an explicit/forced assignment colour if different rows disagree.
        if bool(info.get("forced", False)):
            row["target_rgb"] = rgb

    out: list[dict] = []
    for (_char_id, attr), row in sorted(grouped.items()):
        ids = np.asarray(sorted(set(row["region_ids"])), dtype=np.int32)
        if ids.size == 0:
            continue
        region_mask = np.isin(label_map, ids)
        support = iris_support if attr == "eyes" else hair_support
        if support is not None:
            region_mask &= support > 0
        if not np.any(region_mask):
            continue
        out.append({
            "char_id": int(row["char_id"]),
            "semantic": "iris" if attr == "eyes" else "hair",
            "target_rgb": tuple(row["target_rgb"]),
            "mask": region_mask.astype(np.uint8) * 255,
            "region_ids": ids.tolist(),
        })
    return out

def apply_semantic_mask_overrides(base_bgr: np.ndarray, source_bgr: np.ndarray,
                                  controls: dict | None, *,
                                  parser_backend: str = "auto",
                                  parser_allow_download: bool = False,
                                  masks: SemanticMaskBundle | None = None,
                                  character_overrides: list[dict] | None = None
                                  ) -> tuple[np.ndarray, SemanticMaskBundle | None, dict]:
    """Apply enabled semantic layers as deterministic overwrite operations."""
    cfg = dict(controls or {})
    if not cfg.get("enabled", False):
        return base_bgr.copy(), masks, {"enabled": False, "applied": False}

    manual_masks = dict(cfg.get("manual_masks") or {})
    if masks is None:
        fusion_cfg = dict(cfg.get("fusion") or {})
        masks = build_semantic_masks(
            base_bgr, source_bgr,
            parser_backend=parser_backend,
            parser_allow_download=parser_allow_download,
            manual_masks=manual_masks,
            generator_feature_provider=fusion_cfg.get("generator_feature_provider"),
            generator_semantic_provider=fusion_cfg.get("generator_semantic_provider"),
            use_region_fusion=bool(fusion_cfg.get("enabled", True)),
            use_trappedball=bool(fusion_cfg.get("use_trappedball", True)),
            use_clip_regions=bool(fusion_cfg.get("use_clip_regions", True)),
            clip_allow_download=bool(fusion_cfg.get("clip_allow_download", False)),
            use_cache=bool(fusion_cfg.get("use_cache", True)))
    else:
        probe = next((m for m in (masks.hair_mask, masks.iris_mask, masks.skin_mask,
                                  masks.cloth_mask, masks.glasses_mask, masks.face_shadow_mask,
                                  masks.background_mask, masks.eye_mask, masks.foreground_mask)
                      if m is not None), None)
        if probe is not None and probe.shape[:2] != base_bgr.shape[:2]:
            masks = resize_semantic_masks(masks, base_bgr.shape[:2])

    out = base_bgr.copy()
    changed = {}
    applied_order = []

    bg_cfg = dict(cfg.get("background") or {})
    bg_mode = str(bg_cfg.get("mode", "keep"))
    if bg_cfg.get("enabled", False) and bg_mode not in {"keep", "off", "none"}:
        replacement = render_background_replacement(out, source_bgr, mode=bg_mode)
        out, n = _hard_replace(out, replacement, masks.background_mask)
        changed["background"] = n
        if n:
            applied_order.append("background")

    hair_cfg = dict(cfg.get("hair") or {})
    if hair_cfg.get("enabled", False):
        rgb = tuple(hair_cfg.get("target_rgb") or (24, 24, 24))
        replacement = render_recolor_replacement(
            out, masks.hair_mask, rgb,
            mode=str(hair_cfg.get("mode", "shading")))
        out, n = _hard_replace(out, replacement, masks.hair_mask)
        changed["hair"] = n
        if n:
            applied_order.append("hair")

    character_rows = list(character_overrides or [])
    character_changed = []
    for row in (character_rows if hair_cfg.get("enabled", False) else []):
        if str(row.get("semantic")) != "hair":
            continue
        mask = _binary(row.get("mask"), out.shape[:2])
        if mask is None or not np.any(mask):
            continue
        rgb = tuple(row.get("target_rgb") or (24, 24, 24))
        replacement = render_recolor_replacement(
            out, mask, rgb, mode=str(hair_cfg.get("mode", "shading")))
        out, n = _hard_replace(out, replacement, mask)
        if n:
            key = f"character_{int(row.get('char_id', -1))}_hair"
            changed[key] = n
            applied_order.append(key)
            character_changed.append({"char_id": int(row.get("char_id", -1)),
                                      "semantic": "hair", "pixels": n})

    iris_cfg = dict(cfg.get("iris") or {})
    if iris_cfg.get("enabled", False):
        rgb = tuple(iris_cfg.get("target_rgb") or (70, 135, 220))
        replacement = render_recolor_replacement(
            out, masks.iris_mask, rgb,
            mode=str(iris_cfg.get("mode", "shading")))
        out, n = _hard_replace(out, replacement, masks.iris_mask)
        changed["iris"] = n
        if n:
            applied_order.append("iris")

    for row in (character_rows if iris_cfg.get("enabled", False) else []):
        if str(row.get("semantic")) != "iris":
            continue
        mask = _binary(row.get("mask"), out.shape[:2])
        if mask is None or not np.any(mask):
            continue
        rgb = tuple(row.get("target_rgb") or (70, 135, 220))
        replacement = render_recolor_replacement(
            out, mask, rgb, mode=str(iris_cfg.get("mode", "shading")))
        out, n = _hard_replace(out, replacement, mask)
        if n:
            key = f"character_{int(row.get('char_id', -1))}_iris"
            changed[key] = n
            applied_order.append(key)
            character_changed.append({"char_id": int(row.get("char_id", -1)),
                                      "semantic": "iris", "pixels": n})

    garment_cfg = dict(cfg.get("garment") or cfg.get("clothing") or {})
    if garment_cfg.get("enabled", False):
        # The mask has already passed the line/foreground safety gate in
        # build_semantic_masks.  Recolour only Lab chroma so folds, lighting,
        # screentone and texture remain from the base generator output.
        try:
            from core.garment_uniform import uniform_garment_chroma
            rgb = tuple(garment_cfg.get("target_rgb") or (72, 104, 156))
            replacement = uniform_garment_chroma(
                out, masks.cloth_mask, rgb,
                strength=float(garment_cfg.get("strength", 0.90)),
            )
        except Exception:
            replacement = out.copy()
        out, n = _hard_replace(out, replacement, masks.cloth_mask)
        changed["garment"] = n
        if n:
            applied_order.append("garment")

    # Contract check used both by diagnostics and regression tests.
    union = np.zeros(base_bgr.shape[:2], np.uint8)
    layer_to_mask = {
        "background": masks.background_mask,
        "hair": masks.hair_mask,
        "iris": masks.iris_mask,
        "garment": masks.cloth_mask,
    }
    for name in applied_order:
        mask = layer_to_mask.get(name)
        if mask is not None:
            union[mask > 0] = 255
    for row in character_rows:
        mask = _binary(row.get("mask"), base_bgr.shape[:2])
        if mask is not None:
            union[mask > 0] = 255
    outside = union == 0
    outside_changed = int(np.count_nonzero(np.any(out != base_bgr, axis=2) & outside))

    diagnostics = {
        "enabled": True,
        "applied": bool(applied_order),
        "composition": "hard_replace",
        "applied_order": applied_order,
        "changed_pixels": changed,
        "character_overrides": character_changed,
        "outside_changed_pixels": outside_changed,
        "masks": dict(masks.diagnostics or {}),
    }
    return out, masks, diagnostics
