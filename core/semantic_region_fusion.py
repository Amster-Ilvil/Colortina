"""Multi-evidence semantic region fusion for hard-replace colour masks.

Evidence sources are deliberately independent:
1. anime semantic parser: coarse meaning (hair / eye / foreground);
2. line-art regions: exact boundaries;
3. optional CLIP region classifier: whole-region object label;
4. mc-v2 generator encoder embeddings: propagate identity across fragmented
   regions that the *colourizer itself* represents similarly.

No evidence source may paint pixels directly.  The output is still a binary
mask consumed by semantic_mask_layers._hard_replace().
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable

import cv2
import numpy as np


@dataclass
class RegionEvidence:
    region_id: int
    area: int
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    touches_border: bool
    semantic_hair: float = 0.0
    semantic_eye: float = 0.0
    semantic_foreground: float = 0.0
    semantic_background: float = 0.0
    clip_label: str | None = None
    clip_confidence: float = 0.0
    generator_hair_similarity: float = 0.0
    generator_eye_similarity: float = 0.0
    hair_score: float = 0.0
    eye_score: float = 0.0
    background_score: float = 0.0
    accepted: set[str] = field(default_factory=set)


@dataclass
class RegionFusionResult:
    hair_mask: np.ndarray | None
    eye_mask: np.ndarray | None
    background_mask: np.ndarray | None
    labels: np.ndarray
    regions: list[RegionEvidence]
    diagnostics: dict


def _binary(mask, shape):
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.shape != shape:
        arr = cv2.resize(arr.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return arr > 0


def _fast_labels(source_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    from core.region_map import build_region_map
    rm = build_region_map(source_bgr, line_low=78, gap_close=4)
    return rm.labels.astype(np.int32), {
        "backend": "region_map",
        "region_count": int(rm.labels.max()),
    }


def _region_area_map(labels: np.ndarray) -> np.ndarray:
    return np.bincount(labels.ravel(), minlength=int(labels.max()) + 1)


def _touches_border(labels: np.ndarray, rid: int) -> bool:
    return bool(np.any(labels[0] == rid) or np.any(labels[-1] == rid)
                or np.any(labels[:, 0] == rid) or np.any(labels[:, -1] == rid))


def _fast_is_suspicious(labels: np.ndarray, semantic_masks: list[np.ndarray | None]) -> tuple[bool, str]:
    if labels.max() <= 1:
        return True, "too_few_regions"
    areas = _region_area_map(labels)
    page = max(1, labels.size)
    for prior in semantic_masks:
        if prior is None or not np.any(prior):
            continue
        ids = np.unique(labels[prior])
        for rid in ids.tolist():
            rid = int(rid)
            if rid <= 0 or rid >= len(areas):
                continue
            region_area = int(areas[rid])
            overlap = int(np.count_nonzero((labels == rid) & prior))
            if overlap < 8:
                continue
            # A character semantic prior landing inside a page-border / huge
            # region is the classic open-line leak signal.
            if region_area > page * 0.34 or _touches_border(labels, rid):
                if overlap / max(1.0, float(np.count_nonzero(prior))) > 0.04:
                    return True, "semantic_prior_hits_leaky_region"
    return False, "fast_ok"


def choose_region_labels(source_bgr: np.ndarray, *, hair_prior=None, eye_prior=None,
                         foreground_prior=None, enable_trappedball: bool = True) -> tuple[np.ndarray, dict]:
    labels, diag = _fast_labels(source_bgr)
    priors = [_binary(hair_prior, labels.shape), _binary(eye_prior, labels.shape),
              _binary(foreground_prior, labels.shape)]
    suspicious, reason = _fast_is_suspicious(labels, priors)
    diag["fast_assessment"] = reason
    if suspicious and enable_trappedball:
        try:
            from core.trappedball_regions import segment_trappedball
            robust = segment_trappedball(source_bgr)
            if robust.labels.max() >= 2:
                rdiag = dict(robust.diagnostics)
                rdiag["fallback_from"] = reason
                return robust.labels.astype(np.int32), rdiag
        except Exception as exc:  # fail back to fast map
            diag["trappedball_error"] = f"{type(exc).__name__}: {exc}"
    return labels, diag


def _region_rows(labels: np.ndarray, *, semantic_masks: list[np.ndarray | None],
                 max_regions: int = 180) -> list[tuple[int, int, tuple[int, int, int, int], tuple[float, float]]]:
    areas = _region_area_map(labels)
    page_area = max(1, labels.size)
    semantic_ids: set[int] = set()
    for prior in semantic_masks:
        if prior is not None and np.any(prior):
            semantic_ids.update(int(v) for v in np.unique(labels[prior]).tolist() if int(v) > 0)
    ids = [i for i in range(1, len(areas)) if int(areas[i]) >= 10]
    ids.sort(key=lambda rid: int(areas[rid]), reverse=True)
    keep = set(ids[:max_regions]) | semantic_ids
    rows = []
    for rid in sorted(keep):
        area = int(areas[rid]) if rid < len(areas) else 0
        if area <= 0:
            continue
        yy, xx = np.where(labels == rid)
        if yy.size == 0:
            continue
        x0, x1 = int(xx.min()), int(xx.max())
        y0, y1 = int(yy.min()), int(yy.max())
        rows.append((rid, area, (x0, y0, x1 - x0 + 1, y1 - y0 + 1),
                     (float(xx.mean()), float(yy.mean()))))
    return rows


def _overlap(labels: np.ndarray, rid: int, mask: np.ndarray | None, area: int) -> float:
    if mask is None or area <= 0:
        return 0.0
    return float(np.count_nonzero((labels == rid) & mask)) / float(area)


def _cos(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0
    if a.shape != b.shape:
        n = min(a.size, b.size)
        a, b = a[:n], b[:n]
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Euclidean gap between two rectangles in pixels (0 when touching/overlap)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx = max(0.0, float(max(ax, bx) - min(ax + aw, bx + bw)))
    dy = max(0.0, float(max(ay, by) - min(ay + ah, by + bh)))
    return math.hypot(dx, dy)


def _ink_ratio_between(gray: np.ndarray | None, p0: tuple[float, float],
                       p1: tuple[float, float], threshold: int = 96) -> float:
    """Estimate whether many ink strokes obstruct a region-to-region bridge.

    A single separating contour is expected in manga and is therefore cheap. A
    long path dominated by ink usually means the regions are separated by other
    objects/panels and generator-feature similarity must not jump across it.
    """
    if gray is None or gray.size == 0:
        return 0.0
    h, w = gray.shape[:2]
    dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    n = int(np.clip(round(dist), 8, 96))
    xs = np.clip(np.rint(np.linspace(p0[0], p1[0], n)).astype(np.int32), 0, w - 1)
    ys = np.clip(np.rint(np.linspace(p0[1], p1[1], n)).astype(np.int32), 0, h - 1)
    return float(np.mean(gray[ys, xs] <= int(threshold)))


def _anchor_similarity(region: RegionEvidence, embeddings: dict[int, np.ndarray] | None,
                       anchors: list[RegionEvidence], *, page_diag: float,
                       source_gray: np.ndarray | None = None) -> float:
    """r2r-style conservative region grouping using appearance + edges.

    The mc-v2 encoder supplies appearance/structure similarity. Geometry and
    line-art barriers gate propagation so a distant visually-similar background
    region cannot become hair merely because its embedding is close.
    """
    if not embeddings or region.region_id not in embeddings or not anchors:
        return 0.0
    best = 0.0
    rx, ry = region.center
    for anchor in anchors:
        av = embeddings.get(anchor.region_id)
        if av is None:
            continue
        ax, ay = anchor.center
        spatial = math.hypot(rx - ax, ry - ay) / max(1.0, page_diag)
        gap = _bbox_gap(region.bbox, anchor.bbox) / max(1.0, page_diag)
        # Region correspondence is local. The bbox-gap gate is stricter than
        # the centre-distance gate and handles long thin hair fragments well.
        if spatial > 0.34 or gap > 0.16:
            continue
        ink_ratio = _ink_ratio_between(source_gray, region.center, anchor.center)
        if ink_ratio > 0.42:
            continue
        sim = _cos(embeddings[region.region_id], av)
        spatial_weight = float(np.clip(1.0 - spatial / 0.34, 0.0, 1.0))
        gap_weight = float(np.clip(1.0 - gap / 0.16, 0.0, 1.0))
        # One or two contour crossings are normal between hair fragments.
        # Penalise only paths that are substantially ink-dominated.
        edge_weight = float(np.clip(1.0 - max(0.0, ink_ratio - 0.20) / 0.22, 0.65, 1.0))
        score = sim * (0.68 + 0.20 * spatial_weight + 0.12 * gap_weight) * edge_weight
        best = max(best, score)
    return float(np.clip(best, 0.0, 1.0))


def _mask_from_ids(labels: np.ndarray, ids: set[int]) -> np.ndarray | None:
    if not ids:
        return None
    active = np.isin(labels, np.fromiter(ids, dtype=np.int32))
    return active.astype(np.uint8) * 255


def fuse_semantic_regions(source_bgr: np.ndarray, *, hair_prior=None, eye_prior=None,
                          foreground_prior=None, background_prior=None,
                          clip_classifier=None,
                          generator_feature_provider: Callable[[np.ndarray], dict[int, np.ndarray]] | None = None,
                          enable_trappedball: bool = True) -> RegionFusionResult:
    """Fuse region semantics into line-bounded binary masks.

    ``generator_feature_provider`` receives the chosen label map and returns a
    {region_id: embedding} dict. This keeps the fusion core independent of torch.
    """
    source = np.asarray(source_bgr)
    shape = source.shape[:2]
    hair = _binary(hair_prior, shape)
    eye = _binary(eye_prior, shape)
    fg = _binary(foreground_prior, shape)
    bg = _binary(background_prior, shape)
    if bg is None and fg is not None:
        bg = ~fg

    labels, seg_diag = choose_region_labels(
        source, hair_prior=hair, eye_prior=eye, foreground_prior=fg,
        enable_trappedball=enable_trappedball)
    rows = _region_rows(labels, semantic_masks=[hair, eye, fg, bg])
    page_h, page_w = labels.shape
    page_area = max(1, page_h * page_w)
    page_diag = math.hypot(page_w, page_h)
    if source.ndim == 2:
        source_gray = source.astype(np.uint8)
    else:
        source_gray = cv2.cvtColor(source.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    evidences: list[RegionEvidence] = []
    for rid, area, bbox, center in rows:
        e = RegionEvidence(
            region_id=rid, area=area, bbox=bbox, center=center,
            touches_border=_touches_border(labels, rid),
            semantic_hair=_overlap(labels, rid, hair, area),
            semantic_eye=_overlap(labels, rid, eye, area),
            semantic_foreground=_overlap(labels, rid, fg, area),
            semantic_background=_overlap(labels, rid, bg, area),
        )
        evidences.append(e)

    clip_used = False
    if clip_classifier is not None and evidences:
        try:
            bboxes = [e.bbox for e in evidences]
            labels_out = clip_classifier.classify(source, bboxes)
            if labels_out:
                clip_used = True
                for e, row in zip(evidences, labels_out):
                    e.clip_label, e.clip_confidence = str(row[0]), float(row[1])
        except Exception:
            clip_used = False

    embeddings = None
    if generator_feature_provider is not None and evidences:
        try:
            embeddings = generator_feature_provider(labels) or None
        except Exception:
            embeddings = None

    # First-pass semantic/CLIP anchors. High precision beats recall here.
    hair_anchors = [e for e in evidences if (
        e.semantic_hair >= 0.50 or
        (e.clip_label == "hair" and e.clip_confidence >= 0.58 and e.semantic_foreground >= 0.10) or
        (e.clip_label == "hair" and e.clip_confidence >= 0.74 and not e.touches_border))]
    eye_anchors = [e for e in evidences if (
        e.semantic_eye >= 0.48 or
        (e.clip_label == "eyes" and e.clip_confidence >= 0.72 and not e.touches_border))]

    hair_ids: set[int] = set()
    eye_ids: set[int] = set()
    background_ids: set[int] = set()

    for e in evidences:
        e.generator_hair_similarity = _anchor_similarity(
            e, embeddings, hair_anchors, page_diag=page_diag, source_gray=source_gray)
        e.generator_eye_similarity = _anchor_similarity(
            e, embeddings, eye_anchors, page_diag=page_diag, source_gray=source_gray)
        clip_hair = e.clip_confidence if e.clip_label == "hair" else 0.0
        clip_eye = e.clip_confidence if e.clip_label == "eyes" else 0.0
        clip_bg = e.clip_confidence if e.clip_label in {"background", "sky", "foliage", "stone", "wood", "water"} else 0.0

        gen_hair = max(0.0, (e.generator_hair_similarity - 0.76) / 0.24)
        # Encoder similarity is allowed to recover a missing *fragment* only
        # when another independent cue says it belongs to the character. This
        # is the r2r-style region grouping role; generator features alone never
        # label a distant background region as hair.
        gen_hair_vote = gen_hair * (0.82 if e.semantic_foreground >= 0.16 else 0.18)
        e.hair_score = float(np.clip(
            0.56 * e.semantic_hair + 0.20 * clip_hair + gen_hair_vote, 0.0, 1.0))
        gen_eye = max(0.0, (e.generator_eye_similarity - 0.80) / 0.20)
        e.eye_score = float(np.clip(
            0.72 * e.semantic_eye + 0.18 * clip_eye + 0.10 * gen_eye, 0.0, 1.0))
        border_bg = 0.72 if e.touches_border else 0.0
        e.background_score = float(np.clip(
            max(0.68 * e.semantic_background + 0.22 * clip_bg, border_bg)
            - 0.52 * max(e.semantic_hair, e.semantic_eye, e.semantic_foreground), 0.0, 1.0))

        area_ratio = e.area / float(page_area)
        grouped_hair = bool(
            hair_anchors and e.generator_hair_similarity >= 0.79
            and e.semantic_foreground >= 0.50
            and e.semantic_eye < 0.16
            and e.clip_label not in {"skin", "eyes", "bubble"})
        direct_clip_hair = bool(clip_hair >= 0.74 and not e.touches_border)
        if ((e.hair_score >= 0.46 or grouped_hair or direct_clip_hair) and area_ratio <= 0.20 and
                (not e.touches_border or e.semantic_hair >= 0.72)):
            hair_ids.add(e.region_id)
            if direct_clip_hair and e.hair_score < 0.46 and not grouped_hair:
                e.accepted.add("hair_clip")
            else:
                e.accepted.add("hair_grouped" if grouped_hair and e.hair_score < 0.46 else "hair")
        direct_clip_eye = bool(clip_eye >= 0.72 and not e.touches_border)
        if (e.eye_score >= 0.48 or direct_clip_eye) and area_ratio <= 0.04 and not e.touches_border:
            eye_ids.add(e.region_id)
            e.accepted.add("eye")
        direct_clip_bg = bool(clip_bg >= 0.78 and e.semantic_hair < 0.15 and e.semantic_eye < 0.15)
        if (e.background_score >= 0.58 or direct_clip_bg) and e.semantic_hair < 0.22 and e.semantic_eye < 0.20:
            background_ids.add(e.region_id)
            e.accepted.add("background")

    fused_hair = _mask_from_ids(labels, hair_ids)
    fused_eye = _mask_from_ids(labels, eye_ids)
    fused_bg = _mask_from_ids(labels, background_ids)

    # Preserve semantic recall if the region classifier is intentionally
    # conservative. The priors are still line-protected later by _paintable().
    if fused_hair is None and hair is not None and np.any(hair):
        fused_hair = hair.astype(np.uint8) * 255
    if fused_eye is None and eye is not None and np.any(eye):
        fused_eye = eye.astype(np.uint8) * 255
    if fused_bg is None and bg is not None and np.any(bg):
        fused_bg = bg.astype(np.uint8) * 255

    return RegionFusionResult(
        hair_mask=fused_hair,
        eye_mask=fused_eye,
        background_mask=fused_bg,
        labels=labels,
        regions=evidences,
        diagnostics={
            "segmentation": seg_diag,
            "clip_used": bool(clip_used),
            "generator_features_used": bool(embeddings),
            "candidate_regions": len(evidences),
            "hair_regions": sorted(hair_ids),
            "eye_regions": sorted(eye_ids),
            "background_regions": sorted(background_ids),
            "max_generator_hair_similarity": max(
                [e.generator_hair_similarity for e in evidences] or [0.0]),
        },
    )
