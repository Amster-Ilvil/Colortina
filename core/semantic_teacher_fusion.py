"""Multi-teacher fusion for generator-native semantic-head training.

This module is intentionally training-only.  It combines corrected user masks,
legacy high-precision masks, external parser masks and the stock generator's own
colour response into a dense/partial indexed semantic label.  Runtime inference
still depends only on ``generator.zip`` + ``generator_semantic_head.pth``.

Design goals:
* explicit corrected negatives (glasses/clothes/face shadow/skin/iris) always
  beat a weaker hair teacher;
* hair fusion is recall-oriented, because missing hair means the requested hair
  colour can never be applied;
* untrusted conflicts become IGNORE=255 rather than a guessed class;
* generator colour evidence may recover fine fringe strands, but only inside a
  local scope around an already-confirmed hair object.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import cv2
import numpy as np

from core.generator_semantic_head import DEFAULT_CLASSES
from core.semantic_training_sample import IGNORE_INDEX, build_indexed_semantic_label


@dataclass(frozen=True)
class TeacherFusionConfig:
    class_names: tuple[str, ...] = tuple(DEFAULT_CLASSES)
    human_weight: float = 8.0
    corrected_negative_weight: float = 10.0
    generator_head_weight: float = 2.2
    parser_weight: float = 1.8
    legacy_weight: float = 1.25
    generator_colour_weight: float = 1.45
    min_positive_score: float = 1.05
    conflict_margin: float = 0.18
    hair_recall_threshold: float = 0.42
    hair_colour_threshold: float = 0.44


def _as_prob(mask, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.shape[:2] != shape:
        interp = cv2.INTER_LINEAR if np.issubdtype(arr.dtype, np.floating) else cv2.INTER_NEAREST
        arr = cv2.resize(arr.astype(np.float32), (shape[1], shape[0]), interpolation=interp)
    arr = arr.astype(np.float32)
    if arr.size and float(np.nanmax(arr)) > 1.5:
        arr /= 255.0
    return np.clip(np.nan_to_num(arr, nan=0.0), 0.0, 1.0)


def _binary(mask, shape: tuple[int, int]) -> np.ndarray | None:
    p = _as_prob(mask, shape)
    return None if p is None else (p >= 0.5).astype(np.uint8)


def _bounded_hair_scope(seed: np.ndarray, shape: tuple[int, int], margin_ratio: float = 0.36) -> np.ndarray:
    h, w = shape
    ys, xs = np.where(seed > 0)
    if ys.size == 0:
        return np.zeros(shape, np.uint8)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    mx = max(18, int(round(max(bw, bh) * margin_ratio)))
    my_top = max(18, int(round(bh * 0.28)))
    my_bottom = max(32, int(round(bh * 0.58)))
    scope = np.zeros(shape, np.uint8)
    scope[max(0, y0-my_top):min(h, y1+my_bottom+1),
          max(0, x0-mx):min(w, x1+mx+1)] = 1
    # Keep a connected corridor around the seed but not the entire page bbox.
    k = max(9, int(round(max(bw, bh) * 0.10))) | 1
    grown = cv2.dilate(seed.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), 1)
    return np.maximum(scope, grown > 0).astype(np.uint8)


def generator_colour_hair_probability(
    base_bgr: np.ndarray,
    confirmed_hair: np.ndarray,
    *,
    source_bgr: np.ndarray | None = None,
    exclusion_masks: Mapping[str, np.ndarray | None] | None = None,
    scope_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Build a soft hair teacher from mc-v2's own colour response.

    The confirmed hair seed establishes the colour cluster.  Candidate pixels
    are scored in Lab/HSV and are accepted only in a local object scope.  This
    is useful for fine fringe strands that a corrected binary teacher missed.
    It is *not* used at runtime and therefore cannot make production masks leak.
    """
    base = np.asarray(base_bgr).astype(np.uint8)
    shape = base.shape[:2]
    seed = _binary(confirmed_hair, shape)
    if seed is None or not np.any(seed):
        return np.zeros(shape, np.float32), {"reason": "empty_seed"}

    # Inner seed pixels are less likely to contain antialiased outlines.
    inner = cv2.erode(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), 1)
    if int(np.count_nonzero(inner)) < 64:
        inner = seed.copy()

    lab = cv2.cvtColor(base, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV).astype(np.float32)
    values = lab[inner > 0]
    if values.size == 0:
        return np.zeros(shape, np.float32), {"reason": "empty_inner_seed"}
    center = np.median(values, axis=0)
    mad = np.median(np.abs(values - center[None, :]), axis=0)
    # Keep tolerances generous for shading but tighter on chroma than lightness.
    scales = np.maximum(mad * np.array([3.8, 3.0, 3.0], np.float32),
                        np.array([18.0, 8.0, 8.0], np.float32))
    z = (lab - center[None, None, :]) / scales[None, None, :]
    dist2 = np.sum(z * z, axis=2)
    colour_prob = np.exp(-0.5 * dist2).astype(np.float32)

    # Very dark neutral generator pixels are common for black hair and have an
    # unreliable hue.  Give them a small recall bonus, but only locally.
    sat = hsv[..., 1] / 255.0
    val = hsv[..., 2] / 255.0
    neutral_dark = np.clip((0.34 - sat) / 0.34, 0.0, 1.0) * np.clip((0.55 - val) / 0.55, 0.0, 1.0)
    probability = np.maximum(colour_prob, neutral_dark.astype(np.float32) * 0.72)

    scope = _binary(scope_mask, shape) if scope_mask is not None else _bounded_hair_scope(seed, shape)
    probability *= scope.astype(np.float32)

    # Generator colour alone must never claim explicit semantic negatives.
    exclusions = dict(exclusion_masks or {})
    negative = np.zeros(shape, np.uint8)
    for name in ("eyebrow", "eye", "glasses", "clothes", "face_shadow", "skin", "iris"):
        m = _binary(exclusions.get(name), shape)
        if m is not None:
            negative |= m
    if np.any(negative):
        # Small dilation protects antialiased borders around negative classes.
        negative = cv2.dilate(negative, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1)
        probability[negative > 0] = 0.0

    # If source line art is supplied, pure page-white areas cannot be fine hair.
    if source_bgr is not None:
        src = np.asarray(source_bgr)
        if src.shape[:2] != shape:
            src = cv2.resize(src, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
        gray = src if src.ndim == 2 else cv2.cvtColor(src.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        # Do not zero white hair/highlights completely; just downweight them.
        white = np.clip((gray.astype(np.float32) - 235.0) / 20.0, 0.0, 1.0)
        probability *= (1.0 - 0.52 * white)

    return np.clip(probability, 0.0, 1.0), {
        "reason": "generator_colour_teacher",
        "lab_center": [float(v) for v in center],
        "lab_scale": [float(v) for v in scales],
        "seed_pixels": int(np.count_nonzero(seed)),
        "scope_pixels": int(np.count_nonzero(scope)),
        "nonzero_probability": int(np.count_nonzero(probability > 0.05)),
    }




def refine_glasses_teacher_from_generator_colour(
    base_bgr: np.ndarray,
    source_bgr: np.ndarray,
    coarse_glasses: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Shrink a coarse glasses teacher to generator-supported frame pixels.

    Older bootstrap labels frequently covered nearby bangs/eye shadows together
    with glasses.  The stock generator tends to render glasses/skin-adjacent
    lines with warmer chroma than black hair.  This training-only teacher uses
    that response as a precise core, then admits only a 1-pixel neighbourhood
    of source-ink around it.
    """
    base=np.asarray(base_bgr).astype(np.uint8); shape=base.shape[:2]
    coarse=_binary(coarse_glasses,shape)
    if coarse is None or not np.any(coarse):
        return np.zeros(shape,np.float32), {"reason":"empty_glasses"}
    src=np.asarray(source_bgr)
    if src.shape[:2]!=shape:
        src=cv2.resize(src,(shape[1],shape[0]),interpolation=cv2.INTER_AREA)
    hsv=cv2.cvtColor(base,cv2.COLOR_BGR2HSV); sat=hsv[...,1]
    b,g,r=cv2.split(base)
    warm=((sat>=48)&(r.astype(np.int16)>=b.astype(np.int16)+6)&(coarse>0)).astype(np.uint8)
    gray=src if src.ndim==2 else cv2.cvtColor(src.astype(np.uint8),cv2.COLOR_BGR2GRAY)
    near=cv2.dilate(warm,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),1)>0
    precise=((coarse>0)&near&((gray<=125)|(sat>=35))).astype(np.uint8)
    # Keep a minimal corrected core if the warm response is unusually weak.
    if int(np.count_nonzero(precise)) < max(32,int(np.count_nonzero(coarse)*0.12)):
        precise=((coarse>0)&((gray<=110)&(sat>=20))).astype(np.uint8)
    return precise.astype(np.float32), {
        "reason":"generator_colour_glasses_teacher",
        "coarse_pixels":int(np.count_nonzero(coarse)),
        "warm_core_pixels":int(np.count_nonzero(warm)),
        "teacher_pixels":int(np.count_nonzero(precise)),
    }

def expand_dark_seed_teacher(
    source_bgr: np.ndarray,
    seed_mask: np.ndarray,
    *,
    exclusion_masks: Mapping[str, np.ndarray | None] | None = None,
    threshold: int = 105,
    close_kernel: int = 7,
    probe_kernel: int = 11,
    min_overlap: int = 16,
) -> tuple[np.ndarray, dict]:
    """Expand a corrected dark-class seed along its source-ink component.

    This is intended for *training teachers* such as black clothing/glasses.
    It never runs in production inference.  The expansion is limited to a local
    bbox around the corrected seed and explicit exclusions (for example hair)
    are subtracted before component selection.
    """
    source = np.asarray(source_bgr)
    shape = source.shape[:2]
    seed = _binary(seed_mask, shape)
    if seed is None or not np.any(seed):
        return np.zeros(shape, np.float32), {"reason": "empty_seed"}
    ys, xs = np.where(seed > 0)
    x0, x1 = int(xs.min()), int(xs.max()); y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = x1-x0+1, y1-y0+1
    mx = max(20, int(round(bw * 0.28))); my = max(20, int(round(bh * 0.26)))
    scope = np.zeros(shape, np.uint8)
    scope[max(0,y0-my):min(shape[0],y1+my+1), max(0,x0-mx):min(shape[1],x1+mx+1)] = 1
    gray = source if source.ndim == 2 else cv2.cvtColor(source.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    dark = ((gray <= int(np.clip(threshold, 40, 180))) & (scope > 0)).astype(np.uint8)
    k = max(3, int(close_kernel) | 1)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)),1)
    for _name, raw in dict(exclusion_masks or {}).items():
        m = _binary(raw, shape)
        if m is None or not np.any(m):
            continue
        pad = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),1)
        dark[pad > 0] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    pk = max(3, int(probe_kernel) | 1)
    probe = cv2.dilate(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(pk,pk)),1) > 0
    out = np.zeros(shape, np.uint8)
    accepted = []
    for rid in range(1,n):
        comp = labels == rid
        area = int(stats[rid, cv2.CC_STAT_AREA])
        overlap = int(np.count_nonzero(comp & probe))
        if area < 24 or overlap < max(int(min_overlap), int(min(area, np.count_nonzero(seed)) * 0.002)):
            continue
        out[comp] = 1
        accepted.append({"id": int(rid), "area": area, "overlap": overlap})
    # Corrected pixels are always retained even if antialiased/too light.
    out[seed > 0] = 1
    return out.astype(np.float32), {
        "reason": "dark_seed_teacher",
        "threshold": int(threshold),
        "seed_pixels": int(np.count_nonzero(seed)),
        "teacher_pixels": int(np.count_nonzero(out)),
        "accepted_components": accepted[:24],
    }

def fuse_multiteacher_semantics(
    shape: tuple[int, int],
    *,
    corrected_masks: Mapping[str, np.ndarray | None] | None = None,
    teacher_probabilities: Mapping[str, Mapping[str, np.ndarray | None]] | None = None,
    teacher_weights: Mapping[str, float] | None = None,
    valid_mask: np.ndarray | None = None,
    config: TeacherFusionConfig | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """Fuse multiple semantic teachers into a partial indexed label.

    ``corrected_masks`` are authoritative user/gold masks.  Other teachers are
    soft evidence and never override an explicit corrected negative class.
    """
    cfg = config or TeacherFusionConfig()
    classes = tuple(cfg.class_names)
    lookup = {n: i for i, n in enumerate(classes)}
    h, w = int(shape[0]), int(shape[1])
    scores = {name: np.zeros((h, w), np.float32) for name in classes}
    support = {name: np.zeros((h, w), np.float32) for name in classes}
    corrected = dict(corrected_masks or {})

    # Soft teachers.
    default_weights = {
        "generator_head": cfg.generator_head_weight,
        "parser": cfg.parser_weight,
        "legacy": cfg.legacy_weight,
        "generator_colour": cfg.generator_colour_weight,
    }
    default_weights.update({str(k): float(v) for k, v in dict(teacher_weights or {}).items()})
    teacher_diag = {}
    for teacher_name, class_map in dict(teacher_probabilities or {}).items():
        weight = float(default_weights.get(str(teacher_name), 1.0))
        contributed = {}
        for name, raw in dict(class_map or {}).items():
            if name not in scores:
                continue
            p = _as_prob(raw, (h, w))
            if p is None:
                continue
            scores[name] += p * weight
            support[name] = np.maximum(support[name], p)
            contributed[name] = int(np.count_nonzero(p > 0.05))
        teacher_diag[str(teacher_name)] = {"weight": weight, "pixels": contributed}

    # Corrected/gold masks are extremely strong positive evidence.
    corrected_binary: dict[str, np.ndarray] = {}
    for name, raw in corrected.items():
        if name not in scores:
            continue
        m = _binary(raw, (h, w))
        if m is None:
            continue
        corrected_binary[name] = m
        scores[name][m > 0] += float(cfg.human_weight)
        support[name][m > 0] = 1.0

    # Explicit negative classes are authoritative against hair.  This is the
    # central guard that prevents recall tuning from reintroducing glasses or
    # clothing as hair.
    hair_score = scores.get("hair")
    if hair_score is not None:
        negative_union = np.zeros((h, w), np.uint8)
        for name in ("eyebrow", "eye", "glasses", "clothes", "face_shadow", "skin", "iris"):
            m = corrected_binary.get(name)
            if m is None:
                continue
            negative_union |= m
            scores[name][m > 0] += float(cfg.corrected_negative_weight)
        if np.any(negative_union):
            hair_score[negative_union > 0] = -float(cfg.corrected_negative_weight)

    # Winner/confidence arbitration.  Hair gets a recall-oriented admission
    # rule: a moderate score can beat generic other/background, but not an
    # explicit semantic negative.
    stack = np.stack([scores[n] for n in classes], axis=0)
    order = np.argsort(stack, axis=0)
    winner = order[-1]
    top = np.take_along_axis(stack, winner[None, ...], axis=0)[0]
    second_idx = order[-2]
    second = np.take_along_axis(stack, second_idx[None, ...], axis=0)[0]
    margin = top - second

    label = np.full((h, w), IGNORE_INDEX, np.uint8)
    confident = (top >= float(cfg.min_positive_score)) & (margin >= float(cfg.conflict_margin))
    label[confident] = winner[confident].astype(np.uint8)

    if "hair" in lookup:
        hi = lookup["hair"]
        hs = scores["hair"]
        hair_support = support["hair"]
        hair_candidate = (hair_support >= float(cfg.hair_recall_threshold)) & (hs > 0.0)
        # Only let hair override background/other/IGNORE. Explicit semantic
        # negatives and corrected labels retain priority.
        allowed_existing = np.isin(label, [IGNORE_INDEX,
                                            lookup.get("background", 0),
                                            lookup.get("other", 255)])
        neg = np.zeros((h, w), bool)
        for n in ("eyebrow", "eye", "glasses", "clothes", "face_shadow", "skin", "iris"):
            m = corrected_binary.get(n)
            if m is not None:
                neg |= m > 0
        promote = hair_candidate & allowed_existing & (~neg)
        label[promote] = np.uint8(hi)

    # Corrected classes always win final arbitration in deterministic priority.
    if corrected_binary:
        label = build_indexed_semantic_label(
            (h, w), corrected_binary, class_names=classes,
            valid_mask=valid_mask, ignore_unlabelled=True)
        # Re-apply fused labels only where the corrected map is unlabeled.
        corrected_unlabelled = label == IGNORE_INDEX
        fused_copy = np.full((h, w), IGNORE_INDEX, np.uint8)
        fused_copy[confident] = winner[confident].astype(np.uint8)
        if "hair" in lookup:
            fused_copy[promote] = np.uint8(lookup["hair"])
        label[corrected_unlabelled] = fused_copy[corrected_unlabelled]

    valid = _binary(valid_mask, (h, w)) if valid_mask is not None else None
    if valid is not None:
        label[valid == 0] = np.uint8(IGNORE_INDEX)

    probs = {}
    # Softmax-like normalized score maps are useful for debugging/teacher reuse.
    shifted = stack - np.max(stack, axis=0, keepdims=True)
    ex = np.exp(np.clip(shifted, -20.0, 20.0))
    denom = np.sum(ex, axis=0, keepdims=True) + 1e-8
    norm = ex / denom
    for i, name in enumerate(classes):
        probs[name] = norm[i].astype(np.float32)

    diagnostics = {
        "classes": list(classes),
        "teacher_diagnostics": teacher_diag,
        "labelled_pixels": int(np.count_nonzero(label != IGNORE_INDEX)),
        "ignored_pixels": int(np.count_nonzero(label == IGNORE_INDEX)),
        "class_pixels": {name: int(np.count_nonzero(label == idx)) for name, idx in lookup.items()},
        "hair_recall_threshold": float(cfg.hair_recall_threshold),
        "conflict_margin": float(cfg.conflict_margin),
    }
    return label, probs, diagnostics
