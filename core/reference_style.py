"""Relative reference-style rendering that preserves colour identity.

v4 deliberately ignores ``reference_palette`` during rendering.  Absolute
reference colours belong to CharacterPalette/ScenePalette.  This module changes
only relationships within each segmented region: saturation, shadow tint,
highlight tint, contrast and flattening.  The mean chroma centre of a character
region is preserved, so red hair stays red and blue hair stays blue.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.masks import combined_neutral_mask
from core.region_segmenter import segment_regions
from core.style_descriptor import RegionDescriptor

_CHARACTER_SEMANTICS = {"hair", "skin", "eyes", "clothing",
                        "clothing_primary", "clothing_secondary", "clothing_accent"}
_BACKGROUND_SEMANTICS = {"background", "sky", "foliage", "stone", "wood",
                         "water", "fire", "metal"}


def _semantic_map(context) -> dict[int, tuple[str, float]]:
    if context is None:
        return {}
    regions = getattr(getattr(context, "segmentation", None), "regions", []) or []
    labels = getattr(context, "semantic_labels", []) or []
    return {int(region.label_id): (str(label), float(conf))
            for region, (label, conf) in zip(regions, labels)}


def _region_descriptor(descriptor, semantic: str) -> RegionDescriptor:
    if semantic == "clothing":
        semantic = "clothing_primary"
    return descriptor.region(semantic) if descriptor is not None else RegionDescriptor()


def apply_reference_style(colorized_bgr: np.ndarray, source_bw_bgr: np.ndarray,
                          descriptor, strength: float = 1.0, context=None) -> np.ndarray:
    """Apply region-relative style without replacing identity colours."""
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.001 or descriptor is None:
        return colorized_bgr
    if source_bw_bgr.shape[:2] != colorized_bgr.shape[:2]:
        source_bw_bgr = cv2.resize(
            source_bw_bgr, (colorized_bgr.shape[1], colorized_bgr.shape[0]),
            interpolation=cv2.INTER_AREA)
    gray = (cv2.cvtColor(source_bw_bgr, cv2.COLOR_BGR2GRAY)
            if source_bw_bgr.ndim == 3 else source_bw_bgr)

    segmentation = getattr(context, "segmentation", None)
    if segmentation is None:
        segmentation = segment_regions(gray)
    labels = cv2.resize(segmentation.labels.astype(np.int32),
                        (colorized_bgr.shape[1], colorized_bgr.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
    sem_by_id = _semantic_map(context)
    scope = getattr(descriptor, "style_scope", {}) or {}
    allow_character = bool(scope.get("character_rendering", True))
    allow_background = bool(scope.get("background_rendering", True))
    ambience = float(np.clip(scope.get("global_ambience", 0.20), 0.0, 1.0))

    lab = cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    original_ab = lab[..., 1:3].copy()
    result_ab = original_ab.copy()
    neutral_keep = combined_neutral_mask(gray, line_dilate=1, blur=3).astype(np.float32)

    for region in segmentation.regions:
        rid = int(region.label_id)
        mask = labels == rid
        valid = mask & (neutral_keep > 0.30) & (gray > 20) & (gray < 247)
        if int(np.count_nonzero(valid)) < 12:
            continue
        semantic, semantic_conf = sem_by_id.get(rid, ("unknown", 0.0))
        is_character = semantic in _CHARACTER_SEMANTICS
        is_background = semantic in _BACKGROUND_SEMANTICS
        if is_character and not allow_character:
            continue
        if is_background and not allow_background:
            continue

        rd = _region_descriptor(descriptor, semantic)
        pixels = original_ab[valid]
        center = np.median(pixels, axis=0).astype(np.float32)
        source_l = gray[valid].astype(np.float32)
        q25, q75 = np.quantile(source_l, [0.25, 0.75]) if len(source_l) >= 8 else (90.0, 175.0)
        shadow = valid & (gray <= q25)
        highlight = valid & (gray >= q75)
        mid = valid & ~shadow & ~highlight

        # Saturation/contrast scales deviations around the region's own centre,
        # which preserves the identity hue and separates different characters.
        sat = 1.0 + (float(rd.saturation_scale) * float(descriptor.global_saturation) - 1.0) * strength
        contrast = 1.0 + (float(rd.contrast) - 1.0) * strength
        scale = float(np.clip(sat * contrast, 0.65, 1.55))
        target = center + (original_ab - center) * scale

        # Tier-specific style language.  No absolute palette is involved.
        target[shadow, 0] += float(rd.shadow_hue_rotate) * strength
        target[shadow, 1] += float(rd.shadow_bias) * strength
        target[highlight, 0] += float(rd.highlight_hue_rotate) * strength
        target[highlight, 1] += float(rd.highlight_bias) * strength
        target[mid, 1] += float(rd.warm_bias) * strength * 0.35

        # Optional, intentionally weak ambience only. Character regions receive
        # at most 20% of it; backgrounds can receive the full configured amount.
        ambience_scale = ambience * (0.20 if is_character else 1.0)
        target[valid, 1] += float(descriptor.global_warm_cool) * strength * ambience_scale

        # Preserve the region's average identity centre exactly for character
        # parts and approximately for unknown/background regions.
        target_center = np.median(target[valid], axis=0)
        preserve = 1.0 if is_character else 0.75
        target[valid] -= (target_center - center) * preserve

        alpha = neutral_keep[..., None] * strength
        result_ab[valid] = (original_ab[valid] * (1.0 - alpha[valid]) +
                            target[valid] * alpha[valid])

    lab[..., 1:3] = np.clip(result_ab, 0, 255)
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    # Avoid LAB round-trip changing untouched paper/ink.
    untouched = neutral_keep <= 0.01
    out[untouched] = colorized_bgr[untouched]
    return out
