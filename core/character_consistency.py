"""High-confidence deterministic lock for character identity colours."""

from __future__ import annotations

import cv2
import numpy as np

from core.masks import combined_neutral_mask
from core.line_boundary import boundary_safety_weight

_ATTR_STRENGTH = {
    "hair": 0.92,
    "eyes": 1.00,
    "skin": 0.78,
    "clothing": 0.96,
}


def _rgb_to_lab(rgb: tuple[int, int, int]) -> np.ndarray:
    r, g, b = [int(np.clip(v, 0, 255)) for v in rgb]
    px = np.array([[[b, g, r]]], dtype=np.uint8)
    return cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


def apply_character_palette_lock(result_bgr: np.ndarray, source_bw_bgr: np.ndarray,
                                 character_library=None, strength: float = 1.0,
                                 *, assignments: dict | None = None,
                                 segmentation=None) -> np.ndarray:
    """Lock only assignments that passed v4 confidence and ambiguity gates.

    ``assignments`` and ``segmentation`` should come from PageColorContext.
    The library fallback is retained solely for old callers.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if assignments is None and character_library is not None:
        assignments = getattr(character_library, "last_assignments", None) or {}
    if segmentation is None and character_library is not None:
        segmentation = getattr(character_library, "_last_segmentation", None)
    assignments = assignments or {}
    if strength <= 0.0 or not assignments or segmentation is None:
        return result_bgr

    h, w = result_bgr.shape[:2]
    if source_bw_bgr.shape[:2] != (h, w):
        source_bw_bgr = cv2.resize(source_bw_bgr, (w, h), interpolation=cv2.INTER_AREA)
    source_gray = (cv2.cvtColor(source_bw_bgr, cv2.COLOR_BGR2GRAY)
                   if source_bw_bgr.ndim == 3 else source_bw_bgr)
    labels = cv2.resize(segmentation.labels.astype(np.int32), (w, h),
                        interpolation=cv2.INTER_NEAREST)
    neutral_keep = combined_neutral_mask(source_gray, line_dilate=1, blur=2).astype(np.float32)
    boundary_keep = boundary_safety_weight(
        source_gray, line_low=75, gap_close=4, margin_px=2.2)
    neutral_keep *= boundary_keep

    lab = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ab = lab[..., 1:3]
    touched = np.zeros((h, w), dtype=bool)

    for region_id, info in assignments.items():
        legacy_assignment = "lock_allowed" not in info
        if not legacy_assignment and not info.get("lock_allowed", False):
            continue
        semantic_conf = float(info.get("semantic_confidence", 1.0 if legacy_assignment else 0.0))
        match_score = float(info.get("match_score", 1.0 if legacy_assignment else 0.0))
        margin = float(info.get("margin", 1.0 if legacy_assignment else 0.0))
        forced = bool(info.get("forced", legacy_assignment))
        attr = str(info.get("attribute", ""))
        if not forced:
            if attr in ("eyes", "clothing"):
                if semantic_conf < 0.20 or match_score < 0.34 or margin < 0.02:
                    continue
            elif semantic_conf < 0.32 or match_score < 0.48 or margin < 0.055:
                continue

        mask = (labels == int(region_id)).astype(np.uint8)
        if not mask.any():
            continue
        valid = (mask > 0) & (source_gray > 24) & (source_gray < 247)

        if attr == "eyes":
            # Preserve pupil/eyelashes and eye white.  Only the middle-tone iris
            # pixels are eligible, and the region is eroded one pixel inward.
            eroded = cv2.erode(mask, np.ones((3, 3), np.uint8)) > 0
            valid &= eroded & (source_gray > 42) & (source_gray < 220)
        elif attr == "skin":
            # Avoid strongest shadows/highlights; their relative offsets are
            # handled by style rendering rather than identity replacement.
            valid &= (source_gray > 45) & (source_gray < 232)
        elif attr == "clothing":
            # Clothing colour should read consistently across pages; skip only
            # the most extreme highlights and shadows.
            valid &= (source_gray > 34) & (source_gray < 240)

        if int(valid.sum()) < (3 if attr == "eyes" else 8):
            continue
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        fade_px = 1.5 if attr == "eyes" else (3.6 if attr == "clothing" else 3.0)
        inner = np.clip(distance / fade_px, 0.0, 1.0) * neutral_keep
        inner *= valid.astype(np.float32)

        current = ab[valid]
        median_ab = np.median(current, axis=0).astype(np.float32)
        target_options = [tuple(info["rgb"])]
        extra_slots = info.get("slot_rgbs") or []
        for slot in extra_slots:
            try:
                slot_tuple = tuple(int(v) for v in slot)
            except Exception:
                continue
            if slot_tuple not in target_options:
                target_options.append(slot_tuple)
        variant_target = False
        if attr == "clothing" and len(target_options) > 1:
            has_geometry_slot = ("preferred_slot_index" in info and
                                 bool(info.get("clothing_part")))
            preferred = int(info.get("preferred_slot_index", 0) or 0)
            if has_geometry_slot and 0 <= preferred < len(target_options):
                best_target = target_options[preferred]
            else:
                best_target = min(target_options, key=lambda rgb: float(np.linalg.norm(_rgb_to_lab(rgb)[1:3] - median_ab)))
            variant_target = best_target != target_options[0]
        elif attr == "eyes" and len(target_options) > 1:
            best_target = min(target_options, key=lambda rgb: float(np.linalg.norm(_rgb_to_lab(rgb)[1:3] - median_ab)))
            variant_target = best_target != target_options[0]
        else:
            best_target = target_options[0]
        target_ab = _rgb_to_lab(best_target)[1:3]
        texture_keep = 0.42 if attr == "hair" else (0.06 if attr == "clothing" else (0.0 if attr == "eyes" else 0.28))
        texture = (ab - median_ab) * texture_keep
        desired = target_ab + texture

        attr_strength = _ATTR_STRENGTH.get(attr, 0.75)
        confidence = 1.0 if forced else np.clip(
            0.35 + 0.40 * match_score + 0.25 * min(1.0, margin / 0.15), 0.0, 1.0)
        semantic_gate = np.clip((semantic_conf - 0.25) / 0.55, 0.0, 1.0)
        if attr == "eyes":
            confidence = max(confidence, 0.94)
            semantic_gate = max(semantic_gate, 0.88)
        elif attr == "clothing":
            confidence = max(confidence, 0.86)
            semantic_gate = max(semantic_gate, 0.72)
        alpha_2d = np.clip(
            inner * strength * attr_strength * confidence * semantic_gate,
            0.0, 1.0)
        if attr == "clothing":
            # Prevent outfit colours from drifting between pages: if the whole
            # region is already near the target this stays subtle, otherwise it
            # pushes more decisively toward the locked clothing hue.
            alpha_2d = np.clip(alpha_2d * (1.26 if variant_target else 1.18) + inner * (0.12 if variant_target else 0.08), 0.0, 1.0)
        elif attr == "eyes":
            # Iris colours should stay tightly locked; small regions tolerate
            # stronger convergence than broad skin/hair areas.
            alpha_2d = np.clip(alpha_2d * 1.18 + inner * 0.06, 0.0, 1.0)
        touched |= alpha_2d > 1e-6
        alpha = alpha_2d[..., None]
        ab[:] = ab * (1.0 - alpha) + desired * alpha

    lab[..., 1:3] = np.clip(ab, 0, 255)
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    out[~touched] = result_bgr[~touched]
    return out
