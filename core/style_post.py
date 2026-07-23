"""Style post-processing — turns a StylePreset's knobs into real pixels.

``core/presets.py`` defines ``StylePreset`` (saturation boost, neutral-fade
thresholds, chroma warm/red shift, cel-flatten, lightness gamma/blend, ...)
but nothing in the pipeline ever read those fields — the presets sat there
fully designed and completely unused. This module is that missing code:
one function, ``apply_style_grade``, that grades mc-v2's raw output
according to a StylePreset (built-in, or produced by
``core.style_engine.StyleProfile.to_style_preset()``).

Everything runs in Lab space so lightness (L) and chroma (a, b) can be
tuned independently. Neutral regions (ink lines, speech bubbles, page
gutters) always come from the ORIGINAL black-and-white page via
``core.masks.combined_neutral_mask`` — never from anything mc-v2 painted,
since the model has no ground truth for where paper/ink actually is.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.masks import combined_neutral_mask
from core.presets import StylePreset


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img




_CHARACTER_LABELS = {"hair", "skin", "eyes", "clothing"}
_SCENE_LABELS = {"metal", "wood", "sky", "foliage", "stone", "water", "fire", "background"}


def _fast_person_mask(colorized_bgr: np.ndarray, source_gray: np.ndarray) -> np.ndarray:
    """Cheap character-like mask without face detection or semantic models."""
    h, w = source_gray.shape[:2]
    lab = cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    chroma = np.sqrt(a * a + b * b)
    nonzero = chroma[chroma > 1.0]
    threshold = max(3.0, float(np.percentile(nonzero, 35)) if nonzero.size else 3.0)
    border_ab = np.concatenate([
        lab[:max(1, h // 20), :, 1:3].reshape(-1, 2),
        lab[-max(1, h // 20):, :, 1:3].reshape(-1, 2),
        lab[:, :max(1, w // 20), 1:3].reshape(-1, 2),
        lab[:, -max(1, w // 20):, 1:3].reshape(-1, 2),
    ], axis=0)
    border_median = np.median(border_ab, axis=0).astype(np.float32)
    delta_border = np.linalg.norm(lab[..., 1:3] - border_median[None, None, :], axis=2)
    delta_nonzero = delta_border[delta_border > 1.0]
    delta_threshold = max(7.0, float(np.percentile(delta_nonzero, 40)) if delta_nonzero.size else 7.0)
    high_chroma = max(threshold + 5.0, float(np.percentile(nonzero, 70)) if nonzero.size else threshold + 5.0)
    foreground_evidence = ((delta_border >= delta_threshold) |
                           (chroma >= high_chroma) |
                           (source_gray < 232))
    candidate = ((chroma >= threshold) & foreground_evidence &
                 (source_gray < 248) & (source_gray > 8)).astype(np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    page_area = float(max(1, h * w))
    kept = np.zeros((h, w), np.uint8)
    scored: list[tuple[float, int]] = []
    for label in range(1, num):
        x, y, bw, bh, area = stats[label].tolist()
        frac = float(area) / page_area
        if area < max(32, int(page_area * 0.0012)) or frac > 0.72:
            continue
        component = labels == label
        mean_chroma = float(chroma[component].mean()) if np.any(component) else 0.0
        touches = int(x <= 1) + int(y <= 1) + int(x + bw >= w - 1) + int(y + bh >= h - 1)
        cx, cy = centroids[label]
        center_dist = ((cx / max(1, w) - 0.5) ** 2 + (cy / max(1, h) - 0.5) ** 2) ** 0.5
        score = mean_chroma * (area ** 0.35) * (1.15 - min(0.8, center_dist))
        if touches >= 3 and frac > 0.12:
            score *= 0.20
        elif touches >= 2 and frac > 0.25:
            score *= 0.45
        scored.append((score, label))
    if scored:
        scored.sort(reverse=True)
        best = scored[0][0]
        for score, label in scored:
            if score >= best * 0.28:
                kept[labels == label] = 1
    if not np.any(kept) and np.any(candidate):
        # Never fall back to a page-wide chroma wash. Keep only the central
        # foreground portion so character-only pastel remains meaningful.
        central = np.zeros_like(candidate)
        mx, my = max(1, w // 20), max(1, h // 20)
        central[my:h-my, mx:w-mx] = 1
        kept = candidate & central
    return np.clip(cv2.GaussianBlur(kept.astype(np.float32), (7, 7), 0), 0.0, 1.0)


def _semantic_role_masks(colorized_bgr: np.ndarray, source_gray: np.ndarray, context=None) -> dict[str, np.ndarray]:
    """Return soft masks for character attributes and person coverage.

    These masks are intentionally permissive: monochrome-pastel modes need to
    keep colour across the whole person, including bright hair highlights and
    pale clothing folds, while still allowing the environment to collapse back
    to strict grayscale when required.
    """
    h, w = source_gray.shape[:2]
    sy = max(1, h // 16)
    sx = max(1, w // 16)
    color_sample = colorized_bgr[::sy, ::sx][:16, :16].astype(np.uint64)
    color_signature = int((color_sample[..., 0].sum() * 3 +
                           color_sample[..., 1].sum() * 5 +
                           color_sample[..., 2].sum() * 7) % 4294967291)
    cache_key = (h, w, color_signature,
                 id(getattr(context, "segmentation", None)) if context is not None else 0,
                 len(getattr(context, "character_instances", []) or []) if context is not None else 0)
    if context is not None:
        cached = getattr(context, "_style_role_masks_cache", None)
        if cached is not None and getattr(context, "_style_role_masks_cache_key", None) == cache_key:
            return cached
    masks = {
        "hair": np.zeros((h, w), np.float32),
        "skin": np.zeros((h, w), np.float32),
        "eyes": np.zeros((h, w), np.float32),
        "clothing": np.zeros((h, w), np.float32),
        "person": np.zeros((h, w), np.float32),
        "environment": np.zeros((h, w), np.float32),
    }
    seg = getattr(context, "segmentation", None) if context is not None else None
    labels_full = None
    if seg is not None and getattr(seg, "labels", None) is not None:
        labels_full = cv2.resize(
            seg.labels.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)

    semantic_labels = list(getattr(context, "semantic_labels", []) or []) if context is not None else []
    if labels_full is not None and seg is not None:
        for region, value in zip(getattr(seg, "regions", []), semantic_labels):
            try:
                label, confidence = value
                confidence = float(np.clip(confidence, 0.0, 1.0))
            except Exception:
                continue
            if confidence < 0.12:
                continue
            mask = labels_full == int(region.label_id)
            if label in _CHARACTER_LABELS:
                masks[label][mask] = np.maximum(masks[label][mask], confidence)
                masks["person"][mask] = np.maximum(masks["person"][mask], confidence)
            elif label in _SCENE_LABELS:
                masks["environment"][mask] = np.maximum(masks["environment"][mask], confidence)

    assignments = dict(getattr(context, "identity_assignments", {}) or {}) if context is not None else {}
    if labels_full is not None:
        for region_id, info in assignments.items():
            attr = str(info.get("attribute", ""))
            if attr not in _CHARACTER_LABELS:
                continue
            conf = float(np.clip(info.get("confidence", info.get("semantic_confidence", 0.88)), 0.0, 1.0))
            if not info.get("lock_allowed", False) and not info.get("forced", False):
                conf = min(conf, 0.72)
            mask = labels_full == int(region_id)
            masks[attr][mask] = np.maximum(masks[attr][mask], conf)
            masks["person"][mask] = np.maximum(masks["person"][mask], conf)

    # Broad person fallback from detected body/head boxes.  Restrict it to
    # plausible drawable pixels and regions that mc-v2 already wanted to colour.
    lab = cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    chroma = np.sqrt(a * a + b * b)
    # Fast mode: do not run anime face detection fallbacks.
    instances = list(getattr(context, "character_instances", []) or []) if context is not None else []
    ink_support = (source_gray < 252) & (source_gray > 8)
    for instance in instances:
        box = getattr(instance, "body_bbox", None) or getattr(instance, "head_bbox", None)
        if not box:
            continue
        x, y, bw, bh = map(int, box)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + bw), min(h, y + bh)
        if x1 >= x2 or y1 >= y2:
            continue
        local_person = masks["person"][y1:y2, x1:x2]
        candidate = ink_support[y1:y2, x1:x2] & ((chroma[y1:y2, x1:x2] > 2.0) | (source_gray[y1:y2, x1:x2] < 248))
        fallback_strength = np.where(candidate, 0.42, 0.0).astype(np.float32)
        local_person[:] = np.maximum(local_person, fallback_strength)

    if not np.any(masks["person"] > 0.01):
        masks["person"] = _fast_person_mask(colorized_bgr, source_gray)

    kernel = np.ones((3, 3), np.uint8)
    for key, mask in masks.items():
        if not np.any(mask > 0):
            continue
        solid = (mask > 0.08).astype(np.uint8)
        solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, kernel, iterations=1)
        soft = cv2.GaussianBlur(solid.astype(np.float32), (3, 3), 0)
        if key == "person":
            # person coverage should include all explicit attribute masks
            for attr in ("hair", "skin", "eyes", "clothing"):
                soft = np.maximum(soft, masks[attr])
        masks[key] = np.clip(np.maximum(mask, soft), 0.0, 1.0).astype(np.float32)
    if context is not None:
        try:
            context._style_role_masks_cache_key = cache_key
            context._style_role_masks_cache = masks
        except Exception:
            pass
    return masks


def _propagate_semantic_tint(a: np.ndarray, b: np.ndarray, source_gray: np.ndarray,
                             style: StylePreset,
                             role_masks: dict[str, np.ndarray] | None):
    """Spread existing character tint into low-chroma holes inside the same role mask.

    This is a cheap post-process fallback for monochrome pastel modes: when the
    raw colourization leaves part of a character's hair / clothing / skin nearly
    gray, we softly diffuse nearby valid character colour within the same mask so
    the character reads consistently coloured instead of patchy.
    """
    if role_masks is None:
        return a, b

    h, w = source_gray.shape[:2]
    chroma = np.sqrt(a * a + b * b).astype(np.float32)
    drawable = ((source_gray > 10) & (source_gray < 248)).astype(np.float32)

    def _kernel(radius_hint: int) -> tuple[int, int]:
        k = max(3, int(radius_hint) | 1)
        return (k, k)

    def _spread(base_mask: np.ndarray, *, floor: float, alpha: float, radius_hint: int):
        nonlocal a, b, chroma
        if base_mask is None or not np.any(base_mask > 0.05):
            return
        base = np.clip(base_mask.astype(np.float32), 0.0, 1.0) * drawable
        if not np.any(base > 0.05):
            return
        desired = float(max(2.0, floor))
        seed = base * np.clip((chroma - max(4.0, desired * 0.33)) / max(6.0, desired), 0.0, 1.0)
        if np.count_nonzero(seed > 0.05) < max(32, (h * w) // 1800):
            seed = base * np.clip(chroma / max(8.0, desired), 0.0, 1.0)
        if np.count_nonzero(seed > 0.03) < 8:
            return

        k = _kernel(radius_hint)
        num_a = cv2.GaussianBlur((a * seed).astype(np.float32), k, 0)
        num_b = cv2.GaussianBlur((b * seed).astype(np.float32), k, 0)
        den = cv2.GaussianBlur(seed.astype(np.float32), k, 0) + 1e-4
        mean_a = num_a / den
        mean_b = num_b / den
        mean_chroma = np.sqrt(mean_a * mean_a + mean_b * mean_b)
        valid_mean = mean_chroma > 1.2
        if not np.any(valid_mean & (base > 0.05)):
            return

        deficit = np.clip((desired - chroma) / max(desired, 1.0), 0.0, 1.0)
        # Very bright highlights should retain only part of the propagated tint.
        bright_gate = np.clip((248.0 - source_gray.astype(np.float32)) / 34.0, 0.20, 1.0)
        blend = np.clip(base * deficit * alpha * bright_gate, 0.0, 0.92)
        blend = np.where(valid_mean, blend, 0.0).astype(np.float32)
        if not np.any(blend > 1e-3):
            return
        a = a * (1.0 - blend) + mean_a * blend
        b = b * (1.0 - blend) + mean_b * blend
        chroma = np.sqrt(a * a + b * b).astype(np.float32)

    base_radius = max(5, int(getattr(style, 'guided_filter_radius', 5)) * 2 + 1)
    _spread(role_masks.get('skin'), floor=max(5.0, float(style.skin_chroma_scale) * 12.0),
            alpha=0.42, radius_hint=base_radius)
    _spread(role_masks.get('hair'), floor=max(6.5, float(style.hair_chroma_scale) * 13.5),
            alpha=0.54, radius_hint=base_radius + 2)
    _spread(role_masks.get('clothing'), floor=max(5.5, float(style.clothing_chroma_scale) * 11.5),
            alpha=0.46, radius_hint=base_radius + 2)
    _spread(role_masks.get('eyes'), floor=max(8.0, float(style.eye_chroma_scale) * 18.0),
            alpha=0.62, radius_hint=max(3, base_radius - 2))
    _spread(role_masks.get('person'), floor=max(4.0, float(style.person_chroma_scale) * 9.0),
            alpha=0.24, radius_hint=base_radius + 4)
    return a, b


def _semantic_chroma_scale(colorized_bgr: np.ndarray, source_gray: np.ndarray,
                           style: StylePreset, context=None,
                           role_masks: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """Per-pixel chroma retention for monochrome-pastel modes.

    Semantic regions from PageColorContext take priority. Identity assignments
    and detected character boxes fill gaps so pale hair / clothing highlights
    do not lose colour patch-by-patch.
    """
    h, w = source_gray.shape[:2]
    if getattr(style, "semantic_mode", "all") == "all":
        return np.ones((h, w), np.float32)

    unknown = float(getattr(style, "unknown_chroma_scale", 0.02))
    semantic_mode = str(getattr(style, "semantic_mode", "all"))
    base = unknown
    if semantic_mode == "page_pastel":
        base = max(base, float(getattr(style, "environment_chroma_scale", 0.10)))
    scale = np.full((h, w), base, np.float32)
    masks = role_masks if role_masks is not None else _semantic_role_masks(
        colorized_bgr, source_gray, context=context)

    attr_to_scale = {
        "skin": float(style.skin_chroma_scale),
        "hair": float(style.hair_chroma_scale),
        "eyes": float(style.eye_chroma_scale),
        "clothing": float(style.clothing_chroma_scale),
    }
    for attr, value_scale in attr_to_scale.items():
        mask = masks[attr]
        if np.any(mask > 0):
            scale = np.maximum(scale, mask * value_scale)

    if np.any(masks["environment"] > 0):
        scale = np.where(
            masks["environment"] > 0,
            np.maximum(scale, masks["environment"] * float(style.environment_chroma_scale)),
            scale)

    person_mask = masks["person"]
    if np.any(person_mask > 0):
        explicit_attributes = any(np.any(masks[k] > 0.02)
                                  for k in ("hair", "skin", "eyes", "clothing"))
        if explicit_attributes:
            person_floor = float(getattr(style, "person_chroma_scale", 0.0))
        else:
            person_floor = max(float(getattr(style, "person_chroma_scale", 0.0)),
                               float(getattr(style, "hair_chroma_scale", 0.0)),
                               float(getattr(style, "skin_chroma_scale", 0.0)),
                               float(getattr(style, "clothing_chroma_scale", 0.0)) * 0.92)
        scale = np.maximum(scale, person_mask * person_floor)

    # Conservative skin-colour fallback from mc-v2 output.  This never expands
    # to neighbouring regions, so it cannot tint the environment wholesale.
    ycrcb = cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    skin_like = ((Y > 45) & (Y < 248) & (Cr > 120) & (Cr < 184) &
                 (Cb > 72) & (Cb < 148) &
                 (source_gray > 35) & (source_gray < 238) &
                 (person_mask > 0.02))
    scale[skin_like] = np.maximum(scale[skin_like], float(style.skin_chroma_scale))
    return np.clip(scale, 0.0, 1.0).astype(np.float32)




def _pastel_skin_mask(colorized_bgr: np.ndarray, source_gray: np.ndarray,
                      style: StylePreset, context=None,
                      role_masks: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """Return a conservative skin mask for pastel hue neutralization.

    The mask requires semantic/identity evidence or an anime-face box.  It does
    not globally classify every pink background pixel as skin.  High-chroma
    blush/lips are protected later by a chroma gate.
    """
    h, w = source_gray.shape[:2]
    mask = np.zeros((h, w), np.float32)
    if role_masks is not None and "skin" in role_masks:
        mask = np.maximum(mask, role_masks["skin"].astype(np.float32))
    seg = getattr(context, "segmentation", None) if context is not None else None
    labels_full = None
    if seg is not None and getattr(seg, "labels", None) is not None:
        labels_full = cv2.resize(
            seg.labels.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)

    semantic_labels = list(getattr(context, "semantic_labels", []) or []) if context is not None else []
    if labels_full is not None and seg is not None:
        for region, value in zip(getattr(seg, "regions", []), semantic_labels):
            try:
                label, confidence = value
                confidence = float(confidence)
            except Exception:
                continue
            if label == "skin" and confidence >= 0.20:
                mask[labels_full == int(region.label_id)] = np.maximum(
                    mask[labels_full == int(region.label_id)],
                    float(np.clip(confidence, 0.0, 1.0)))

    assignments = dict(getattr(context, "identity_assignments", {}) or {}) if context is not None else {}
    if labels_full is not None:
        for region_id, info in assignments.items():
            if str(info.get("attribute", "")) != "skin":
                continue
            if not info.get("lock_allowed", False) and not info.get("forced", False):
                continue
            confidence = float(np.clip(info.get("confidence", 0.85), 0.0, 1.0))
            mask[labels_full == int(region_id)] = np.maximum(
                mask[labels_full == int(region_id)], confidence)

    ycrcb = cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    skin_like = ((Y > 55) & (Y < 250) & (Cr > 120) & (Cr < 178) &
                 (Cb > 76) & (Cb < 142) &
                 (source_gray > 35) & (source_gray < 246))

    # Fast mode: do not run anime face detection fallbacks.
    instances = list(getattr(context, "character_instances", []) or []) if context is not None else []
    if role_masks is not None and "person" in role_masks:
        person_like = role_masks["person"] > 0.02
        mask[skin_like & person_like] = np.maximum(mask[skin_like & person_like], 0.72)

    for instance in instances:
        box = getattr(instance, "head_bbox", None) or getattr(instance, "body_bbox", None)
        if not box:
            continue
        x, y, bw, bh = map(int, box)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + bw), min(h, y + bh)
        if x1 >= x2 or y1 >= y2:
            continue
        local_skin = skin_like[y1:y2, x1:x2]
        local_mask = mask[y1:y2, x1:x2]
        local_mask[local_skin] = np.maximum(local_mask[local_skin], 0.72)

    # Soften the confidence edge without expanding beyond a few pixels.
    if np.any(mask > 0):
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def _reinforce_semantic_chroma(a: np.ndarray, b: np.ndarray, source_gray: np.ndarray,
                               style: StylePreset,
                               role_masks: dict[str, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Make semantic sliders visibly affect the page.

    The raw model often leaves character regions with a correct hue direction but
    too little chroma for the UI controls to feel responsive.  This pass boosts
    or suppresses chroma *within the existing hue direction* per semantic role,
    using soft floors/ceilings instead of a hard recolor.
    """
    if role_masks is None:
        return a, b

    chroma = np.sqrt(a * a + b * b).astype(np.float32)
    gray = source_gray.astype(np.float32)
    editable = ((gray > 12.0) & (gray < 248.0)).astype(np.float32)

    def _apply(mask: np.ndarray | None, *, floor: float, ceiling: float, weight: float,
               neutral_floor: float = 0.0, bright_protect: bool = True):
        nonlocal a, b, chroma
        if mask is None:
            return
        m = np.clip(mask.astype(np.float32), 0.0, 1.0) * editable
        if not np.any(m > 0.01):
            return
        active = m > 0.02
        if not np.any(active):
            return
        gate = np.ones_like(m, dtype=np.float32)
        if bright_protect:
            gate *= np.clip((244.0 - gray) / 26.0, 0.18, 1.0)
        # Near-neutral hues should not be amplified into dirty colour noise.
        gate *= np.clip((chroma - neutral_floor) / max(6.0, floor * 0.85 + 6.0), 0.0, 1.0)
        gate = np.maximum(gate, np.clip((chroma - 5.0) / 12.0, 0.0, 1.0) * 0.35)

        target = chroma.copy()
        need_up = active & (chroma < floor)
        if np.any(need_up):
            blend = np.clip((floor - chroma[need_up]) / max(floor, 1.0), 0.0, 1.0)
            target[need_up] = chroma[need_up] + (floor - chroma[need_up]) * blend * weight
        need_down = active & (chroma > ceiling)
        if np.any(need_down):
            blend = np.clip((chroma[need_down] - ceiling) / max(ceiling, 1.0), 0.0, 1.0)
            target[need_down] = chroma[need_down] - (chroma[need_down] - ceiling) * blend * weight

        ratio = np.ones_like(chroma, dtype=np.float32)
        nz = chroma > 1e-3
        ratio[nz] = target[nz] / np.maximum(chroma[nz], 1e-3)
        ratio[~nz] = 1.0
        ratio = 1.0 + (ratio - 1.0) * np.clip(m * gate, 0.0, 1.0)
        a *= ratio
        b *= ratio
        chroma = np.sqrt(a * a + b * b).astype(np.float32)

    person_scale = float(getattr(style, 'person_chroma_scale', 0.0))
    env_scale = float(getattr(style, 'environment_chroma_scale', 0.0))
    unknown_scale = float(getattr(style, 'unknown_chroma_scale', 0.0))
    # Tuned so 0..200 UI range is visually distinct on manga pages.
    _apply(role_masks.get('hair'), floor=max(4.5, float(style.hair_chroma_scale) * 18.0),
           ceiling=max(7.5, float(style.hair_chroma_scale) * 38.0), weight=0.82, neutral_floor=2.5)
    _apply(role_masks.get('skin'), floor=max(3.4, float(style.skin_chroma_scale) * 13.5),
           ceiling=max(6.0, float(style.skin_chroma_scale) * 29.0), weight=0.74, neutral_floor=1.0)
    _apply(role_masks.get('eyes'), floor=max(6.0, float(style.eye_chroma_scale) * 22.0),
           ceiling=max(10.0, float(style.eye_chroma_scale) * 44.0), weight=0.92, neutral_floor=2.0, bright_protect=False)
    _apply(role_masks.get('clothing'), floor=max(4.0, float(style.clothing_chroma_scale) * 16.0),
           ceiling=max(7.0, float(style.clothing_chroma_scale) * 34.0), weight=0.86, neutral_floor=2.0)
    if str(getattr(style, 'semantic_mode', 'all')) != 'all':
        _apply(role_masks.get('person'), floor=max(2.2, person_scale * 9.5),
               ceiling=max(4.2, person_scale * 22.0), weight=0.34, neutral_floor=0.8)
        _apply(role_masks.get('environment'), floor=max(0.3, env_scale * 4.5),
               ceiling=max(1.2, env_scale * 16.5), weight=0.78, neutral_floor=0.0)
        _apply(role_masks.get('unknown'), floor=max(0.2, unknown_scale * 3.5),
               ceiling=max(1.0, unknown_scale * 14.0), weight=0.68, neutral_floor=0.0)
    return a, b


def _cel_flatten_chroma(a: np.ndarray, b: np.ndarray, source_gray: np.ndarray,
                        amount: float) -> tuple[np.ndarray, np.ndarray]:
    """Snap each lineart-bounded region's chroma toward its own mean.

    This is what makes a page read as hand-colored manhwa (flat fills)
    instead of a single soft gradient wash — `amount` 0 leaves chroma
    untouched, 1 makes every region a single flat color.
    """
    if amount <= 0:
        return a, b
    from core.region_segmenter import segment_regions

    seg = segment_regions(source_gray)
    if not seg.regions:
        return a, b

    labels_full = cv2.resize(
        seg.labels.astype(np.int32), (source_gray.shape[1], source_gray.shape[0]),
        interpolation=cv2.INTER_NEAREST)

    a_out, b_out = a.copy(), b.copy()
    for region in seg.regions:
        m = labels_full == region.label_id
        if not np.any(m):
            continue
        mean_a = float(a[m].mean())
        mean_b = float(b[m].mean())
        a_out[m] = a[m] * (1.0 - amount) + mean_a * amount
        b_out[m] = b[m] * (1.0 - amount) + mean_b * amount
    return a_out, b_out


def enforce_character_only_color(colorized_bgr: np.ndarray,
                                 source_bw_bgr: np.ndarray,
                                 context=None,
                                 role_masks: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """Keep colour only inside the detected character mask.

    This final constraint is intentionally reusable by the pipeline after
    style-strength blending and reference grading, both of which could
    otherwise reintroduce faint scene colour outside the person.
    """
    if source_bw_bgr.shape[:2] != colorized_bgr.shape[:2]:
        source_bw_bgr = cv2.resize(
            source_bw_bgr, (colorized_bgr.shape[1], colorized_bgr.shape[0]),
            interpolation=cv2.INTER_AREA)
    source_gray = _to_gray(source_bw_bgr).astype(np.uint8)
    role_masks = role_masks if role_masks is not None else _semantic_role_masks(
        colorized_bgr, source_gray, context=context)
    person_mask = np.clip(role_masks["person"], 0.0, 1.0)
    gray_bgr = cv2.cvtColor(source_gray, cv2.COLOR_GRAY2BGR)
    if not np.any(person_mask > 0.01):
        return gray_bgr
    edge = cv2.GaussianBlur(person_mask, (3, 3), 0)[..., None]
    return np.clip(
        colorized_bgr.astype(np.float32) * edge +
        gray_bgr.astype(np.float32) * (1.0 - edge),
        0, 255).astype(np.uint8)


def apply_style_grade(colorized_bgr: np.ndarray, source_bw_bgr: np.ndarray,
                      style: StylePreset, context=None) -> np.ndarray:
    """Grade mc-v2's raw colorized output according to `style`.

    Parameters
    ----------
    colorized_bgr : np.ndarray
        mc-v2's raw BGR output for this page.
    source_bw_bgr : np.ndarray
        The ORIGINAL black-and-white page (any size — resized to match
        `colorized_bgr` automatically). Supplies the lineart/bubble/
        gutter neutral mask and the fallback L channel for
        `l_blend_alpha`.
    style : StylePreset
        From ``core.presets.get_style()`` or
        ``StyleProfile.to_style_preset()``.
    """
    if source_bw_bgr.shape[:2] != colorized_bgr.shape[:2]:
        source_bw_bgr = cv2.resize(
            source_bw_bgr, (colorized_bgr.shape[1], colorized_bgr.shape[0]),
            interpolation=cv2.INTER_AREA)
    source_gray = _to_gray(source_bw_bgr)
    semantic_mode = getattr(style, "semantic_mode", "all")
    needs_role_masks = (semantic_mode != "all" or
                        bool(getattr(style, "force_environment_grayscale", False)))
    role_masks = (_semantic_role_masks(colorized_bgr, source_gray, context=context)
                  if needs_role_masks else None)

    lab = cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]                # 0-255 (OpenCV's 8-bit Lab: true L * 255/100)
    a = lab[..., 1] - 128.0        # centered at 0
    b = lab[..., 2] - 128.0

    # 1. Saturation — boost (>1) is vibrance-style: stronger on
    #    already-low-chroma pixels so it reads "more colorful" without
    #    blowing out saturated areas.  A value BELOW 1 means a pale /
    #    ink-wash style: scale ALL chroma down uniformly, and softly
    #    diffuse it so colors read as translucent washes.
    chroma = np.sqrt(a * a + b * b)
    if style.saturation_boost < 1.0:
        a = a * style.saturation_boost
        b = b * style.saturation_boost
        k = max(3, (min(a.shape) // 200) | 1)   # odd kernel, scale-aware
        a = cv2.GaussianBlur(a, (k, k), 0)
        b = cv2.GaussianBlur(b, (k, k), 0)
        if getattr(style, "key", "") == "light":
            # Keep the watercolor feel, but avoid drifting all the way to
            # near-monochrome.  A slight warm tint is enough.
            b = b + 1.2
    else:
        boost = 1.0 + (style.saturation_boost - 1.0) * np.clip(1.0 - chroma / 40.0, 0.0, 1.0)
        a = a * boost
        b = b * boost

    # 2. Global chroma shifts (warm/cool via b, red/green via a).
    b = b + style.chroma_warm_shift
    a = a + style.chroma_red_shift

    # Monochrome pastel modes keep character colour selectively while the
    # environment remains almost grayscale.  Low-chroma skin is first pulled
    # toward a warm ivory base; high-chroma blush/lips keep their local red.
    skin_correction = None
    pastel_skin_mask = None
    pastel_input_chroma = None
    if getattr(style, "semantic_mode", "all") != "all" and getattr(style, "skin_neutralize", 0.0) > 0:
        skin_mask = _pastel_skin_mask(
            colorized_bgr, source_gray, style, context=context, role_masks=role_masks)
        current_chroma = np.sqrt(a * a + b * b)
        pastel_input_chroma = current_chroma
        low_chroma_gate = np.clip((25.0 - current_chroma) / 15.0, 0.0, 1.0)
        pastel_skin_mask = skin_mask
        skin_correction = np.clip(
            skin_mask * low_chroma_gate * float(style.skin_neutralize), 0.0, 1.0)

    if role_masks is not None and getattr(style, "semantic_mode", "all") != "all":
        a, b = _propagate_semantic_tint(a, b, source_gray, style, role_masks)

    semantic_scale = _semantic_chroma_scale(
        colorized_bgr, source_gray, style, context=context, role_masks=role_masks)
    if getattr(style, "key", "") == "light":
        semantic_scale = np.clip(semantic_scale * 0.88 + 0.08, 0.0, 1.0)
    if pastel_skin_mask is not None and pastel_input_chroma is not None:
        # Preserve local high-chroma accents such as blush and lips while the
        # surrounding low-chroma skin is neutralized toward warm ivory.
        accent_gate = np.clip((pastel_input_chroma - 22.0) / 24.0, 0.0, 1.0)
        accent_scale = pastel_skin_mask * accent_gate * 0.58
        semantic_scale = np.maximum(semantic_scale, accent_scale)
    a *= semantic_scale
    b *= semantic_scale
    if skin_correction is not None:
        # Apply the warm-ivory target after semantic desaturation.  Applying it
        # before the scale made the already-small target collapse to neutral
        # gray, while retaining mc-v2's pink direction on the rest of the face.
        a = a * (1.0 - skin_correction) + float(style.skin_target_a) * skin_correction
        b = b * (1.0 - skin_correction) + float(style.skin_target_b) * skin_correction

    if role_masks is not None and getattr(style, "semantic_mode", "all") != "all":
        a, b = _reinforce_semantic_chroma(a, b, source_gray, style, role_masks=role_masks)

    # 3. Cel flattening.
    if style.cel_flatten > 0:
        a, b = _cel_flatten_chroma(a, b, source_gray, style.cel_flatten)

    # 4. Lightness — gamma curve, optionally blended back toward the
    #    original B&W L so lineart/ink contrast isn't washed out by
    #    whatever L the model guessed.
    L_norm = np.clip(L / 255.0, 0.0, 1.0)
    L_gamma = np.power(L_norm, style.l_gamma) * 255.0
    if style.l_blend_alpha < 1.0:
        L_source = source_gray.astype(np.float32)
        L_final = L_source * (1.0 - style.l_blend_alpha) + L_gamma * style.l_blend_alpha
    else:
        L_final = L_gamma

    # 5. Neutral fade — force ink/bubbles/gutters to zero chroma, and
    #    soft-fade chroma near white/black per the style's thresholds,
    #    never dropping below `neutral_fade_floor` (keeps bright
    #    surfaces like skin highlights or light hair from going gray).
    keep_mask = combined_neutral_mask(source_gray, line_dilate=1, blur=3)

    g = source_gray.astype(np.float32)
    transition = max(1.0, float(style.neutral_transition))
    white_fade = np.clip((style.white_threshold - g) / transition, 0.0, 1.0)
    black_fade = np.clip((g - style.black_threshold) / transition, 0.0, 1.0)
    tone_fade = np.maximum(white_fade * black_fade, style.neutral_fade_floor)
    if pastel_skin_mask is not None:
        # Bright skin often sits near the page-white threshold.  Preserve a
        # restrained amount of its warm ivory/blush chroma without granting the
        # same privilege to speech bubbles or white background.
        tone_fade = np.maximum(tone_fade, pastel_skin_mask * 0.78)

    keep_override = np.zeros_like(tone_fade, np.float32)
    if semantic_mode != "all" and role_masks is not None:
        attr_floor = np.zeros_like(tone_fade, np.float32)
        attr_floor = np.maximum(attr_floor, role_masks["skin"] * max(0.32, float(style.skin_chroma_scale) * 0.64))
        attr_floor = np.maximum(attr_floor, role_masks["hair"] * max(0.46, float(style.hair_chroma_scale) * 0.74))
        attr_floor = np.maximum(attr_floor, role_masks["eyes"] * max(0.60, float(style.eye_chroma_scale) * 0.82))
        attr_floor = np.maximum(attr_floor, role_masks["clothing"] * max(0.34, float(style.clothing_chroma_scale) * 0.56))
        attr_floor = np.maximum(attr_floor, role_masks["person"] * max(0.24, float(style.person_chroma_scale) * 0.54))
        tone_fade = np.maximum(tone_fade, attr_floor)

        # White hair highlights / pale folds are often mistaken for paper by
        # the neutral mask.  Grant explicit character regions a minimum keep
        # factor so the chroma floor above can actually survive.
        keep_override = np.maximum(keep_override, role_masks["skin"] * 0.36)
        keep_override = np.maximum(keep_override, role_masks["hair"] * 0.72)
        keep_override = np.maximum(keep_override, role_masks["eyes"] * 0.82)
        keep_override = np.maximum(keep_override, role_masks["clothing"] * 0.42)
        keep_override = np.maximum(keep_override, role_masks["person"] * 0.28)

    final_keep = np.maximum(keep_mask * tone_fade, keep_override)
    a = a * final_keep
    b = b * final_keep

    out_lab = np.dstack([
        np.clip(L_final, 0, 255),
        np.clip(a + 128.0, 0, 255),
        np.clip(b + 128.0, 0, 255),
    ]).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)

    if bool(getattr(style, "force_environment_grayscale", False)):
        out_bgr = enforce_character_only_color(
            out_bgr, source_bw_bgr, context=context, role_masks=role_masks)
    return out_bgr
