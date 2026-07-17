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


def apply_style_grade(colorized_bgr: np.ndarray, source_bw_bgr: np.ndarray,
                      style: StylePreset) -> np.ndarray:
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
    else:
        boost = 1.0 + (style.saturation_boost - 1.0) * np.clip(1.0 - chroma / 40.0, 0.0, 1.0)
        a = a * boost
        b = b * boost

    # 2. Global chroma shifts (warm/cool via b, red/green via a).
    b = b + style.chroma_warm_shift
    a = a + style.chroma_red_shift

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

    final_keep = keep_mask * tone_fade
    a = a * final_keep
    b = b * final_keep

    out_lab = np.dstack([
        np.clip(L_final, 0, 255),
        np.clip(a + 128.0, 0, 255),
        np.clip(b + 128.0, 0, 255),
    ]).astype(np.uint8)
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)
