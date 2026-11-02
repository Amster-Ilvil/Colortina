"""Hue-faithful, source-ink-aware manga wash rendering.

The Generator owns the colour palette, composition and shading. This renderer
never attempts to segment hair/skin/clothes or invent a new palette. It applies
only chroma attenuation, a restrained paper-lit lift and source black-ink lock.

Important: no desaturation from style grading is allowed *before* this step.
"""
from __future__ import annotations

import cv2
import numpy as np

# The three named styles share one generator-first colour policy.  These
# targets are intentionally not near zero: the previous values (0.53/.39/.27)
# pushed legitimate skin/clothing/hair colours into gray on toned manga pages.
# They control pigment transparency, not the underlying character palette.
# At strength=0 the renderer is an exact identity.
_CHROMA_TARGET = {"light2": 0.77, "light2_soft": 0.64, "light3": 0.50}


def render_inkwash(source_bgr: np.ndarray, color_bgr: np.ndarray, *,
                   preset: str = "light2", strength: float = 0.78,
                   grain: float = 0.08,
                   preserve_mask: np.ndarray | None = None,
                   light3_intensity: float = 1.0) -> np.ndarray:
    """Return a subtle wash retaining mc-v2 hue, facial shading and ink.

    Source gray is only a black-ink / white-paper *selective* guide, never a
    replacement luminance layer: screen-toned source scans otherwise desaturate
    into dirty gray when composed over the native generator output.
    """
    if (source_bgr.shape != color_bgr.shape or source_bgr.ndim != 3
            or source_bgr.shape[2] != 3 or source_bgr.dtype != np.uint8
            or color_bgr.dtype != np.uint8):
        raise ValueError("source/color must have equal HxWx3 uint8 BGR dimensions")
    s = float(np.clip(strength, 0, 1))
    if s == 0:
        return color_bgr.copy()
    if preserve_mask is not None and np.asarray(preserve_mask).shape != source_bgr.shape[:2]:
        raise ValueError("preserve_mask must match the source HxW dimensions")

    gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    src_l = gray * (100.0 / 255.0)
    lab = cv2.cvtColor(color_bgr.astype(np.float32) / 255.0,
                       cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    chroma = np.sqrt(a * a + b * b)

    # Chroma is attenuated by a common scalar in Lab. The a:b direction
    # (therefore hue) is kept, even on unusual hairstyles/garments.
    target = _CHROMA_TARGET.get(str(preset), _CHROMA_TARGET['light2'])
    # The existing GUI's 极淡水墨 intensity slider historically affected only
    # the legacy style-grade path. Generator-first rendering skips that grade,
    # so make the control effective here instead of silently ignoring it.
    if str(preset) == "light3":
        intensity = float(np.clip(light3_intensity, 0.0, 2.0))
        target *= 0.65 + 0.35 * intensity
    pigment = 1.0 - s * (1.0 - target)
    # Saturated generator colours encode useful character/garment identity.
    # Retain a little more of them while continuing to soften weaker washes.
    # The cap remains below 1 and the three presets keep their order.
    pigment += (1.0 - pigment) * s * 0.09 * np.clip(chroma / 44.0, 0.0, 1.0)
    # Dark near-black hair frequently has blue-violet highlight noise; turn
    # that *dark* chroma down without guessing whether an object is hair.
    deep_ink = np.clip((37.0 - l) / 25.0, 0.0, 1.0)
    deep_ink *= np.clip((110.0 - gray) / 90.0, 0.0, 1.0)
    pigment_field = pigment * (1.0 - deep_ink * s * 0.36)
    a *= pigment_field
    b *= pigment_field

    # Slight edge-aware pigment mixing prevents spotty chroma gradients while
    # retaining the generator's silhouette; L/ink details are never smoothed.
    # This is purposefully much weaker than legacy whole-page colour blurring.
    colour_smooth = s * 0.12
    if colour_smooth > 0:
        a_mix = cv2.bilateralFilter(a.astype(np.float32), 5, 5.0, 3.0)
        b_mix = cv2.bilateralFilter(b.astype(np.float32), 5, 5.0, 3.0)
        a = a * (1.0 - colour_smooth) + a_mix * colour_smooth
        b = b * (1.0 - colour_smooth) + b_mix * colour_smooth

    # Only in chromatic *deep* shadows, gently mute blue-violet noise. Do not
    # remap blue/violet garments, saturated highlights, or neutral black hair.
    violet = np.clip((a - 7.0) / 15.0, 0.0, 1.0)
    violet *= np.clip((-b - 5.0) / 17.0, 0.0, 1.0)
    violet *= np.clip((48.0 - l) / 23.0, 0.0, 1.0)
    violet *= np.clip((gray - 65.0) / 90.0, 0.0, 1.0)
    violet_alpha = violet * s * 0.17
    a *= (1.0 - violet_alpha)
    b *= (1.0 - violet_alpha)

    # Retain generator shading as the baseline. Only moderately dark colored
    # pools receive a small lift toward the source paper, and never black ink.
    paint = np.clip(chroma / 13.0, 0.0, 1.0)
    bright_source = np.clip((gray - 90.0) / 100.0, 0.0, 1.0)
    mid_shadow = np.clip((73.0 - l) / 36.0, 0.0, 1.0)
    # Lighter variants are fractionally more paper-lit; all three keep the
    # original shading and never turn source gray halftones into final gray.
    lift_gain = {"light2": 0.19, "light2_soft": 0.22, "light3": 0.25}.get(str(preset), 0.19)
    lift = s * lift_gain * paint * bright_source * mid_shadow * (1.0 - deep_ink)
    lift_target = np.maximum(l, np.minimum(93.0, src_l + 4.0))
    l += (lift_target - l) * lift

    # Protect original printed black contours (not whole gray screentone
    # areas) without replacing the model's chromatic interiors.
    ink = np.clip((74.0 - gray) / 67.0, 0.0, 1.0) ** 2.1
    ink *= s
    # Absolute-black printed pixels remain black at every nonzero wash strength;
    # partially restoring them would leave tinted rims on manga lettering.
    ink = np.maximum(ink, (gray <= 10.0).astype(np.float32))
    l = l * (1.0 - ink) + src_l * ink
    a *= (1.0 - ink)
    b *= (1.0 - ink)

    # Speech balloons and unprinted white paper stay white, but highlights
    # intentionally coloured by the generator remain tinted.
    paper = ((gray >= 253.0) & (l >= 96.0) & (chroma < 9.0))
    l[paper] = l[paper] * (1.0 - s) + src_l[paper] * s
    a[paper] *= 1.0 - s
    b[paper] *= 1.0 - s

    # A very faint luminance-only pigment grain, anchored to coloured areas
    # and deterministic. Avoid adding dots on screen tones / linework.
    grain = float(np.clip(grain, 0, 1))
    if grain > 0:
        rng = np.random.default_rng(20261102)
        n = rng.standard_normal(l.shape).astype(np.float32)
        wash = paint * np.clip((gray - 156.0) / 100.0, 0.0, 1.0)
        l += n * grain * s * 0.17 * wash

    out_lab = cv2.merge((np.clip(l, 0, 100),
                         np.clip(a, -127, 127),
                         np.clip(b, -127, 127))).astype(np.float32)
    out = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)
    rendered = np.clip(np.rint(out * 255), 0, 255).astype(np.uint8)
    if preserve_mask is not None:
        rendered[np.asarray(preserve_mask) > 0] = color_bgr[np.asarray(preserve_mask) > 0]
    return rendered
