"""Natural, edge-aware post-color filters for manga pages.

The previous implementation applied global Lab/HSV shifts directly to every
pixel.  That is fast, but it can look like a translucent glass sheet placed over
the page because flat areas, line art, highlights and shadows all move together.

This implementation separates the image into an edge-aware tone base and a
fine-detail residual, edits tonal zones on the base only, then restores the
original detail.  Saturation and warmth use content-aware masks and protect
paper white / black ink.  The result is closer to a photographic tone equalizer
than a uniform overlay.
"""
from __future__ import annotations

import cv2
import numpy as np


_FILTER_KEYS = (
    "brightness", "contrast", "saturation", "warmth",
    "shadow_lift", "highlight",
)


def _slider_factor(tuning: dict | None, key: str, default: float = 100.0) -> float:
    try:
        value = float((tuning or {}).get(key, default))
    except Exception:
        value = default
    return float(np.clip(value, 0.0, 200.0)) / 100.0


def _smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    if edge1 <= edge0:
        return (value >= edge1).astype(np.float32)
    t = np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _guided_filter(guidance: np.ndarray, src: np.ndarray,
                   radius: int, eps: float) -> np.ndarray:
    """Small dependency-free grayscale guided filter.

    ``cv2.ximgproc.guidedFilter`` is not guaranteed to be installed, so the
    standard box-filter formulation is kept here.  Complexity is linear in the
    number of pixels and edges in ``guidance`` are retained in the base layer.
    """
    guidance = np.asarray(guidance, dtype=np.float32)
    src = np.asarray(src, dtype=np.float32)
    radius = max(2, int(radius))
    ksize = (radius * 2 + 1, radius * 2 + 1)

    mean_i = cv2.boxFilter(guidance, cv2.CV_32F, ksize, normalize=True,
                           borderType=cv2.BORDER_REFLECT)
    mean_p = cv2.boxFilter(src, cv2.CV_32F, ksize, normalize=True,
                           borderType=cv2.BORDER_REFLECT)
    corr_i = cv2.boxFilter(guidance * guidance, cv2.CV_32F, ksize,
                           normalize=True, borderType=cv2.BORDER_REFLECT)
    corr_ip = cv2.boxFilter(guidance * src, cv2.CV_32F, ksize,
                            normalize=True, borderType=cv2.BORDER_REFLECT)
    var_i = np.maximum(corr_i - mean_i * mean_i, 0.0)
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + float(max(eps, 1e-7)))
    b = mean_p - a * mean_i
    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize, normalize=True,
                           borderType=cv2.BORDER_REFLECT)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize, normalize=True,
                           borderType=cv2.BORDER_REFLECT)
    return np.clip(mean_a * guidance + mean_b, 0.0, 1.0).astype(np.float32)


def _piecewise_contrast(x: np.ndarray, gain: float, pivot: float) -> np.ndarray:
    """Continuous, endpoint-preserving contrast curve around ``pivot``."""
    gain = float(np.clip(gain, 0.35, 2.8))
    pivot = float(np.clip(pivot, 0.28, 0.72))
    below = x <= pivot
    out = np.empty_like(x, dtype=np.float32)
    left = np.clip(x / max(pivot, 1e-5), 0.0, 1.0)
    right = np.clip((1.0 - x) / max(1.0 - pivot, 1e-5), 0.0, 1.0)
    out[below] = pivot * np.power(left[below], gain)
    out[~below] = 1.0 - (1.0 - pivot) * np.power(right[~below], gain)
    return np.clip(out, 0.0, 1.0)


def _source_guidance(source_bw_bgr: np.ndarray | None,
                     shape: tuple[int, int], fallback_luma: np.ndarray) -> np.ndarray:
    h, w = shape
    if source_bw_bgr is None or getattr(source_bw_bgr, "size", 0) == 0:
        return fallback_luma.astype(np.float32)
    source = source_bw_bgr
    if source.shape[:2] != (h, w):
        source = cv2.resize(source, (w, h), interpolation=cv2.INTER_AREA)
    if source.ndim == 2:
        gray = source
    else:
        gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    return (gray.astype(np.float32) / 255.0)


def _protection_maps(source_luma: np.ndarray,
                     result_luma: np.ndarray,
                     chroma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(tone_protection, color_permission)`` maps.

    Speech-bubble/paper whites and solid ink should not drift.  Strong source
    edges also receive partial protection to prevent bright/dark bands along
    line art after shadow/highlight edits.
    """
    paper = _smoothstep(0.90, 0.985, source_luma)
    ink = 1.0 - _smoothstep(0.025, 0.16, source_luma)

    gx = cv2.Sobel(source_luma, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(source_luma, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)
    edge = np.clip(edge / 1.25, 0.0, 1.0)
    edge = cv2.GaussianBlur(edge, (3, 3), 0)

    near_neutral = 1.0 - _smoothstep(2.0, 12.0, chroma)
    paper_neutral = paper * (0.45 + 0.55 * near_neutral)
    tone_protection = np.clip(np.maximum(ink * 0.96, paper_neutral * 0.92), 0.0, 1.0)
    tone_protection = np.maximum(tone_protection, edge * 0.48)

    midtone = _smoothstep(0.035, 0.16, result_luma) * (1.0 - _smoothstep(0.91, 0.995, result_luma))
    color_permission = midtone * (1.0 - ink * 0.98) * (1.0 - paper_neutral * 0.92)
    color_permission *= (1.0 - edge * 0.18)
    return np.clip(tone_protection, 0.0, 1.0), np.clip(color_permission, 0.0, 1.0)


def apply_image_filter(result_bgr: np.ndarray,
                       tuning: dict | None,
                       *,
                       style_strength: float = 1.0,
                       is_styled: bool = True,
                       source_bw_bgr: np.ndarray | None = None) -> np.ndarray:
    """Apply a natural edge-aware filter to a colored page.

    Tonal controls operate on an edge-aware illumination/base layer.  Fine
    texture and line detail are restored afterward.  Saturation behaves like
    vibrance (low-chroma colors move more than already-saturated colors), and
    warmth is restricted to drawable colored regions instead of tinting paper
    and ink uniformly.
    """
    if result_bgr is None or result_bgr.size == 0:
        return result_bgr
    tuning = dict(tuning or {})
    if all(int(tuning.get(key, 100)) == 100 for key in _FILTER_KEYS):
        return result_bgr

    brightness = _slider_factor(tuning, "brightness")
    contrast = _slider_factor(tuning, "contrast")
    saturation = _slider_factor(tuning, "saturation")
    warmth = _slider_factor(tuning, "warmth")
    shadow_lift = _slider_factor(tuning, "shadow_lift")
    highlight = _slider_factor(tuning, "highlight")

    # Float Lab avoids repeated uint8 conversions and gives a perceptual L axis.
    bgr_float = result_bgr.astype(np.float32) / 255.0
    lab = cv2.cvtColor(bgr_float, cv2.COLOR_BGR2LAB).astype(np.float32)
    luma = np.clip(lab[..., 0] / 100.0, 0.0, 1.0)
    a = lab[..., 1].copy()
    b = lab[..., 2].copy()
    chroma = np.sqrt(a * a + b * b).astype(np.float32)

    h, w = luma.shape
    source_luma = _source_guidance(source_bw_bgr, (h, w), luma)
    # Blend source line-art guidance with the colored result's luminance.  This
    # keeps line boundaries hard without forcing all modeled shading back to B&W.
    guidance = np.clip(source_luma * 0.68 + luma * 0.32, 0.0, 1.0)
    radius = int(np.clip(round(min(h, w) / 95.0), 6, 30))
    base = _guided_filter(guidance, luma, radius=radius, eps=0.0065)
    detail = np.clip(luma - base, -0.20, 0.20)

    tone_protection, color_permission = _protection_maps(source_luma, luma, chroma)
    editable_tone = 1.0 - tone_protection
    adjusted = base.copy()

    # Exposure-like brightness.  The asymmetric power curve preserves black and
    # white endpoints rather than adding a uniform gray veil.
    bright_delta = brightness - 1.0
    if abs(bright_delta) > 1e-5:
        if bright_delta > 0.0:
            exponent = 1.0 / (1.0 + bright_delta * 1.25)
        else:
            exponent = 1.0 + (-bright_delta) * 1.35
        bright_base = np.power(np.clip(adjusted, 0.0, 1.0), exponent)
        adjusted += (bright_base - adjusted) * editable_tone

    # Edge-aware tonal zones: masks come from the smooth base, not raw pixels.
    shadow_delta = shadow_lift - 1.0
    if abs(shadow_delta) > 1e-5:
        shadow_mask = (1.0 - _smoothstep(0.18, 0.60, base)) * editable_tone
        if shadow_delta > 0.0:
            change = shadow_delta * 0.28 * shadow_mask * (1.0 - adjusted)
        else:
            change = shadow_delta * 0.22 * shadow_mask * adjusted
        adjusted = np.clip(adjusted + change, 0.0, 1.0)

    highlight_delta = highlight - 1.0
    if abs(highlight_delta) > 1e-5:
        highlight_mask = _smoothstep(0.42, 0.88, base) * editable_tone
        if highlight_delta > 0.0:
            change = highlight_delta * 0.20 * highlight_mask * (1.0 - adjusted)
        else:
            change = highlight_delta * 0.24 * highlight_mask * adjusted
        adjusted = np.clip(adjusted + change, 0.0, 1.0)

    contrast_delta = contrast - 1.0
    if abs(contrast_delta) > 1e-5:
        valid = base[(source_luma > 0.08) & (source_luma < 0.94)]
        pivot = float(np.median(valid)) if valid.size else 0.5
        gain = float(np.exp(contrast_delta * 0.86))
        curved = _piecewise_contrast(adjusted, gain=gain, pivot=pivot)
        adjusted += (curved - adjusted) * editable_tone

    # Recombine the untouched detail layer. Contrast can gently reinforce local
    # detail, but is deliberately limited to avoid halos/crunchy line art.
    detail_gain = float(np.clip(1.0 + contrast_delta * 0.10, 0.90, 1.10))
    luma_out = np.clip(adjusted + detail * detail_gain, 0.0, 1.0)
    luma_out = luma_out * (1.0 - tone_protection) + luma * tone_protection

    # Style strength already blends raw mc-v2 and style grade.  Color-filter
    # strength follows that share, but never drops to zero so original mc-v2 can
    # still be adjusted intentionally.
    color_weight = 1.0 if not is_styled else 0.42 + 0.58 * float(np.clip(style_strength, 0.0, 1.0))

    saturation_delta = saturation - 1.0
    if abs(saturation_delta) > 1e-5:
        if saturation_delta > 0.0:
            # Vibrance: low/medium chroma receives more gain than colors already
            # near saturation, reducing clipping and cartoon-like blocks.
            vibrance = 1.0 - 0.70 * _smoothstep(18.0, 62.0, chroma)
        else:
            vibrance = np.ones_like(chroma, np.float32)
        sat_gain = 1.0 + saturation_delta * color_weight * vibrance * color_permission
        a *= np.clip(sat_gain, 0.12, 1.85)
        b *= np.clip(sat_gain, 0.12, 1.85)

    warmth_delta = warmth - 1.0
    if abs(warmth_delta) > 1e-5:
        # Subtle opponent-color balance, strongest in low/medium chroma and
        # midtones.  No uniform yellow cast over speech bubbles or black ink.
        neutral_room = 0.35 + 0.65 * (1.0 - _smoothstep(24.0, 70.0, chroma))
        warm_mask = color_permission * neutral_room
        amount = warmth_delta * color_weight
        a += amount * 2.0 * warm_mask
        b += amount * 6.2 * warm_mask

    lab[..., 0] = luma_out * 100.0
    lab[..., 1] = np.clip(a, -127.0, 127.0)
    lab[..., 2] = np.clip(b, -127.0, 127.0)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    out = np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)
    return out
