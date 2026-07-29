"""Perceptual recoloring primitives shared by manual brush and region fill.

The goal is to redirect an existing AI colour field toward a selected colour
without replacing the field with a flat LAB value.  Local chroma variation,
soft gradients, small warm/cool shifts and luminance texture are retained.
"""
from __future__ import annotations

import cv2
import numpy as np


def robust_ab_center(lab: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Return a robust chroma centre from editable mid-tone pixels."""
    valid = active.astype(bool)
    if lab.ndim != 3 or lab.shape[2] < 3 or not np.any(valid):
        return np.array([128.0, 128.0], dtype=np.float32)
    light = lab[..., 0]
    preferred = valid & (light >= 42.0) & (light <= 228.0)
    pixels = lab[..., 1:3][preferred] if np.any(preferred) else lab[..., 1:3][valid]
    return np.median(pixels, axis=0).astype(np.float32)


def perceptual_target_ab(
    lab: np.ndarray,
    active: np.ndarray,
    target_ab: np.ndarray,
    *,
    texture_retention: float = 0.62,
    chroma_retention: float = 0.82,
) -> np.ndarray:
    """Build a target A/B field while retaining AI-authored colour texture.

    Residual chroma is decomposed into radial (saturation) and tangential
    (small hue variation) components around the region's robust colour centre.
    The field is then re-oriented around the chosen target colour.  This keeps
    shading-related colour variation and avoids a uniform painted patch.
    """
    current = lab[..., 1:3].astype(np.float32)
    centre = robust_ab_center(lab, active)
    residual = current - centre[None, None, :]

    source_vec = centre - 128.0
    target_vec = np.asarray(target_ab, dtype=np.float32) - 128.0
    source_mag = float(np.linalg.norm(source_vec))
    target_mag = float(np.linalg.norm(target_vec))

    if source_mag > 1e-4 and target_mag > 1e-4:
        source_unit = source_vec / source_mag
        target_unit = target_vec / target_mag
        cos_a = float(np.clip(np.dot(source_unit, target_unit), -1.0, 1.0))
        sin_a = float(source_unit[0] * target_unit[1] - source_unit[1] * target_unit[0])
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        residual = residual @ rotation.T

    # Do not amplify noisy chroma when the selected colour is very saturated.
    scale = np.clip((target_mag + 12.0) / (source_mag + 12.0), 0.65, 1.25)
    retained = residual * float(np.clip(texture_retention, 0.0, 1.0)) * scale

    # Preserve a portion of the original chroma field as a second safeguard
    # against flattening; target centring still determines the visible hue.
    original_detail = (current - centre[None, None, :]) * float(
        np.clip(chroma_retention, 0.0, 1.0))
    detail = retained * 0.72 + original_detail * 0.28
    detail = np.clip(detail, -42.0, 42.0)
    return np.clip(np.asarray(target_ab, dtype=np.float32)[None, None, :] + detail,
                   0.0, 255.0)


def recolor_hls_toward_rgb(
    image_bgr: np.ndarray,
    rgb: tuple[int, int, int],
    alpha: np.ndarray,
    *,
    active: np.ndarray | None = None,
    hue_strength: float = 1.0,
    saturation_strength: float = 1.0,
    saturation_texture: float = 0.18,
    lightness_strength: float = 0.0,
) -> np.ndarray:
    """Recolour toward an RGB target while preserving visible shading.

    LAB A/B replacement can push saturated reds, blues and violets toward a
    different visible hue after gamut clipping (for example, red can become
    orange and blue can become purple). HLS lets us preserve the existing
    lightness field while steering hue directly toward the user's chosen colour.

    ``alpha`` is a per-pixel 0..1 strength map. Pixels outside ``active`` (or
    where alpha is zero) are copied byte-for-byte from the input image.

    Dark / very light / nearly neutral swatches automatically receive stronger
    lightness matching. That makes manual tools and the new colour filter obey
    the chosen swatch much more faithfully (for example, black no longer stays
    flat grey just because the source shading was bright).
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    if alpha is None:
        return image_bgr.copy()

    alpha = np.asarray(alpha, dtype=np.float32)
    if alpha.shape != image_bgr.shape[:2]:
        raise ValueError("alpha shape must match image height/width")
    alpha = np.clip(alpha, 0.0, 1.0)
    if active is None:
        active_mask = alpha > 1e-6
    else:
        active_mask = np.asarray(active, dtype=bool) & (alpha > 1e-6)
    if not np.any(active_mask):
        return image_bgr.copy()

    r, g, b = [int(np.clip(v, 0, 255)) for v in rgb]
    image_f = image_bgr.astype(np.float32) / 255.0
    hls = cv2.cvtColor(image_f, cv2.COLOR_BGR2HLS)
    target_bgr = np.array([[[b / 255.0, g / 255.0, r / 255.0]]], dtype=np.float32)
    target_h, target_l, target_s = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HLS)[0, 0]

    target_l = float(target_l)
    target_s = float(target_s)
    target_neutral = target_s < 0.08
    target_dark = target_l < 0.16
    target_light = target_l > 0.90

    # Exact swatches such as black, white or near-grey need stronger lightness
    # authority than vivid colours, otherwise preserving shading keeps the fill
    # looking like the old colour with only a slight tint.
    sat_texture = float(np.clip(saturation_texture, 0.0, 1.0))
    hue_strength = float(np.clip(hue_strength, 0.0, 2.0))
    saturation_strength = float(np.clip(saturation_strength, 0.0, 2.0))
    lightness_strength = float(np.clip(lightness_strength, 0.0, 1.0))
    if target_neutral:
        sat_texture *= 0.12
        lightness_strength = max(lightness_strength, 0.80 if (target_dark or target_light) else 0.48)
    elif target_dark:
        sat_texture *= 0.30
        lightness_strength = max(lightness_strength, 0.84)
    elif target_light:
        sat_texture *= 0.52
        lightness_strength = max(lightness_strength, 0.58)
    else:
        lightness_strength = max(lightness_strength, 0.05)

    current_h = hls[..., 0]
    current_l = hls[..., 1]
    current_s = hls[..., 2]

    # Hue is undefined for neutral pixels. Starting those pixels at the target
    # hue prevents a grey/white source from travelling through an arbitrary hue
    # while saturation is being introduced.
    start_h = np.where(current_s < 0.025, float(target_h), current_h)
    delta_h = (float(target_h) - start_h + 180.0) % 360.0 - 180.0
    hue_alpha = np.clip(alpha * hue_strength, 0.0, 1.0)
    new_h = (start_h + delta_h * hue_alpha) % 360.0

    # Retain small local saturation variations so gradients and material texture
    # survive, but centre the result on the selected swatch's saturation. For
    # nearly neutral targets keep only a small residual; otherwise bright/dark
    # grey fills inherit too much colour from the original page.
    median_s = float(np.median(current_s[active_mask]))
    sat_residual = current_s - median_s
    target_s_field = np.clip(target_s + sat_residual * sat_texture, 0.0, 1.0)
    sat_alpha = np.clip(alpha * saturation_strength, 0.0, 1.0)
    new_s = current_s * (1.0 - sat_alpha) + target_s_field * sat_alpha

    # The user's selected hue should be exact, while the AI-authored light/dark
    # modelling remains dominant unless the target colour demands a stronger
    # luminance match (e.g. black, white or grey).
    light_alpha = np.clip(alpha * lightness_strength, 0.0, 1.0)
    new_l = current_l * (1.0 - light_alpha) + target_l * light_alpha

    shifted_hls = hls.copy()
    shifted_hls[..., 0] = new_h
    shifted_hls[..., 1] = new_l
    shifted_hls[..., 2] = new_s
    shifted = cv2.cvtColor(shifted_hls, cv2.COLOR_HLS2BGR)
    shifted_u8 = np.clip(shifted * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)

    out = image_bgr.copy()
    out[active_mask] = shifted_u8[active_mask]
    return out


def normalize_recolor_mode(mode: str | None) -> str:
    """Return the canonical manual recolour mode.

    Older project files and internal callers used several labels for the same
    intent.  Normalising them here keeps brush, bucket, lasso and rectangle
    tools on exactly the same implementation.
    """
    value = str(mode or "shift").strip().lower()
    if value in {"flat", "solid", "pure", "pure_color"}:
        return "flat"
    if value in {"shading", "uniform", "uniform_hue", "keep_shading"}:
        return "shading"
    return "shift"


def pupil_natural_blend_alpha(
    image_bgr: np.ndarray,
    alpha: np.ndarray,
    *,
    active: np.ndarray | None = None,
) -> np.ndarray:
    """Protect iris highlights and the dark pupil while tinting mid-tones.

    The option is intentionally implemented as an alpha-field refinement, not
    a second colour overlay.  Existing mc-v2 highlights, eyelashes and pupil
    depth therefore remain part of the original image, while the coloured iris
    body receives the selected hue.  The returned field is always zero outside
    the caller's authoritative mask.
    """
    alpha_f = np.asarray(alpha, dtype=np.float32)
    if alpha_f.shape != image_bgr.shape[:2]:
        raise ValueError("alpha shape must match image height/width")
    alpha_f = np.clip(alpha_f, 0.0, 1.0)
    active_mask = alpha_f > 1e-6
    if active is not None:
        active_mask &= np.asarray(active, dtype=bool)
    if int(np.count_nonzero(active_mask)) < 8:
        return np.where(active_mask, alpha_f, 0.0).astype(np.float32)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    light = lab[..., 0]
    values = light[active_mask]
    p10, p35, p65, p92 = [float(v) for v in np.percentile(
        values, (10.0, 35.0, 65.0, 92.0))]

    # A nearly flat patch has no distinct pupil/highlight structure to protect.
    if p92 - p10 < 18.0:
        return np.where(active_mask, alpha_f, 0.0).astype(np.float32)

    dark_span = max(8.0, p35 - p10)
    light_span = max(8.0, p92 - p65)
    dark_gate = np.clip((light - p10) / dark_span, 0.0, 1.0)
    highlight_gate = np.clip((p92 - light) / light_span, 0.0, 1.0)
    midtone_gate = dark_gate * highlight_gate

    # Keep a small tint in protected pixels so the result still feels unified,
    # but make the iris body authoritative.  Blur only the tonal weighting; the
    # hard edit boundary is restored immediately afterwards.
    weight = 0.18 + 0.82 * midtone_gate
    weight = cv2.GaussianBlur(weight.astype(np.float32), (0, 0), 0.7)
    refined = alpha_f * np.clip(weight, 0.14, 1.0)
    return np.where(active_mask, refined, 0.0).astype(np.float32)


def _hue_delta_degrees(target: np.ndarray | float,
                       source: np.ndarray | float) -> np.ndarray:
    return (np.asarray(target, dtype=np.float32)
            - np.asarray(source, dtype=np.float32) + 180.0) % 360.0 - 180.0


def _circular_mean_degrees(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if values.size == 0 or float(np.sum(weights)) <= 1e-6:
        return 0.0
    radians = np.deg2rad(values)
    x = float(np.sum(np.cos(radians) * weights))
    y = float(np.sum(np.sin(radians) * weights))
    if abs(x) + abs(y) <= 1e-8:
        return float(np.median(values))
    return float(np.rad2deg(np.arctan2(y, x)) % 360.0)


def _interpolate_hue(current_h: np.ndarray,
                     target_h: np.ndarray | float,
                     amount: np.ndarray) -> np.ndarray:
    return (current_h + _hue_delta_degrees(target_h, current_h) * amount) % 360.0


def _shading_lightness_field(current_l: np.ndarray,
                             active_mask: np.ndarray,
                             target_l: float,
                             target_s: float) -> np.ndarray:
    """Preserve modelling while making neutral/dark/light swatches authoritative."""
    values = current_l[active_mask]
    if values.size == 0:
        return current_l.copy()

    # Ordinary chromatic colours preserve the *shape* of the authored luminance
    # field while recentering its median on the selected swatch.  A flat source
    # therefore reaches the literal selected colour in one operation, whereas
    # folds and highlights keep their relative light/dark offsets.
    if target_s >= 0.08 and 0.16 < target_l < 0.84:
        median = float(np.median(values))
        return np.clip(current_l + (target_l - median) * 0.90, 0.0, 1.0)

    p08, p92 = np.percentile(values, [8.0, 92.0])
    spread = float(p92 - p08)
    if spread < 0.018:
        normalised = np.full_like(current_l, 0.5, dtype=np.float32)
    else:
        normalised = np.clip((current_l - float(p08)) / spread, 0.0, 1.0)

    if target_l <= 0.16:
        # Black/dark swatches become a dark tonal family, not unchanged grey.
        low = max(0.0, target_l * 0.35)
        high = min(0.30, target_l + 0.24)
        return low + normalised * (high - low)
    if target_l >= 0.84:
        # White/light swatches keep folds through a compressed highlight range.
        low = max(0.68, target_l - 0.22)
        high = min(1.0, target_l + 0.06)
        return low + normalised * (high - low)

    # Mid-grey: remove hue, centre the region on the requested grey and retain
    # a controlled amount of its original contrast.
    median = float(np.median(values))
    return np.clip(target_l + (current_l - median) * 0.62,
                   max(0.0, target_l - 0.28), min(1.0, target_l + 0.28))


def _target_lab_from_rgb(rgb: tuple[int, int, int]) -> np.ndarray:
    """Convert an RGB swatch to OpenCV LAB without changing channel order."""
    r, g, b = [int(np.clip(v, 0, 255)) for v in rgb]
    pixel = np.array([[[b, g, r]]], dtype=np.uint8)
    return cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


def _natural_lightness_field(current_l: np.ndarray,
                             active_mask: np.ndarray,
                             target_l: float,
                             target_chroma: float) -> np.ndarray:
    """Keep authored modelling while making black/white/grey choices visible.

    The reference implementation intentionally changes luminance only slightly
    for ordinary chromatic swatches.  Neutral and extreme-lightness swatches are
    the exception: preserving the old luminance literally would make black stay
    grey and white stay muddy, so their tonal range is compressed around the
    requested colour while retaining folds and highlights.
    """
    values = current_l[active_mask]
    if values.size == 0:
        return current_l.copy()

    median = float(np.median(values))
    if target_chroma >= 5.0 and 38.0 < target_l < 224.0:
        # Equivalent in spirit to the reference package's tiny target-L mix:
        # colour changes decisively, but the AI-authored light/dark field remains.
        return np.clip(current_l + (float(target_l) - median) * 0.10,
                       0.0, 255.0)

    p08, p92 = np.percentile(values, [8.0, 92.0])
    spread = float(p92 - p08)
    if spread < 3.0:
        normalised = np.full_like(current_l, 0.5, dtype=np.float32)
    else:
        normalised = np.clip((current_l - float(p08)) / spread, 0.0, 1.0)

    if target_l <= 42.0:
        low = max(0.0, float(target_l) * 0.20)
        high = min(78.0, float(target_l) + 58.0)
        return low + normalised * (high - low)
    if target_l >= 224.0:
        low = max(176.0, float(target_l) - 62.0)
        high = min(255.0, float(target_l) + 10.0)
        return low + normalised * (high - low)

    # Mid-grey: remove old colour while retaining controlled local contrast.
    return np.clip(float(target_l) + (current_l - median) * 0.64,
                   max(0.0, float(target_l) - 70.0),
                   min(255.0, float(target_l) + 70.0))


def recolor_with_mode(
    image_bgr: np.ndarray,
    rgb: tuple[int, int, int],
    alpha: np.ndarray,
    *,
    active: np.ndarray | None = None,
    mode: str = "shift",
    pupil_blend: bool = False,
) -> np.ndarray:
    """Apply one of the three independent manual recolour algorithms.

    ``shift`` — natural hue migration
        Ports the stable LAB colour-field migration used by the supplied
        reference package.  The robust region colour centre is redirected to
        the selected swatch while bounded local A/B variation, gradients and
        material texture are retained.

    ``shading`` — uniform hue, preserve shading
        Uses a separate HLS path.  Hue and saturation become uniform at full
        strength, while the existing lightness field is recentered rather than
        flattened.  This is deliberately not a parameter variant of ``shift``.

    ``flat`` — fully uniform pure colour
        Alpha-composites the literal selected RGB.  Full-strength pixels are
        byte-exact; only the inward feather or brush edge is blended.
    """
    if image_bgr is None or image_bgr.size == 0 or alpha is None:
        return image_bgr

    alpha_f = np.asarray(alpha, dtype=np.float32)
    if alpha_f.shape != image_bgr.shape[:2]:
        raise ValueError("alpha shape must match image height/width")
    alpha_f = np.clip(alpha_f, 0.0, 1.0)
    if active is None:
        active_mask = alpha_f > 1e-6
    else:
        active_mask = np.asarray(active, dtype=bool) & (alpha_f > 1e-6)
    if not np.any(active_mask):
        return image_bgr.copy()

    canonical = normalize_recolor_mode(mode)

    # Never let a non-zero feather outside the caller's authoritative mask
    # influence conversion.  All three tools therefore share identical bounds.
    alpha_f = np.where(active_mask, alpha_f, 0.0).astype(np.float32)
    if pupil_blend and canonical != "flat":
        alpha_f = pupil_natural_blend_alpha(
            image_bgr, alpha_f, active=active_mask)
        active_mask &= alpha_f > 1e-6
        if not np.any(active_mask):
            return image_bgr.copy()
    r, g, b = [int(np.clip(v, 0, 255)) for v in rgb]
    target_bgr_u8 = np.array([b, g, r], dtype=np.uint8)

    if canonical == "flat":
        source_f = image_bgr.astype(np.float32)
        a3 = alpha_f[..., None]
        target = target_bgr_u8.astype(np.float32)[None, None, :]
        blended = source_f * (1.0 - a3) + target * a3
        out = image_bgr.copy()
        blended_u8 = np.clip(blended + 0.5, 0.0, 255.0).astype(np.uint8)
        out[active_mask] = blended_u8[active_mask]
        exact = active_mask & (alpha_f >= 0.999)
        out[exact] = target_bgr_u8
        return out

    if canonical == "shift":
        # V3 restores the simpler, stable LAB/HLS-era natural migration.  It
        # steers chroma toward the selected colour while preserving the current
        # mc-v2 texture and almost all of its luminance modelling.
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = _target_lab_from_rgb((r, g, b))
        blend_alpha = np.power(alpha_f, 0.75)[..., None]
        lab[..., 1:3] = (
            lab[..., 1:3] * (1.0 - blend_alpha)
            + target_lab[1:3] * blend_alpha * 0.95
        )

        desired_l = lab[..., 0] * 0.88 + float(target_lab[0]) * 0.12
        light_alpha = blend_alpha[..., 0]
        lab[..., 0] = (
            lab[..., 0] * (1.0 - light_alpha)
            + desired_l * light_alpha
        )

        converted = cv2.cvtColor(
            np.clip(lab, 0.0, 255.0).astype(np.uint8), cv2.COLOR_LAB2BGR)
        out = image_bgr.copy()
        out[active_mask] = converted[active_mask]
        return out

    # Uniform-hue mode intentionally remains a separate HLS implementation.
    # Constant H/S gives a genuinely unified colour family across light and dark
    # folds, while _shading_lightness_field retains the modelled luminance shape.
    image_f = image_bgr.astype(np.float32) / 255.0
    hls = cv2.cvtColor(image_f, cv2.COLOR_BGR2HLS)
    target_px = np.array([[[b / 255.0, g / 255.0, r / 255.0]]], dtype=np.float32)
    target_h, target_l, target_s = [float(v) for v in cv2.cvtColor(
        target_px, cv2.COLOR_BGR2HLS)[0, 0]]

    current_h = hls[..., 0]
    current_l = hls[..., 1]
    current_s = hls[..., 2]
    desired_l = _shading_lightness_field(
        current_l, active_mask, target_l, target_s)

    # Neutral source pixels have no meaningful hue.  Starting them directly at
    # the target avoids a transient red/green cast during partial-strength edits.
    start_h = np.where(current_s < 0.025, target_h, current_h)
    chroma_alpha = np.power(alpha_f, 0.72)
    new_h = _interpolate_hue(start_h, target_h, chroma_alpha)
    new_s = current_s * (1.0 - chroma_alpha) + target_s * chroma_alpha
    new_l = current_l * (1.0 - alpha_f) + desired_l * alpha_f

    shifted_hls = hls.copy()
    shifted_hls[..., 0] = new_h
    shifted_hls[..., 1] = np.clip(new_l, 0.0, 1.0)
    shifted_hls[..., 2] = np.clip(new_s, 0.0, 1.0)
    shifted = cv2.cvtColor(shifted_hls, cv2.COLOR_HLS2BGR)
    shifted_u8 = np.clip(shifted * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)

    out = image_bgr.copy()
    out[active_mask] = shifted_u8[active_mask]
    return out
