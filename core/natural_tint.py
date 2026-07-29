"""Natural perceptual tinting shared by global colour controls.

This module is deliberately independent from mc-v2 and the manual editing
stack.  It changes only the already-colourized result and therefore cannot
alter model inference, hint placement, line-region maps, or undo behaviour.

The implementation redirects the low-frequency Lab chroma field toward a
chosen swatch while rotating and retaining bounded local colour detail.  The
original luminance field remains authoritative except for neutral or extreme
black/white swatches, where a gentle tone-family adjustment makes the selected
colour visibly meaningful without producing a flat overlay.
"""
from __future__ import annotations

import cv2
import numpy as np


def _target_lab_float(rgb: tuple[int, int, int]) -> np.ndarray:
    r, g, b = (int(np.clip(v, 0, 255)) for v in rgb)
    pixel = np.array([[[b / 255.0, g / 255.0, r / 255.0]]], np.float32)
    return cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, np.float32)
    target = np.asarray(target, np.float32)
    source_norm = float(np.linalg.norm(source))
    target_norm = float(np.linalg.norm(target))
    if source_norm <= 1e-5 or target_norm <= 1e-5:
        return np.eye(2, dtype=np.float32)
    source_u = source / source_norm
    target_u = target / target_norm
    cosine = float(np.clip(np.dot(source_u, target_u), -1.0, 1.0))
    sine = float(source_u[0] * target_u[1] - source_u[1] * target_u[0])
    return np.array([[cosine, -sine], [sine, cosine]], np.float32)


def _effective_alpha(alpha: np.ndarray, authority: float) -> np.ndarray:
    """Scale a mask while allowing useful 100-200% control without clipping.

    Values up to 100% scale linearly.  The second 100% progressively consumes
    the remaining distance to full authority, so 200% is strong but still
    bounded and cannot create an invalid blend coefficient.
    """
    base = np.clip(np.asarray(alpha, np.float32), 0.0, 1.0)
    authority = float(np.clip(authority, 0.0, 2.0))
    if authority <= 1.0:
        return base * authority
    first = base
    extra = authority - 1.0
    return np.clip(first + (1.0 - first) * extra * 0.72, 0.0, 1.0)


def _tone_family_field(current_l: np.ndarray,
                       active: np.ndarray,
                       target_l: float) -> np.ndarray:
    values = current_l[active]
    if values.size == 0:
        return current_l.copy()
    median = float(np.median(values))

    if target_l <= 18.0:
        # Multiplicative compression preserves every fold/highlight instead of
        # replacing the page with uniform black or grey.
        factor = 0.42 + 0.22 * (target_l / 18.0)
        return np.clip(current_l * factor, 0.0, 100.0)
    if target_l >= 88.0:
        factor = 0.46 + 0.20 * ((100.0 - target_l) / 12.0)
        return np.clip(100.0 - (100.0 - current_l) * factor, 0.0, 100.0)

    # Ordinary chromatic colours only recenter the illumination slightly.
    return np.clip(current_l + (target_l - median) * 0.14, 0.0, 100.0)


def apply_natural_tint(image_bgr: np.ndarray,
                       rgb: tuple[int, int, int],
                       alpha: np.ndarray,
                       *,
                       active: np.ndarray | None = None,
                       authority: float = 1.0,
                       texture_retention: float = 0.62,
                       chroma_retention: float = 0.34,
                       tone_strength: float = 0.04) -> np.ndarray:
    """Bias an image toward ``rgb`` without a flat colour-overlay appearance.

    ``alpha`` defines the authoritative editable area. ``authority`` accepts
    0..2, matching a 0..200% UI control. Local A/B residuals are rotated toward
    the target colour and retained, so mc-v2 gradients and material texture do
    not collapse into a single colour block.
    """
    if image_bgr is None or image_bgr.size == 0 or alpha is None:
        return image_bgr

    alpha_f = np.asarray(alpha, np.float32)
    if alpha_f.shape != image_bgr.shape[:2]:
        raise ValueError("alpha shape must match image height/width")
    if active is None:
        active_mask = alpha_f > 1e-6
    else:
        active_mask = np.asarray(active, bool) & (alpha_f > 1e-6)
    if not np.any(active_mask):
        return image_bgr.copy()

    mix = _effective_alpha(alpha_f, authority)
    mix = np.where(active_mask, mix, 0.0).astype(np.float32)
    if float(mix.max()) <= 1e-6:
        return image_bgr.copy()

    image_f = image_bgr.astype(np.float32) / 255.0
    lab = cv2.cvtColor(image_f, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = _target_lab_float(rgb)

    current_ab = lab[..., 1:3].copy()
    centre = np.median(current_ab[active_mask], axis=0).astype(np.float32)
    residual = current_ab - centre[None, None, :]
    rotation = _rotation_between(centre, target_lab[1:3])
    rotated = residual @ rotation.T

    texture_retention = float(np.clip(texture_retention, 0.0, 1.0))
    chroma_retention = float(np.clip(chroma_retention, 0.0, 1.0))
    # Low-frequency variation carries material and illumination colour; small
    # high-frequency variation carries texture. Bound both to prevent colourful
    # noise from being amplified by a strong 150-200% setting.
    sigma = float(np.clip(min(image_bgr.shape[:2]) / 180.0, 0.8, 3.2))
    smooth_residual = np.empty_like(rotated)
    for channel in range(2):
        smooth_residual[..., channel] = cv2.GaussianBlur(
            rotated[..., channel], (0, 0), sigma)
    fine_residual = rotated - smooth_residual
    detail = (smooth_residual * texture_retention
              + fine_residual * chroma_retention)
    detail = np.clip(detail, -28.0, 28.0)
    desired_ab = target_lab[1:3][None, None, :] + detail

    a3 = mix[..., None]
    lab[..., 1:3] = current_ab * (1.0 - a3) + desired_ab * a3

    target_chroma = float(np.linalg.norm(target_lab[1:3]))
    extreme = max(
        np.clip((18.0 - float(target_lab[0])) / 18.0, 0.0, 1.0),
        np.clip((float(target_lab[0]) - 88.0) / 12.0, 0.0, 1.0),
    )
    neutral = float(np.clip((10.0 - target_chroma) / 10.0, 0.0, 1.0))
    adaptive_tone = max(float(np.clip(tone_strength, 0.0, 1.0)),
                        0.58 * extreme + 0.18 * neutral)
    if adaptive_tone > 1e-6:
        desired_l = _tone_family_field(lab[..., 0], active_mask, float(target_lab[0]))
        l_mix = np.clip(mix * adaptive_tone, 0.0, 1.0)
        lab[..., 0] = lab[..., 0] * (1.0 - l_mix) + desired_l * l_mix

    converted = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    converted = np.clip(converted * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    out = image_bgr.copy()
    out[active_mask] = converted[active_mask]
    return out
