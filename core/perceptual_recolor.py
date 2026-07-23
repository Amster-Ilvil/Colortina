"""Perceptual recoloring primitives shared by manual brush and region fill.

The goal is to redirect an existing AI colour field toward a selected colour
without replacing the field with a flat LAB value.  Local chroma variation,
soft gradients, small warm/cool shifts and luminance texture are retained.
"""
from __future__ import annotations

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
