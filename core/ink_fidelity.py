"""Optional source-ink fidelity pass for real scans and screentone manga.

Inspired by source luminance/edge retention used by open-source manga
colorizers.  It deliberately does NOT infer character identity or recolor
hair.  It treats the mc-v2 output as a color proposal, retains source ink
and attenuates excessive chroma without altering page dimensions.
"""
from __future__ import annotations

import cv2
import numpy as np


def restore_ink_fidelity(source_bgr: np.ndarray,
                         color_bgr: np.ndarray,
                         *, strength: float = 0.88,
                         chroma_factor: float = 0.79) -> np.ndarray:
    """Preserve original ink/screentone luminance in a colorized BGR page.

    Never brightens original dark strokes; untouched areas retain the model's
    tonal shading.  No segmentation/weights/network needed. Return a new image.
    """
    if source_bgr.shape != color_bgr.shape or source_bgr.ndim != 3 or source_bgr.shape[2] != 3:
        raise ValueError("source and colored image must have identical HxWx3 BGR shapes")
    if source_bgr.dtype != np.uint8 or color_bgr.dtype != np.uint8:
        raise ValueError("source and colored image must be uint8")
    strength = float(np.clip(strength, 0, 1))
    chroma_factor = float(np.clip(chroma_factor, 0, 1))
    if strength <= 0 and chroma_factor >= 1.0:
        return color_bgr.copy()
    gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lab = cv2.cvtColor(color_bgr.astype(np.float32) / 255.0,
                       cv2.COLOR_BGR2LAB)
    ink = np.power(np.clip((185.0 - gray) / 160.0, 0.0, 1.0), 1.3)
    src_l = gray * (100.0 / 255.0)
    dst_l = lab[..., 0]
    # Preserve all model-generated colored shading where source ink is absent.
    dst_l += (np.minimum(src_l, dst_l) - dst_l) * ink * strength
    lab[..., 1] *= chroma_factor * (1.0 - 0.40 * ink)
    lab[..., 2] *= chroma_factor * (1.0 - 0.40 * ink)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return np.clip(bgr * 255.0, 0, 255).astype(np.uint8)
