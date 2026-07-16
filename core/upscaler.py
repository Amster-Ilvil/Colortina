"""Final upscale pass for the Ultra quality preset.

``core.model_downloader.ensure_esrgan_downloaded`` fetches Real-ESRGAN
anime weights but nothing ever called it and ``Config`` had no
``ESRGAN_MODEL_PATH``/``ESRGAN_MODEL_URL`` to point it at — a dangling
feature. This module completes it, and degrades gracefully:

  1. If the ``realesrgan`` package is installed AND the weights file
     exists, run real Real-ESRGAN super-resolution.
  2. Otherwise, fall back to a high-quality Lanczos resize — still a
     correct (if less detailed) 4x upscale, and never fails or blocks
     on a missing optional dependency / model download.
"""

from __future__ import annotations

import os

import cv2
import numpy as np


def upscale(image_bgr: np.ndarray, scale: int = 4,
           weights_path: str | None = None) -> np.ndarray:
    """Upscale `image_bgr` by `scale`x. Never raises."""
    if weights_path and os.path.exists(weights_path):
        try:
            return _upscale_realesrgan(image_bgr, scale, weights_path)
        except Exception as exc:  # noqa: BLE001 — any failure falls back
            print(f"[upscaler] Real-ESRGAN unavailable ({exc}); "
                 f"falling back to Lanczos upscale")
    h, w = image_bgr.shape[:2]
    return cv2.resize(image_bgr, (w * scale, h * scale),
                      interpolation=cv2.INTER_LANCZOS4)


def _upscale_realesrgan(image_bgr: np.ndarray, scale: int,
                        weights_path: str) -> np.ndarray:
    """Real-ESRGAN anime-tuned super-resolution (optional dependency)."""
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    # anime6B checkpoint: 6 RRDB blocks (vs. 23 for the general x4plus model)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6,
                    num_grow_ch=32, scale=scale)
    upsampler = RealESRGANer(
        scale=scale, model_path=weights_path, model=model,
        tile=400, tile_pad=10, pre_pad=0, half=True)
    out, _ = upsampler.enhance(image_bgr, outscale=scale)
    return out
