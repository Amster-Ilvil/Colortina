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


_esrgan_model = None
_esrgan_device = None


def _load_esrgan(weights_path: str):
    """Load anime6B into the vendored torch-only RRDBNet (cached)."""
    global _esrgan_model, _esrgan_device
    if _esrgan_model is not None:
        return _esrgan_model, _esrgan_device
    import torch
    from vendor.realesrgan_min.rrdbnet import RRDBNet

    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    state = state.get("params_ema") or state.get("params") or state
    # anime6B checkpoint: 6 RRDB blocks (vs. 23 for the general x4plus model)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6,
                    num_grow_ch=32, scale=4)
    model.load_state_dict(state, strict=True)
    model.eval()
    device = torch.device("cpu")
    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_built() and mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
    except Exception:
        pass
    model.to(device)
    _esrgan_model, _esrgan_device = model, device
    return model, device


def _upscale_realesrgan(image_bgr: np.ndarray, scale: int,
                        weights_path: str) -> np.ndarray:
    """Real-ESRGAN anime6B super-resolution, torch-only, tiled."""
    import torch

    model, device = _load_esrgan(weights_path)
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = img.shape[:2]
    tile, pad = 256, 8
    out = np.zeros((h * 4, w * 4, 3), np.float32)
    with torch.inference_mode():
        for y0 in range(0, h, tile):
            for x0 in range(0, w, tile):
                y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                py0, px0 = max(0, y0 - pad), max(0, x0 - pad)
                py1, px1 = min(h, y1 + pad), min(w, x1 + pad)
                patch = torch.from_numpy(
                    img[py0:py1, px0:px1].transpose(2, 0, 1)
                ).unsqueeze(0).to(device)
                sr = model(patch)[0].clamp_(0, 1).float().cpu().numpy()
                sr = sr.transpose(1, 2, 0)
                oy, ox = (y0 - py0) * 4, (x0 - px0) * 4
                out[y0 * 4:y1 * 4, x0 * 4:x1 * 4] = (
                    sr[oy:oy + (y1 - y0) * 4, ox:ox + (x1 - x0) * 4])
    out_bgr = cv2.cvtColor(
        np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR)
    if scale != 4:
        out_bgr = cv2.resize(out_bgr, (w * scale, h * scale),
                             interpolation=cv2.INTER_AREA)
    return out_bgr
