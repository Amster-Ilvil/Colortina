import os

import torch
from torchvision.transforms import ToTensor
import numpy as np
from .networks.models import Colorizer
from .denoising.denoiser import FFDNetDenoiser
from .utils.utils import resize_pad


def _normalize_state_dict(state):
    """Accept generator weights in any of the common release formats.

    Compatibility idea from Manga-Colorization-FJ (zip/pt weight compat):
    checkpoints in the wild come as raw state dicts (the .zip release),
    plain .pt/.pth state dicts, wrapped dicts ({'state_dict': ...},
    {'generator': ...}, {'model': ...}), and DataParallel-saved dicts
    whose keys carry a 'module.' prefix.  Normalize all of them to a
    plain state dict the model can load directly.
    """
    # Unwrap common container keys
    if isinstance(state, dict):
        for key in ('state_dict', 'generator', 'model', 'net'):
            inner = state.get(key)
            if isinstance(inner, dict) and inner and all(
                    hasattr(v, 'shape') for v in list(inner.values())[:3]):
                state = inner
                break
    # Strip DataParallel 'module.' prefixes
    if isinstance(state, dict) and any(k.startswith('module.') for k in state):
        state = {k[len('module.'):] if k.startswith('module.') else k: v
                 for k, v in state.items()}
    return state


class MangaColorizator:
    def __init__(self, device, generator_path='networks/generator.zip',
                 extractor_path='networks/extractor.pth',
                 denoiser_weights_dir='denoising/models/'):
        self.device = device
        self.colorizer = Colorizer().to(device)

        # Always deserialize to CPU first, then let load_state_dict() copy
        # into the already-on-device model. Loading straight to an MPS
        # device via map_location=device can fail outright on some
        # checkpoints (MPS has no float64 support, and some saved buffers
        # are float64) — CPU deserialization always works, and
        # load_state_dict's per-tensor copy_() handles the cross-device
        # move safely regardless of target device.
        generator_state = _normalize_state_dict(
            torch.load(generator_path, map_location='cpu', weights_only=False))
        self.colorizer.generator.load_state_dict(generator_state)


        # Load separate extractor weights only if available.
        # When the generator state dict already contains encoder.* keys
        # (which it does for the standard release), this is not needed.
        if extractor_path and os.path.isfile(extractor_path):
            extractor_state = _normalize_state_dict(
                torch.load(extractor_path, map_location='cpu', weights_only=False))
            self.colorizer.generator.encoder.load_state_dict(extractor_state)

        self.colorizer.eval()

        self.denoiser = FFDNetDenoiser(
            _device=device if isinstance(device, str) else str(device),
            _weights_dir=denoiser_weights_dir
        )

        self.current_image = None
        self.current_hint = None
        self.current_pad = None

    def set_image(self, image, size=576, apply_denoise=True, denoise_sigma=25,
                  transform=ToTensor()):
        if size % 32 != 0:
            raise RuntimeError("size is not divisible by 32")

        # Resize to model size FIRST, then denoise — denoising at the model
        # resolution instead of up-to-1200px avoids wasted FFDNet compute.
        image, self.current_pad = resize_pad(image, size)

        if apply_denoise and denoise_sigma > 0:
            # resize_pad returns (H, W, 1); the denoiser expects 2-D or (H, W, 3)
            den_in = image[:, :, 0] if (image.ndim == 3 and image.shape[2] == 1) else image
            image = self.denoiser.get_denoised_image(den_in, sigma=denoise_sigma,
                                                     max_edge=None)
            if image.ndim == 2:
                image = image[:, :, None]
            elif image.shape[2] > 1:
                image = image[:, :, :1]

        self.current_image = transform(image).unsqueeze(0).to(self.device)
        self.current_hint = torch.zeros(1, 4, self.current_image.shape[2],
                                        self.current_image.shape[3]).to(self.device)

    def update_hint(self, hint, mask):
        if isinstance(hint, np.ndarray):
            hint = hint.astype('float32')
            if hint.max() > 1.0:
                hint = hint / 255.0
            hint = (hint - 0.5) / 0.5
            hint = torch.FloatTensor(hint).permute(2, 0, 1).unsqueeze(0).to(self.device)

        if isinstance(mask, np.ndarray):
            mask = mask.astype('float32')
            if mask.max() > 1.0:
                mask = mask / 255.0
            mask = torch.FloatTensor(mask).unsqueeze(0).unsqueeze(0).to(self.device)

        # Match the original project: only masked pixels carry hint colour.
        # Without the multiply, unmasked pixels normalize to -1 and act like
        # a full-page black hint.
        self.current_hint = torch.cat([hint * mask, mask], 1)

    @torch.no_grad()
    def colorize(self):
        fake_color, _ = self.colorizer(torch.cat([self.current_image, self.current_hint], 1))
        result = fake_color[0].detach().cpu().permute(1, 2, 0).numpy()
        result = result * 0.5 + 0.5

        if self.current_pad[0] > 0:
            result = result[:-self.current_pad[0]]
        if self.current_pad[1] > 0:
            result = result[:, :-self.current_pad[1]]

        return result
