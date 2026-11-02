import os

import torch
from torchvision.transforms import ToTensor
import numpy as np
import cv2
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
                 denoiser_weights_dir='denoising/models/',
                 region_adapter_path=None, denoiser_device=None,
                 tunnel2_adapter_path=None):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.denoiser_device = (torch.device(denoiser_device)
                                if isinstance(denoiser_device, str)
                                else (denoiser_device or self.device))
        # Build and load entirely on CPU first.  Moving the initialized model to
        # MPS only after all checkpoints are loaded avoids macOS/torch combinations
        # that fail while copying mixed checkpoint buffers into an already-MPS model.
        self.colorizer = Colorizer()

        # Always deserialize to CPU. Some checkpoints contain buffers whose
        # original dtype cannot be materialized directly on MPS.
        generator_state = _normalize_state_dict(
            torch.load(generator_path, map_location='cpu', weights_only=True))
        self.colorizer.generator.load_state_dict(generator_state, strict=True)

        # Optional adapters are a separate checkpoint, so stock weights remain
        # byte-compatible with the upstream generator.zip.
        if region_adapter_path and os.path.isfile(region_adapter_path):
            # ``enable_region_adapters`` creates modules lazily after the stock
            # checkpoint has loaded. Because the generator was already moved to
            # CUDA/MPS above, newly created modules must explicitly follow the
            # generator device; otherwise the first controlled inference fails
            # with a CPU-vs-device tensor mismatch.
            adapters = self.colorizer.generator.enable_region_adapters()
            adapter_state = _normalize_state_dict(
                torch.load(region_adapter_path, map_location='cpu',
                           weights_only=True))
            prefix = 'region_adapters.'
            if isinstance(adapter_state, dict) and any(
                    str(k).startswith(prefix) for k in adapter_state):
                adapter_state = {
                    str(k)[len(prefix):]: v for k, v in adapter_state.items()
                    if str(k).startswith(prefix)
                }
            adapters.load_state_dict(adapter_state, strict=True)
            model_parameter = next(self.colorizer.generator.parameters())
            adapters.to(device=model_parameter.device, dtype=model_parameter.dtype)
            adapters.eval()

        # Load separate extractor weights only if available.
        # When the generator state dict already contains encoder.* keys
        # (which it does for the standard release), this is not needed.
        if extractor_path and os.path.isfile(extractor_path):
            extractor_state = _normalize_state_dict(
                torch.load(extractor_path, map_location='cpu', weights_only=True))
            self.colorizer.generator.encoder.load_state_dict(extractor_state)

        # This adapter is deliberately kept outside Generator so loading it
        # cannot add keys to or mutate the stock 645-key checkpoint contract.
        # Bad/missing optional adapter files fail closed to the direct residual
        # path instead of breaking ordinary colorization.
        self.tunnel2_semantic_adapter = None
        self.tunnel2_semantic_adapter_error = None
        self.tunnel2_semantic_adapter_path = str(tunnel2_adapter_path or "")
        if self.tunnel2_semantic_adapter_path and os.path.isfile(
                self.tunnel2_semantic_adapter_path):
            try:
                from core.generator_feature_adapter import (
                    load_tunnel2_adapter_checkpoint)
                adapter, _payload = load_tunnel2_adapter_checkpoint(
                    self.tunnel2_semantic_adapter_path, device=self.device,
                    eval_mode=True)
                self.tunnel2_semantic_adapter = adapter
            except Exception as exc:  # optional control must fail closed
                self.tunnel2_semantic_adapter_error = (
                    f"{type(exc).__name__}: {exc}")

        self.colorizer.to(self.device)
        self.colorizer.eval()

        self.denoiser = FFDNetDenoiser(
            _device=str(self.denoiser_device),
            _weights_dir=denoiser_weights_dir
        )

        self.current_image = None
        self.current_hint = None
        self.current_pad = None
        self._control_context = None

    def prepare_control_context(self):
        """Cache one encoder/base-decoder pass for semantic consumers."""
        if self.current_image is None or self.current_hint is None:
            raise RuntimeError("set_image must be called before preparing control context")
        if self._control_context is None:
            base = torch.cat([self.current_image, self.current_hint], 1)
            self._control_context = self.colorizer.generator.prepare_control_context(base)
        return self._control_context

    def _region_mask_tensor(self, region_mask):
        if region_mask is None or self.current_image is None:
            return None
        mask = np.asarray(region_mask)
        if mask.ndim == 3:
            mask = np.max(mask, axis=2)
        mask = (mask > 0).astype('float32')
        model_h, model_w = self.current_image.shape[-2:]
        pad_h, pad_w = self.current_pad or (0, 0)
        valid_h = max(1, model_h - int(pad_h))
        valid_w = max(1, model_w - int(pad_w))
        mask = cv2.resize(mask, (valid_w, valid_h), interpolation=cv2.INTER_AREA)
        if pad_h or pad_w:
            mask = np.pad(mask, ((0, int(pad_h)), (0, int(pad_w))),
                          mode='constant', constant_values=0.0)
        return torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(
            device=self.device, dtype=self.current_image.dtype)

    def _controlled_inputs(self, region_control):
        base = torch.cat([self.current_image, self.current_hint], 1)
        if not region_control or not bool(region_control.get('enabled', False)):
            return base, None, None
        mask = self._region_mask_tensor(region_control.get('mask'))
        if mask is None or not bool(torch.any(mask > 0)):
            return base, None, None
        rgb = region_control.get('target_rgb', region_control.get('rgb', (28, 28, 32)))
        hint_rgb = region_control.get('generator_hint_rgb') or rgb
        target = torch.tensor([float(rgb[0]), float(rgb[1]), float(rgb[2])],
                              device=self.device, dtype=self.current_image.dtype
                              ).view(1, 3, 1, 1) / 127.5 - 1.0
        hint_target = torch.tensor(
            [float(hint_rgb[0]), float(hint_rgb[1]), float(hint_rgb[2])],
            device=self.device, dtype=self.current_image.dtype
        ).view(1, 3, 1, 1) / 127.5 - 1.0
        opacity = float(np.clip(region_control.get('hint_opacity', 0.50), 0.0, 1.0))
        effective = mask * opacity
        old_rgb = self.current_hint[:, :3]
        old_alpha = self.current_hint[:, 3:4]
        controlled_rgb = old_rgb * (1.0 - effective) + hint_target * effective
        controlled_alpha = torch.maximum(old_alpha, effective)
        controlled_hint = torch.cat([controlled_rgb, controlled_alpha], 1)

        controlled_image = self.current_image
        # Dark, near-achromatic targets (typical commanded black hair) benefit
        # from a source-side bias: keep line art untouched but pull the bright
        # interior of the selected sketch toward a mid gray so the generator is
        # less tempted to hallucinate orange / purple hair before the final lock.
        bias_strength = float(np.clip(region_control.get('source_bias_strength', 0.0), 0.0, 1.0))
        if bias_strength > 0.0:
            tone = float(np.clip(region_control.get('source_bias_tone', 170), 0.0, 255.0)) / 255.0
            # Only brighter-than-target pixels are pulled down. Dark line art is
            # effectively preserved even before the later deterministic restore.
            headroom = torch.clamp(self.current_image - tone, min=0.0)
            denom = max(1e-4, 1.0 - tone)
            emphasis = torch.clamp(headroom / denom, 0.0, 1.0)
            image_alpha = mask * bias_strength * emphasis
            controlled_image = self.current_image * (1.0 - image_alpha) + tone * image_alpha

        controlled = torch.cat([controlled_image, controlled_hint], 1)
        target_map = target.expand(-1, -1, mask.shape[-2], mask.shape[-1])
        return base, controlled, (mask, target_map)

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
        self._control_context = None

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
        self._control_context = None

    @torch.no_grad()
    def extract_region_embeddings(self, image, labels, *, size=576):
        """Pool the generator's pretrained encoder pyramid over region ids.

        ``image`` is RGB/gray numpy input and ``labels`` is a full-resolution
        int32 region map. Returned vectors are L2-normalized numpy arrays keyed
        by region id. No checkpoint parameters are added or changed.
        """
        labels = np.asarray(labels, dtype=np.int32)
        if labels.ndim != 2:
            raise ValueError("labels must be a 2-D region map")
        self.set_image(image, size=int(size), apply_denoise=False, denoise_sigma=0)
        features = self.colorizer.generator.extract_structure_features(self.current_image)
        model_h, model_w = self.current_image.shape[-2:]
        pad_h, pad_w = self.current_pad or (0, 0)
        valid_h = max(1, model_h - int(pad_h))
        valid_w = max(1, model_w - int(pad_w))
        model_labels = cv2.resize(labels.astype(np.float32), (valid_w, valid_h),
                                  interpolation=cv2.INTER_NEAREST).astype(np.int32)
        if pad_h or pad_w:
            model_labels = np.pad(model_labels, ((0, int(pad_h)), (0, int(pad_w))),
                                  mode='constant', constant_values=0)

        ids = [int(v) for v in np.unique(labels).tolist() if int(v) > 0]
        chunks = {rid: [] for rid in ids}
        for fmap in features:
            _b, _c, fh, fw = fmap.shape
            scaled = cv2.resize(model_labels.astype(np.float32), (int(fw), int(fh)),
                                interpolation=cv2.INTER_NEAREST).astype(np.int32)
            for rid in ids:
                yy, xx = np.where(scaled == rid)
                if yy.size == 0:
                    # Preserve vector dimensionality when a tiny region vanishes
                    # at a deep stage.
                    pooled = torch.zeros((_c,), device=fmap.device, dtype=fmap.dtype)
                else:
                    mask = torch.from_numpy((scaled == rid)).to(device=fmap.device)
                    values = fmap[0, :, mask]
                    pooled = values.mean(dim=1) if values.numel() else torch.zeros(
                        (_c,), device=fmap.device, dtype=fmap.dtype)
                # Channel-wise normalization makes shallow/deep stages comparable
                # before concatenation and suppresses absolute activation scale.
                pooled = pooled.float()
                pooled = (pooled - pooled.mean()) / pooled.std().clamp_min(1e-5)
                chunks[rid].append(pooled.cpu().numpy())

        out = {}
        for rid, parts in chunks.items():
            if not parts:
                continue
            vec = np.concatenate(parts).astype(np.float32, copy=False)
            norm = float(np.linalg.norm(vec))
            if norm > 1e-8:
                vec = vec / norm
            out[rid] = vec
        return out

    @torch.no_grad()
    def colorize(self, region_control=None):
        base, controlled, tensors = self._controlled_inputs(region_control)
        if controlled is None or tensors is None:
            fake_color, _ = self.colorizer(base)
        else:
            mask, target_map = tensors
            cfg = dict(region_control or {})
            control_mode = str(cfg.get('generator_control_mode', 'direct_tunnel2')).strip().lower()
            if control_mode in {'direct_tunnel2', 'direct', 'tunnel2'}:
                # The control is now part of Generator.forward itself.  The
                # command sketch carries the normal region-control hint/source
                # bias, while x1..x4 are still computed once from the stock
                # grayscale input.
                context = self.prepare_control_context()
                semantic_control = {
                    'mask': mask,
                    'target_rgb': target_map,
                    'command_sketch': controlled,
                    'strength': float(cfg.get('shallow_strength', 0.94)),
                    'feature_gain': float(cfg.get('feature_gain', 1.75)),
                    'mask_feather_px': int(cfg.get('mask_feather_px', 3)),
                    'hint_opacity': float(cfg.get('hint_opacity', 0.50)),
                    'strict_output_mask': bool(cfg.get('strict_output_mask', True)),
                    'cached_features': context,
                }
                if self.tunnel2_semantic_adapter is not None and bool(
                        cfg.get('use_adapter', False)):
                    semantic_control['tunnel2_adapter'] = self.tunnel2_semantic_adapter
                fake_color, _ = self.colorizer.generator(
                    base, semantic_control=semantic_control)
            elif control_mode in {'dual_route', 'legacy_dual_route'}:
                fake_color, _ = self.colorizer.generator.forward_controlled(
                    base, controlled, mask, target_rgb=target_map,
                    deep_strength=float(cfg.get('deep_strength', 0.38)),
                    middle_strength=float(cfg.get('middle_strength', 0.74)),
                    shallow_strength=float(cfg.get('shallow_strength', 0.94)),
                    feature_gain=float(cfg.get('feature_gain', 1.75)),
                    deep_shrink_px=int(cfg.get('deep_shrink_px', 8)),
                    middle_shrink_px=int(cfg.get('middle_shrink_px', 3)),
                    shallow_shrink_px=int(cfg.get('shallow_shrink_px', 0)),
                    mask_feather_px=int(cfg.get('mask_feather_px', 3)),
                    use_adapter=bool(cfg.get('use_adapter', False)),
                )
            else:
                raise ValueError(f'unsupported generator_control_mode: {control_mode}')
        result = fake_color[0].detach().cpu().permute(1, 2, 0).numpy()
        result = result * 0.5 + 0.5

        if self.current_pad[0] > 0:
            result = result[:-self.current_pad[0]]
        if self.current_pad[1] > 0:
            result = result[:, :-self.current_pad[1]]

        return result
