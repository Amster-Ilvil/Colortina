"""Read-only adapter exposing the stock mc-v2 generator encoder pyramid."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


TUNNEL2_ADAPTER_FORMAT = "colortina-tunnel2-semantic-color-adapter"
TUNNEL2_ADAPTER_VERSION = 1

from core.generator_semantic_head import DEFAULT_FEATURE_CHANNELS


@dataclass(frozen=True)
class GeneratorFeatureSpec:
    channels: tuple[int, ...]
    spatial_shapes: tuple[tuple[int, int], ...]


class GeneratorFeatureAdapter(nn.Module):
    """Validate and expose generator structure features without adding weights.

    The adapter owns no trainable parameter and does not register the generator
    as a child module (``object.__setattr__`` is intentional), so using it can
    never alter the stock ``generator.zip`` state_dict.
    """

    def __init__(self, generator, expected_channels: Sequence[int] = DEFAULT_FEATURE_CHANNELS):
        super().__init__()
        object.__setattr__(self, "_generator_ref", generator)
        self.expected_channels = tuple(int(v) for v in expected_channels)

    @property
    def generator(self):
        return object.__getattribute__(self, "_generator_ref")

    def forward(self, sketch: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = tuple(self.generator.extract_structure_features(sketch))
        if len(features) != len(self.expected_channels):
            raise RuntimeError(
                f"generator feature pyramid changed: expected {len(self.expected_channels)} stages, "
                f"got {len(features)}")
        channels = tuple(int(f.shape[1]) for f in features)
        if channels != self.expected_channels:
            raise RuntimeError(
                f"generator semantic feature contract changed: expected {self.expected_channels}, got {channels}")
        return features

    def inspect(self, sketch: torch.Tensor) -> GeneratorFeatureSpec:
        with torch.inference_mode():
            features = self.forward(sketch)
        return GeneratorFeatureSpec(
            channels=tuple(int(f.shape[1]) for f in features),
            spatial_shapes=tuple(tuple(int(v) for v in f.shape[-2:]) for f in features),
        )


class Tunnel2SemanticColorAdapter(nn.Module):
    """Small trainable delta head for the frozen generator ``tunnel2``.

    The module is intentionally external to ``Generator``.  It therefore adds
    no keys to the stock generator state_dict and can be loaded independently.
    Its output is an inward-mask-gated feature delta; it cannot write outside
    the supplied semantic region even in native mode.
    """

    def __init__(self, feature_channels: int = 64, hidden_channels: int = 32):
        super().__init__()
        self.feature_channels = int(feature_channels)
        self.hidden_channels = int(hidden_channels)
        if self.feature_channels <= 0 or self.hidden_channels <= 0:
            raise ValueError("adapter channel counts must be positive")
        # feature + mask + masked target RGB + optional hint residual
        in_channels = self.feature_channels * 2 + 4
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_channels, 1, bias=False),
            nn.GroupNorm(8 if self.hidden_channels % 8 == 0 else 1, self.hidden_channels),
            nn.GELU(),
        )
        self.depthwise = nn.Conv2d(
            self.hidden_channels, self.hidden_channels, 3, padding=1,
            groups=self.hidden_channels, bias=False)
        self.activation = nn.GELU()
        self.output = nn.Conv2d(self.hidden_channels, self.feature_channels, 1)
        # A fresh adapter is an exact no-op until trained or loaded.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _mask_tensor(mask, feature: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask, device=feature.device, dtype=feature.dtype)
        else:
            mask = mask.to(device=feature.device, dtype=feature.dtype)
        if mask.ndim == 2:
            mask = mask[None, None]
        elif mask.ndim == 3:
            mask = mask[:, None]
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError("adapter mask must be [H,W], [B,H,W] or [B,1,H,W]")
        if mask.shape[0] == 1 and feature.shape[0] > 1:
            mask = mask.expand(feature.shape[0], -1, -1, -1)
        if mask.shape[0] != feature.shape[0]:
            raise ValueError("adapter mask batch size does not match feature")
        return F.interpolate(mask, size=feature.shape[-2:], mode="bilinear",
                             align_corners=False).clamp(0.0, 1.0)

    @staticmethod
    def _target_tensor(target_rgb, feature: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(target_rgb):
            target_rgb = torch.as_tensor(target_rgb, device=feature.device,
                                         dtype=feature.dtype)
        else:
            target_rgb = target_rgb.to(device=feature.device, dtype=feature.dtype)
        if target_rgb.ndim == 1 and target_rgb.numel() == 3:
            target_rgb = target_rgb.view(1, 3, 1, 1)
        elif target_rgb.ndim == 2 and target_rgb.shape[-1] == 3:
            target_rgb = target_rgb.view(target_rgb.shape[0], 3, 1, 1)
        elif target_rgb.ndim != 4 or target_rgb.shape[1] != 3:
            raise ValueError("adapter target_rgb must be [3], [B,3] or [B,3,H,W]")
        if float(target_rgb.detach().abs().max()) > 1.5:
            target_rgb = target_rgb / 127.5 - 1.0
        if target_rgb.shape[0] == 1 and feature.shape[0] > 1:
            target_rgb = target_rgb.expand(feature.shape[0], -1, -1, -1)
        if target_rgb.shape[0] != feature.shape[0]:
            raise ValueError("adapter target_rgb batch size does not match feature")
        return F.interpolate(target_rgb, size=feature.shape[-2:], mode="bilinear",
                             align_corners=False)

    def forward(self, feature: torch.Tensor, mask, target_rgb,
                hint_residual: torch.Tensor | None = None,
                strength: float = 1.0) -> torch.Tensor:
        if feature.ndim != 4 or int(feature.shape[1]) != self.feature_channels:
            raise ValueError(
                f"adapter feature must be BCHW with {self.feature_channels} channels")
        mask_t = self._mask_tensor(mask, feature)
        target_t = self._target_tensor(target_rgb, feature)
        if hint_residual is None:
            residual = torch.zeros_like(feature)
        else:
            residual = hint_residual.to(device=feature.device, dtype=feature.dtype)
            if residual.shape != feature.shape:
                raise ValueError("adapter hint_residual must match feature shape")
        conditioned = torch.cat([
            feature, mask_t, target_t * mask_t, residual,
        ], dim=1)
        delta = self.output(self.activation(self.depthwise(self.project(conditioned))))
        return delta * mask_t * float(max(0.0, min(2.0, strength)))

    def metadata(self) -> dict:
        return {
            "format": TUNNEL2_ADAPTER_FORMAT,
            "version": TUNNEL2_ADAPTER_VERSION,
            "feature_channels": self.feature_channels,
            "hidden_channels": self.hidden_channels,
        }


def save_tunnel2_adapter_checkpoint(path, adapter: Tunnel2SemanticColorAdapter,
                                    *, training: dict | None = None,
                                    metrics: dict | None = None) -> str:
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = dict(adapter.metadata())
    payload["state_dict"] = adapter.state_dict()
    payload["training"] = dict(training or {})
    payload["metrics"] = dict(metrics or {})
    torch.save(payload, path)
    return path


def load_tunnel2_adapter_checkpoint(path, *, device="cpu", eval_mode=True):
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != TUNNEL2_ADAPTER_FORMAT:
        raise RuntimeError("not a Colortina tunnel2 semantic adapter checkpoint")
    if int(payload.get("version", 0)) != TUNNEL2_ADAPTER_VERSION:
        raise RuntimeError("unsupported tunnel2 adapter checkpoint version")
    adapter = Tunnel2SemanticColorAdapter(
        feature_channels=int(payload.get("feature_channels", 64)),
        hidden_channels=int(payload.get("hidden_channels", 32)),
    )
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("tunnel2 adapter checkpoint is missing state_dict")
    adapter.load_state_dict(state, strict=True)
    adapter.to(device)
    if eval_mode:
        adapter.eval()
    return adapter, payload
