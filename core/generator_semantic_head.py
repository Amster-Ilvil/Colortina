"""Lightweight semantic head for manga-colorization-v2 generator features.

The stock ``generator.zip`` stays byte-for-byte untouched.  This module attaches
an independent FPN-style decoder to the pretrained monochrome encoder pyramid:

    generator encoder x1/x2/x3/x4 -> semantic head -> per-pixel class logits

Only this head is trained for the first production stage.  That means a bad or
missing semantic checkpoint can never damage the normal mc-v2 colour output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F


CHECKPOINT_FORMAT = "colortina-generator-semantic-head"
CHECKPOINT_VERSION = 1
DEFAULT_FEATURE_CHANNELS = (64, 256, 512, 1024)
DEFAULT_CLASSES = (
    "background",
    "hair",
    "iris",
    "eye",
    "eyebrow",
    "skin",
    "clothes",
    "glasses",
    "face_shadow",
    "other",
)


class _ConvGNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        padding = kernel_size // 2
        # GroupNorm is stable for the batch-size=1 training regime used by
        # variable-aspect manga pages and works on CPU/CUDA/MPS.
        groups = 8
        while out_channels % groups and groups > 1:
            groups //= 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )


class GeneratorSemanticHead(nn.Module):
    """Small FPN decoder over mc-v2's frozen SEResNeXt feature pyramid.

    The head deliberately consumes *only* the generator's monochrome structure
    encoder.  It has no dependency on hand-written HairMask heuristics and does
    not modify any generator parameter.
    """

    def __init__(
        self,
        class_names: Sequence[str] = DEFAULT_CLASSES,
        feature_channels: Sequence[int] = DEFAULT_FEATURE_CHANNELS,
        fpn_width: int = 64,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.class_names = tuple(str(v) for v in class_names)
        self.feature_channels = tuple(int(v) for v in feature_channels)
        self.fpn_width = int(fpn_width)
        if len(self.feature_channels) != 4:
            raise ValueError("GeneratorSemanticHead expects exactly four encoder stages")
        if len(self.class_names) < 2:
            raise ValueError("at least two semantic classes are required")

        self.lateral = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, self.fpn_width, 1, bias=False),
                nn.GroupNorm(8 if self.fpn_width % 8 == 0 else 1, self.fpn_width),
                nn.GELU(),
            )
            for ch in self.feature_channels
        ])
        # Concatenate all scales rather than only summing them.  This gives tiny
        # classes such as iris/glasses access to shallow detail while hair and
        # background retain deep semantic context.
        self.fuse = nn.Sequential(
            _ConvGNAct(self.fpn_width * 4, self.fpn_width, 3),
            nn.Dropout2d(float(max(0.0, min(0.5, dropout)))),
            _ConvGNAct(self.fpn_width, self.fpn_width, 3),
        )
        self.classifier = nn.Conv2d(self.fpn_width, len(self.class_names), 1)

    def forward(self, features: Sequence[torch.Tensor], output_size=None) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError(f"expected four feature maps, got {len(features)}")
        target_hw = tuple(int(v) for v in features[0].shape[-2:])
        rows = []
        for i, (feature, lateral) in enumerate(zip(features, self.lateral)):
            if feature.ndim != 4:
                raise ValueError(f"feature {i} must be BCHW")
            expected = self.feature_channels[i]
            if int(feature.shape[1]) != expected:
                raise ValueError(
                    f"feature {i} channel mismatch: expected {expected}, got {feature.shape[1]}")
            x = lateral(feature)
            if tuple(x.shape[-2:]) != target_hw:
                x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
            rows.append(x)
        logits = self.classifier(self.fuse(torch.cat(rows, dim=1)))
        if output_size is not None:
            output_size = tuple(int(v) for v in output_size)
            if tuple(logits.shape[-2:]) != output_size:
                logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits

    def metadata(self) -> dict:
        return {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "class_names": list(self.class_names),
            "feature_channels": list(self.feature_channels),
            "fpn_width": int(self.fpn_width),
        }


@dataclass(frozen=True)
class SemanticHeadCheckpointInfo:
    path: str
    class_names: tuple[str, ...]
    feature_channels: tuple[int, ...]
    fpn_width: int
    version: int


def save_semantic_head_checkpoint(
    path: str | Path,
    head: GeneratorSemanticHead,
    *,
    training: dict | None = None,
    metrics: dict | None = None,
    inference: dict | None = None,
) -> str:
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = dict(head.metadata())
    payload["state_dict"] = head.state_dict()
    payload["training"] = dict(training or {})
    payload["metrics"] = dict(metrics or {})
    payload["inference"] = dict(inference or {})
    torch.save(payload, path)
    return path


def _load_payload(path: str | Path, map_location="cpu") -> dict:
    payload = torch.load(str(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError("semantic head checkpoint must contain a dictionary")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise RuntimeError("not a Colortina generator semantic-head checkpoint")
    version = int(payload.get("version", 0))
    if version != CHECKPOINT_VERSION:
        raise RuntimeError(f"unsupported semantic-head checkpoint version: {version}")
    return payload


def inspect_semantic_head_checkpoint(path: str | Path) -> SemanticHeadCheckpointInfo:
    payload = _load_payload(path, map_location="cpu")
    return SemanticHeadCheckpointInfo(
        path=str(path),
        class_names=tuple(payload.get("class_names") or ()),
        feature_channels=tuple(int(v) for v in payload.get("feature_channels") or ()),
        fpn_width=int(payload.get("fpn_width", 64)),
        version=int(payload.get("version", 0)),
    )


def load_semantic_head_checkpoint(
    path: str | Path,
    *,
    device="cpu",
    eval_mode: bool = True,
) -> tuple[GeneratorSemanticHead, dict]:
    payload = _load_payload(path, map_location="cpu")
    head = GeneratorSemanticHead(
        class_names=tuple(payload["class_names"]),
        feature_channels=tuple(int(v) for v in payload["feature_channels"]),
        fpn_width=int(payload.get("fpn_width", 64)),
    )
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("semantic head checkpoint is missing state_dict")
    missing, unexpected = head.load_state_dict(state, strict=True)
    if missing or unexpected:  # strict=True normally raises before here
        raise RuntimeError(f"semantic head mismatch: missing={missing}, unexpected={unexpected}")
    head.to(device)
    if eval_mode:
        head.eval()
    return head, payload
