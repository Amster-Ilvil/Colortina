"""Structured colour-hint metadata used throughout the v4 pipeline.

The model ultimately consumes only an RGB hint image and a mask, but decisions
such as priority, density, confidence gates and retry degradation must happen
*before* that information is discarded.  ``HintSpec`` therefore remains the
canonical representation until rasterization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import numpy as np

HintSource = Literal[
    "manual",          # local model hint; never expands to an entire region
    "manual_region",   # explicit whole-region hint (advanced/legacy only)
    "eyedropper_hint", # eyedropper-sampled model hint from a colored result
    "character_identity",
    "scene_palette",
    "auto_instance",
    "style_only",
]

_SOURCE_PRIORITY: dict[str, int] = {
    "style_only": 0,
    "scene_palette": 20,
    "auto_instance": 40,
    "character_identity": 70,
    "manual": 100,
    "manual_region": 100,
    "eyedropper_hint": 100,
}

_SOURCE_DEFAULT_STRENGTH: dict[str, float] = {
    "style_only": 0.12,
    "scene_palette": 0.28,
    "auto_instance": 0.32,
    "character_identity": 0.58,
    "manual": 0.72,
    "manual_region": 1.00,
    "eyedropper_hint": 0.88,
}


@dataclass
class HintSpec:
    x_norm: float
    y_norm: float
    rgb: tuple[int, int, int]
    radius_norm: float = 0.004
    strength: float = 1.0
    source: HintSource = "auto_instance"
    region_id: int | None = None
    semantic: str | None = None
    character_id: int | None = None
    confidence: float = 1.0
    priority: int | None = None

    def __post_init__(self) -> None:
        self.x_norm = float(np.clip(self.x_norm, 0.0, 1.0))
        self.y_norm = float(np.clip(self.y_norm, 0.0, 1.0))
        self.rgb = tuple(int(np.clip(v, 0, 255)) for v in self.rgb)
        self.radius_norm = max(0.0005, float(self.radius_norm))
        self.strength = float(np.clip(self.strength, 0.0, 1.0))
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))
        if self.priority is None:
            self.priority = _SOURCE_PRIORITY.get(str(self.source), 0)

    @property
    def effective_strength(self) -> float:
        return float(np.clip(self.strength * self.confidence, 0.0, 1.0))

    def as_legacy_point(self) -> tuple[float, float, tuple[int, int, int], float]:
        return (self.x_norm, self.y_norm, self.rgb, self.radius_norm)

    def clone(self, **changes: Any) -> "HintSpec":
        return replace(self, **changes)

    def remap_to_crop(self, crop_x: int, crop_y: int, crop_w: int, crop_h: int,
                      page_w: int, page_h: int, *, canvas_w: int | None = None,
                      canvas_h: int | None = None) -> "HintSpec | None":
        if crop_w <= 0 or crop_h <= 0:
            return None
        canvas_w = int(canvas_w or crop_w)
        canvas_h = int(canvas_h or crop_h)
        px = self.x_norm * page_w
        py = self.y_norm * page_h
        if not (crop_x <= px < crop_x + crop_w and crop_y <= py < crop_y + crop_h):
            return None
        return self.clone(
            x_norm=(px - crop_x) / canvas_w,
            y_norm=(py - crop_y) / canvas_h,
            radius_norm=self.radius_norm * page_w / max(1, canvas_w),
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["rgb"] = list(self.rgb)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "HintSpec":
        payload = dict(data)
        if "color" in payload and "rgb" not in payload:
            payload["rgb"] = payload.pop("color")
        payload["rgb"] = tuple(payload.get("rgb", (128, 128, 128)))
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in payload.items() if k in known})

    @classmethod
    def from_any(cls, value: Any, *, default_source: HintSource = "auto_instance",
                 default_strength: float | None = None) -> "HintSpec":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            payload = dict(value)
            payload.setdefault("source", default_source)
            if default_strength is not None:
                payload.setdefault("strength", default_strength)
            return cls.from_dict(payload)
        # Backwards compatible current Hint dataclass.
        if hasattr(value, "x_norm") and hasattr(value, "y_norm"):
            rgb = getattr(value, "rgb", getattr(value, "color", (128, 128, 128)))
            source = getattr(value, "source", default_source)
            strength = getattr(value, "strength", default_strength)
            if strength is None:
                strength = _SOURCE_DEFAULT_STRENGTH.get(str(source), 1.0)
            return cls(
                x_norm=getattr(value, "x_norm"),
                y_norm=getattr(value, "y_norm"),
                rgb=tuple(rgb),
                radius_norm=getattr(value, "radius_norm", 0.004),
                strength=strength,
                source=source,
                region_id=getattr(value, "region_id", None),
                semantic=getattr(value, "semantic", None),
                character_id=getattr(value, "character_id", None),
                confidence=getattr(value, "confidence", 1.0),
                priority=getattr(value, "priority", None),
            )
        if isinstance(value, (tuple, list)):
            if len(value) == 4:
                x, y, rgb, radius = value
            elif len(value) == 3:
                x, y, rgb = value
                radius = 0.004
            else:
                raise ValueError(f"unsupported hint tuple length: {len(value)}")
            strength = (default_strength if default_strength is not None
                        else _SOURCE_DEFAULT_STRENGTH.get(default_source, 1.0))
            return cls(float(x), float(y), tuple(rgb), float(radius),
                       strength=strength, source=default_source)
        raise TypeError(f"unsupported hint type: {type(value).__name__}")


def source_default_strength(source: str) -> float:
    return _SOURCE_DEFAULT_STRENGTH.get(source, 1.0)
