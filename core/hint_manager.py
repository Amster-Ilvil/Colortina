"""Unified automatic/manual colour hints with structured v4 metadata."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from core.hint_composer import HintComposer
from core.hint_spec import HintSpec
from core.region_map import RegionMap, build_region_map
from core.style_director import tiered_from_rgb

HintPoint = tuple[float, float, tuple[int, int, int], float]


@dataclass
class Hint:
    x_norm: float
    y_norm: float
    color: tuple[int, int, int]
    radius_norm: float = 0.004
    priority: int = 0
    region_id: int | None = None
    source: str = "auto_instance"
    semantic: str | None = None
    character_id: int | None = None
    confidence: float = 1.0
    strength: float = 1.0

    @classmethod
    def from_spec(cls, spec: HintSpec) -> "Hint":
        return cls(spec.x_norm, spec.y_norm, spec.rgb, spec.radius_norm,
                   int(spec.priority or 0), spec.region_id, spec.source,
                   spec.semantic, spec.character_id, spec.confidence,
                   spec.strength)

    def to_spec(self) -> HintSpec:
        return HintSpec(
            self.x_norm, self.y_norm, self.color, self.radius_norm,
            strength=self.strength, source=self.source,
            region_id=self.region_id, semantic=self.semantic,
            character_id=self.character_id, confidence=self.confidence,
            priority=self.priority)

    def as_point(self, radius_scale: float = 1.0) -> HintPoint:
        return (float(self.x_norm), float(self.y_norm), tuple(map(int, self.color)),
                max(0.0005, float(self.radius_norm) * float(radius_scale)))

    def to_dict(self) -> dict:
        return self.to_spec().to_dict()

    @classmethod
    def from_dict(cls, data: dict) -> "Hint":
        return cls.from_spec(HintSpec.from_dict(data))


@dataclass
class HintManager:
    auto_hints: list[Hint] = field(default_factory=list)
    manual_hints: list[Hint] = field(default_factory=list)
    _region_map: RegionMap | None = field(default=None, init=False, repr=False)
    _source_key: tuple | None = field(default=None, init=False, repr=False)

    @property
    def region_map(self) -> RegionMap | None:
        return self._region_map

    def bind_source_image(self, image_bgr: np.ndarray, *, line_low: int = 75,
                          gap_close: int = 4) -> RegionMap:
        key = (id(image_bgr), image_bgr.shape[:2], int(line_low), int(gap_close))
        if self._region_map is None or self._source_key != key:
            self._region_map = build_region_map(
                image_bgr, line_low=line_low, gap_close=gap_close)
            self._source_key = key
            for hint in self.auto_hints + self.manual_hints:
                hint.region_id = self._region_map.region_at_norm(
                    hint.x_norm, hint.y_norm) or None
        return self._region_map

    def set_auto_hints(self, points: list) -> None:
        hints: list[Hint] = []
        for value in points or []:
            spec = HintSpec.from_any(value, default_source="auto_instance")
            if self._region_map is not None:
                # The guided segmentation and the interactive RegionMap use
                # different thresholds/label numbering.  Composer overrides
                # must use the interactive map so manual strokes suppress the
                # correct enclosed model hints.
                spec.region_id = self._region_map.region_at_norm(
                    spec.x_norm, spec.y_norm) or None
            hints.append(Hint.from_spec(spec))
        self.auto_hints = hints

    def add_manual_hint(self, x_norm: float, y_norm: float,
                        color: tuple[int, int, int],
                        radius_norm: float = 0.015) -> bool:
        x_norm = float(np.clip(x_norm, 0.0, 1.0))
        y_norm = float(np.clip(y_norm, 0.0, 1.0))
        radius_norm = max(0.001, float(radius_norm))
        color = tuple(int(np.clip(v, 0, 255)) for v in color)
        region_id = (self._region_map.region_at_norm(x_norm, y_norm) or None
                     if self._region_map is not None else None)
        merge_distance = max(0.003, radius_norm * 0.65)
        for existing in reversed(self.manual_hints[-16:]):
            if existing.region_id != region_id or existing.color != color:
                continue
            dx, dy = existing.x_norm - x_norm, existing.y_norm - y_norm
            if dx * dx + dy * dy <= merge_distance * merge_distance:
                existing.x_norm = (existing.x_norm + x_norm) * 0.5
                existing.y_norm = (existing.y_norm + y_norm) * 0.5
                existing.radius_norm = max(existing.radius_norm, radius_norm)
                return False
        self.manual_hints.append(Hint(
            x_norm, y_norm, color, radius_norm=radius_norm, priority=100,
            region_id=region_id, source="manual", strength=1.00))
        self._thin_region_samples(region_id, color, max_samples=8)
        return True

    def _thin_region_samples(self, region_id: int | None,
                             color: tuple[int, int, int], max_samples: int) -> None:
        indices = [i for i, h in enumerate(self.manual_hints)
                   if h.region_id == region_id and h.color == color]
        if len(indices) <= max_samples:
            return
        keep_positions = set(np.linspace(0, len(indices) - 1,
                                         max_samples, dtype=int).tolist())
        remove = {idx for pos, idx in enumerate(indices) if pos not in keep_positions}
        self.manual_hints = [h for i, h in enumerate(self.manual_hints) if i not in remove]

    def clear_manual_hints(self) -> None:
        self.manual_hints = []

    def undo_last_manual(self) -> None:
        if self.manual_hints:
            self.manual_hints.pop()

    def merge_specs(self, *, image_bgr: np.ndarray | None = None,
                    style_descriptor=None, style_strength: float = 1.0,
                    manual_strength: float = 1.0) -> list[HintSpec]:
        auto = [h.to_spec() for h in self.auto_hints]
        manual = [h.to_spec() for h in self.manual_hints]
        # Empty auto runs do not need a full connected-region map. Build it
        # only when a real hint exists and composition actually needs regions.
        if image_bgr is not None and self._region_map is None and (auto or manual):
            self.bind_source_image(image_bgr)
        composed = HintComposer(self._region_map).compose(
            auto, manual, manual_strength=manual_strength)

        out: list[HintSpec] = []
        for spec in composed:
            if spec.source == "manual_region" and spec.region_id and image_bgr is not None:
                out.extend(self._region_tier_specs(
                    int(spec.region_id), spec, image_bgr,
                    style_descriptor=style_descriptor,
                    style_strength=style_strength,
                    manual_strength=manual_strength))
            else:
                out.append(spec)
        return out

    def merge(self, suppress_radius_norm: float = 0.02, *,
              image_bgr: np.ndarray | None = None,
              style_descriptor=None, style_strength: float = 1.0,
              manual_strength: float = 1.0) -> list[HintPoint]:
        # Legacy API used by older tests/callers.
        return [h.as_legacy_point() for h in self.merge_specs(
            image_bgr=image_bgr, style_descriptor=style_descriptor,
            style_strength=style_strength, manual_strength=manual_strength)]

    def _region_tier_specs(self, region_id: int, manual: HintSpec,
                           image_bgr: np.ndarray, *, style_descriptor=None,
                           style_strength: float = 1.0,
                           manual_strength: float = 1.0) -> list[HintSpec]:
        if self._region_map is None:
            return [manual]
        mask = self._region_map.labels == int(region_id)
        area = int(np.count_nonzero(mask))
        if area == 0:
            return [manual]
        tiers = tiered_from_rgb(manual.rgb, descriptor=style_descriptor,
                                strength=style_strength)
        h, w = mask.shape
        radius = max(0.002, manual.radius_norm * (0.55 + 1.25 * manual_strength))
        if area < 96:
            return [manual.clone(rgb=tiers.mid_rgb, radius_norm=radius)]
        gray = (image_bgr if image_bgr.ndim == 2 else
                cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY))
        distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        valid = mask & (distance >= 1.5) & (gray >= 45) & (gray <= 245)
        ys, xs = np.nonzero(valid)
        if ys.size == 0:
            return [manual.clone(rgb=tiers.mid_rgb, radius_norm=radius)]
        if ys.size > 24000:
            step = int(np.ceil(ys.size / 24000))
            ys, xs = ys[::step], xs[::step]
        values = gray[ys, xs].astype(np.float32)
        dist_values = distance[ys, xs].astype(np.float32)
        quantiles = [0.50, 0.80, 0.20]
        colors = [tiers.mid_rgb, tiers.highlight_rgb, tiers.shadow_rgb]
        out: list[HintSpec] = []
        used: set[int] = set()
        for q, color in zip(quantiles, colors):
            target = float(np.quantile(values, q))
            score = np.abs(values - target) + 8.0 / np.maximum(dist_values, 0.5)
            for idx in np.argsort(score)[:32]:
                idx = int(idx)
                if idx in used:
                    continue
                used.add(idx)
                out.append(manual.clone(
                    x_norm=float(xs[idx] / w), y_norm=float(ys[idx] / h),
                    rgb=color, radius_norm=radius,
                    strength=float(np.clip(manual_strength, 0.0, 1.0))))
                break
        return out or [manual]

    def preview_hints(self) -> list[Hint]:
        return [*self.auto_hints, *self.manual_hints]

    def to_dict(self) -> dict:
        return {
            "version": 4,
            "auto_hints": [h.to_dict() for h in self.auto_hints],
            "manual_hints": [h.to_dict() for h in self.manual_hints],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HintManager":
        return cls(
            auto_hints=[Hint.from_dict(item) for item in data.get("auto_hints", [])],
            manual_hints=[Hint.from_dict(item) for item in data.get("manual_hints", [])],
        )

    def reset(self) -> None:
        self.auto_hints = []
        self.manual_hints = []
