"""Priority, density and safety rules for structured hints."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from core.hint_spec import HintSpec


class HintComposer:
    """Compose automatic and manual hints without losing metadata.

    Rules:
    - ordinary manual brush hints stay local and only suppress nearby auto hints;
    - only explicit ``manual_region`` hints override an enclosed region;
    - no ``style_only`` absolute-colour point is emitted;
    - per-region and page-wide density limits are source aware;
    - low-confidence identity hints are dropped rather than force-applied.
    """

    MAX_PAGE_AUTO = 120
    REGION_LIMITS = ((0.006, 1), (0.04, 2), (1.01, 3))

    def __init__(self, region_map=None):
        self.region_map = region_map

    def compose(self, auto_hints: list[HintSpec], manual_hints: list[HintSpec],
                *, manual_strength: float = 1.0) -> list[HintSpec]:
        manual_strength = float(np.clip(manual_strength, 0.0, 1.0))
        auto = [h for h in auto_hints if h.source != "style_only" and h.effective_strength > 0.01]
        if manual_strength > 0.001:
            strength_gain = 0.20 + 1.55 * manual_strength
            radius_gain = 0.65 + 1.10 * manual_strength
            manual = [h.clone(
                strength=float(np.clip(h.strength * strength_gain, 0.0, 1.0)),
                radius_norm=max(0.0005, h.radius_norm * radius_gain),
            ) for h in manual_hints]
        else:
            manual = []

        region_override: dict[int, HintSpec] = {}
        point_manual: list[HintSpec] = []
        for h in manual:
            if h.source == "manual_region" and h.region_id:
                region_override[int(h.region_id)] = h
            else:
                point_manual.append(h)

        kept: list[HintSpec] = []
        for h in auto:
            if h.region_id and int(h.region_id) in region_override:
                continue
            if any(self._overlaps(h, m) for m in point_manual):
                continue
            kept.append(h)

        kept = self._confidence_gate(kept)
        kept = self._limit_density(kept)
        return [*kept, *region_override.values(), *point_manual]

    @staticmethod
    def _overlaps(a: HintSpec, b: HintSpec) -> bool:
        radius = max(0.012, a.radius_norm + b.radius_norm)
        dx = a.x_norm - b.x_norm
        dy = a.y_norm - b.y_norm
        return dx * dx + dy * dy <= radius * radius

    @staticmethod
    def _confidence_gate(hints: list[HintSpec]) -> list[HintSpec]:
        out = []
        for h in hints:
            if h.source == "character_identity" and h.confidence < 0.56:
                continue
            if h.source == "auto_instance" and h.confidence < 0.36:
                continue
            out.append(h)
        return out

    def _region_limit(self, region_id: int | None) -> int:
        if not region_id or self.region_map is None:
            return 2
        labels = self.region_map.labels
        frac = float(np.count_nonzero(labels == int(region_id)) / max(1, labels.size))
        for upper, limit in self.REGION_LIMITS:
            if frac <= upper:
                return limit
        return 3

    def _limit_density(self, hints: list[HintSpec]) -> list[HintSpec]:
        grouped: dict[tuple, list[HintSpec]] = defaultdict(list)
        for h in hints:
            grouped[(h.region_id, h.source, h.semantic, h.character_id)].append(h)

        selected: list[HintSpec] = []
        for key, items in grouped.items():
            region_id = key[0]
            limit = self._region_limit(region_id)
            items = sorted(items, key=lambda h: (h.priority or 0, h.effective_strength), reverse=True)
            # Spatially diversify same-region points rather than keeping a cluster.
            chosen: list[HintSpec] = []
            for h in items:
                if all((h.x_norm - c.x_norm) ** 2 + (h.y_norm - c.y_norm) ** 2 > 0.0025 ** 2
                       for c in chosen):
                    chosen.append(h)
                if len(chosen) >= limit:
                    break
            selected.extend(chosen)

        auto = [h for h in selected if h.source not in ("manual", "manual_region", "eyedropper_hint")]
        manual = [h for h in selected if h.source in ("manual", "manual_region", "eyedropper_hint")]
        auto = sorted(auto, key=lambda h: (h.priority or 0, h.effective_strength), reverse=True)
        return [*auto[:self.MAX_PAGE_AUTO], *manual]


def degrade_for_retry(hints: list[HintSpec]) -> list[HintSpec]:
    """One-shot safe retry policy after hint-blob detection."""
    out: list[HintSpec] = []
    for h in hints:
        if h.source in ("manual", "manual_region", "eyedropper_hint"):
            out.append(h)
        elif h.source == "character_identity":
            out.append(h.clone(
                radius_norm=max(0.001, h.radius_norm * 0.55),
                strength=h.strength * 0.72,
            ))
        # Scene/auto hints are deliberately omitted during the retry.
    return out
