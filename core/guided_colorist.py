"""Page analysis and sparse, source-aware colour guidance.

v4 separates three responsibilities:
- StyleDescriptor changes rendering relationships only;
- CharacterLibrary supplies absolute identity colours;
- ScenePalette optionally supplies absolute environment colours.

Unknown character regions no longer receive a shared default hair/skin/clothing
colour.  In that case mc-v2 is allowed to choose the base colour itself.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

from core.hint_spec import HintSpec
from core.page_color_context import PageColorContext
from core.region_classifier import RegionClassifier
from core.region_segmenter import segment_regions

_classifier_lock = threading.Lock()
_shared_classifier: RegionClassifier | None = None

_MIN_CONF = 0.25
_LAYER_RADIUS = 0.0035
_CHARACTER_LABELS = {"hair", "skin", "eyes", "clothing"}
_SCENE_LABELS = {"metal", "wood", "sky", "foliage", "stone", "water", "fire", "background"}


def _get_classifier() -> RegionClassifier:
    global _shared_classifier
    with _classifier_lock:
        if _shared_classifier is None:
            _shared_classifier = RegionClassifier()
        return _shared_classifier


class GuidedColorist:
    def __init__(self, config, style_descriptor=None, style_profile=None,
                 character_memories: dict | None = None,
                 character_library=None, scene_palette=None,
                 style_strength: float = 1.0,
                 reference_strength: float = 1.0):
        self._cfg = config
        self._character_library = character_library
        self._scene_palette = scene_palette
        self._style_strength = float(np.clip(style_strength, 0.0, 1.0))
        self._reference_strength = float(np.clip(reference_strength, 0.0, 1.0))
        self._classifier = _get_classifier()
        self._character_memories = character_memories or {}
        self._descriptor = style_descriptor
        if self._descriptor is None and style_profile is not None:
            self._descriptor = _profile_to_descriptor(style_profile)
        self._style_director = None
        if self._descriptor is not None:
            from core.style_director import StyleDirector
            self._style_director = StyleDirector(
                self._descriptor, strength=self._style_strength)
        self._last_context: PageColorContext | None = None

    @property
    def available(self) -> bool:
        # Character identity can still operate through the OpenCV face
        # fallback when the optional CLIP model has not been installed.
        has_library = bool(getattr(self._character_library, "characters", None))
        return bool(self._classifier.available or has_library)

    @property
    def script(self) -> dict | None:
        # Kept for old UI/debug code.  No absolute default palette is exposed.
        return {"source": "v4-structured", "palette": {}}

    @property
    def last_context(self) -> PageColorContext | None:
        return self._last_context

    def prepare(self, sample_pages_bgr: list[np.ndarray]) -> None:
        # Analysis is page-local in v4; retained as a no-op for compatibility.
        return None

    def analyze_page(self, image_bgr: np.ndarray,
                     forced_matches: dict[int, int] | None = None) -> PageColorContext:
        h, w = image_bgr.shape[:2]
        gray = (image_bgr if image_bgr.ndim == 2
                else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY))
        seg = segment_regions(image_bgr)
        if not seg.regions:
            context = PageColorContext(segmentation=seg,
                                       diagnostics={"reason": "no_regions"})
            self._last_context = context
            return context

        classifier_available = bool(self._classifier.available)
        labels = None
        if classifier_available:
            labels = self._classifier.classify(
                image_bgr, [r.bbox for r in seg.regions])

        assignments: dict[int, dict] = {}
        instances = []
        match_diag: dict = {}

        if self._character_library is not None:
            if labels:
                assignments, instances, match_diag = self._character_library.match_page(
                    seg.regions, labels, segmentation=seg, gray_page=gray,
                    page_bgr=image_bgr, classifier=self._classifier,
                    forced_matches=forced_matches)
                # Dense manga pages sometimes produce valid CLIP labels but no
                # usable hair anchor.  Fall back to face geometry rather than
                # silently losing all identity colours.
                if not instances and match_diag.get("reason") == "no_hair_anchor":
                    assignments, instances, match_diag, labels =                         self._character_library.match_page_fallback(
                            segmentation=seg, page_bgr=image_bgr,
                            classifier=self._classifier,
                            forced_matches=forced_matches)
            else:
                assignments, instances, match_diag, labels =                     self._character_library.match_page_fallback(
                        segmentation=seg, page_bgr=image_bgr,
                        classifier=self._classifier,
                        forced_matches=forced_matches)

        if not labels:
            labels = [("unknown", 0.0) for _ in seg.regions]

        memory_rgb: dict[int, tuple[int, int, int]] = {}
        # CharacterMemory and ScenePalette depend on semantic labels.  They are
        # disabled in pure face-fallback mode; explicit CharacterLibrary
        # assignments remain available and confidence-gated.
        if classifier_available:
            for label_key, memory in self._character_memories.items():
                same_label = [r for r, (lab, conf) in zip(seg.regions, labels)
                              if lab == label_key and conf >= _MIN_CONF]
                for region_id, value in memory.assign(same_label, gray).items():
                    try:
                        s = value.lstrip("#")
                        memory_rgb[int(region_id)] = (
                            int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
                    except Exception:
                        continue

        hints: list[HintSpec] = []
        for region, (label, semantic_conf) in zip(seg.regions, labels):
            semantic_conf = float(semantic_conf)
            if label == "bubble" or semantic_conf < _MIN_CONF:
                continue
            if region.mean_gray >= 242 or region.mean_gray < 55:
                continue

            info = assignments.get(int(region.label_id))
            base_rgb = None
            source = None
            character_id = None
            confidence = semantic_conf
            source_strength = 0.0

            if info is not None and info.get("lock_allowed", False):
                base_rgb = tuple(info["rgb"])
                source = "character_identity"
                character_id = int(info["char_id"])
                confidence = min(semantic_conf, float(info.get("match_score", 0.0)))
                source_strength = 0.58 * self._reference_strength
            elif int(region.label_id) in memory_rgb:
                base_rgb = memory_rgb[int(region.label_id)]
                source = "auto_instance"
                confidence = semantic_conf * 0.75
                source_strength = 0.30 * self._reference_strength
            elif (classifier_available and self._scene_palette is not None
                  and label in _SCENE_LABELS):
                base_rgb = self._scene_palette.color_for(label)
                if base_rgb is not None:
                    source = "scene_palette"
                    confidence = semantic_conf
                    source_strength = 0.28 * float(
                        getattr(self._scene_palette, "strength", 0.7))

            # Style-only regions do not receive an absolute default colour.
            if base_rgb is None or source is None:
                continue

            key = "clothing_primary" if label == "clothing" else label
            if self._style_director is not None:
                tiered = self._style_director.get_tiered(key, base_rgb=base_rgb)
            else:
                from core.style_director import tiered_from_rgb
                tiered = tiered_from_rgb(base_rgb, descriptor=None, strength=0.0)
            hints.extend(_emit_tiered_specs(
                region, seg, gray, w, h, tiered,
                source=source, semantic=label, character_id=character_id,
                confidence=confidence, strength=source_strength,
            ))

        diagnostics = {
            "classifier_available": classifier_available,
            "regions": len(seg.regions),
            "semantic_confident": sum(1 for _lab, conf in labels if conf >= _MIN_CONF),
            "identity_assignments": len(assignments),
            "hint_count": len(hints),
            "hint_sources": _count_sources(hints),
            **match_diag,
        }
        context = PageColorContext(
            segmentation=seg, semantic_labels=labels,
            character_instances=instances,
            identity_assignments=assignments,
            hints=hints, diagnostics=diagnostics)
        self._last_context = context
        return context

    def hints_for_page(self, image_bgr: np.ndarray,
                       forced_matches: dict[int, int] | None = None) -> list[HintSpec]:
        return self.analyze_page(image_bgr, forced_matches=forced_matches).hints


def _count_sources(hints: list[HintSpec]) -> dict[str, int]:
    out: dict[str, int] = {}
    for hint in hints:
        out[hint.source] = out.get(hint.source, 0) + 1
    return out


def _emit_tiered_specs(region, seg, gray: np.ndarray, w: int, h: int, tc,
                       *, source: str, semantic: str,
                       character_id: int | None, confidence: float,
                       strength: float) -> list[HintSpec]:
    mask = seg.labels == int(region.label_id)
    if not np.any(mask):
        return []
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    valid = mask & (distance >= 1.25)
    valid &= (gray >= 55) & (gray <= 242)
    ys, xs = np.nonzero(valid)
    if ys.size == 0:
        px, py = region.interior_point
        return [HintSpec(px / w, py / h, tc.mid_rgb, _LAYER_RADIUS,
                         strength=strength, source=source,
                         region_id=int(region.label_id), semantic=semantic,
                         character_id=character_id, confidence=confidence)]

    if ys.size > 20000:
        step = int(np.ceil(ys.size / 20000))
        ys, xs = ys[::step], xs[::step]
    values = gray[ys, xs].astype(np.float32)
    dist = distance[ys, xs].astype(np.float32)
    area_frac = float(region.frac)
    max_points = 1 if area_frac <= 0.006 else (2 if area_frac <= 0.04 else 3)
    tiers = [
        (0.50, tc.mid_rgb),
        (0.80, tc.highlight_rgb),
        (0.20, tc.shadow_rgb),
    ][:max_points]
    out: list[HintSpec] = []
    used: set[int] = set()
    for quantile, rgb in tiers:
        target = float(np.quantile(values, quantile))
        score = np.abs(values - target) + 7.0 / np.maximum(dist, 0.5)
        for idx in np.argsort(score)[:24]:
            idx = int(idx)
            if idx in used:
                continue
            used.add(idx)
            out.append(HintSpec(
                float(xs[idx] / w), float(ys[idx] / h), rgb,
                radius_norm=_LAYER_RADIUS, strength=strength,
                source=source, region_id=int(region.label_id),
                semantic=semantic, character_id=character_id,
                confidence=confidence))
            break
    return out


def _profile_to_descriptor(profile):
    """Convert a legacy StyleProfile into relative rendering language only."""
    from core.style_descriptor import StyleDescriptor, RegionDescriptor

    warm_bias = {"warm": 4.0, "cool": -4.0, "neutral": 0.0}.get(
        getattr(profile, "temperature", "neutral"), 0.0)
    gradient = getattr(profile, "gradient", 0.5)
    contrast = getattr(profile, "contrast", 0.75)
    sat = getattr(profile, "saturation", 0.85)

    def rd(extra_warm: float = 0.0, sat_mult: float = 1.0) -> RegionDescriptor:
        return RegionDescriptor(
            warm_bias=round(warm_bias + extra_warm, 2),
            shadow_bias=round(-warm_bias * 0.3, 2),
            shadow_desat=round((1.0 - gradient) * 0.3, 3),
            highlight_bias=round(warm_bias * 0.2, 2),
            saturation_scale=round(sat * 1.2 * sat_mult, 3),
            gradient=round(gradient, 3),
            contrast=round(max(0.7, contrast * 1.2), 3),
            hint_layers=3,
        )

    return StyleDescriptor(
        name=getattr(profile, "name", "Legacy"),
        description=getattr(profile, "description", "Converted StyleProfile"),
        source="extracted",
        global_warm_cool=round(warm_bias * 0.3, 2),
        global_saturation=round(sat * 1.2, 3),
        global_contrast=round(contrast * 1.2, 3),
        global_shadow_lift=round(getattr(profile, "shadow_strength", 0.5) * 0.3, 3),
        cel_flatten=round(max(0.0, 1.0 - gradient * 1.5), 3),
        style_scope={"character_rendering": True,
                     "background_rendering": True,
                     "global_ambience": 0.15},
        hair=rd(), eyes=rd(sat_mult=1.05), skin=rd(extra_warm=3.0, sat_mult=0.85),
        sky=rd(extra_warm=-5.0, sat_mult=0.8), foliage=rd(extra_warm=2.0),
        clothing_primary=rd(), clothing_secondary=rd(), clothing_accent=rd(),
        background=rd(sat_mult=0.65), metal=rd(sat_mult=0.5),
        water=rd(extra_warm=-6.0, sat_mult=0.85),
        fire=rd(extra_warm=8.0, sat_mult=1.25), stone=rd(sat_mult=0.45),
        wood=rd(extra_warm=4.0, sat_mult=0.8),
        palette={}, saturation=sat, contrast=contrast,
        temperature=getattr(profile, "temperature", "neutral"),
        shadow_strength=getattr(profile, "shadow_strength", 0.6),
        gradient=gradient,
    )
