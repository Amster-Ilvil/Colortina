"""Guided colorist — segment, identify, assign, hint.

Orchestrates the "digital colorist" pipeline for auto (mc-v2) mode:

1. Segment the page into lineart-bounded regions (region_segmenter).
2. Label each region via CLIP zero-shot (region_classifier).
3. Look up the job's StyleDescriptor to get TieredColor per region.
4. Emit LAYERED sparse hint points — highlight / mid / shadow — that
   feed mc-v2's native hint channel.

Why layered hints matter
────────────────────────
mc-v2 was trained on real colored manga where a hint sampled from hair
would naturally land on a darker pixel sometimes and a brighter pixel
other times, producing a range of colors the model could interpolate.
When we feed it a single flat color, it has nothing to interpolate and
produces a color wash.

By placing hints on pixels whose actual luminance matches the tier
(bright pixels get the highlight color, dark pixels get the shadow
color), we recreate the natural hint distribution the model was trained
on — and style becomes visible.

Backwards compatibility
───────────────────────
If no StyleDescriptor is active (style_key == "neutral", no custom
profile), the code falls back to the old single-color-per-region
behavior so existing sessions are unaffected.
"""

import threading
from typing import Optional

import numpy as np

from core.color_director import ColorDirector, hex_to_bgr
from core.region_classifier import RegionClassifier
from core.region_segmenter import segment_regions

_classifier_lock = threading.Lock()
_shared_classifier: RegionClassifier | None = None


def _get_classifier() -> RegionClassifier:
    global _shared_classifier
    with _classifier_lock:
        if _shared_classifier is None:
            _shared_classifier = RegionClassifier()
        return _shared_classifier


# A hint point: (x_norm, y_norm, (r, g, b), radius_norm)
HintPoint = tuple[float, float, tuple[int, int, int], float]

_AUTO_RADIUS_NORM = 0.006   # single-dot radius (fraction of image width)
_LAYER_RADIUS     = 0.005   # slightly smaller for tier sub-dots

_DIRECT_KEYS   = {"skin", "hair", "metal", "wood", "sky", "foliage",
                  "stone", "water", "fire", "background"}
_CLOTHING_KEYS = ("clothing_primary", "clothing_secondary", "clothing_accent")

_MIN_CONF = 0.25

# Luminance thresholds for tier classification (0-255, grayscale of source page)
_L_HI_MIN = 175   # pixel qualifies as "highlight zone"
_L_SH_MAX =  90   # pixel qualifies as "shadow zone"


class GuidedColorist:
    """Per-job color guide.  Build once per job, call hints_for_page per page.

    Parameters
    ----------
    style_descriptor : StyleDescriptor | None
        New-style color language object.  When given, layered hints are
        produced.  Takes priority over style_profile.
    style_profile : StyleProfile | None
        Legacy color profile (palette + global stats).  Used if no
        style_descriptor.  Converted to a StyleDescriptor internally.
    character_memories : dict | None
        {label: CharacterMemory} for book-level per-character color slots.
    """

    def __init__(self, config, use_llm: bool | None = None,
                 style_descriptor=None,
                 style_profile=None,
                 character_memories: dict | None = None):
        self._cfg = config
        self._use_llm = use_llm
        self._classifier = _get_classifier()
        self._director_plain = ColorDirector(config)
        self._script: dict | None = None
        self._character_memories = character_memories or {}

        # Resolve descriptor — new path > legacy profile > None
        self._descriptor = style_descriptor
        if self._descriptor is None and style_profile is not None:
            self._descriptor = _profile_to_descriptor(style_profile)

        # StyleDirector (new path) — created lazily once we have a descriptor
        self._style_director = None
        if self._descriptor is not None:
            from core.style_director import StyleDirector
            self._style_director = StyleDirector(
                self._descriptor,
                palette=style_profile.palette if style_profile else None)

    @property
    def available(self) -> bool:
        return self._classifier.available

    @property
    def script(self) -> dict | None:
        return self._script

    def prepare(self, sample_pages_bgr: list[np.ndarray]) -> None:
        """Build the job's color script from a few sample pages."""
        if self._descriptor is not None:
            # Use descriptor's palette as seed for the plain director script
            palette = self._descriptor.palette or {}
            from core.color_director import DEFAULT_PALETTE
            self._script = {
                "palette": {**DEFAULT_PALETTE, **palette},
                "mood": self._descriptor.description or "descriptor style",
                "source": "descriptor",
            }
            return

        counts: dict[str, int] = {}
        if self._classifier.available:
            for page in sample_pages_bgr:
                seg = segment_regions(page)
                if not seg.regions:
                    continue
                labels = self._classifier.classify(
                    page, [r.bbox for r in seg.regions])
                if not labels:
                    continue
                for (label, conf) in labels:
                    if conf >= _MIN_CONF:
                        counts[label] = counts.get(label, 0) + 1

        summary = {"pages_sampled": len(sample_pages_bgr),
                   "region_counts": counts,
                   "content_hint": "black-and-white manga/manhwa chapter"}
        self._script = self._director_plain.build_script(
            summary, use_llm=self._use_llm)

    def hints_for_page(self, image_bgr: np.ndarray) -> list[HintPoint]:
        """Segment + label one page and return its color hint points.

        When a StyleDescriptor is active:
          Each region generates up to 3 tiers of hints (hi / mid / shadow),
          placed on page pixels whose luminance matches that tier.

        Without a StyleDescriptor:
          Falls back to the original single-color-per-region behavior.
        """
        if not self._classifier.available:
            return []
        if self._script is None:
            self.prepare([image_bgr])

        h, w = image_bgr.shape[:2]
        gray = (image_bgr if image_bgr.ndim == 2
                else np.mean(image_bgr, axis=2).astype(np.uint8))
        seg  = segment_regions(image_bgr)
        if not seg.regions:
            return []

        labels = self._classifier.classify(image_bgr, [r.bbox for r in seg.regions])
        if not labels:
            return []

        def _on_tone(px: int, py: int) -> bool:
            v = int(gray[min(h - 1, max(0, py)), min(w - 1, max(0, px))])
            return 70 <= v < 238

        def _pixel_tier(px: int, py: int) -> str:
            v = int(gray[min(h - 1, max(0, py)), min(w - 1, max(0, px))])
            if v >= _L_HI_MIN:
                return "hi"
            if v <= _L_SH_MAX:
                return "sh"
            return "mid"

        palette = self._script["palette"]

        # Character Memory per-instance overrides
        instance_rgb: dict[int, tuple[int, int, int]] = {}
        for label_key, memory in self._character_memories.items():
            same_label = [r for r, (lab, conf) in zip(seg.regions, labels)
                         if lab == label_key and conf >= _MIN_CONF]
            for region_id, hex_color in memory.assign(same_label, gray).items():
                b, g, r = hex_to_bgr(hex_color)
                instance_rgb[region_id] = (r, g, b)

        points: list[HintPoint] = []
        clothing_rank = 0

        for region, (label, conf) in zip(seg.regions, labels):
            if label == "bubble" or conf < _MIN_CONF:
                continue
            if region.mean_gray >= 238 or region.mean_gray < 70:
                continue

            # Resolve palette key
            if label == "clothing":
                key = _CLOTHING_KEYS[clothing_rank % len(_CLOTHING_KEYS)]
                clothing_rank += 1
            elif label in _DIRECT_KEYS:
                key = label
            else:
                continue

            # ── New path: layered tiered hints via StyleDirector ──────
            if self._style_director is not None and region.label_id not in instance_rgb:
                tc = self._style_director.get_tiered(key)
                _emit_tiered_hints(
                    points, region, seg, gray, w, h,
                    tc, _on_tone, _pixel_tier)
                continue

            # ── Legacy / CharacterMemory path: single flat color ──────
            if region.label_id in instance_rgb:
                rgb = instance_rgb[region.label_id]
            else:
                hex_color = palette.get(key)
                if not hex_color:
                    continue
                b, g, r = hex_to_bgr(hex_color)
                rgb = (r, g, b)

            px, py = region.interior_point
            if _on_tone(px, py):
                points.append((px / w, py / h, rgb, _AUTO_RADIUS_NORM))

            if region.frac > 0.008:
                for (ex, ey) in seg.interior_points(region):
                    if _on_tone(ex, ey):
                        points.append((ex / w, ey / h, rgb, _AUTO_RADIUS_NORM))

        return points


# ── Tiered hint emitter ───────────────────────────────────────────────────────

def _emit_tiered_hints(points: list,
                       region,
                       seg,
                       gray: np.ndarray,
                       w: int, h: int,
                       tc,            # TieredColor
                       on_tone,       # callable(px, py) -> bool
                       pixel_tier,    # callable(px, py) -> "hi"|"mid"|"sh"
                       ) -> None:
    """Place hi/mid/shadow hint dots on pixels whose luminance matches
    each tier.  For small regions we fall back to a single mid-tone dot."""

    tier_map = {"hi": tc.highlight_rgb, "mid": tc.mid_rgb, "sh": tc.shadow_rgb}

    # Interior point of the region — classify its tier
    px, py = region.interior_point
    if on_tone(px, py):
        tier = pixel_tier(px, py) if tc.hint_layers >= 3 else "mid"
        points.append((px / w, py / h, tier_map[tier], _LAYER_RADIUS))

    if region.frac <= 0.008:
        return   # small region: one dot is enough

    # Extra interior points for large regions — place each at the
    # correct tier colour so mc-v2 sees the full gradient range.
    tier_counts = {"hi": 0, "mid": 0, "sh": 0}
    for (ex, ey) in seg.interior_points(region, spacing=90, max_points=14):
        if not on_tone(ex, ey):
            continue
        tier = pixel_tier(ex, ey) if tc.hint_layers >= 3 else "mid"
        # Cap each tier so we don't flood the model with one colour only
        if tier_counts[tier] >= 4:
            continue
        points.append((ex / w, ey / h, tier_map[tier], _LAYER_RADIUS))
        tier_counts[tier] += 1


# ── Helper: convert legacy StyleProfile to StyleDescriptor ───────────────────

def _profile_to_descriptor(profile):
    """Wrap an old StyleProfile in a StyleDescriptor so the new code path
    works.  The descriptor fields come from the profile's global stats;
    per-region language is synthesised from temperature/gradient/contrast."""
    from core.style_descriptor import StyleDescriptor, RegionDescriptor

    warm_bias = {"warm": 4.0, "cool": -4.0, "neutral": 0.0}.get(
        getattr(profile, "temperature", "neutral"), 0.0)
    gradient = getattr(profile, "gradient", 0.5)
    contrast = getattr(profile, "contrast",  0.75)
    sat      = getattr(profile, "saturation", 0.85)

    def _rd(extra_warm: float = 0.0, sat_mult: float = 1.0) -> RegionDescriptor:
        return RegionDescriptor(
            warm_bias=round(warm_bias + extra_warm, 2),
            shadow_bias=round(-warm_bias * 0.3, 2),
            shadow_desat=round((1.0 - gradient) * 0.3, 3),
            highlight_bias=round(warm_bias * 0.2, 2),
            saturation_scale=round(sat * 1.5 * sat_mult, 3),
            gradient=round(gradient, 3),
            contrast=round(max(0.7, contrast * 1.3), 3),
            hint_layers=3,
        )

    return StyleDescriptor(
        name=getattr(profile, "name", "Legacy"),
        description=getattr(profile, "description", "Converted from StyleProfile"),
        source="extracted",
        global_warm_cool=round(warm_bias * 0.3, 2),
        global_saturation=round(sat * 1.4, 3),
        global_contrast=round(contrast * 1.3, 3),
        global_shadow_lift=round(getattr(profile, "shadow_strength", 0.5) * 0.3, 3),
        cel_flatten=round(max(0.0, 1.0 - gradient * 1.5), 3),
        hair=_rd(extra_warm=warm_bias * 0.1),
        skin=_rd(extra_warm=3.0, sat_mult=0.85),
        sky=_rd(extra_warm=-5.0, sat_mult=0.8),
        foliage=_rd(extra_warm=2.0, sat_mult=0.9),
        clothing_primary=_rd(),
        clothing_secondary=_rd(),
        clothing_accent=_rd(),
        background=_rd(sat_mult=0.6),
        metal=_rd(sat_mult=0.45),
        water=_rd(extra_warm=-6.0, sat_mult=0.85),
        fire=_rd(extra_warm=8.0, sat_mult=1.3),
        stone=_rd(sat_mult=0.4),
        wood=_rd(extra_warm=4.0, sat_mult=0.8),
        palette=getattr(profile, "palette", {}),
        saturation=sat,
        contrast=contrast,
        temperature=getattr(profile, "temperature", "neutral"),
        shadow_strength=getattr(profile, "shadow_strength", 0.6),
        gradient=gradient,
    )
