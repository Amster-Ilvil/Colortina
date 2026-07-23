"""StyleDescriptor — the new heart of Colortina's style system.

Instead of storing raw hex colors ("hair = #FFD200"), a StyleDescriptor
stores *color language*: the relationships, biases, and distribution
shapes that make a style recognisable regardless of what specific
characters appear in the comic.

Design principles (from requirements doc):
  - Base Color (identity)  is decided by mc-v2 — we NEVER overwrite it.
  - Style Color (language) is decided by StyleDescriptor — we shift/tint
    the mc-v2 base, never replace it.
  - A highlight is NOT a fixed RGB; it is a direction in LAB space
    (warmer, lighter, more saturated) relative to the mid-tone.

Saved as .ccstyle (JSON) alongside the palette field kept for backwards
compat with old StyleProfile files.

Used by:
  core/style_analyzer.py  — populates from a reference image
  core/style_director.py  — converts to per-region hint colours
  core/guided_colorist.py — feeds into Hint Generator
  core/style_post.py      — drives post-processing grade
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

_VERSION = 4


@dataclass
class RegionDescriptor:
    """Style language for ONE semantic region class (hair, skin, sky…).

    All values are *relative* — they say HOW to shade, not WHAT color.
    """
    # Hue character — warm (+) tilts toward red/yellow, cool (-) toward blue.
    warm_bias: float = 0.0          # LAB b-channel shift applied to mid-tones
    # Shadow treatment
    shadow_bias: float = 0.0        # extra LAB b-shift for dark sub-region
    shadow_desat: float = 0.0       # 0-1: desaturate shadows (cinematic look)
    shadow_hue_rotate: float = 0.0  # LAB a-shift for shadow (e.g. purplish shadows)
    # Highlight treatment
    highlight_bias: float = 0.0     # extra LAB b-shift for light sub-region
    highlight_desat: float = 0.0    # 0-1: desaturate highlights
    highlight_hue_rotate: float = 0.0  # LAB a-shift for highlight
    # Overall contrast within this region
    contrast: float = 1.0           # chroma contrast multiplier (>1 = more vivid)
    # Saturation character
    saturation_scale: float = 1.0   # per-region saturation multiplier
    # Gradient vs flat-fill (0 = flat, 1 = maximum gradient preserved)
    gradient: float = 0.5
    # How many hint layers to generate (3 = hi/mid/lo, 1 = single mid-tone)
    hint_layers: int = 3


@dataclass
class StyleDescriptor:
    """Complete style description as color language, not pixel values.

    Can be:
      - extracted from reference images (StyleAnalyzer)
      - hand-authored (built-in presets)
      - loaded from a .ccstyle file
    """
    name: str = "Custom"
    description: str = ""
    source: str = "extracted"       # "extracted" | "builtin" | "manual"
    version: int = _VERSION
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%d"))

    # ── Global atmosphere ─────────────────────────────────────────────
    global_warm_cool: float = 0.0    # overall temperature shift (LAB b-axis)
    global_saturation: float = 1.0   # overall chroma multiplier
    global_contrast: float = 1.0     # overall luminance contrast multiplier
    global_shadow_lift: float = 0.0  # lift shadows (0=none, +=cinematic)
    cel_flatten: float = 0.3         # 0 = full gradient, 1 = flat fills

    # Reference-wide chroma signature.  Unlike the relative per-region
    # language above, these fields preserve the visible atmosphere of the
    # supplied colour pages even when semantic CLIP labelling is imperfect.
    # Values are OpenCV LAB [L, A, B] statistics and dominant RGB hex colours.
    reference_lab_mean: list[float] = field(default_factory=list)
    reference_lab_std: list[float] = field(default_factory=list)
    # Preview-only palette.  v4 deliberately never uses these absolute colours
    # to recolour character regions.
    reference_palette: list[str] = field(default_factory=list)

    # Which parts of the page may receive relative style treatment.  Absolute
    # identity colours are never stored here.
    style_scope: dict = field(default_factory=lambda: {
        "character_rendering": True,
        "background_rendering": True,
        "global_ambience": 0.20,
    })
    revision: int = 0

    # ── Per-region descriptors ────────────────────────────────────────
    hair:       RegionDescriptor = field(default_factory=RegionDescriptor)
    eyes:       RegionDescriptor = field(default_factory=lambda: RegionDescriptor(
                    contrast=1.2, saturation_scale=1.05, gradient=0.35))
    skin:       RegionDescriptor = field(default_factory=lambda: RegionDescriptor(
                    warm_bias=3.0, shadow_bias=-1.0, shadow_hue_rotate=2.0,
                    highlight_bias=1.0, saturation_scale=0.9))
    sky:        RegionDescriptor = field(default_factory=lambda: RegionDescriptor(
                    warm_bias=-4.0, gradient=0.8, saturation_scale=0.85))
    foliage:    RegionDescriptor = field(default_factory=lambda: RegionDescriptor(
                    warm_bias=2.0, saturation_scale=0.9))
    clothing_primary:   RegionDescriptor = field(default_factory=RegionDescriptor)
    clothing_secondary: RegionDescriptor = field(default_factory=RegionDescriptor)
    clothing_accent:    RegionDescriptor = field(default_factory=RegionDescriptor)
    background: RegionDescriptor = field(default_factory=lambda: RegionDescriptor(
                    saturation_scale=0.7, gradient=0.4))
    metal:      RegionDescriptor = field(default_factory=lambda: RegionDescriptor(
                    saturation_scale=0.5, contrast=1.3))
    water:      RegionDescriptor = field(default_factory=lambda: RegionDescriptor(
                    warm_bias=-5.0, gradient=0.9))
    fire:       RegionDescriptor = field(default_factory=lambda: RegionDescriptor(
                    warm_bias=8.0, shadow_bias=4.0, saturation_scale=1.3))
    stone:      RegionDescriptor = field(default_factory=lambda: RegionDescriptor(
                    saturation_scale=0.4, gradient=0.5))
    wood:       RegionDescriptor = field(default_factory=lambda: RegionDescriptor(
                    warm_bias=4.0, saturation_scale=0.8))

    # Number of pixels/references that actually contributed to each region.
    # Missing semantic classes are excluded when several references are merged
    # instead of blending in a neutral default and diluting the real style.
    region_samples: dict = field(default_factory=dict)

    # ── Backwards-compat palette (kept for old .ccstyle files) ───────
    # Not used for Hint generation; only as a fallback seed colour when
    # the StyleDirector needs an absolute reference.
    palette: dict = field(default_factory=dict)

    # ── Global scalar stats (used by to_style_preset post-processing) ─
    saturation: float = 0.85
    contrast: float = 0.75
    temperature: str = "warm"
    shadow_strength: float = 0.6
    gradient: float = 0.4

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, path: str) -> str:
        if not path.endswith(".ccstyle"):
            path += ".ccstyle"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._to_json(), f, indent=2, ensure_ascii=False)
        return path

    def _to_json(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def load(cls, path: str) -> "StyleDescriptor":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "StyleDescriptor":
        # v1-v3 files are upgraded conservatively.  Old absolute palettes remain
        # available for UI preview but are not used as character identity colours.
        from core.schema_migration import migrate_ccstyle
        data, _notes = migrate_ccstyle(data)
        region_keys = {"hair", "eyes", "skin", "sky", "foliage", "background",
                       "metal", "water", "fire", "stone", "wood",
                       "clothing_primary", "clothing_secondary", "clothing_accent"}
        kwargs: dict = {}
        for k, v in data.items():
            if k in region_keys and isinstance(v, dict):
                kwargs[k] = RegionDescriptor(**{
                    kk: vv for kk, vv in v.items()
                    if kk in RegionDescriptor.__dataclass_fields__})
            elif k in cls.__dataclass_fields__:
                kwargs[k] = v
        return cls(**kwargs)

    def region(self, key: str) -> RegionDescriptor:
        """Look up a RegionDescriptor by palette key, falling back to
        a neutral default so callers never get KeyError."""
        return getattr(self, key, RegionDescriptor())

    # ── Conversion to legacy StyleProfile (backwards compat) ─────────

    def to_style_profile(self):
        """Return a legacy StyleProfile so old code that calls
        style_profile.to_style_preset() / as_palette_override() still works."""
        from core.style_engine import StyleProfile
        return StyleProfile(
            name=self.name,
            palette=self.palette,
            saturation=self.saturation,
            contrast=self.contrast,
            temperature=self.temperature,
            shadow_strength=self.shadow_strength,
            gradient=self.gradient,
            description=self.description,
            source=self.source,
        )

    def to_style_preset(self, key: str | None = None):
        """Map descriptor global stats onto a StylePreset for style_post."""
        from core.presets import StylePreset
        ambience = float((self.style_scope or {}).get("global_ambience", 0.20))
        warm_shift = {"warm": 3.0, "cool": -3.0, "neutral": 0.0}.get(
            self.temperature, 0.0)
        import numpy as np
        ambience = float(np.clip(ambience, 0.0, 1.0))
        return StylePreset(
            key=key or f"descriptor_{self.name.lower().replace(' ', '_')}",
            label=f"{self.name} (descriptor)",
            description=self.description or self.name,
            saturation_boost=float(np.clip(1.0 + (self.saturation - 0.7) * 0.45, 0.85, 1.35)),
            white_threshold=int(np.clip(210 + self.contrast * 25, 205, 238)),
            black_threshold=int(np.clip(20 + self.shadow_strength * 25, 15, 45)),
            l_gamma=float(np.clip(1.15 - self.contrast * 0.3, 0.85, 1.15)),
            chroma_warm_shift=(warm_shift + self.global_warm_cool * 0.25) * ambience,
            cel_flatten=float(np.clip(self.cel_flatten, 0.0, 0.85)),
            neutral_fade_floor=float(np.clip(0.25 + self.gradient * 0.3, 0.25, 0.6)),
            denoise_sigma=15,
        )
