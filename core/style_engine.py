"""Style Engine — extract, save, load, and manage reusable styles.

New in v2:
  - extract_from_reference / extract_from_references now return a
    StyleDescriptor (color language) instead of just a hex palette.
  - StyleProfile is kept for backwards-compat: it wraps a StyleDescriptor
    when loaded from an old .ccstyle file, and old callers that only use
    ``as_palette_override()`` / ``to_style_preset()`` still work.

File format (.ccstyle):
  Still JSON, now with a "version" field.  v1 files (old palette-only
  format) are read and silently upgraded to v2 StyleDescriptors.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

from core.color_director import DEFAULT_PALETTE
from core.presets import StylePreset
from core.style_descriptor import StyleDescriptor
from core.style_analyzer import StyleAnalyzer

_CCSTYLE_VERSION = 3


# ── Legacy StyleProfile (kept for backwards compat) ───────────────────────────

@dataclass
class StyleProfile:
    """Legacy wrapper.  New code should use StyleDescriptor directly.

    When loaded from a v1 .ccstyle file, a StyleDescriptor is synthesised
    from the old palette + global stats so the hint pipeline still benefits
    from tiered hints even for old saved styles.
    """

    name: str = "Custom"
    palette: dict = field(default_factory=lambda: dict(DEFAULT_PALETTE))
    saturation: float = 0.85
    contrast: float = 0.75
    temperature: str = "warm"
    shadow_strength: float = 0.6
    gradient: float = 0.4
    description: str = ""
    source: str = "extracted"
    version: int = _CCSTYLE_VERSION
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%d"))

    # Attached StyleDescriptor (None for v1 files until first access)
    _descriptor: StyleDescriptor | None = field(
        default=None, repr=False, compare=False)

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, path: str) -> str:
        if not path.endswith(".ccstyle"):
            path += ".ccstyle"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = asdict(self)
        data.pop("_descriptor", None)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, path: str) -> "StyleProfile":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # External v2 files used to be silently truncated to the legacy scalar
        # fields here.  Detect the full descriptor and retain every per-region
        # shadow/highlight parameter.
        if data.get("version", 1) >= 2 and "global_warm_cool" in data:
            desc = StyleDescriptor._from_dict(data)
            profile = _descriptor_to_profile(desc)
            profile._descriptor = desc
            return profile
        known = {k: v for k, v in data.items()
                 if k in cls.__dataclass_fields__ and k != "_descriptor"}
        return cls(**known)

    # ── Application ──────────────────────────────────────────────────

    def as_palette_override(self) -> dict:
        merged = dict(DEFAULT_PALETTE)
        merged.update(self.palette)
        return merged

    def to_style_preset(self, key: str | None = None) -> StylePreset:
        warm_shift = {"warm": 3.0, "cool": -3.0, "neutral": 0.0}[self.temperature]
        return StylePreset(
            key=key or f"extracted_{self.name.lower().replace(' ', '_')}",
            label=f"{self.name} (extracted)",
            description=self.description or f"Extracted from reference: {self.name}",
            saturation_boost=float(np.clip(0.9 + self.saturation * 0.9, 1.0, 1.9)),
            white_threshold=int(np.clip(210 + self.contrast * 25, 205, 238)),
            black_threshold=int(np.clip(20 + self.shadow_strength * 25, 15, 45)),
            l_gamma=float(np.clip(1.15 - self.contrast * 0.3, 0.85, 1.15)),
            chroma_warm_shift=warm_shift,
            cel_flatten=float(np.clip(1.0 - self.gradient, 0.0, 0.85)),
            neutral_fade_floor=float(np.clip(0.25 + self.gradient * 0.3, 0.25, 0.6)),
            denoise_sigma=15,
        )

    def get_descriptor(self) -> StyleDescriptor:
        """Return the attached StyleDescriptor, synthesising one if needed."""
        if self._descriptor is None:
            # Synthesise from legacy palette + global stats
            from core.guided_colorist import _profile_to_descriptor
            self._descriptor = _profile_to_descriptor(self)
        return self._descriptor


# ── StyleEngine ───────────────────────────────────────────────────────────────

class StyleEngine:
    """Extracts StyleDescriptors and manages an on-disk style library."""

    def __init__(self, styles_dir: str = "styles"):
        self.styles_dir = styles_dir
        os.makedirs(styles_dir, exist_ok=True)
        self._analyzer = StyleAnalyzer()

    # ── Extraction (new path — returns StyleDescriptor) ───────────────

    def extract_descriptor(self, color_bgr: np.ndarray,
                            name: str = "Extracted",
                            classifier=None) -> StyleDescriptor:
        """Analyze one reference image and return a StyleDescriptor."""
        return self._analyzer.analyze(color_bgr, name=name, classifier=classifier)

    def extract_descriptors(self, color_images: list,
                             name: str = "Extracted",
                             classifier=None,
                             weights: list | None = None) -> StyleDescriptor:
        """Analyze multiple reference images and merge into one StyleDescriptor."""
        return self._analyzer.analyze_many(
            color_images, name=name, classifier=classifier, weights=weights)

    # ── Extraction (legacy path — returns StyleProfile wrapping a descriptor) ─

    def extract_from_reference(self, color_bgr: np.ndarray,
                                name: str = "Extracted",
                                classifier=None) -> StyleProfile:
        """Legacy API — returns a StyleProfile that internally holds a
        StyleDescriptor so the hint pipeline works at full quality."""
        desc = self.extract_descriptor(color_bgr, name=name, classifier=classifier)
        profile = _descriptor_to_profile(desc)
        profile._descriptor = desc
        return profile

    def extract_from_references(self, color_images: list,
                                 name: str = "Extracted",
                                 classifier=None,
                                 weights: list | None = None) -> StyleProfile:
        """Legacy multi-image API."""
        desc = self.extract_descriptors(
            color_images, name=name, classifier=classifier, weights=weights)
        profile = _descriptor_to_profile(desc)
        profile._descriptor = desc
        return profile

    # ── Persistence ───────────────────────────────────────────────────

    def save_style(self, obj, filename: str | None = None) -> str:
        """Save a StyleDescriptor or StyleProfile (auto-detects type)."""
        if isinstance(obj, StyleDescriptor):
            filename = filename or f"{obj.name.lower().replace(' ', '_')}.ccstyle"
            return obj.save(os.path.join(self.styles_dir, filename))
        # StyleProfile — save it; it carries palette + global stats
        filename = filename or f"{obj.name.lower().replace(' ', '_')}.ccstyle"
        path = os.path.join(self.styles_dir, filename)
        # If there is an attached descriptor, save the full descriptor JSON
        desc = getattr(obj, "_descriptor", None)
        if desc is not None:
            return desc.save(path)
        return obj.save(path)

    def load_style(self, filename: str) -> StyleProfile:
        """Load a .ccstyle file; always returns a StyleProfile for backwards
        compat.  The StyleDescriptor (v2) is attached transparently."""
        path = filename if os.path.isabs(filename) else os.path.join(self.styles_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        version = data.get("version", 1)
        if version >= 2 and "global_warm_cool" in data:
            # v2 file — contains a full StyleDescriptor
            desc = StyleDescriptor._from_dict(data)
            profile = _descriptor_to_profile(desc)
            profile._descriptor = desc
            return profile
        else:
            # v1 legacy file — plain palette + global stats
            profile = StyleProfile.load(path)
            return profile

    def load_descriptor(self, filename: str) -> StyleDescriptor:
        """Load and return a StyleDescriptor directly (no legacy wrapper)."""
        profile = self.load_style(filename)
        return profile.get_descriptor()

    def list_styles(self) -> list:
        if not os.path.isdir(self.styles_dir):
            return []
        return sorted(f for f in os.listdir(self.styles_dir) if f.endswith(".ccstyle"))

    def list_styles_with_names(self) -> list[tuple[str, str]]:
        out = []
        for filename in self.list_styles():
            try:
                with open(os.path.join(self.styles_dir, filename),
                          "r", encoding="utf-8") as f:
                    data = json.load(f)
                name = data.get("name", os.path.splitext(filename)[0])
            except Exception:
                name = os.path.splitext(filename)[0]
            out.append((filename, name))
        return out

    def delete_style(self, filename: str) -> None:
        path = filename if os.path.isabs(filename) else os.path.join(self.styles_dir, filename)
        if os.path.exists(path):
            os.remove(path)

    def mix_profiles(self, a: StyleProfile, b: StyleProfile,
                     weight_b: float = 0.5, name: str = "Mixed") -> StyleProfile:
        """Linear-blend two profiles (global stats only; descriptors merged)."""
        w = float(np.clip(weight_b, 0.0, 1.0))
        palette = dict(a.palette)
        if w >= 0.5:
            palette.update(b.palette)
        return StyleProfile(
            name=name,
            palette=palette,
            saturation=a.saturation * (1 - w) + b.saturation * w,
            contrast=a.contrast * (1 - w) + b.contrast * w,
            shadow_strength=a.shadow_strength * (1 - w) + b.shadow_strength * w,
            gradient=a.gradient * (1 - w) + b.gradient * w,
            temperature=b.temperature if w >= 0.5 else a.temperature,
            description=f"Mix of {a.name} ({(1-w)*100:.0f}%) and {b.name} ({w*100:.0f}%)",
            source="mixed",
        )


# ── Private helpers ───────────────────────────────────────────────────────────

def _descriptor_to_profile(desc: StyleDescriptor) -> StyleProfile:
    """Build a legacy-compatible StyleProfile from a StyleDescriptor."""
    return StyleProfile(
        name=desc.name,
        palette=desc.palette,
        saturation=desc.saturation,
        contrast=desc.contrast,
        temperature=desc.temperature,
        shadow_strength=desc.shadow_strength,
        gradient=desc.gradient,
        description=desc.description,
        source=desc.source,
        version=_CCSTYLE_VERSION,
    )
