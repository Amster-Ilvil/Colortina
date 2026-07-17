"""Character Library — Character-aware Color Assignment (fully local).

Architecture: colors belong to CHARACTERS, not to region categories.

    彩页 → Character Extractor → Character Library
    黑白页 → Character Matcher → Character Memory → Region Assign → Hints

Previously the pipeline decided "hair = palette['hair']" — every head in
the book got the same hex, and a reference page could only warm/cool the
result, never change WHICH color a character's hair is.  This module
fixes the assignment layer:

  * ``extract_from_reference`` reads a color reference page, treats each
    hair region as a character anchor, and records that character's full
    attribute palette (hair / skin / eyes / clothing) by spatially
    associating nearby labeled regions with the anchor.
  * ``assign_page`` matches a black-and-white page's hair regions to
    known characters (by the same practical tone signal CharacterMemory
    uses — manga encodes character identity in deliberate screentone
    levels) and then propagates that character's OWN skin and clothing
    colors to the regions spatially belonging to that character.

StyleDirector stays responsible for RENDERING (highlight/shadow/warmth);
this module is responsible for ASSIGNMENT (which color).  No embeddings,
no networks, no APIs — everything runs on CPU in milliseconds.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

# Attribute labels a character "owns".  Regions of these classes that sit
# inside a character's body area take the character's colors instead of
# the global palette.
_CHARACTER_ATTRS = ("hair", "skin", "eyes", "clothing")

_MIN_CONF = 0.25


def _bgr_median_hex(color_bgr: np.ndarray, seg, region) -> str | None:
    m = seg.labels == region.label_id
    if not np.any(m):
        return None
    ys, xs = np.nonzero(m)
    ys = np.clip((ys / seg.scale).astype(int), 0, color_bgr.shape[0] - 1)
    xs = np.clip((xs / seg.scale).astype(int), 0, color_bgr.shape[1] - 1)
    b, g, r = np.median(color_bgr[ys, xs].astype(np.float32), axis=0)
    return "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))


@dataclass
class CharacterProfile:
    char_id: int
    hair_tone: float                    # grayscale anchor for matching
    colors: dict = field(default_factory=dict)   # attr -> "#rrggbb"
    name: str = ""
    hits: int = 1


class CharacterLibrary:
    """Book-level character database with tone-anchored matching."""

    def __init__(self, max_characters: int = 8, tone_tolerance: float = 34.0):
        self.characters: list[CharacterProfile] = []
        self.max_characters = max_characters
        self.tone_tolerance = tone_tolerance

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, path: str) -> str:
        data = {"max_characters": self.max_characters,
                "tone_tolerance": self.tone_tolerance,
                "characters": [asdict(c) for c in self.characters]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "CharacterLibrary":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        lib = cls(max_characters=data.get("max_characters", 8),
                  tone_tolerance=data.get("tone_tolerance", 34.0))
        lib.characters = [CharacterProfile(**c) for c in data.get("characters", [])]
        return lib

    # ── Character Extractor (runs once per reference page) ───────────

    def extract_from_reference(self, color_bgr: np.ndarray, classifier=None) -> int:
        """Build character profiles from one color reference page.

        Each hair region anchors one character.  Skin / eyes / clothing
        regions are attached to the nearest anchor whose body area
        (hair bbox grown downward ~3.5x its height) contains their
        centroid.  Returns how many characters were added.
        """
        from core.region_segmenter import segment_regions

        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        seg = segment_regions(gray)
        if not seg.regions:
            return 0

        labels = None
        if classifier is not None and classifier.available:
            labels = classifier.classify(color_bgr, [r.bbox for r in seg.regions])
        if not labels:
            return 0

        tagged = [(r, lab) for r, (lab, conf) in zip(seg.regions, labels)
                  if conf >= _MIN_CONF and lab in _CHARACTER_ATTRS]
        anchors = [(r, lab) for r, lab in tagged if lab == "hair"]
        if not anchors:
            return 0

        added = 0
        for hair_region, _ in anchors:
            if len(self.characters) >= self.max_characters:
                break
            hair_hex = _bgr_median_hex(color_bgr, seg, hair_region)
            if hair_hex is None:
                continue
            hx, hy, hw, hh = hair_region.bbox
            # Character body area: hair bbox widened and grown downward.
            body = (hx - hw * 0.4, hy, hw * 1.8, hh * 4.5)

            colors = {"hair": hair_hex}
            for region, lab in tagged:
                if lab == "hair" or lab in colors:
                    continue
                cx = region.bbox[0] + region.bbox[2] / 2
                cy = region.bbox[1] + region.bbox[3] / 2
                bx, by, bw, bh = body
                if bx <= cx <= bx + bw and by <= cy <= by + bh:
                    hex_c = _bgr_median_hex(color_bgr, seg, region)
                    if hex_c:
                        colors[lab] = hex_c

            self.characters.append(CharacterProfile(
                char_id=len(self.characters),
                hair_tone=float(hair_region.mean_gray),
                colors=colors))
            added += 1
        return added

    # ── Character Matcher + Region Assign (every B&W page) ───────────

    def assign_page(self, regions: list, labels: list) -> dict:
        """Assign character-owned colors to one page's regions.

        `regions` / `labels` are the page's full segmentation output
        (region list and matching [(label, conf), ...]).  Returns
        {region.label_id: (r, g, b)} for every region that belongs to a
        matched character — hair AND that character's skin / clothing.
        """
        if not self.characters or not regions:
            return {}

        tagged = [(r, lab) for r, (lab, conf) in zip(regions, labels)
                  if conf >= _MIN_CONF and lab in _CHARACTER_ATTRS]
        hair_regions = [r for r, lab in tagged if lab == "hair"]
        if not hair_regions:
            return {}

        # Greedy nearest-tone matching, darkest hair first (mirrors
        # CharacterMemory.assign so the two systems agree).
        out: dict = {}
        used: set = set()
        for hair in sorted(hair_regions, key=lambda r: r.mean_gray):
            best, best_d = None, self.tone_tolerance
            for ci, ch in enumerate(self.characters):
                if ci in used:
                    continue
                d = abs(ch.hair_tone - hair.mean_gray)
                if d < best_d:
                    best_d, best = d, ci
            if best is None:
                continue
            used.add(best)
            ch = self.characters[best]

            def _rgb(hex_c: str) -> tuple[int, int, int]:
                s = hex_c.lstrip("#")
                return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))

            out[hair.label_id] = _rgb(ch.colors["hair"])

            hx, hy, hw, hh = hair.bbox
            bx, by = hx - hw * 0.4, hy
            bw, bh = hw * 1.8, hh * 4.5
            for region, lab in tagged:
                if lab == "hair" or lab not in ch.colors:
                    continue
                if region.label_id in out:
                    continue
                cx = region.bbox[0] + region.bbox[2] / 2
                cy = region.bbox[1] + region.bbox[3] / 2
                if bx <= cx <= bx + bw and by <= cy <= by + bh:
                    out[region.label_id] = _rgb(ch.colors[lab])
        return out
