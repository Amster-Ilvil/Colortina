"""Character Memory — keeps distinct region instances (typically hair,
optionally skin) visually consistent across an entire book.

The rest of the pipeline (``ColorDirector`` / ``GuidedColorist``)
produces ONE color per semantic label ("hair" -> one hex) for the whole
job. That's correct for backgrounds and props, but wrong the moment two
characters have different hair colors: every head in the book gets
painted the same shade.

This does NOT attempt face recognition or cross-page identity tracking —
both are fragile across manga's wide range of art styles (Q-version,
realistic, extreme angles). Instead it uses the same practical signal
manga art itself encodes: hair (and to a lesser extent skin) is drawn
with a deliberate, distinct TONE level per character even in
black-and-white — screentone density and line weight say "this is
character A's hair, that's character B's" long before any color exists.

  1. Rank same-label region instances on one page by mean tone (gray).
  2. Match each instance to the nearest known "slot" (a persistent tone
     anchor -> hex color) by tone proximity.
  3. Unmatched instances spawn a new slot (up to `max_slots`), taking
     the next color off a rotating fallback palette.
  4. Slots persist for the whole book (save/load JSON), so "the
     character with tone ~210" gets the same hex on page 1 and page 400.

Known limitation (documented deliberately, not glossed over): tone alone
cannot separate two characters whose hair is a similar BRIGHTNESS but
different HUE (e.g. isekai pink vs. green hair, which can convert to
near-identical grayscale). ``seed_from_reference`` is the fix for that
case — when a color reference page exists, real hues bind directly to
slots instead of the rotating fallback palette. Without a reference,
same-tone/different-hue characters will collapse onto one slot; a user
can still split them by editing the offending page's hint manually.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

# Rotating fallback palette for slots with no reference color — spans
# hues an colorist reaches for most often, ordered so neighboring slots
# don't land on near-identical hues.
_FALLBACK_COLORS = [
    "#2c2c2c", "#f6c945", "#5b8fd6", "#e8746b", "#7fbf6a",
    "#c07fd6", "#e89b4a", "#4ac0c0", "#d6567f", "#8a6a48",
]


@dataclass
class CharacterSlot:
    slot_id: int
    tone: float             # running mean grayscale (0-255) of matched instances
    color_hex: str
    name: str = ""           # user-assigned name (e.g. "鸣人") — empty = unnamed
    locked: bool = False     # True once bound to a real reference color or set
                             # by the user; tone no longer drifts via EMA
    hits: int = 1            # instances matched so far (for the tone EMA)


@dataclass
class CharacterMemory:
    """Per-label slot bank, persisted as one JSON file per book/project."""

    label: str = "hair"
    slots: list = field(default_factory=list)
    max_slots: int = 8
    tone_tolerance: float = 28.0  # gray distance to still count as "the same slot"

    # ── Persistence ──────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {"label": self.label, "max_slots": self.max_slots,
                "tone_tolerance": self.tone_tolerance,
                "slots": [asdict(s) for s in self.slots]}

    @classmethod
    def from_dict(cls, data: dict) -> "CharacterMemory":
        mem = cls(label=data.get("label", "hair"),
                  max_slots=data.get("max_slots", 8),
                  tone_tolerance=data.get("tone_tolerance", 28.0))
        mem.slots = [CharacterSlot(**s) for s in data.get("slots", [])]
        return mem

    def save(self, path: str) -> str:
        out_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(out_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, path: str) -> "CharacterMemory":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # ── Seeding from a color reference ────────────────────────────────

    def seed_from_reference(self, color_bgr: np.ndarray, classifier=None) -> int:
        """Populate slots from real colors in a color reference page.

        Segments + labels the reference exactly like the B&W pipeline
        does, keeps only regions matching `self.label`, ranks them by
        luminance (brightest first) and creates one slot per instance
        (capped at `max_slots`), bound and locked to the real color.
        Returns how many slots were seeded.
        """
        from core.region_segmenter import segment_regions

        self.slots = []
        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        seg = segment_regions(gray)
        if not seg.regions:
            return 0

        candidates = seg.regions
        if classifier is not None and classifier.available:
            labels = classifier.classify(color_bgr, [r.bbox for r in seg.regions])
            if labels:
                candidates = [r for r, (lab, conf) in zip(seg.regions, labels)
                             if lab == self.label and conf >= 0.25]

        instances = []
        for region in candidates:
            m = seg.labels == region.label_id
            if not np.any(m):
                continue
            ys, xs = np.nonzero(m)
            ys = np.clip((ys / seg.scale).astype(int), 0, color_bgr.shape[0] - 1)
            xs = np.clip((xs / seg.scale).astype(int), 0, color_bgr.shape[1] - 1)
            pixels = color_bgr[ys, xs].astype(np.float32)
            bgr = np.median(pixels, axis=0)
            luminance = 0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2]
            instances.append((luminance, bgr))

        instances.sort(key=lambda t: -t[0])  # brightest first
        for i, (luminance, bgr) in enumerate(instances[:self.max_slots]):
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(bgr[2]), int(bgr[1]), int(bgr[0]))
            self.slots.append(CharacterSlot(
                slot_id=i, tone=float(luminance), color_hex=hex_color, locked=True))
        return len(self.slots)

    # ── Per-page assignment ──────────────────────────────────────────

    def assign(self, regions: list, gray_page: np.ndarray) -> dict:
        """Match this page's regions (already filtered to `self.label`)
        to slots. Returns {region.label_id: hex_color}.

        Regions are matched nearest-tone-first (greedy) so two same-page
        instances don't collide on one slot when they don't have to.
        `gray_page` is accepted for API symmetry / future use (per-region
        tone already lives on `region.mean_gray`).
        """
        if not regions:
            return {}

        order = sorted(range(len(regions)), key=lambda i: regions[i].mean_gray)
        used_slots: set = set()
        out: dict = {}

        for i in order:
            region = regions[i]
            tone = region.mean_gray
            best = None
            best_dist = self.tone_tolerance
            for si, slot in enumerate(self.slots):
                if si in used_slots:
                    continue
                d = abs(slot.tone - tone)
                if d < best_dist:
                    best_dist = d
                    best = si

            if best is not None:
                slot = self.slots[best]
                used_slots.add(best)
                out[region.label_id] = slot.color_hex
                if not slot.locked:
                    # EMA toward this page's tone — slow drift (scan
                    # quality, printing) doesn't fragment the slot.
                    slot.tone = (slot.tone * slot.hits + tone) / (slot.hits + 1)
                    slot.hits += 1
            elif len(self.slots) < self.max_slots:
                color = _FALLBACK_COLORS[len(self.slots) % len(_FALLBACK_COLORS)]
                slot = CharacterSlot(slot_id=len(self.slots), tone=tone, color_hex=color)
                self.slots.append(slot)
                used_slots.add(len(self.slots) - 1)
                out[region.label_id] = color
            else:
                # Bank is full — use the closest slot rather than leave
                # the region uncolored.
                closest = min(range(len(self.slots)),
                             key=lambda si: abs(self.slots[si].tone - tone))
                out[region.label_id] = self.slots[closest].color_hex

        return out

    # ── Manual override ────────────────────────────────────────────

    def set_slot_color(self, slot_id: int, hex_color: str, name: str = "") -> None:
        for slot in self.slots:
            if slot.slot_id == slot_id:
                slot.color_hex = hex_color
                slot.locked = True
                if name:
                    slot.name = name
                return
        raise ValueError(f"no slot with id {slot_id}")
