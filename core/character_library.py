"""Book-level character colour library with persistent identity matching.

Reference pages create canonical character profiles.  Profiles from multiple
reference pages are merged instead of being appended as duplicates.  Target
pages match each character instance independently (the same character may
appear in several panels) using CLIP appearance embeddings plus manga tone and
shape features.  The detailed assignments are reused by both hint generation
and deterministic post-colour locking.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field

from core.character_instance import CharacterInstance
from core.character_matcher import decide_match

import cv2
import numpy as np

_CHARACTER_ATTRS = ("hair", "skin", "eyes", "clothing")
_MIN_CONF = 0.20


def _rgb(hex_c: str) -> tuple[int, int, int]:
    s = hex_c.lstrip("#")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def _rgb_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*[int(np.clip(v, 0, 255)) for v in rgb])


def _blend_hex(a: str, b: str, weight_b: float = 0.35) -> str:
    """Blend identity colours in LAB so lighting variants converge stably."""
    try:
        aa = np.array([[[_rgb(a)[2], _rgb(a)[1], _rgb(a)[0]]]], dtype=np.uint8)
        bb = np.array([[[_rgb(b)[2], _rgb(b)[1], _rgb(b)[0]]]], dtype=np.uint8)
        la = cv2.cvtColor(aa, cv2.COLOR_BGR2LAB).astype(np.float32)
        lb = cv2.cvtColor(bb, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab = la * (1.0 - weight_b) + lb * weight_b
        out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)[0, 0]
        return _rgb_hex((int(out[2]), int(out[1]), int(out[0])))
    except Exception:
        return b


def _hex_lab(value: str) -> np.ndarray | None:
    try:
        r, g, b = _rgb(value)
        px = np.array([[[b, g, r]]], dtype=np.uint8)
        return cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    except Exception:
        return None


def _hex_distance(a: str | None, b: str | None) -> float | None:
    if not a or not b:
        return None
    la = _hex_lab(a)
    lb = _hex_lab(b)
    if la is None or lb is None:
        return None
    return float(np.linalg.norm(la - lb))


def _color_compatibility(candidate: dict, existing: dict) -> float:
    """Average LAB distance for overlapping identity colours."""
    ds = []
    for key in ("hair", "eyes", "skin"):
        d = _hex_distance(candidate.get(key), existing.get(key))
        if d is not None:
            ds.append(d)
    if not ds:
        return 0.0
    return float(sum(ds) / len(ds))


def _normalize_slot_palette(value, *, max_slots: int = 3) -> list[str]:
    out: list[str] = []
    for item in (value or []):
        if not isinstance(item, str):
            continue
        item = item.strip().lower()
        if not (len(item) == 7 and item.startswith("#")):
            continue
        if item not in out:
            out.append(item)
        if len(out) >= max_slots:
            break
    return out


def _merge_slot_palette(existing, candidate, *, max_slots: int = 3) -> list[str]:
    slots = _normalize_slot_palette(existing, max_slots=max_slots)
    if isinstance(candidate, str):
        candidates = _normalize_slot_palette([candidate], max_slots=max_slots)
    else:
        candidates = _normalize_slot_palette(candidate, max_slots=max_slots)
    for color in candidates:
        if color in slots:
            continue
        nearest = None
        nearest_d = 1e9
        for idx, slot in enumerate(slots):
            d = _hex_distance(slot, color)
            if d is not None and d < nearest_d:
                nearest_d = d
                nearest = idx
        if nearest is not None and nearest_d <= 12.0:
            slots[nearest] = _blend_hex(slots[nearest], color, 0.35)
        elif len(slots) < max_slots:
            slots.append(color)
        else:
            # Keep the farthest slot family representative stable by replacing
            # only when the new color is closer to one existing slot than that
            # slot is to the palette centre.
            centre = _normalize_slot_palette(slots[:1], max_slots=1)
            if nearest is not None and centre:
                centre_d = _hex_distance(slots[nearest], centre[0]) or 999.0
                if nearest_d < centre_d:
                    slots[nearest] = _blend_hex(slots[nearest], color, 0.45)
    return _normalize_slot_palette(slots, max_slots=max_slots)


def _init_color_slots(colors: dict) -> dict:
    out: dict[str, list[str]] = {}
    for attr, value in (colors or {}).items():
        if attr in _CHARACTER_ATTRS and isinstance(value, str):
            out[attr] = _normalize_slot_palette([value])
    return out


def _classify_clothing_part(region, body_bbox: tuple[int, int, int, int] | None,
                            page_shape: tuple[int, int] | None = None) -> str:
    """Classify a clothing region as upper/lower/accessory using stable geometry.

    This is intentionally lightweight: the goal is not semantic fashion
    understanding, but a repeatable slot assignment across pages.
    """
    x, y, w, h = region.bbox
    cx = x + w / 2.0
    cy = y + h / 2.0
    if body_bbox is None:
        # Small, narrow or tiny regions are more likely ties, ribbons, cuffs or
        # accessories than the main garment body.
        if float(getattr(region, "frac", 0.0)) < 0.006 or min(w, h) <= 8:
            return "accessory"
        return "upper"
    bx, by, bw, bh = body_bbox
    rel_x = (cx - bx) / max(1.0, float(bw))
    rel_y = (cy - by) / max(1.0, float(bh))
    area_ratio = (w * h) / max(1.0, float(bw * bh))
    aspect = w / max(1.0, float(h))
    # Accessories tend to be small or thin and near the centre/head/edges.
    if (area_ratio < 0.035 or min(w, h) <= 8 or aspect > 4.2 or aspect < 0.23):
        if 0.12 <= rel_x <= 0.88 and 0.08 <= rel_y <= 0.82:
            return "accessory"
    return "upper" if rel_y < 0.56 else "lower"


def _preferred_slot_index(part: str, slots: list | tuple | None) -> int:
    count = len(slots or [])
    if count <= 1:
        return 0
    if part == "lower":
        return 1 if count >= 2 else 0
    if part == "accessory":
        return 2 if count >= 3 else (1 if count >= 2 else 0)
    return 0


def _assignment_target_rgb(info: dict) -> tuple[int, int, int] | None:
    """Resolve the canonical RGB for diagnostics / drift measurement."""
    slots = list(info.get("slot_rgbs") or [])
    if slots:
        idx = int(info.get("preferred_slot_index", 0) or 0)
        idx = min(max(idx, 0), len(slots) - 1)
        try:
            return tuple(int(v) for v in slots[idx])
        except Exception:
            pass
    rgb = info.get("rgb")
    if rgb is None:
        return None
    try:
        return tuple(int(v) for v in rgb)
    except Exception:
        return None


def _region_median_rgb(result_bgr: np.ndarray, labels: np.ndarray, region_id: int) -> tuple[int, int, int] | None:
    mask = labels == int(region_id)
    if not np.any(mask):
        return None
    pixels = result_bgr[mask]
    if pixels.size == 0:
        return None
    b, g, r = np.median(pixels.astype(np.float32), axis=0)
    return int(r), int(g), int(b)


def _bgr_median_hex(color_bgr: np.ndarray, seg, region) -> str | None:
    m = seg.labels == region.label_id
    if not np.any(m):
        return None
    ys, xs = np.nonzero(m)
    ys = np.clip((ys / seg.scale).astype(int), 0, color_bgr.shape[0] - 1)
    xs = np.clip((xs / seg.scale).astype(int), 0, color_bgr.shape[1] - 1)
    pixels = color_bgr[ys, xs].astype(np.uint8)
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    useful = pixels[(hsv[:, 1] > 12) & (hsv[:, 2] > 18) & (hsv[:, 2] < 248)]
    if len(useful) >= 8:
        pixels = useful
    if not len(pixels):
        return None
    b, g, r = np.median(pixels.astype(np.float32), axis=0)
    return _rgb_hex((int(r), int(g), int(b)))


def _region_pixels(gray_page: np.ndarray, seg, region) -> np.ndarray:
    m = seg.labels == region.label_id
    if not np.any(m):
        return np.empty(0, dtype=np.uint8)
    ys, xs = np.nonzero(m)
    ys = np.clip((ys / seg.scale).astype(int), 0, gray_page.shape[0] - 1)
    xs = np.clip((xs / seg.scale).astype(int), 0, gray_page.shape[1] - 1)
    return gray_page[ys, xs].astype(np.uint8)


def _hair_features(gray_page: np.ndarray, seg, region) -> tuple[list[float], float, float]:
    pixels = _region_pixels(gray_page, seg, region)
    if pixels.size:
        hist, _ = np.histogram(pixels, bins=12, range=(0, 256))
        hist = hist.astype(np.float32)
        hist /= max(1.0, float(hist.sum()))
        hist_list = hist.tolist()
    else:
        hist_list = []
    _x, _y, w, h = region.bbox
    return hist_list, float(w / max(1.0, h)), float(region.frac)


def _hist_distance(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.5
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    return float(np.sqrt(max(0.0, 1.0 - float(np.sum(np.sqrt(aa * bb))))))


def _cosine_distance(a: list[float], b: list[float]) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-8:
        return None
    return float(np.clip(1.0 - float(np.dot(aa, bb) / denom), 0.0, 2.0) / 2.0)


def _body_bbox(region, page_shape: tuple[int, int] | None = None) -> tuple[int, int, int, int]:
    x, y, w, h = region.bbox
    bx, by = int(x - w * 0.65), int(y - h * 0.20)
    bw, bh = int(w * 2.30), int(h * 5.20)
    if page_shape is not None:
        H, W = page_shape
        bx, by = max(0, bx), max(0, by)
        bw, bh = min(W - bx, bw), min(H - by, bh)
    return bx, by, max(1, bw), max(1, bh)


def _head_bbox(region, page_shape: tuple[int, int] | None = None) -> tuple[int, int, int, int]:
    x, y, w, h = region.bbox
    bx, by = int(x - w * 0.55), int(y - h * 0.45)
    bw, bh = int(w * 2.10), int(h * 2.10)
    if page_shape is not None:
        H, W = page_shape
        bx, by = max(0, bx), max(0, by)
        bw, bh = min(W - bx, bw), min(H - by, bh)
    return bx, by, max(1, bw), max(1, bh)


def _embedding_average(a: list[float], b: list[float], weight_b: float) -> list[float]:
    if not a:
        return list(b)
    if not b or len(a) != len(b):
        return list(a)
    v = np.asarray(a, dtype=np.float32) * (1.0 - weight_b) + np.asarray(b, dtype=np.float32) * weight_b
    norm = float(np.linalg.norm(v))
    if norm > 1e-8:
        v /= norm
    return v.astype(float).tolist()


@dataclass
class CharacterProfile:
    char_id: int
    hair_tone: float
    colors: dict = field(default_factory=dict)
    name: str = ""
    hits: int = 1
    hair_hist: list[float] = field(default_factory=list)
    hair_aspect: float = 1.0
    hair_area_frac: float = 0.0
    appearance_embedding: list[float] = field(default_factory=list)
    # Dependency-free grayscale/edge descriptor used when CLIP is unavailable
    # or a page has no reliable semantic hair anchor.
    lineart_embedding: list[float] = field(default_factory=list)
    reference_samples: int = 1
    manual: bool = False
    color_slots: dict = field(default_factory=dict)


class CharacterLibrary:
    def __init__(self, max_characters: int = 24, tone_tolerance: float = 40.0,
                 match_threshold: float = 0.58, merge_threshold: float = 0.18,
                 min_match_score: float = 0.48, min_margin: float = 0.055,
                 min_semantic_conf: float = 0.32):
        self.characters: list[CharacterProfile] = []
        self.max_characters = max_characters
        self.tone_tolerance = tone_tolerance
        self.match_threshold = match_threshold
        self.merge_threshold = merge_threshold
        self.min_match_score = min_match_score
        self.min_margin = min_margin
        self.min_semantic_conf = min_semantic_conf
        self.revision = 0
        # Kept only for old UI compatibility.  v4 pipeline consumes explicit
        # PageColorContext assignments instead of this hidden mutable state.
        self.last_assignments: dict[int, dict] = {}
        self.last_instances: list[CharacterInstance] = []
        self._last_segmentation = None
        self._last_source_shape = None

    def to_dict(self) -> dict:
        return {
            "version": 5,
            "max_characters": self.max_characters,
            "tone_tolerance": self.tone_tolerance,
            "match_threshold": self.match_threshold,
            "merge_threshold": self.merge_threshold,
            "min_match_score": self.min_match_score,
            "min_margin": self.min_margin,
            "min_semantic_conf": self.min_semantic_conf,
            "revision": self.revision,
            "characters": [asdict(c) for c in self.characters],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CharacterLibrary":
        from core.schema_migration import migrate_ccpalette
        data, _notes = migrate_ccpalette(data)
        lib = cls(max_characters=data.get("max_characters", 24),
                  tone_tolerance=data.get("tone_tolerance", 40.0),
                  match_threshold=data.get("match_threshold", 0.58),
                  merge_threshold=data.get("merge_threshold", 0.18),
                  min_match_score=data.get("min_match_score", 0.48),
                  min_margin=data.get("min_margin", 0.055),
                  min_semantic_conf=data.get("min_semantic_conf", 0.32))
        lib.revision = int(data.get("revision", 0))
        known = CharacterProfile.__dataclass_fields__
        lib.characters = [CharacterProfile(**{
            k: v for k, v in item.items() if k in known
        }) for item in data.get("characters", [])]
        for ch in lib.characters:
            if not getattr(ch, "color_slots", None):
                ch.color_slots = _init_color_slots(ch.colors)
            else:
                ch.color_slots = {k: _normalize_slot_palette(v)
                                  for k, v in dict(ch.color_slots).items() if k in _CHARACTER_ATTRS}
        return lib

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "CharacterLibrary":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def _feature_distance(self, tone: float, hist: list[float], aspect: float,
                          area_frac: float, embedding: list[float],
                          lineart_embedding: list[float],
                          character: CharacterProfile) -> float:
        tone_d = min(1.0, abs(character.hair_tone - tone) / 100.0)
        hist_d = _hist_distance(character.hair_hist, hist)
        aspect_d = min(1.0, abs(math.log(
            max(aspect, 1e-3) / max(character.hair_aspect, 1e-3))) / 1.7)
        area_d = min(1.0, abs(math.log(
            max(area_frac, 1e-6) / max(character.hair_area_frac, 1e-6))) / 3.2)
        appearance_d = _cosine_distance(character.appearance_embedding, embedding)
        lineart_d = _cosine_distance(character.lineart_embedding, lineart_embedding)
        if appearance_d is not None and lineart_d is not None:
            return (0.42 * appearance_d + 0.28 * lineart_d +
                    0.10 * tone_d + 0.10 * hist_d +
                    0.06 * aspect_d + 0.04 * area_d)
        if appearance_d is not None:
            return (0.56 * appearance_d + 0.14 * tone_d + 0.15 * hist_d +
                    0.10 * aspect_d + 0.05 * area_d)
        if lineart_d is not None:
            return (0.55 * lineart_d + 0.16 * tone_d + 0.16 * hist_d +
                    0.08 * aspect_d + 0.05 * area_d)
        return 0.30 * tone_d + 0.35 * hist_d + 0.22 * aspect_d + 0.13 * area_d

    def _upsert_profile(self, candidate: CharacterProfile) -> bool:
        best = None
        best_d = 999.0
        for ch in self.characters:
            d = self._feature_distance(
                candidate.hair_tone, candidate.hair_hist, candidate.hair_aspect,
                candidate.hair_area_frac, candidate.appearance_embedding,
                candidate.lineart_embedding, ch)
            if d < best_d:
                best_d, best = d, ch
        if best is not None and best_d <= self.merge_threshold:
            color_d = _color_compatibility(candidate.colors, best.colors)
            hair_d = _hex_distance(candidate.colors.get("hair"), best.colors.get("hair"))
            if (hair_d is None or hair_d <= 18.0) and color_d <= 26.0:
                n = max(1, best.reference_samples)
                w = 1.0 / (n + 1.0)
                best.hair_tone = best.hair_tone * (1.0 - w) + candidate.hair_tone * w
                best.hair_aspect = best.hair_aspect * (1.0 - w) + candidate.hair_aspect * w
                best.hair_area_frac = best.hair_area_frac * (1.0 - w) + candidate.hair_area_frac * w
                if best.hair_hist and candidate.hair_hist and len(best.hair_hist) == len(candidate.hair_hist):
                    h = np.asarray(best.hair_hist) * (1.0 - w) + np.asarray(candidate.hair_hist) * w
                    best.hair_hist = (h / max(float(h.sum()), 1e-8)).astype(float).tolist()
                best.appearance_embedding = _embedding_average(
                    best.appearance_embedding, candidate.appearance_embedding, w)
                best.lineart_embedding = _embedding_average(
                    best.lineart_embedding, candidate.lineart_embedding, w)
                for attr, color in candidate.colors.items():
                    best.colors[attr] = (_blend_hex(best.colors[attr], color, min(0.4, w + 0.15))
                                         if attr in best.colors else color)
                    best.color_slots[attr] = _merge_slot_palette(
                        best.color_slots.get(attr),
                        candidate.color_slots.get(attr, [color]) if getattr(candidate, "color_slots", None) else [color])
                best.reference_samples += 1
                self.revision += 1
                return False
        if len(self.characters) >= self.max_characters:
            return False
        candidate.char_id = max((c.char_id for c in self.characters), default=-1) + 1
        if not getattr(candidate, "color_slots", None):
            candidate.color_slots = _init_color_slots(candidate.colors)
        self.characters.append(candidate)
        self.revision += 1
        return True

    def add_manual_reference(self, color_bgr: np.ndarray,
                             head_bbox: tuple[int, int, int, int], *,
                             colors: dict[str, str], name: str = "",
                             rotation: int = 0, classifier=None,
                             merge_same_name: bool = True) -> CharacterProfile:
        """Create/update one explicit character identity from a selected head.

        Manual enrolment is the safe path for complex covers and rotated or
        occluded characters.  It never merges different unnamed characters by
        visual similarity.  A profile is merged only when the user supplies
        the same non-empty name and ``merge_same_name`` is enabled.
        """
        if color_bgr is None or color_bgr.size == 0:
            raise ValueError("reference image is empty")
        clean_colors = {}
        for key, value in (colors or {}).items():
            if key not in _CHARACTER_ATTRS or not isinstance(value, str):
                continue
            value = value.strip().lower()
            if len(value) == 7 and value.startswith("#"):
                try:
                    int(value[1:], 16)
                except ValueError:
                    continue
                clean_colors[key] = value
        if "hair" not in clean_colors:
            raise ValueError("manual character reference requires a hair colour")

        from core.manual_reference import clip_bbox, manual_head_features, rotate_crop
        h, w = color_bgr.shape[:2]
        bbox = clip_bbox(head_bbox, (h, w))
        tone, hist, aspect, area_frac, lineart = manual_head_features(
            color_bgr, bbox, rotation=rotation)

        appearance = []
        if classifier is not None and getattr(classifier, "available", False):
            try:
                x, y, bw, bh = bbox
                crop = rotate_crop(color_bgr[y:y + bh, x:x + bw], rotation)
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                embeds = classifier.embed_bboxes(
                    gray_bgr, [(0, 0, gray_bgr.shape[1], gray_bgr.shape[0])])
                if embeds is not None and len(embeds):
                    appearance = embeds[0].astype(float).tolist()
            except Exception:
                appearance = []

        candidate = CharacterProfile(
            char_id=-1, hair_tone=tone, colors=clean_colors,
            name=str(name or "").strip(), hair_hist=hist,
            hair_aspect=aspect, hair_area_frac=area_frac,
            appearance_embedding=appearance, lineart_embedding=lineart,
            reference_samples=1, manual=True,
            color_slots=_init_color_slots(clean_colors))

        existing = None
        if merge_same_name and candidate.name:
            folded = candidate.name.casefold()
            existing = next((ch for ch in self.characters
                             if ch.name.strip().casefold() == folded), None)
        if existing is None:
            if len(self.characters) >= self.max_characters:
                raise ValueError("character library is full")
            candidate.char_id = max((c.char_id for c in self.characters),
                                    default=-1) + 1
            self.characters.append(candidate)
            self.revision += 1
            return candidate

        n = max(1, existing.reference_samples)
        weight = 1.0 / (n + 1.0)
        existing.hair_tone = existing.hair_tone * (1.0 - weight) + tone * weight
        existing.hair_aspect = existing.hair_aspect * (1.0 - weight) + aspect * weight
        existing.hair_area_frac = (existing.hair_area_frac * (1.0 - weight) +
                                   area_frac * weight)
        if existing.hair_hist and len(existing.hair_hist) == len(hist):
            mixed = (np.asarray(existing.hair_hist, dtype=np.float32) * (1.0 - weight) +
                     np.asarray(hist, dtype=np.float32) * weight)
            existing.hair_hist = (mixed / max(float(mixed.sum()), 1e-8)).tolist()
        else:
            existing.hair_hist = list(hist)
        existing.lineart_embedding = _embedding_average(
            existing.lineart_embedding, lineart, weight)
        existing.appearance_embedding = _embedding_average(
            existing.appearance_embedding, appearance, weight)
        for attr, value in clean_colors.items():
            existing.colors[attr] = (_blend_hex(existing.colors[attr], value,
                                                 min(0.38, weight + 0.12))
                                     if attr in existing.colors else value)
            existing.color_slots[attr] = _merge_slot_palette(existing.color_slots.get(attr), value)
        existing.reference_samples += 1
        existing.manual = True
        self.revision += 1
        return existing

    def extract_from_reference(self, color_bgr: np.ndarray, classifier=None) -> int:
        from core.region_segmenter import segment_regions

        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        seg = segment_regions(gray)
        classifier_available = bool(classifier is not None and classifier.available)
        if not seg.regions or not classifier_available:
            return self._extract_from_reference_faces(color_bgr, classifier=classifier)
        semantic_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        labels = classifier.classify(semantic_bgr, [r.bbox for r in seg.regions])
        if not labels:
            return self._extract_from_reference_faces(color_bgr, classifier=classifier)
        tagged = [(r, lab, conf) for r, (lab, conf) in zip(seg.regions, labels)
                  if conf >= _MIN_CONF and lab in _CHARACTER_ATTRS]
        anchors = [r for r, lab, _conf in tagged if lab == "hair"]
        if not anchors:
            return self._extract_from_reference_faces(color_bgr, classifier=classifier)
        body_boxes = [_body_bbox(r, color_bgr.shape[:2]) for r in anchors]
        head_boxes = [_head_bbox(r, color_bgr.shape[:2]) for r in anchors]
        embeddings = classifier.embed_bboxes(semantic_bgr, head_boxes)

        added = 0
        for index, hair_region in enumerate(anchors):
            hair_hex = _bgr_median_hex(color_bgr, seg, hair_region)
            if hair_hex is None:
                continue
            bx, by, bw, bh = body_boxes[index]
            colors = {"hair": hair_hex}
            # Multiple eye/skin fragments are common.  Use the largest reliable
            # component of each attribute as the canonical identity colour.
            best_attr: dict[str, tuple[float, str]] = {}
            for region, lab, conf in tagged:
                if lab == "hair":
                    continue
                cx = region.bbox[0] + region.bbox[2] / 2
                cy = region.bbox[1] + region.bbox[3] / 2
                if bx <= cx <= bx + bw and by <= cy <= by + bh:
                    value = _bgr_median_hex(color_bgr, seg, region)
                    score = float(region.area) * max(0.1, conf)
                    if value and (lab not in best_attr or score > best_attr[lab][0]):
                        best_attr[lab] = (score, value)
            colors.update({lab: value for lab, (_score, value) in best_attr.items()})
            # Tiny iris regions often escape CLIP; supplement them with
            # head-local geometric/color sampling without overwriting a more
            # confident semantic sample.
            try:
                from core.face_parts import sample_face_palette
                fallback = sample_face_palette(
                    color_bgr, seg, hair_region, head_boxes[index])
                for attr, value in fallback.items():
                    colors.setdefault(attr, value)
            except Exception:
                pass
            hist, aspect, area_frac = _hair_features(gray, seg, hair_region)
            embedding = (embeddings[index].astype(float).tolist()
                         if embeddings is not None and index < len(embeddings) else [])
            from core.anime_face_detector import lineart_descriptor
            lineart = lineart_descriptor(semantic_bgr, head_boxes[index])
            candidate = CharacterProfile(
                char_id=-1, hair_tone=float(hair_region.mean_gray), colors=colors,
                hair_hist=hist, hair_aspect=aspect, hair_area_frac=area_frac,
                appearance_embedding=embedding, lineart_embedding=lineart)
            if self._upsert_profile(candidate):
                added += 1
        return added

    def _extract_from_reference_faces(self, color_bgr: np.ndarray,
                                      classifier=None) -> int:
        """Fallback identity enrolment from detected face/head crops.

        This path is especially useful for dense colour covers where connected
        line regions do not produce reliable semantic hair anchors.
        """
        from core.anime_face_detector import (
            detect_anime_faces, hair_tone_features, lineart_descriptor,
            sample_reference_face_palette)

        faces = detect_anime_faces(color_bgr)
        if not faces:
            return 0
        embeddings = None
        if classifier is not None and getattr(classifier, "available", False):
            try:
                gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
                gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                embeddings = classifier.embed_bboxes(
                    gray_bgr, [face.head_bbox for face in faces])
            except Exception:
                embeddings = None

        added = 0
        for index, face in enumerate(faces):
            colors = sample_reference_face_palette(color_bgr, face)
            # Hair is the minimum identity colour; a plausible skin sample
            # filters most text/logo false positives on colour covers.
            if "hair" not in colors or "skin" not in colors:
                continue
            tone, hist, aspect, area_frac = hair_tone_features(color_bgr, face)
            appearance = (embeddings[index].astype(float).tolist()
                          if embeddings is not None and index < len(embeddings) else [])
            candidate = CharacterProfile(
                char_id=-1, hair_tone=tone, colors=colors,
                hair_hist=hist, hair_aspect=aspect,
                hair_area_frac=area_frac,
                appearance_embedding=appearance,
                lineart_embedding=lineart_descriptor(
                    cv2.cvtColor(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY),
                                 cv2.COLOR_GRAY2BGR), face.head_bbox,
                    rotation=getattr(face, "rotation", 0)),
                color_slots=_init_color_slots(colors))
            if self._upsert_profile(candidate):
                added += 1
        return added

    def match_page_fallback(self, *, segmentation, page_bgr: np.ndarray,
                            classifier=None,
                            forced_matches: dict[int, int] | None = None
                            ) -> tuple[dict[int, dict], list[CharacterInstance], dict, list]:
        """Face-geometry identity matching when semantic CLIP is unavailable.

        Returns assignments, instances, diagnostics and a region-aligned list
        of fallback semantic labels.
        """
        from core.anime_face_detector import (
            detect_anime_faces, hair_tone_features, lineart_descriptor,
            map_regions_to_face)
        if not self.characters or segmentation is None or page_bgr is None:
            labels = [("unknown", 0.0) for _ in getattr(segmentation, "regions", [])]
            return {}, [], {"matched": 0, "ambiguous": 0, "unmatched": 0,
                            "face_fallback": True}, labels

        faces = detect_anime_faces(page_bgr)
        regions = list(segmentation.regions or [])
        region_by_id = {int(r.label_id): r for r in regions}
        labels_by_id: dict[int, tuple[str, float]] = {}
        if not faces:
            return {}, [], {"matched": 0, "ambiguous": 0, "unmatched": 0,
                            "face_fallback": True, "reason": "no_face"}, [
                ("unknown", 0.0) for _ in regions]

        gray = (page_bgr if page_bgr.ndim == 2
                else cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY))
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        embeddings = None
        if classifier is not None and getattr(classifier, "available", False):
            try:
                embeddings = classifier.embed_bboxes(
                    gray_bgr, [face.head_bbox for face in faces])
            except Exception:
                embeddings = None

        forced_matches = forced_matches or {}
        out: dict[int, dict] = {}
        instances: list[CharacterInstance] = []
        matched_count = ambiguous = unmatched = 0
        claimed: set[int] = set()

        # Stronger/larger faces claim overlapping regions first.
        ordered = sorted(enumerate(faces),
                         key=lambda item: (item[1].confidence,
                                           item[1].bbox[2] * item[1].bbox[3]),
                         reverse=True)
        for original_index, face in ordered:
            mapped = map_regions_to_face(segmentation, face, page_bgr.shape[:2])
            for attr in mapped:
                mapped[attr] = [rid for rid in mapped[attr] if rid not in claimed]
            if not mapped["hair"]:
                continue
            anchor = int(mapped["hair"][0])
            forced_id = forced_matches.get(anchor)
            tone, hist, aspect, area_frac = hair_tone_features(page_bgr, face)
            lineart = lineart_descriptor(gray_bgr, face.head_bbox,
                                          rotation=getattr(face, "rotation", 0))
            appearance = (embeddings[original_index].astype(float).tolist()
                          if embeddings is not None and original_index < len(embeddings) else [])
            scored = sorted(((self._feature_distance(
                tone, hist, aspect, area_frac, appearance, lineart, ch), ch)
                for ch in self.characters), key=lambda item: item[0])

            if forced_id == -1:
                from core.character_matcher import MatchDecision
                decision = MatchDecision(None, 1.0, 1.0, 0.0, 0.0, 0.0,
                                         False, False, True)
            else:
                forced_ch = (next((c for c in self.characters
                                   if c.char_id == forced_id), None)
                             if forced_id is not None else None)
                # Dependency-free lineart matching is useful for proposing a
                # candidate, but pose and screentone can make two characters
                # deceptively similar.  Automatic locking therefore uses a
                # deliberately stricter gate than the CLIP path.  A user-forced
                # page binding still overrides these thresholds.
                decision = decide_match(
                    scored,
                    match_threshold=min(self.match_threshold, 0.42),
                    min_match_score=max(0.62, self.min_match_score),
                    min_margin=max(0.075, self.min_margin),
                    semantic_confidence=face.confidence,
                    min_semantic_conf=0.30,
                    forced_character=forced_ch)

            if not decision.matched:
                unmatched += 1
            elif not decision.lock_allowed:
                ambiguous += 1
            else:
                matched_count += 1
            ch = decision.character
            instance = CharacterInstance(
                instance_id=f"face-{anchor}", head_bbox=face.head_bbox,
                body_bbox=face.body_bbox, face_embedding=appearance,
                lineart_embedding=lineart,
                hair_regions=list(mapped["hair"]),
                skin_regions=list(mapped["skin"]),
                eye_regions=list(mapped["eyes"]),
                clothing_regions=list(mapped["clothing"]),
                confidence=float(face.confidence),
                matched_character_id=(ch.char_id if ch is not None else None),
                top1_score=decision.score, top2_score=decision.second_score,
                margin=decision.margin, lock_allowed=decision.lock_allowed)
            instances.append(instance)

            if ch is None:
                continue
            for attr in ("hair", "skin", "eyes", "clothing"):
                if attr not in ch.colors:
                    continue
                attr_allowed = bool(decision.lock_allowed)
                # Eye regions are tiny and text-like false positives are common;
                # demand a stronger face score unless the binding was forced.
                if attr == "eyes" and not decision.forced and face.confidence < 0.42:
                    attr_allowed = False
                for rid in mapped[attr]:
                    rid = int(rid)
                    labels_by_id[rid] = (attr, float(face.confidence))
                    slots = [_rgb(v) for v in ch.color_slots.get(attr, [ch.colors[attr]])]
                    part = None
                    preferred = 0
                    if attr == "clothing":
                        region_obj = region_by_id.get(rid)
                        part = (_classify_clothing_part(
                            region_obj, face.body_bbox, page_bgr.shape[:2])
                            if region_obj is not None else "upper")
                        preferred = _preferred_slot_index(part, slots)
                        instance.clothing_parts[rid] = part
                    out[rid] = {
                        "rgb": _rgb(ch.colors[attr]), "attribute": attr,
                        "slot_rgbs": slots,
                        "preferred_slot_index": preferred,
                        "clothing_part": part,
                        "char_id": ch.char_id, "distance": decision.distance,
                        "match_score": decision.score,
                        "top2_score": decision.second_score,
                        "margin": decision.margin,
                        "semantic_confidence": float(face.confidence),
                        "lock_allowed": attr_allowed,
                        "forced": decision.forced,
                    }
                    claimed.add(rid)
            ch.hits += 1

        labels = [labels_by_id.get(int(region.label_id), ("unknown", 0.0))
                  for region in regions]
        diagnostics = {
            "matched": matched_count, "ambiguous": ambiguous,
            "unmatched": unmatched, "faces_detected": len(faces),
            "face_fallback": True,
            "lock_regions": sum(1 for item in out.values()
                                if item.get("lock_allowed")),
            "skipped_lock_regions": sum(1 for item in out.values()
                                        if not item.get("lock_allowed")),
        }
        return out, instances, diagnostics, labels

    def match_page(self, regions: list, labels: list, *, segmentation=None,
                   gray_page: np.ndarray | None = None,
                   page_bgr: np.ndarray | None = None, classifier=None,
                   forced_matches: dict[int, int] | None = None
                   ) -> tuple[dict[int, dict], list[CharacterInstance], dict]:
        """Return explicit assignments, instances and diagnostics for one page.

        A low distance alone is not enough for a hard lock: Top-2 margin,
        semantic confidence and geometry must all pass their safety gates.
        """
        if not self.characters or not regions:
            return {}, [], {"matched": 0, "ambiguous": 0, "unmatched": 0}
        tagged = [(r, lab, float(conf)) for r, (lab, conf) in zip(regions, labels)
                  if lab in _CHARACTER_ATTRS]
        hair_tagged = [(r, conf) for r, lab, conf in tagged
                       if lab == "hair" and conf >= max(0.18, self.min_semantic_conf * 0.65)]
        # Within a plausible head box, a paired geometric eye candidate can
        # override a weak generic CLIP label.  Strong semantic labels remain.
        if hair_tagged and segmentation is not None and page_bgr is not None:
            try:
                from core.face_parts import find_eye_region_ids
                by_id = {int(r.label_id): (r, lab, conf) for r, lab, conf in tagged}
                region_by_id = {int(r.label_id): r for r in regions}
                existing_eye_ids = {rid for rid, (_r, lab, conf) in by_id.items()
                                    if lab == "eyes" and conf >= self.min_semantic_conf}
                for hair, _hair_conf in hair_tagged:
                    ids = find_eye_region_ids(
                        segmentation, hair, page_bgr.shape[:2],
                        _head_bbox(hair, page_bgr.shape[:2]), existing_eye_ids)
                    for rid in ids:
                        current = by_id.get(rid)
                        if current is None or current[2] < 0.55:
                            by_id[rid] = (region_by_id[rid], "eyes", 0.46)
                            existing_eye_ids.add(rid)
                tagged = list(by_id.values())
            except Exception:
                pass
        if not hair_tagged:
            return {}, [], {"matched": 0, "ambiguous": 0, "unmatched": 0,
                            "reason": "no_hair_anchor"}
        forced_matches = forced_matches or {}

        hair_regions = [item[0] for item in hair_tagged]
        target_embeddings = None
        if classifier is not None and page_bgr is not None and classifier.available:
            gray_embed_page = (cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
                               if page_bgr.ndim == 3 else page_bgr)
            gray_embed_page = cv2.cvtColor(gray_embed_page, cv2.COLOR_GRAY2BGR)
            target_embeddings = classifier.embed_bboxes(
                gray_embed_page, [_head_bbox(r, gray_embed_page.shape[:2]) for r in hair_regions])

        from core.anime_face_detector import lineart_descriptor
        lineart_page = (cv2.cvtColor(gray_page, cv2.COLOR_GRAY2BGR)
                        if gray_page is not None and gray_page.ndim == 2
                        else page_bgr)
        target_lineart = [lineart_descriptor(
            lineart_page, _head_bbox(r, lineart_page.shape[:2]))
            for r in hair_regions] if lineart_page is not None else [[] for _ in hair_regions]

        matches: list[dict] = []
        instances: list[CharacterInstance] = []
        ambiguous = 0
        unmatched = 0
        for i, (hair, hair_sem_conf) in enumerate(hair_tagged):
            forced_id = forced_matches.get(int(hair.label_id))
            hist, aspect, area_frac = (_hair_features(gray_page, segmentation, hair)
                                       if segmentation is not None and gray_page is not None
                                       else ([], hair.bbox[2] / max(1, hair.bbox[3]), hair.frac))
            emb = (target_embeddings[i].astype(float).tolist()
                   if target_embeddings is not None and i < len(target_embeddings) else [])
            lineart_emb = target_lineart[i] if i < len(target_lineart) else []
            scored = sorted(((self._feature_distance(
                float(hair.mean_gray), hist, aspect, area_frac, emb,
                lineart_emb, ch), ch)
                for ch in self.characters), key=lambda item: item[0])

            if forced_id == -1:
                # Explicit user rule: keep the model's colour and never lock
                # this page-local character instance.
                from core.character_matcher import MatchDecision
                decision = MatchDecision(
                    None, 1.0, 1.0, 0.0, 0.0, 0.0,
                    matched=False, lock_allowed=False, forced=True)
            else:
                forced_ch = (next((c for c in self.characters if c.char_id == forced_id), None)
                             if forced_id is not None else None)
                decision = decide_match(
                    scored, match_threshold=self.match_threshold,
                    min_match_score=self.min_match_score,
                    min_margin=self.min_margin,
                    semantic_confidence=hair_sem_conf,
                    min_semantic_conf=self.min_semantic_conf,
                    forced_character=forced_ch)
            first_distance = decision.distance
            first_ch = decision.character
            top1_score = decision.score
            top2_score = decision.second_score
            margin = decision.margin
            lock_allowed = decision.lock_allowed
            matched = decision.matched
            forced = decision.forced
            if not matched:
                unmatched += 1
            elif not lock_allowed:
                ambiguous += 1

            instance = CharacterInstance(
                instance_id=f"hair-{hair.label_id}",
                head_bbox=_head_bbox(hair, page_bgr.shape[:2] if page_bgr is not None else None),
                body_bbox=_body_bbox(hair, page_bgr.shape[:2] if page_bgr is not None else None),
                face_embedding=emb,
                lineart_embedding=lineart_emb,
                hair_regions=[int(hair.label_id)],
                confidence=top1_score,
                matched_character_id=first_ch.char_id if matched else None,
                top1_score=top1_score,
                top2_score=top2_score,
                margin=margin,
                lock_allowed=lock_allowed,
            )
            instances.append(instance)
            if matched:
                matches.append({
                    "hair": hair, "character": first_ch,
                    "distance": float(first_distance), "top1_score": top1_score,
                    "top2_score": top2_score, "margin": margin,
                    "semantic_confidence": hair_sem_conf,
                    "lock_allowed": lock_allowed, "forced": forced,
                    "instance": instance,
                })

        out: dict[int, dict] = {}
        for match in matches:
            hair, ch = match["hair"], match["character"]
            if "hair" in ch.colors:
                out[int(hair.label_id)] = {
                    "rgb": _rgb(ch.colors["hair"]), "attribute": "hair",
                    "slot_rgbs": [_rgb(v) for v in ch.color_slots.get("hair", [ch.colors["hair"]])],
                    "char_id": ch.char_id, "distance": match["distance"],
                    "match_score": match["top1_score"],
                    "top2_score": match["top2_score"], "margin": match["margin"],
                    "semantic_confidence": match["semantic_confidence"],
                    "lock_allowed": match["lock_allowed"], "forced": match["forced"],
                }

        for region, lab, semantic_conf in tagged:
            if lab == "hair" or region.label_id in out:
                continue
            cx = region.bbox[0] + region.bbox[2] / 2
            cy = region.bbox[1] + region.bbox[3] / 2
            options = []
            for match in matches:
                hair, ch = match["hair"], match["character"]
                if lab not in ch.colors:
                    continue
                bx, by, bw, bh = _body_bbox(hair)
                hx0, hy0, hw, hh = _head_bbox(hair)
                inside_body = bx <= cx <= bx + bw and by <= cy <= by + bh
                inside_head = hx0 <= cx <= hx0 + hw and hy0 <= cy <= hy0 + hh
                if lab == "eyes" and not inside_head:
                    continue
                if lab == "skin" and not (inside_head or inside_body):
                    continue
                if lab == "clothing" and not inside_body:
                    continue
                if not (inside_body or inside_head):
                    continue
                hx = hair.bbox[0] + hair.bbox[2] / 2
                hy = hair.bbox[1] + hair.bbox[3] / 2
                norm_w = max(1, hw if lab in ("eyes", "skin") else bw)
                norm_h = max(1, hh if lab in ("eyes", "skin") else bh)
                spatial = math.hypot((cx - hx) / norm_w, (cy - hy) / norm_h)
                options.append((spatial + match["distance"] * 0.5, match))
            if not options:
                continue
            _score, match = min(options, key=lambda item: item[0])
            ch = match["character"]
            allowed = bool(match["lock_allowed"] and semantic_conf >= self.min_semantic_conf)
            # Tiny eye candidates require stronger semantic evidence.
            if lab == "eyes" and semantic_conf < max(0.42, self.min_semantic_conf):
                allowed = False
            instance = match["instance"]
            slots = [_rgb(v) for v in ch.color_slots.get(lab, [ch.colors[lab]])]
            clothing_part = None
            preferred_slot = 0
            if lab == "clothing":
                clothing_part = _classify_clothing_part(
                    region, instance.body_bbox, page_bgr.shape[:2] if page_bgr is not None else None)
                preferred_slot = _preferred_slot_index(clothing_part, slots)
            out[int(region.label_id)] = {
                "rgb": _rgb(ch.colors[lab]), "attribute": lab,
                "slot_rgbs": slots,
                "preferred_slot_index": preferred_slot,
                "clothing_part": clothing_part,
                "char_id": ch.char_id, "distance": match["distance"],
                "match_score": match["top1_score"],
                "top2_score": match["top2_score"], "margin": match["margin"],
                "semantic_confidence": semantic_conf,
                "lock_allowed": allowed, "forced": match["forced"],
            }
            attr_list = {"skin": "skin_regions", "eyes": "eye_regions",
                         "clothing": "clothing_regions"}.get(lab)
            if attr_list:
                getattr(instance, attr_list).append(int(region.label_id))
            if lab == "clothing":
                instance.clothing_parts[int(region.label_id)] = clothing_part or "upper"

        for match in matches:
            match["character"].hits += 1
        diagnostics = {
            "matched": len(matches), "ambiguous": ambiguous,
            "unmatched": unmatched,
            "lock_regions": sum(1 for item in out.values() if item.get("lock_allowed")),
            "skipped_lock_regions": sum(1 for item in out.values() if not item.get("lock_allowed")),
        }
        return out, instances, diagnostics

    def assign_page_detailed(self, regions: list, labels: list, *, segmentation=None,
                             gray_page: np.ndarray | None = None,
                             page_bgr: np.ndarray | None = None, classifier=None,
                             forced_matches: dict[int, int] | None = None) -> dict[int, dict]:
        out, instances, _diagnostics = self.match_page(
            regions, labels, segmentation=segmentation, gray_page=gray_page,
            page_bgr=page_bgr, classifier=classifier, forced_matches=forced_matches)
        self.last_assignments = out
        self.last_instances = instances
        self._last_segmentation = segmentation
        self._last_source_shape = tuple(gray_page.shape[:2]) if gray_page is not None else None
        return out

    def assign_page(self, regions: list, labels: list, *, segmentation=None,
                    gray_page: np.ndarray | None = None,
                    page_bgr: np.ndarray | None = None, classifier=None,
                    forced_matches: dict[int, int] | None = None) -> dict:
        detailed = self.assign_page_detailed(
            regions, labels, segmentation=segmentation, gray_page=gray_page,
            page_bgr=page_bgr, classifier=classifier, forced_matches=forced_matches)
        return {region_id: item["rgb"] for region_id, item in detailed.items()
                if item.get("lock_allowed", False)}

    def diagnostic_rows(self, page_context=None, result_bgr: np.ndarray | None = None,
                        *, max_rows: int = 8) -> list[dict]:
        """Return compact per-character lock diagnostics for the UI.

        When a rendered page is supplied, actual region colours are compared
        with their identity targets in LAB space.  This exposes silent clothing
        or iris drift instead of only reporting that a region was "locked".
        """
        rows: list[dict] = []
        char_by_id = {int(ch.char_id): ch for ch in self.characters}
        assignments = (getattr(page_context, "identity_assignments", None) or {}
                       if page_context is not None else {})
        segmentation = getattr(page_context, "segmentation", None) if page_context is not None else None
        labels = None
        if result_bgr is not None and segmentation is not None:
            labels = segmentation.labels.astype(np.int32)
            if labels.shape[:2] != result_bgr.shape[:2]:
                labels = cv2.resize(labels, (result_bgr.shape[1], result_bgr.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)

        if assignments:
            grouped: dict[int, dict] = {}
            thresholds = {"eyes": 16.0, "clothing": 20.0,
                          "skin": 18.0, "hair": 24.0}
            for region_id, info in assignments.items():
                char_id = info.get("char_id")
                if char_id is None or int(char_id) not in char_by_id:
                    continue
                char_id = int(char_id)
                row = grouped.setdefault(char_id, {
                    "char_id": char_id,
                    "active_regions": 0,
                    "locked_regions": 0,
                    "attributes": set(),
                    "part_counts": {},
                    "drift_alerts": [],
                    "max_delta_e": 0.0,
                })
                row["active_regions"] += 1
                if info.get("lock_allowed", False):
                    row["locked_regions"] += 1
                attr = str(info.get("attribute", ""))
                if attr:
                    row["attributes"].add(attr)
                part = str(info.get("clothing_part") or "")
                if attr == "clothing" and part:
                    row["part_counts"][part] = row["part_counts"].get(part, 0) + 1

                if labels is not None and result_bgr is not None and info.get("lock_allowed", False):
                    actual = _region_median_rgb(result_bgr, labels, int(region_id))
                    target = _assignment_target_rgb(info)
                    if actual is not None and target is not None:
                        actual_lab = _hex_lab(_rgb_hex(actual))
                        target_lab = _hex_lab(_rgb_hex(target))
                        if actual_lab is not None and target_lab is not None:
                            delta_e = float(np.linalg.norm(actual_lab - target_lab))
                            row["max_delta_e"] = max(row["max_delta_e"], delta_e)
                            threshold = thresholds.get(attr, 24.0)
                            if delta_e > threshold:
                                row["drift_alerts"].append({
                                    "attribute": attr,
                                    "part": part or None,
                                    "delta_e": round(delta_e, 1),
                                    "region_id": int(region_id),
                                    "actual": _rgb_hex(actual),
                                    "target": _rgb_hex(target),
                                })

            for char_id, row in grouped.items():
                ch = char_by_id[char_id]
                clothing_slots = list(ch.color_slots.get("clothing", []))[:3]
                rows.append({
                    "char_id": char_id,
                    "name": ch.name or f"#{char_id}",
                    "hair": ch.colors.get("hair"),
                    "eyes": ch.colors.get("eyes"),
                    "skin": ch.colors.get("skin"),
                    "clothing": ch.colors.get("clothing"),
                    "clothing_slots": clothing_slots,
                    "active_regions": row["active_regions"],
                    "locked_regions": row["locked_regions"],
                    "attributes": sorted(row["attributes"]),
                    "part_counts": dict(row["part_counts"]),
                    "drift_alerts": list(row["drift_alerts"]),
                    "max_delta_e": round(float(row["max_delta_e"]), 1),
                    "reference_samples": ch.reference_samples,
                })
        if not rows:
            for ch in self.characters[:max_rows]:
                rows.append({
                    "char_id": int(ch.char_id),
                    "name": ch.name or f"#{ch.char_id}",
                    "hair": ch.colors.get("hair"),
                    "eyes": ch.colors.get("eyes"),
                    "skin": ch.colors.get("skin"),
                    "clothing": ch.colors.get("clothing"),
                    "clothing_slots": list(ch.color_slots.get("clothing", []))[:3],
                    "active_regions": 0,
                    "locked_regions": 0,
                    "attributes": [],
                    "part_counts": {},
                    "drift_alerts": [],
                    "max_delta_e": 0.0,
                    "reference_samples": ch.reference_samples,
                })
        rows.sort(key=lambda item: (
            -len(item.get("drift_alerts", [])),
            -int(item.get("locked_regions", 0)),
            -int(item.get("active_regions", 0)),
            item.get("char_id", 0)))
        return rows[:max_rows]


    def update_character(self, char_id: int, *, name: str | None = None,
                         colors: dict | None = None,
                         color_slots: dict | None = None) -> None:
        ch = next((c for c in self.characters if c.char_id == char_id), None)
        if ch is None:
            raise ValueError(f"unknown character id: {char_id}")
        if name is not None:
            ch.name = str(name)
        if colors:
            for key, value in colors.items():
                if key in _CHARACTER_ATTRS and isinstance(value, str):
                    ch.colors[key] = value.lower()
                    ch.color_slots[key] = _merge_slot_palette(ch.color_slots.get(key), value)
        if color_slots:
            for key, values in color_slots.items():
                if key not in _CHARACTER_ATTRS:
                    continue
                normalized = _normalize_slot_palette(values)
                if normalized:
                    ch.color_slots[key] = normalized
                    if key not in ch.colors:
                        ch.colors[key] = normalized[0]
        self.revision += 1

    def learn_from_colorized_page(self, page_context, result_bgr: np.ndarray, *,
                                 strength: float = 1.0) -> int:
        """Stabilize identity colours from a successfully rendered page.

        Only high-confidence assignments are learned, and the observed colour
        must already be sufficiently chromatic.  This lets later pages reuse
        more stable clothing / hair / eye variants without drifting toward
        grayscale or background contamination.
        """
        if page_context is None or result_bgr is None or result_bgr.size == 0:
            return 0
        strength = float(np.clip(strength, 0.0, 1.0))
        if strength <= 0.02 or not getattr(self, 'characters', None):
            return 0
        assignments = dict(getattr(page_context, 'identity_assignments', {}) or {})
        seg = getattr(page_context, 'segmentation', None)
        if not assignments or seg is None or getattr(seg, 'labels', None) is None:
            return 0

        h, w = result_bgr.shape[:2]
        labels = cv2.resize(seg.labels.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)
        changed = 0

        for region_id, info in assignments.items():
            char_id = info.get('char_id')
            attr = str(info.get('attribute', ''))
            if char_id is None or attr not in _CHARACTER_ATTRS:
                continue
            forced = bool(info.get('forced', False))
            lock_allowed = bool(info.get('lock_allowed', False))
            match_score = float(info.get('match_score', 0.0))
            semantic_conf = float(info.get('semantic_confidence', 0.0))
            margin = float(info.get('margin', 0.0))
            if not forced:
                if not lock_allowed:
                    continue
                if attr == 'eyes':
                    if match_score < 0.48 or semantic_conf < 0.18:
                        continue
                elif attr == 'clothing':
                    if match_score < 0.44 or semantic_conf < 0.16 or margin < 0.015:
                        continue
                elif match_score < 0.55 or semantic_conf < 0.22 or margin < 0.02:
                    continue

            mask = labels == int(region_id)
            if int(np.count_nonzero(mask)) < (4 if attr == 'eyes' else 14):
                continue
            pixels = result_bgr[mask]
            if pixels.size == 0:
                continue
            hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV).reshape(-1, 3)
            useful = pixels[(hsv[:, 1] > (10 if attr == 'skin' else 16)) &
                            (hsv[:, 2] > 20) & (hsv[:, 2] < 248)]
            if useful.shape[0] < max(4, pixels.shape[0] // 10):
                useful = pixels
            if useful.size == 0:
                continue
            b, g, r = np.median(useful.astype(np.float32), axis=0)
            rgb = (int(r), int(g), int(b))
            # Skip near-neutral observations; learning should not dilute identity.
            px = np.array([[[rgb[2], rgb[1], rgb[0]]]], dtype=np.uint8)
            lab = cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
            chroma = float(np.linalg.norm(lab[1:3] - 128.0))
            chroma_floor = 4.0 if attr == 'skin' else (6.0 if attr == 'clothing' else 7.0)
            if chroma < chroma_floor:
                continue
            color_hex = _rgb_hex(rgb)
            ch = next((c for c in self.characters if c.char_id == int(char_id)), None)
            if ch is None:
                continue
            slot_list = _merge_slot_palette(ch.color_slots.get(attr), color_hex)
            old_primary = ch.colors.get(attr)
            old_slots = list(ch.color_slots.get(attr, []))
            if attr == 'clothing' and 'preferred_slot_index' in info and slot_list:
                idx = min(max(int(info.get('preferred_slot_index', 0) or 0), 0), len(slot_list) - 1)
                ch.colors[attr] = slot_list[idx]
            else:
                primary = ch.colors.get(attr, slot_list[0] if slot_list else color_hex)
                ch.colors[attr] = _blend_hex(primary, color_hex, min(0.16 + 0.24 * strength, 0.36))
            ch.color_slots[attr] = slot_list
            if ch.colors.get(attr) != old_primary or ch.color_slots.get(attr, []) != old_slots:
                changed += 1

        if changed:
            self.revision += 1
        return changed

    def remove_character(self, char_id: int) -> None:
        before = len(self.characters)
        self.characters = [c for c in self.characters if c.char_id != char_id]
        if len(self.characters) != before:
            self.revision += 1
