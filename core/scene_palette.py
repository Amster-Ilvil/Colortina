"""Absolute colours for environment regions, separate from visual style."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

_ENVIRONMENT = {"sky", "foliage", "stone", "wood", "water", "fire", "metal", "background"}


@dataclass
class ScenePalette:
    name: str = "Scene"
    colors: dict[str, str] = field(default_factory=dict)
    strength: float = 0.7
    revision: int = 0

    def color_for(self, semantic: str) -> tuple[int, int, int] | None:
        value = self.colors.get(semantic)
        if not value:
            return None
        try:
            s = value.lstrip("#")
            return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
        except Exception:
            return None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "ScenePalette | None":
        if not data:
            return None
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: str) -> str:
        if not path.endswith(".ccscene"):
            path += ".ccscene"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "ScenePalette":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        palette = cls.from_dict(data)
        if palette is None:
            raise ValueError("invalid scene palette")
        return palette

    @classmethod
    def extract_from_references(cls, images: list[np.ndarray], classifier,
                                name: str = "Scene") -> "ScenePalette":
        """Extract only environment colours; character parts are ignored."""
        from core.region_segmenter import segment_regions

        samples: dict[str, list[tuple[np.ndarray, float]]] = {}
        if classifier is None or not classifier.available:
            return cls(name=name)
        for image in images:
            seg = segment_regions(image)
            if not seg.regions:
                continue
            labels = classifier.classify(image, [r.bbox for r in seg.regions])
            if not labels:
                continue
            for region, (label, conf) in zip(seg.regions, labels):
                if label not in _ENVIRONMENT or conf < 0.30:
                    continue
                mask = seg.labels == int(region.label_id)
                ys, xs = np.nonzero(mask)
                if ys.size < 12:
                    continue
                ys = np.clip((ys / seg.scale).astype(int), 0, image.shape[0] - 1)
                xs = np.clip((xs / seg.scale).astype(int), 0, image.shape[1] - 1)
                pixels = image[ys, xs]
                hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
                valid = pixels[(hsv[:, 1] > 18) & (hsv[:, 2] > 20) & (hsv[:, 2] < 248)]
                if len(valid) >= 8:
                    pixels = valid
                if not len(pixels):
                    continue
                median = np.median(pixels.astype(np.float32), axis=0)
                samples.setdefault(label, []).append((median, float(region.area) * float(conf)))

        colors: dict[str, str] = {}
        for label, values in samples.items():
            weights = np.asarray([max(1.0, w) for _c, w in values], dtype=np.float32)
            bgr = np.average(np.asarray([c for c, _w in values], dtype=np.float32),
                             axis=0, weights=weights)
            b, g, r = np.clip(bgr, 0, 255).astype(np.uint8)
            colors[label] = f"#{int(r):02x}{int(g):02x}{int(b):02x}"
        return cls(name=name, colors=colors, strength=0.7, revision=1)
