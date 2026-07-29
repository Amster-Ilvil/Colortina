"""Detection of small, regular colour blobs around automatic hint centres."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.hint_spec import HintSpec


@dataclass
class HintArtifactReport:
    score: float
    suspicious: int
    checked: int
    threshold: float = 14.0

    @property
    def should_retry(self) -> bool:
        return self.checked >= 2 and self.score >= self.threshold


def _delta_e_mean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.astype(np.float32) - b.astype(np.float32), axis=-1).mean())


def detect_hint_blobs(result_bgr: np.ndarray, hints, *, threshold: float = 14.0) -> HintArtifactReport:
    specs = [HintSpec.from_any(h) for h in (hints or [])
             if HintSpec.from_any(h).source not in ("manual", "eyedropper_hint", "style_only")]
    if not specs:
        return HintArtifactReport(0.0, 0, 0, threshold)
    lab = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2LAB)
    h, w = lab.shape[:2]
    scores: list[float] = []
    suspicious = 0
    for spec in specs[:160]:
        x = int(np.clip(round(spec.x_norm * w), 0, w - 1))
        y = int(np.clip(round(spec.y_norm * h), 0, h - 1))
        r = int(np.clip(round(spec.radius_norm * w), 2, 8))
        y1, y2 = max(0, y - r * 3), min(h, y + r * 3 + 1)
        x1, x2 = max(0, x - r * 3), min(w, x + r * 3 + 1)
        patch = lab[y1:y2, x1:x2]
        if min(patch.shape[:2]) < 5:
            continue
        yy, xx = np.mgrid[y1:y2, x1:x2]
        d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        inner = patch[d <= max(1.5, r * 0.8)]
        ring = patch[(d >= r * 1.35) & (d <= r * 2.5)]
        if len(inner) < 4 or len(ring) < 8:
            continue
        score = float(np.linalg.norm(np.median(inner, axis=0) - np.median(ring, axis=0)))
        scores.append(score)
        if score >= threshold:
            suspicious += 1
    aggregate = float(np.percentile(scores, 80)) if scores else 0.0
    return HintArtifactReport(aggregate, suspicious, len(scores), threshold)
