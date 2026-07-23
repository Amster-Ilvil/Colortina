"""Fast, model-free quality diagnostics for colorized manga pages.

The score is intentionally heuristic: it flags likely monochrome washes,
insufficient palette separation and colored ink/line bleed so users know which
pages deserve a re-roll or manual correction. It never blocks export.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass
class QualityReport:
    score: int
    colorfulness: float
    hue_diversity: float
    dominant_hue_ratio: float
    ink_chroma: float
    wash_detected: bool
    line_bleed_detected: bool
    reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def assess_colorization(original_bgr: np.ndarray,
                        result_bgr: np.ndarray) -> QualityReport:
    """Assess one result without requiring a reference/ground-truth image."""
    if original_bgr.shape[:2] != result_bgr.shape[:2]:
        original_bgr = cv2.resize(original_bgr, (result_bgr.shape[1], result_bgr.shape[0]),
                                  interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Evaluate actual drawable areas, excluding paper, speech bubbles and ink.
    content = (gray >= 35) & (gray <= 242)
    if not np.any(content):
        content = np.ones(gray.shape, dtype=bool)
    sat = hsv[..., 1][content]
    hue = hsv[..., 0][content]
    chroma = np.sqrt((lab[..., 1] - 128.0) ** 2 +
                     (lab[..., 2] - 128.0) ** 2)[content]
    colorfulness = float(np.median(chroma)) if chroma.size else 0.0

    colorful = sat >= 24
    if np.any(colorful):
        hist, _ = np.histogram(hue[colorful], bins=12, range=(0, 180))
        total = max(1, int(hist.sum()))
        dominant = float(hist.max() / total)
        occupied = int(np.count_nonzero(hist >= max(2, total * 0.025)))
        hue_diversity = float(occupied / 12.0)
    else:
        dominant, hue_diversity = 1.0, 0.0

    # Black ink should remain close to neutral. High chroma on ink is a useful
    # proxy for color leaking across line art.
    ink = gray < 55
    if np.any(ink):
        ink_chroma = float(np.percentile(np.sqrt(
            (lab[..., 1][ink] - 128.0) ** 2 +
            (lab[..., 2][ink] - 128.0) ** 2), 75))
    else:
        ink_chroma = 0.0

    reasons: list[str] = []
    wash = ((dominant > 0.80 and hue_diversity < 0.34) or
            (colorfulness < 7.0))
    if wash:
        reasons.append("monochrome_wash")
    if hue_diversity < 0.20:
        reasons.append("low_palette_diversity")
    line_bleed = ink_chroma > 12.0
    if line_bleed:
        reasons.append("colored_ink_or_edge_bleed")

    score = 100
    if wash:
        score -= 35
    if hue_diversity < 0.20:
        score -= 15
    elif hue_diversity < 0.34:
        score -= 7
    if line_bleed:
        score -= min(25, int((ink_chroma - 12.0) * 1.5) + 8)
    if colorfulness > 45:
        score -= 5  # likely oversaturated, but do not over-penalize stylized art
    score = int(np.clip(score, 0, 100))

    return QualityReport(
        score=score,
        colorfulness=round(colorfulness, 2),
        hue_diversity=round(hue_diversity, 3),
        dominant_hue_ratio=round(dominant, 3),
        ink_chroma=round(ink_chroma, 2),
        wash_detected=wash,
        line_bleed_detected=line_bleed,
        reasons=reasons,
    )
