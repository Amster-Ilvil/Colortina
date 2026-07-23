"""Quantitative helpers for real-page regression evaluation.

These functions do not require the model and can evaluate previously generated
pages.  They are intentionally lightweight so the same report can run on a Mac
mini after a normal colorization session.
"""

from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

from core.hint_artifact import detect_hint_blobs


def _rgb_to_lab(rgb: tuple[int, int, int]) -> np.ndarray:
    r, g, b = [int(np.clip(v, 0, 255)) for v in rgb]
    px = np.array([[[b, g, r]]], dtype=np.uint8)
    return cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


def delta_e(rgb_a: tuple[int, int, int], rgb_b: tuple[int, int, int]) -> float:
    """OpenCV-LAB Euclidean distance for two RGB colours."""
    return float(np.linalg.norm(_rgb_to_lab(rgb_a) - _rgb_to_lab(rgb_b)))


def median_rgb_at_points(image_bgr: np.ndarray, points: list,
                         radius: int = 4) -> tuple[int, int, int] | None:
    """Robustly sample RGB around normalized or pixel coordinate points."""
    if image_bgr is None or not points:
        return None
    h, w = image_bgr.shape[:2]
    pixels = []
    for point in points:
        if len(point) < 2:
            continue
        x, y = float(point[0]), float(point[1])
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            x, y = x * w, y * h
        px, py = int(round(x)), int(round(y))
        x1, x2 = max(0, px - radius), min(w, px + radius + 1)
        y1, y2 = max(0, py - radius), min(h, py + radius + 1)
        patch = image_bgr[y1:y2, x1:x2]
        if patch.size:
            pixels.append(patch.reshape(-1, 3))
    if not pixels:
        return None
    bgr = np.median(np.concatenate(pixels, axis=0).astype(np.float32), axis=0)
    b, g, r = np.clip(bgr, 0, 255).astype(np.uint8)
    return int(r), int(g), int(b)


def identity_delta_metrics(samples: list[dict]) -> dict:
    """Compute same/different-character Delta-E from sampled RGB records.

    Each record should contain ``character``, ``attribute`` and ``rgb``.
    """
    grouped: dict[tuple[str, str], list[tuple[int, int, int]]] = defaultdict(list)
    for sample in samples:
        rgb = sample.get("rgb")
        if rgb is None or len(rgb) != 3:
            continue
        grouped[(str(sample.get("character", "")),
                 str(sample.get("attribute", "")))].append(tuple(map(int, rgb)))

    same_distances = []
    centers: dict[tuple[str, str], tuple[int, int, int]] = {}
    for key, colors in grouped.items():
        labs = np.asarray([_rgb_to_lab(c) for c in colors], dtype=np.float32)
        center_lab = np.median(labs, axis=0)
        same_distances.extend(float(np.linalg.norm(lab - center_lab)) for lab in labs)
        rgb_center = np.median(np.asarray(colors, dtype=np.float32), axis=0)
        centers[key] = tuple(np.clip(rgb_center, 0, 255).astype(np.uint8).tolist())

    different = []
    keys = list(centers)
    for i, key_a in enumerate(keys):
        for key_b in keys[i + 1:]:
            if key_a[1] == key_b[1] and key_a[0] != key_b[0]:
                different.append(delta_e(centers[key_a], centers[key_b]))
    return {
        "same_character_delta_e_mean": (float(np.mean(same_distances))
                                         if same_distances else None),
        "same_character_delta_e_max": (float(np.max(same_distances))
                                        if same_distances else None),
        "different_character_delta_e_min": (float(np.min(different))
                                              if different else None),
        "sample_groups": len(grouped),
    }


def line_bleed_ratio(source_bw_bgr: np.ndarray, result_bgr: np.ndarray) -> float:
    """Fraction of dark source-line pixels that gained visible chroma."""
    if source_bw_bgr.shape[:2] != result_bgr.shape[:2]:
        source_bw_bgr = cv2.resize(source_bw_bgr,
                                   (result_bgr.shape[1], result_bgr.shape[0]),
                                   interpolation=cv2.INTER_AREA)
    gray = (cv2.cvtColor(source_bw_bgr, cv2.COLOR_BGR2GRAY)
            if source_bw_bgr.ndim == 3 else source_bw_bgr)
    ink = gray < 58
    if not np.any(ink):
        return 0.0
    lab = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.linalg.norm(lab[..., 1:3] - 128.0, axis=2)
    return float(np.count_nonzero(ink & (chroma > 11.0)) / np.count_nonzero(ink))


def assignment_metrics(diagnostics: list[dict]) -> dict:
    matched = sum(int(d.get("matched", 0) or 0) for d in diagnostics)
    ambiguous = sum(int(d.get("ambiguous", 0) or 0) for d in diagnostics)
    unmatched = sum(int(d.get("unmatched", 0) or 0) for d in diagnostics)
    locks = sum(int(d.get("lock_regions", 0) or 0) for d in diagnostics)
    skipped = sum(int(d.get("skipped_lock_regions", 0) or 0) for d in diagnostics)
    candidates = matched + unmatched
    return {
        "identity_assignment_coverage": (matched / candidates if candidates else None),
        "ambiguous_match_ratio": (ambiguous / matched if matched else None),
        "lock_region_coverage": (locks / (locks + skipped) if locks + skipped else None),
        "matched_instances": matched,
        "ambiguous_instances": ambiguous,
        "unmatched_instances": unmatched,
    }


def style_separation_retention(before_bgr: np.ndarray, after_bgr: np.ndarray,
                               masks: list[np.ndarray]) -> float:
    """Ratio of the minimum inter-region LAB distance after vs before style."""
    if len(masks) < 2:
        return 1.0

    def centers(img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        return [np.median(lab[m.astype(bool)], axis=0) for m in masks if np.any(m)]

    before = centers(before_bgr)
    after = centers(after_bgr)
    if len(before) < 2 or len(after) < 2:
        return 1.0

    def min_dist(values):
        return min(float(np.linalg.norm(values[i] - values[j]))
                   for i in range(len(values))
                   for j in range(i + 1, len(values)))

    return min_dist(after) / max(1e-6, min_dist(before))


hint_blob_score = detect_hint_blobs
