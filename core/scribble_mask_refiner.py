"""Scribble-guided line-art mask correction.

Inspired by pixiv/manga-segment's interaction model (MIT): coarse positive and
negative strokes identify objects, then region structure expands the edit. This
Python implementation is intentionally Colortina-native and operates on the
same line-art region labels used by automatic semantic masks; no browser code
or JavaScript is included.
"""
from __future__ import annotations

import cv2
import numpy as np


def _to_mask(value, shape):
    if value is None:
        return np.zeros(shape, np.uint8)
    arr = np.asarray(value)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.shape != shape:
        arr = cv2.resize(arr.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (arr > 0).astype(np.uint8)


def refine_mask_with_scribbles(labels: np.ndarray, *, positive=None, negative=None,
                               prior=None, source_bgr=None,
                               positive_overlap: float = 0.015,
                               negative_overlap: float = 0.01,
                               neighbour_steps: int = 1) -> tuple[np.ndarray, dict]:
    """Expand rough user strokes to complete line-bounded regions.

    Positive strokes include a region, negative strokes veto it. A semantic
    prior may seed additional regions but never overrides an explicit negative
    stroke. One optional adjacency step fills tiny fragmented neighbours when
    their grayscale statistics are close to an already accepted region.
    """
    lbl = np.asarray(labels, dtype=np.int32)
    if lbl.ndim != 2:
        raise ValueError("labels must be 2-D")
    shape = lbl.shape
    pos = _to_mask(positive, shape)
    neg = _to_mask(negative, shape)
    pri = _to_mask(prior, shape)
    max_id = int(lbl.max())
    areas = np.bincount(lbl.ravel(), minlength=max_id + 1)
    accepted: set[int] = set()
    vetoed: set[int] = set()
    scores: dict[int, float] = {}

    for rid in range(1, max_id + 1):
        area = int(areas[rid]) if rid < len(areas) else 0
        if area <= 0:
            continue
        region = lbl == rid
        p = float(np.count_nonzero(region & (pos > 0))) / float(area)
        n = float(np.count_nonzero(region & (neg > 0))) / float(area)
        s = float(np.count_nonzero(region & (pri > 0))) / float(area)
        scores[rid] = max(p * 4.0, s)
        if n >= negative_overlap or np.count_nonzero(region & (neg > 0)) >= 3:
            vetoed.add(rid)
            continue
        if p >= positive_overlap or np.count_nonzero(region & (pos > 0)) >= 2:
            accepted.add(rid)
        elif s >= 0.56:
            accepted.add(rid)

    # Optional tiny-fragment growth. This mirrors the interaction idea of
    # redrawing a rough line instead of manually painting every small lock.
    if source_bgr is not None and neighbour_steps > 0 and accepted:
        gray = (cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)
                if np.asarray(source_bgr).ndim == 3 else np.asarray(source_bgr)).astype(np.uint8)
        means = {}
        for rid in range(1, max_id + 1):
            pixels = gray[lbl == rid]
            if pixels.size:
                means[rid] = float(np.mean(pixels))
        cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        for _ in range(int(neighbour_steps)):
            add = set()
            for rid in list(accepted):
                region = (lbl == rid).astype(np.uint8)
                border = cv2.dilate(region, cross, iterations=1).astype(bool) & ~region.astype(bool)
                for nid in np.unique(lbl[border]).tolist():
                    nid = int(nid)
                    if nid <= 0 or nid in accepted or nid in vetoed:
                        continue
                    if nid >= len(areas) or areas[nid] <= 0:
                        continue
                    # Only pull small fragments with similar line tone.
                    if areas[nid] <= max(3000, areas[rid] * 0.35) and abs(means.get(nid, 255) - means.get(rid, 255)) <= 34:
                        if scores.get(nid, 0.0) >= 0.18:
                            add.add(nid)
            if not add:
                break
            accepted.update(add)

    accepted -= vetoed
    out = np.isin(lbl, np.fromiter(accepted, dtype=np.int32) if accepted else np.array([], np.int32))
    return out.astype(np.uint8) * 255, {
        "accepted_regions": sorted(accepted),
        "vetoed_regions": sorted(vetoed),
        "positive_pixels": int(np.count_nonzero(pos)),
        "negative_pixels": int(np.count_nonzero(neg)),
    }
