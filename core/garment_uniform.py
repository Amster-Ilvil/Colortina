"""Conservative, line-bounded garment recolouring.

The semantic parser is only a proposal on monochrome manga pages.  This
module turns that proposal into an editable mask only when it agrees with a
closed line-art region and, when available, a foreground prior.  Ambiguous
page/background regions are rejected instead of being painted.

The recolour itself changes only Lab chroma.  Lightness, ink, screentone and
texture remain from the generator output, and pixels outside the accepted
mask are byte-for-byte untouched.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SafeGarmentMask:
    mask: np.ndarray
    diagnostics: dict


def _binary(value, shape: tuple[int, int]) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim == 3:
        arr = np.max(arr, axis=2)
    if arr.shape[:2] != shape:
        arr = cv2.resize(arr.astype(np.uint8), (shape[1], shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    return (arr > 0).astype(np.uint8)


def build_safe_garment_mask(
    source_bgr: np.ndarray,
    garment_prior: np.ndarray | None,
    *,
    foreground_mask: np.ndarray | None = None,
    negative_masks: dict[str, np.ndarray | None] | None = None,
    min_region_support: float = 0.55,
    max_region_area_ratio: float = 0.20,
    min_overlap_pixels: int = 24,
    line_low: int = 78,
    gap_close: int = 2,
) -> SafeGarmentMask:
    """Build a fail-closed garment mask from independent evidence.

    A region must be enclosed by the shared line map, must not touch the page
    border, and must have strong support from the garment prior.  A foreground
    prior limits the proposal to detected character pixels.  Explicit negative
    classes always win.
    """
    source = np.asarray(source_bgr)
    shape = source.shape[:2]
    prior = _binary(garment_prior, shape)
    empty = np.zeros(shape, dtype=np.uint8)
    if prior is None or not np.any(prior):
        return SafeGarmentMask(empty, {
            "reason": "empty_garment_prior",
            "prior_pixels": 0,
            "accepted_pixels": 0,
            "accepted_regions": 0,
        })

    candidate = prior.copy()
    fg = _binary(foreground_mask, shape)
    if fg is not None and np.any(fg):
        candidate &= fg

    negatives = np.zeros(shape, dtype=np.uint8)
    for name, raw in dict(negative_masks or {}).items():
        if name not in {"hair", "skin", "face", "eyes", "iris", "glasses",
                        "face_shadow", "background", "text", "lineart"}:
            continue
        mask = _binary(raw, shape)
        if mask is not None:
            negatives |= mask
    if np.any(negatives):
        negatives = cv2.dilate(negatives, np.ones((3, 3), np.uint8), 1)
        candidate[negatives > 0] = 0

    try:
        from core.region_map import build_region_map
        region_map = build_region_map(source, line_low=int(line_low),
                                      gap_close=max(0, int(gap_close)))
    except Exception as exc:  # fail closed if the shared boundary map fails
        return SafeGarmentMask(empty, {
            "reason": f"region_map_error:{type(exc).__name__}",
            "prior_pixels": int(np.count_nonzero(prior)),
            "candidate_pixels": int(np.count_nonzero(candidate)),
            "accepted_pixels": 0,
            "accepted_regions": 0,
        })

    labels = region_map.labels
    page_area = max(1, int(labels.size))
    accepted = np.zeros(shape, dtype=np.uint8)
    accepted_rows = []
    rejected_rows = []
    for rid in np.unique(labels[candidate > 0]).tolist():
        rid = int(rid)
        if rid <= 0:
            continue
        region = labels == rid
        area = int(np.count_nonzero(region))
        overlap = int(np.count_nonzero(region & (candidate > 0)))
        support = overlap / float(max(1, area))
        border = bool(region_map.touches_border(rid))
        ok = (not border and area <= page_area * float(max_region_area_ratio)
              and support >= float(min_region_support)
              and overlap >= int(min_overlap_pixels))
        row = {
            "region_id": rid,
            "area": area,
            "overlap": overlap,
            "support": round(float(support), 4),
            "touches_border": border,
        }
        if ok:
            accepted[region] = 1
            accepted_rows.append(row)
        else:
            rejected_rows.append(row)

    # Never paint a detected line pixel.  This is an extra guard even though
    # region_map labels normally exclude the barrier already.
    if region_map.barrier is not None:
        accepted[np.asarray(region_map.barrier) > 0] = 0

    return SafeGarmentMask(accepted, {
        "reason": "accepted" if np.any(accepted) else "no_region_passed_gate",
        "prior_pixels": int(np.count_nonzero(prior)),
        "candidate_pixels": int(np.count_nonzero(candidate)),
        "accepted_pixels": int(np.count_nonzero(accepted)),
        "accepted_regions": len(accepted_rows),
        "rejected_regions": len(rejected_rows),
        "accepted": accepted_rows[:64],
        "rejected": rejected_rows[:64],
        "line_guard_pixels": int(np.count_nonzero(region_map.barrier)) if region_map.barrier is not None else 0,
    })


def uniform_garment_chroma(
    base_bgr: np.ndarray,
    mask: np.ndarray | None,
    target_rgb: tuple[int, int, int],
    *,
    strength: float = 0.9,
) -> np.ndarray:
    """Apply one garment colour family while preserving L/shading/texture."""
    base = np.asarray(base_bgr).copy()
    active = _binary(mask, base.shape[:2])
    if active is None or not np.any(active):
        return base
    target = np.asarray(tuple(target_rgb), dtype=np.uint8).reshape(1, 1, 3)
    target_bgr = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(base, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    alpha = float(np.clip(strength, 0.0, 1.0))
    if alpha <= 0.0:
        return base
    # Target hue/chroma; source L is intentionally untouched.
    lab[..., 1][active > 0] = (
        lab[..., 1][active > 0] * (1.0 - alpha)
        + target_lab[1] * alpha
    )
    lab[..., 2][active > 0] = (
        lab[..., 2][active > 0] * (1.0 - alpha)
        + target_lab[2] * alpha
    )
    recoloured = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    out = base.copy()
    out[active > 0] = recoloured[active > 0]
    return out
