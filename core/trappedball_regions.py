"""Leak-resistant line-art region segmentation for Colortina.

The core trapped-ball morphology is adapted from hepesu/LineFiller (MIT).
Colortina wraps it in a bounded, deterministic API and keeps the original
line-art pixels as region id 0.  It is used as a *fallback* when the faster
RegionMap looks suspicious, not for every page.

Upstream: https://github.com/hepesu/LineFiller
License: MIT (see THIRD_PARTY_NOTICES.md)
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class TrappedBallResult:
    labels: np.ndarray
    binary: np.ndarray
    fills: int
    diagnostics: dict


def _ball(radius: int) -> np.ndarray:
    r = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _unfilled_points(image: np.ndarray, radius: int = 0) -> np.ndarray:
    work = image
    if radius > 0:
        work = cv2.morphologyEx(work, cv2.MORPH_ERODE, _ball(radius), iterations=1)
    y, x = np.where(work == 255)
    if x.size == 0:
        return np.empty((0, 2), np.int32)
    return np.stack((x.astype(np.int32), y.astype(np.int32)), axis=-1)


def _fill_single(image: np.ndarray, seed: tuple[int, int], radius: int) -> np.ndarray:
    """One trapped-ball fill. 0 means the selected fill in the return image."""
    ball = _ball(radius)
    pass1 = np.full(image.shape, 255, np.uint8)
    inv = cv2.bitwise_not(image)
    mask1 = cv2.copyMakeBorder(inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
    cv2.floodFill(pass1, mask1, (int(seed[0]), int(seed[1])), 0, 0, 0, 4)
    pass1 = cv2.morphologyEx(pass1, cv2.MORPH_DILATE, ball, iterations=1)
    pass2 = np.full(image.shape, 255, np.uint8)
    mask2 = cv2.copyMakeBorder(pass1, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
    cv2.floodFill(pass2, mask2, (int(seed[0]), int(seed[1])), 0, 0, 0, 4)
    return cv2.morphologyEx(pass2, cv2.MORPH_ERODE, ball, iterations=1)


def _trapped_multi(image: np.ndarray, radius: int, *, method: str | None,
                   max_iter: int = 1600) -> list[tuple[np.ndarray, np.ndarray]]:
    unfilled = image.copy()
    fills: list[tuple[np.ndarray, np.ndarray]] = []
    sizes: list[int] = []
    for _ in range(max(1, int(max_iter))):
        pts = _unfilled_points(unfilled, radius)
        if pts.size == 0:
            break
        fill = _fill_single(unfilled, tuple(pts[0]), radius)
        yy, xx = np.where(fill == 0)
        if yy.size == 0:
            break
        unfilled[yy, xx] = 0
        fills.append((yy, xx))
        sizes.append(int(yy.size))
    if not fills or method is None:
        return fills
    arr = np.asarray(sizes, dtype=np.float32)
    if method == "max":
        threshold = float(arr.max())
    elif method == "median":
        threshold = float(np.median(arr))
    elif method == "mean":
        threshold = float(arr.mean())
    else:
        threshold = 0.0
    return [f for f, s in zip(fills, sizes) if s >= threshold]


def _flood_single(image: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    pass1 = np.full(image.shape, 255, np.uint8)
    inv = cv2.bitwise_not(image)
    mask = cv2.copyMakeBorder(inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
    cv2.floodFill(pass1, mask, (int(seed[0]), int(seed[1])), 0, 0, 0, 4)
    return pass1


def _flood_multi(image: np.ndarray, *, max_iter: int = 20000) -> list[tuple[np.ndarray, np.ndarray]]:
    unfilled = image.copy()
    fills: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(max(1, int(max_iter))):
        pts = _unfilled_points(unfilled)
        if pts.size == 0:
            break
        fill = _flood_single(unfilled, tuple(pts[0]))
        yy, xx = np.where(fill == 0)
        if yy.size == 0:
            break
        unfilled[yy, xx] = 0
        fills.append((yy, xx))
    return fills


def _mark(image: np.ndarray, fills: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    out = image.copy()
    for fill in fills:
        out[fill] = 0
    return out


def _build_map(shape: tuple[int, int], fills: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    labels = np.zeros(shape, np.int32)
    for idx, fill in enumerate(fills, start=1):
        labels[fill] = idx
    return labels


def _merge_tiny(labels: np.ndarray, *, max_iter: int = 6,
                tiny_area: int = 220) -> np.ndarray:
    """Merge tiny fill fragments into the neighbour with most border contact."""
    out = labels.copy().astype(np.int32)
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(max(1, int(max_iter))):
        changed = False
        ids, counts = np.unique(out, return_counts=True)
        areas = dict(zip(ids.tolist(), counts.tolist()))
        for rid in ids.tolist():
            if rid <= 0 or areas.get(rid, 0) > tiny_area:
                continue
            region = (out == rid).astype(np.uint8)
            border = cv2.dilate(region, cross, iterations=1).astype(bool) & ~region.astype(bool)
            neighbours = out[border]
            neighbours = neighbours[neighbours > 0]
            if neighbours.size == 0:
                continue
            nids, ncounts = np.unique(neighbours, return_counts=True)
            target = int(nids[int(np.argmax(ncounts))])
            if target != rid:
                out[out == rid] = target
                changed = True
        if not changed:
            break
    # Relabel densely for predictable downstream arrays.
    result = np.zeros_like(out)
    for new_id, old_id in enumerate([i for i in np.unique(out).tolist() if i > 0], start=1):
        result[out == old_id] = new_id
    return result


def segment_trappedball(gray_or_bgr: np.ndarray, *, threshold: int = 220,
                        radii: tuple[int, ...] = (3, 2, 1),
                        max_edge: int = 1600,
                        merge_tiny: bool = True) -> TrappedBallResult:
    """Return leak-resistant region labels at the input resolution.

    Large pages are processed on a bounded working copy and labels are resized
    back with nearest-neighbour interpolation.  This keeps batch jobs predictable.
    """
    src = np.asarray(gray_or_bgr)
    gray = (cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) if src.ndim == 3 else src).astype(np.uint8)
    h, w = gray.shape[:2]
    scale = min(1.0, float(max_edge) / max(h, w)) if max_edge else 1.0
    work = gray
    if scale < 1.0:
        work = cv2.resize(gray, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                          interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(work, int(np.clip(threshold, 64, 252)), 255, cv2.THRESH_BINARY)

    fills: list[tuple[np.ndarray, np.ndarray]] = []
    result = binary.copy()
    for index, radius in enumerate(tuple(int(r) for r in radii if int(r) > 0)):
        method = "max" if index == 0 else "mean"
        batch = _trapped_multi(result, radius, method=method)
        fills.extend(batch)
        result = _mark(result, batch)
    rest = _flood_multi(result)
    fills.extend(rest)

    labels = _build_map(result.shape[:2], fills)
    if merge_tiny and labels.max() > 0:
        labels = _merge_tiny(labels)
    if labels.shape != (h, w):
        labels = cv2.resize(labels.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    # Never assign ink to a fill id after upscaling.
    ink = gray < int(np.clip(threshold, 64, 252))
    labels[ink] = 0
    return TrappedBallResult(
        labels=labels,
        binary=binary,
        fills=int(labels.max()),
        diagnostics={
            "backend": "trapped_ball",
            "threshold": int(threshold),
            "radii": list(radii),
            "work_scale": float(scale),
            "region_count": int(labels.max()),
        },
    )
