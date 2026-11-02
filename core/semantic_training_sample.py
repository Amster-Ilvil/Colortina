"""Utilities for turning corrected Colortina semantic masks into training labels.

The generator-native semantic head is meant to learn from *corrected* masks,
not from random weights or blindly trusted heuristics.  This module converts a
set of class masks into one indexed PNG label map using deterministic class
priority.  Unlabelled pixels may remain IGNORE=255 so a user only needs to
correct regions that matter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

from core.generator_semantic_head import DEFAULT_CLASSES

IGNORE_INDEX = 255
DEFAULT_PRIORITY = (
    "background",
    "other",
    "skin",
    "hair",
    "clothes",
    "face_shadow",
    "glasses",
    "iris",
    "eye",
    "eyebrow",
)


def _binary(mask, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if tuple(arr.shape[:2]) != tuple(shape):
        arr = cv2.resize(arr.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (arr > 0).astype(np.uint8)


def build_indexed_semantic_label(
    shape: tuple[int, int],
    masks: Mapping[str, np.ndarray | None],
    *,
    class_names: Sequence[str] = DEFAULT_CLASSES,
    valid_mask: np.ndarray | None = None,
    ignore_unlabelled: bool = True,
    priority: Sequence[str] = DEFAULT_PRIORITY,
) -> np.ndarray:
    """Build a uint8 indexed semantic label from corrected binary masks.

    Later classes in ``priority`` overwrite earlier ones.  This makes iris beat
    hair, glasses beat skin, and hair beat generic ``other`` while keeping the
    operation deterministic.  When ``ignore_unlabelled`` is true, pixels not
    covered by any supplied class stay at 255 and do not affect training loss.
    """
    h, w = (int(shape[0]), int(shape[1]))
    classes = tuple(str(v) for v in class_names)
    lookup = {name: idx for idx, name in enumerate(classes)}
    fill = IGNORE_INDEX if ignore_unlabelled else lookup.get("background", 0)
    label = np.full((h, w), int(fill), np.uint8)
    valid = _binary(valid_mask, (h, w)) if valid_mask is not None else None

    ordered = list(priority) + [name for name in classes if name not in priority]
    for name in ordered:
        if name not in lookup:
            continue
        mask = _binary(masks.get(name), (h, w))
        if mask is None:
            continue
        active = mask > 0
        if valid is not None:
            active &= valid > 0
        label[active] = np.uint8(lookup[name])
    if valid is not None:
        label[valid == 0] = np.uint8(IGNORE_INDEX)
    return label


def save_training_pair(
    out_dir: str | Path,
    stem: str,
    image_bgr: np.ndarray,
    label: np.ndarray,
) -> tuple[str, str]:
    out = Path(out_dir)
    images = out / "images"
    labels = out / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    image_path = images / f"{stem}.png"
    label_path = labels / f"{stem}.png"
    if not cv2.imwrite(str(image_path), image_bgr):
        raise RuntimeError(f"failed to save image: {image_path}")
    if not cv2.imwrite(str(label_path), label.astype(np.uint8)):
        raise RuntimeError(f"failed to save label: {label_path}")
    return str(image_path), str(label_path)
