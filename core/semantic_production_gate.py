"""Production acceptance gate for Generator Semantic Head checkpoints."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Mapping, Sequence
import numpy as np

IGNORE_INDEX = 255


@dataclass(frozen=True)
class ProductionGateThresholds:
    hair_recall_min: float = 0.990
    hair_precision_min: float = 0.980
    hair_iou_min: float = 0.970
    glasses_to_hair_max: float = 0.005
    clothes_to_hair_max: float = 0.005
    face_shadow_to_hair_max: float = 0.005
    skin_to_hair_max: float = 0.008


def evaluate_production_gate(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    class_names: Sequence[str],
    thresholds: ProductionGateThresholds | None = None,
) -> dict:
    th = thresholds or ProductionGateThresholds()
    pred = np.asarray(predicted)
    gt = np.asarray(target)
    if pred.shape != gt.shape:
        raise ValueError("predicted and target shapes must match")
    lookup = {str(n): i for i, n in enumerate(class_names)}
    if "hair" not in lookup:
        raise ValueError("hair class is required")
    valid = gt != IGNORE_INDEX
    hi = lookup["hair"]
    p_hair = (pred == hi) & valid
    g_hair = (gt == hi) & valid
    tp = int(np.count_nonzero(p_hair & g_hair))
    fp = int(np.count_nonzero(p_hair & ~g_hair))
    fn = int(np.count_nonzero(~p_hair & g_hair))
    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    iou = tp / max(1, tp + fp + fn)

    leak = {}
    for name in ("glasses", "clothes", "face_shadow", "skin"):
        idx = lookup.get(name)
        if idx is None:
            continue
        region = (gt == idx) & valid
        denom = int(np.count_nonzero(region))
        leak[name] = 0.0 if denom == 0 else float(np.count_nonzero(p_hair & region)) / denom

    checks = {
        "hair_recall": recall >= th.hair_recall_min,
        "hair_precision": precision >= th.hair_precision_min,
        "hair_iou": iou >= th.hair_iou_min,
        "glasses_to_hair": leak.get("glasses", 0.0) <= th.glasses_to_hair_max,
        "clothes_to_hair": leak.get("clothes", 0.0) <= th.clothes_to_hair_max,
        "face_shadow_to_hair": leak.get("face_shadow", 0.0) <= th.face_shadow_to_hair_max,
        "skin_to_hair": leak.get("skin", 0.0) <= th.skin_to_hair_max,
    }
    return {
        "production_ready": bool(all(checks.values())),
        "checks": checks,
        "thresholds": asdict(th),
        "metrics": {
            "hair_recall": float(recall),
            "hair_precision": float(precision),
            "hair_iou": float(iou),
            "glasses_to_hair": float(leak.get("glasses", 0.0)),
            "clothes_to_hair": float(leak.get("clothes", 0.0)),
            "face_shadow_to_hair": float(leak.get("face_shadow", 0.0)),
            "skin_to_hair": float(leak.get("skin", 0.0)),
            "hair_tp": tp, "hair_fp": fp, "hair_fn": fn,
        },
    }
