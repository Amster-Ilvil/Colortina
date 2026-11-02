"""Losses for high-recall generator semantic-head training."""
from __future__ import annotations

from typing import Sequence
import torch
import torch.nn.functional as F

IGNORE_INDEX = 255


def _valid_one_hot(target: torch.Tensor, num_classes: int):
    valid = target != IGNORE_INDEX
    safe = target.clone()
    safe[~valid] = 0
    one_hot = F.one_hot(safe.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    one_hot *= valid[:, None].float()
    return valid, one_hot


def multiclass_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    class_names: Sequence[str],
    hair_alpha: float = 0.22,
    hair_beta: float = 0.78,
    default_alpha: float = 0.50,
    default_beta: float = 0.50,
    eps: float = 1.0,
) -> torch.Tensor:
    num_classes = len(tuple(class_names))
    valid, one_hot = _valid_one_hot(target, num_classes)
    probs = torch.softmax(logits.float(), dim=1) * valid[:, None].float()
    present = one_hot.sum(dim=(0, 2, 3)) > 0
    if not present.any():
        return logits.sum() * 0.0
    tp = (probs * one_hot).sum(dim=(0, 2, 3))
    fp = (probs * (1.0 - one_hot) * valid[:, None]).sum(dim=(0, 2, 3))
    fn = ((1.0 - probs) * one_hot).sum(dim=(0, 2, 3))
    alpha = torch.full_like(tp, float(default_alpha))
    beta = torch.full_like(tp, float(default_beta))
    try:
        hi = list(class_names).index("hair")
        alpha[hi] = float(hair_alpha)
        beta[hi] = float(hair_beta)
    except ValueError:
        pass
    score = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1.0 - score[present].mean()


def boundary_weight_map(target: torch.Tensor, *, radius: int = 2, boost: float = 2.0) -> torch.Tensor:
    """Return per-pixel CE weights with extra emphasis around semantic edges."""
    valid = target != IGNORE_INDEX
    safe = target.clone().float()
    safe[~valid] = -1.0
    edge = torch.zeros_like(valid, dtype=torch.bool)
    # Compare neighbouring labels; ignore padded/unsupervised pairs.
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rolled = torch.roll(target, shifts=(dy, dx), dims=(-2, -1))
        rolled_valid = torch.roll(valid, shifts=(dy, dx), dims=(-2, -1))
        edge |= valid & rolled_valid & (target != rolled)
    e = edge[:, None].float()
    if radius > 0:
        k = int(radius) * 2 + 1
        e = F.max_pool2d(e, kernel_size=k, stride=1, padding=radius)
    return (1.0 + e[:, 0] * float(boost)) * valid.float()


def high_recall_semantic_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    class_names: Sequence[str],
    class_weights: torch.Tensor | None = None,
    ce_weight: float = 1.0,
    tversky_weight: float = 0.85,
    boundary_weight: float = 0.22,
) -> tuple[torch.Tensor, dict]:
    ce_raw = F.cross_entropy(
        logits, target, weight=class_weights, ignore_index=IGNORE_INDEX, reduction="none")
    bw = boundary_weight_map(target, radius=2, boost=2.0)
    valid = target != IGNORE_INDEX
    ce = (ce_raw * (1.0 + float(boundary_weight) * (bw - 1.0))).sum() / valid.sum().clamp_min(1)
    tv = multiclass_tversky_loss(logits, target, class_names=class_names)
    total = float(ce_weight) * ce + float(tversky_weight) * tv
    return total, {"ce": ce.detach(), "tversky": tv.detach()}
