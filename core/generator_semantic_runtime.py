"""Inference helpers for the generator-native semantic head."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from core.generator_semantic_head import (
    DEFAULT_CLASSES,
    GeneratorSemanticHead,
    load_semantic_head_checkpoint,
)


@dataclass
class GeneratorSemanticPrediction:
    masks: dict[str, np.ndarray]
    probabilities: dict[str, np.ndarray]
    class_map: np.ndarray
    diagnostics: dict

    def get(self, name: str, default=None):
        return self.masks.get(str(name), default)


class GeneratorSemanticRuntime:
    """Lazy checkpoint-backed segmentation runtime.

    A runtime is considered available only when a *trained checkpoint file*
    exists and loads strictly.  Randomly initialized weights are never used for
    user-facing masks.
    """

    def __init__(self, weights_path: str, device="cpu", *, allow_experimental: bool = False):
        self.weights_path = str(weights_path or "")
        self.device = torch.device(device)
        # GUI / ordinary pipeline must never use single-page research weights.
        self.allow_experimental = bool(allow_experimental)
        self.head: GeneratorSemanticHead | None = None
        self.payload: dict | None = None
        self.load_error: str | None = None

    @property
    def available(self) -> bool:
        return bool(self.weights_path and Path(self.weights_path).is_file())

    def ensure_loaded(self) -> GeneratorSemanticHead:
        if self.head is not None:
            return self.head
        if not self.available:
            raise FileNotFoundError(self.weights_path or "generator_semantic_head.pth")
        try:
            head, payload = load_semantic_head_checkpoint(
                self.weights_path, device=self.device, eval_mode=True)
            gate = dict(dict(payload.get("metrics") or {}).get("production_gate") or {})
            if not self.allow_experimental and not bool(gate.get("production_ready", False)):
                raise RuntimeError(
                    "Semantic checkpoint is research-only or lacks validated production gate; "
                    "requires multi-page independent hair/skin/eyebrow validation")
            self.head = head
            self.payload = payload
            self.load_error = None
            return head
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            raise

    def to(self, device) -> None:
        self.device = torch.device(device)
        if self.head is not None:
            self.head.to(self.device)

    @torch.inference_mode()
    def predict_from_features(
        self,
        features: Sequence[torch.Tensor],
        *,
        model_hw: tuple[int, int],
        valid_hw: tuple[int, int],
        output_hw: tuple[int, int],
        confidence_threshold: float = 0.45,
        class_thresholds: Mapping[str, float] | None = None,
    ) -> GeneratorSemanticPrediction:
        head = self.ensure_loaded()
        feats = [f.to(self.device) for f in features]
        logits = head(feats, output_size=model_hw)
        valid_h, valid_w = (int(valid_hw[0]), int(valid_hw[1]))
        logits = logits[..., :valid_h, :valid_w]
        if tuple(logits.shape[-2:]) != tuple(output_hw):
            logits = F.interpolate(
                logits, size=tuple(int(v) for v in output_hw),
                mode="bilinear", align_corners=False)
        probs = torch.softmax(logits.float(), dim=1)[0]
        winner_prob, winner = torch.max(probs, dim=0)
        class_names = tuple(head.class_names)
        masks: dict[str, np.ndarray] = {}
        probability_maps: dict[str, np.ndarray] = {}
        checkpoint_inference = dict((self.payload or {}).get("inference") or {})
        checkpoint_thresholds = dict(checkpoint_inference.get("class_thresholds") or {})
        thresholds = {str(k): float(v) for k, v in checkpoint_thresholds.items()}
        thresholds.update({str(k): float(v) for k, v in dict(class_thresholds or {}).items()})
        mask_policy = str(checkpoint_inference.get("mask_policy", "winner")).strip().lower()
        veto_thresholds = {str(k): float(v) for k, v in
                           dict(checkpoint_inference.get("negative_veto_thresholds") or {}).items()}
        winner_np = winner.detach().cpu().numpy().astype(np.uint8)
        winner_prob_np = winner_prob.detach().cpu().numpy().astype(np.float32)
        for idx, name in enumerate(class_names):
            probability_maps[name] = probs[idx].detach().cpu().numpy().astype(np.float32)

        if mask_policy in {"semantic_priority", "priority_threshold", "high_recall"}:
            # Independent threshold admission improves recall for thin hair that
            # slightly loses argmax to generic other/background.  Strong
            # semantic negatives still veto hair, so this cannot turn glasses
            # or clothing into hair merely by lowering the hair threshold.
            for idx, name in enumerate(class_names):
                p = probability_maps[name]
                threshold = float(thresholds.get(name, confidence_threshold))
                masks[name] = (p >= threshold).astype(np.uint8) * 255
            hair = masks.get("hair")
            if hair is not None:
                active_hair = hair > 0
                for negative_name in ("eyebrow_eye", "eyebrow", "eye", "glasses", "clothes", "face_shadow", "skin", "iris"):
                    pneg = probability_maps.get(negative_name)
                    if pneg is None:
                        continue
                    default_veto = max(float(thresholds.get(negative_name, confidence_threshold)), 0.34)
                    vt = float(veto_thresholds.get(negative_name, default_veto))
                    active_hair &= pneg < vt
                masks["hair"] = active_hair.astype(np.uint8) * 255
            # Resolve overlaps deterministically for the class map/debug view.
            priority = ["background", "other", "skin", "hair", "clothes",
                        "face_shadow", "glasses", "iris"]
            resolved = winner_np.copy()
            lookup = {n: i for i, n in enumerate(class_names)}
            for name in priority:
                if name in masks and name in lookup:
                    resolved[masks[name] > 0] = np.uint8(lookup[name])
            winner_np = resolved
        else:
            for idx, name in enumerate(class_names):
                p = probability_maps[name]
                threshold = float(thresholds.get(name, confidence_threshold))
                active = (winner_np == idx) & (p >= threshold)
                masks[name] = active.astype(np.uint8) * 255
        diagnostics = {
            "backend": "generator_semantic_head",
            "weights_path": self.weights_path,
            "class_names": list(class_names),
            "confidence_threshold": float(confidence_threshold),
            "class_thresholds": dict(thresholds),
            "mask_policy": mask_policy,
            "negative_veto_thresholds": dict(veto_thresholds),
            "production_ready": bool(dict((self.payload or {}).get("metrics") or {}).get("production_gate", {}).get("production_ready", False)),
            "calibrated_checkpoint": bool(
                str(dict((self.payload or {}).get("training") or {}).get("mode", "")).startswith("bootstrap")),
            "model_hw": [int(v) for v in model_hw],
            "valid_hw": [int(v) for v in valid_hw],
            "output_hw": [int(v) for v in output_hw],
            "pixels": {k: int(np.count_nonzero(v)) for k, v in masks.items()},
            "mean_winner_confidence": float(winner_prob_np.mean()),
        }
        return GeneratorSemanticPrediction(
            masks=masks,
            probabilities=probability_maps,
            class_map=winner_np,
            diagnostics=diagnostics,
        )


def prediction_to_parser_dict(prediction: GeneratorSemanticPrediction | dict | None) -> dict:
    """Normalize runtime output for ``build_semantic_masks`` integration."""
    if prediction is None:
        return {}
    if isinstance(prediction, dict):
        # Test/connector-friendly form: either already parser-shaped or
        # {masks:{...}, diagnostics:{...}}.
        if "masks" not in prediction:
            return dict(prediction)
        masks = dict(prediction.get("masks") or {})
        diagnostics = dict(prediction.get("diagnostics") or {})
    else:
        masks = prediction.masks
        diagnostics = dict(prediction.diagnostics or {})
    return {
        "hair_mask": masks.get("hair"),
        "iris_mask": masks.get("iris"),
        "skin_mask": masks.get("skin"),
        "cloth_mask": masks.get("clothes"),
        "background_mask": masks.get("background"),
        "glasses_mask": masks.get("glasses"),
        "face_shadow_mask": masks.get("face_shadow"),
        "diagnostics": diagnostics,
    }
