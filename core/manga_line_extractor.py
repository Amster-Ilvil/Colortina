"""Manga structural-line AI inference for closed-region selection.

This adapter integrates the official MIT-licensed
``MangaLineExtraction_PyTorch`` network without changing mc-v2.  It runs only
when the user requests rectangle/lasso closed-region detection, on the original
black-and-white source crop.  The returned value is a full-resolution 0..1
line-probability map consumed by :mod:`core.structural_line_detector`.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import cv2
import numpy as np


class MangaLineModelError(RuntimeError):
    """Raised when the structural-line model cannot be loaded or executed."""


@dataclass(frozen=True)
class MangaLineInference:
    probability: np.ndarray
    device: str
    crop_bbox: tuple[int, int, int, int]


_MODEL = None
_MODEL_KEY: tuple[str, str] | None = None
_MODEL_LOCK = threading.RLock()


def _choose_device(requested: str = "auto") -> str:
    import torch

    value = str(requested or "auto").strip().lower()
    if value not in {"auto", "cpu", "mps", "cuda"}:
        value = "auto"
    if value == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "mps":
        backend = getattr(torch.backends, "mps", None)
        return "mps" if backend is not None and backend.is_available() else "cpu"
    if value == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    backend = getattr(torch.backends, "mps", None)
    if backend is not None and backend.is_available():
        return "mps"
    return "cpu"


def model_ready(weights_path: str) -> bool:
    try:
        return os.path.isfile(weights_path) and os.path.getsize(weights_path) >= 64 * 1024 * 1024
    except OSError:
        return False


def _unwrap_state_dict(payload):
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                payload = candidate
                break
    if not isinstance(payload, dict):
        raise MangaLineModelError("漫画线稿模型权重格式无效")
    cleaned = {}
    for key, value in payload.items():
        name = str(key)
        if name.startswith("module."):
            name = name[7:]
        cleaned[name] = value
    return cleaned


def _load_model(weights_path: str, device: str):
    global _MODEL, _MODEL_KEY
    import torch
    from vendor.manga_line_extraction import res_skip

    path = os.path.abspath(weights_path)
    if not model_ready(path):
        raise MangaLineModelError(f"未找到完整的漫画线稿模型：{path}")
    key = (path, device)
    with _MODEL_LOCK:
        if _MODEL is not None and _MODEL_KEY == key:
            return _MODEL
        model = res_skip()
        try:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:  # PyTorch < 2.0 compatibility
                payload = torch.load(path, map_location="cpu")
            state = _unwrap_state_dict(payload)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise MangaLineModelError(
                    "漫画线稿权重与网络结构不匹配："
                    f"缺少 {len(missing)} 项，多出 {len(unexpected)} 项")
            model.eval()
            model.to(device)
        except MangaLineModelError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise MangaLineModelError(f"加载漫画线稿模型失败：{exc}") from exc
        _MODEL = model
        _MODEL_KEY = key
        return model


def release_model() -> None:
    """Release the cached model and accelerator buffers."""
    global _MODEL, _MODEL_KEY
    with _MODEL_LOCK:
        _MODEL = None
        _MODEL_KEY = None
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        backend = getattr(torch.backends, "mps", None)
        if backend is not None and backend.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _selection_bbox(mask: np.ndarray | None, shape: tuple[int, int],
                    pad: int = 40) -> tuple[int, int, int, int]:
    h, w = shape
    if mask is None or mask.size == 0 or not np.any(mask > 0):
        return 0, 0, w, h
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.nonzero(mask > 0)
    return (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(w, int(xs.max()) + pad + 1),
        min(h, int(ys.max()) + pad + 1),
    )


def _fit_inference_size(gray: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = gray.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return gray, 1.0
    scale = float(max_side) / float(longest)
    resized = cv2.resize(
        gray, (max(16, int(round(w * scale))), max(16, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA)
    return resized, scale


def _output_to_line_probability(output: np.ndarray,
                                input_gray: np.ndarray) -> np.ndarray:
    """Convert the network grayscale output to a line probability.

    Official weights normally emit black structural lines on white.  The small
    orientation check keeps the adapter robust to converted/quantized weights
    that invert the output convention.
    """
    out = np.nan_to_num(np.asarray(output, np.float32), nan=255.0,
                        posinf=255.0, neginf=0.0)
    out = np.clip(out, 0.0, 255.0)
    dark_candidate = 1.0 - out / 255.0
    light_candidate = out / 255.0

    gray = np.asarray(input_gray, np.float32)
    dark_cut = float(np.percentile(gray, 18.0))
    light_cut = float(np.percentile(gray, 82.0))
    ink = gray <= dark_cut
    paper = gray >= light_cut

    def score(candidate: np.ndarray) -> float:
        if not np.any(ink) or not np.any(paper):
            return 0.0
        return float(candidate[ink].mean() - candidate[paper].mean())

    probability = (dark_candidate
                   if score(dark_candidate) >= score(light_candidate)
                   else light_candidate)
    low = float(np.percentile(probability, 8.0))
    high = float(np.percentile(probability, 99.4))
    if high - low > 1e-5:
        probability = (probability - low) / (high - low)
    probability = np.clip(probability, 0.0, 1.0)
    return cv2.GaussianBlur(probability.astype(np.float32), (0, 0), 0.35)


def _infer_once(gray: np.ndarray, weights_path: str, device: str) -> np.ndarray:
    import torch

    model = _load_model(weights_path, device)
    h, w = gray.shape[:2]
    rows = int(np.ceil(h / 16.0)) * 16
    cols = int(np.ceil(w / 16.0)) * 16
    patch = np.full((1, 1, rows, cols), 255.0, dtype=np.float32)
    patch[0, 0, :h, :w] = gray.astype(np.float32)
    tensor = torch.from_numpy(patch).to(device)
    with _MODEL_LOCK, torch.inference_mode():
        output = model(tensor)
        result = output.detach().to("cpu").numpy()[0, 0, :h, :w]
    return result


def extract_line_probability(
    source_bgr: np.ndarray,
    selection_mask: np.ndarray | None,
    *,
    weights_path: str,
    requested_device: str = "auto",
    max_side: int = 1024,
    context_pad: int = 40,
) -> MangaLineInference:
    """Run AI line extraction on the selected source crop.

    The source is always converted from the original imported page, never from
    the mc-v2 colour result.  Only the rectangle plus context is inferred, which
    keeps latency and memory much lower than full-page extraction.
    """
    if source_bgr is None or source_bgr.size == 0:
        raise MangaLineModelError("没有可用于线条识别的原始图片")
    if source_bgr.ndim == 2:
        gray_full = source_bgr.astype(np.uint8)
    else:
        gray_full = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray_full.shape[:2]
    bbox = _selection_bbox(selection_mask, (h, w), pad=max(8, int(context_pad)))
    x1, y1, x2, y2 = bbox
    crop = gray_full[y1:y2, x1:x2]
    if crop.size == 0:
        raise MangaLineModelError("矩形选区为空，无法识别线条")
    infer_gray, scale = _fit_inference_size(crop, max(256, int(max_side)))
    device = _choose_device(requested_device)
    try:
        raw = _infer_once(infer_gray, weights_path, device)
    except Exception as exc:  # noqa: BLE001
        # Some old operators/weights can fail on a particular MPS/CUDA build.
        # A deterministic CPU retry is preferable to silently dropping AI lines.
        if device != "cpu":
            release_model()
            device = "cpu"
            try:
                raw = _infer_once(infer_gray, weights_path, device)
            except Exception as cpu_exc:  # noqa: BLE001
                raise MangaLineModelError(
                    f"漫画线稿模型推理失败：{cpu_exc}") from cpu_exc
        else:
            raise MangaLineModelError(f"漫画线稿模型推理失败：{exc}") from exc
    probability = _output_to_line_probability(raw, infer_gray)
    if scale != 1.0:
        probability = cv2.resize(
            probability, (crop.shape[1], crop.shape[0]),
            interpolation=cv2.INTER_LINEAR)
    full = np.zeros((h, w), np.float32)
    full[y1:y2, x1:x2] = np.clip(probability, 0.0, 1.0)
    if selection_mask is not None:
        selection = selection_mask
        if selection.shape[:2] != (h, w):
            selection = cv2.resize(selection, (w, h), interpolation=cv2.INTER_NEAREST)
        # Keep a narrow context halo for gap repair, but prevent remote AI lines
        # from affecting another panel.
        halo = cv2.dilate((selection > 0).astype(np.uint8),
                          np.ones((9, 9), np.uint8), iterations=1)
        full[halo == 0] = 0.0
    return MangaLineInference(full, device, bbox)
