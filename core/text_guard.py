"""Text / speech-bubble protection mask (comic-text-detector ONNX).

Detects text pixels on the ORIGINAL black-and-white page so the final
colour composite can keep dialogue crisp: bubbles stay paper-white and
glyphs stay ink-black instead of picking up model colour washes.

Model: ``comictextdetector.pt.onnx`` (dmMaze/comic-text-detector, GPL-3,
distributed via manga-image-translator releases). Only the ``seg`` output
is used — no OCR, no boxes, nothing leaves the machine. The model file is
optional: when missing, protection quietly becomes a no-op.
"""
from __future__ import annotations

import os
import threading
import zlib

import cv2
import numpy as np

_INPUT_SIZE = 1024
_lock = threading.Lock()
_session = None
_tried = False
_mask_cache: dict[tuple, np.ndarray] = {}


def _model_path() -> str:
    from config import Config
    return getattr(Config, "TEXT_DETECTOR_PATH",
                   os.path.join(Config.WEIGHTS_DIR, "comictextdetector.pt.onnx"))


def available() -> bool:
    return _ensure_session() is not None


def _ensure_session():
    global _session, _tried
    if _tried:
        return _session
    with _lock:
        if _tried:
            return _session
        _tried = True
        path = _model_path()
        if not os.path.isfile(path) or os.path.getsize(path) < 1024 * 1024:
            print("[text_guard] 未找到 comictextdetector.pt.onnx，"
                  "文字气泡保护自动停用")
            return None
        try:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.log_severity_level = 3
            _session = ort.InferenceSession(
                path, sess_options=so, providers=["CPUExecutionProvider"])
            print("[text_guard] 文字检测模型已就绪（本地 ONNX）")
        except Exception as exc:  # noqa: BLE001 — optional feature, never fatal
            print(f"[text_guard] 文字检测模型加载失败（{exc}），保护停用")
            _session = None
    return _session


def _fingerprint(image_bgr: np.ndarray) -> tuple:
    h, w = image_bgr.shape[:2]
    sy, sx = max(1, h // 24), max(1, w // 24)
    sample = np.ascontiguousarray(image_bgr[::sy, ::sx][:24, :24])
    return (id(image_bgr), h, w, int(zlib.adler32(sample.tobytes())))


def text_mask(image_bgr: np.ndarray, *, threshold: float = 0.3,
              dilate_px: int = 3, feather_px: int = 2) -> np.ndarray | None:
    """Return a float32 0..1 protection mask at page resolution, or None."""
    session = _ensure_session()
    if session is None or image_bgr is None:
        return None
    key = _fingerprint(image_bgr) + (round(threshold, 3), dilate_px, feather_px)
    cached = _mask_cache.get(key)
    if cached is not None:
        return cached

    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    inp = cv2.resize(rgb, (_INPUT_SIZE, _INPUT_SIZE),
                     interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    inp = inp.transpose(2, 0, 1)[None]
    try:
        outputs = session.run(None, {session.get_inputs()[0].name: inp})
    except Exception as exc:  # noqa: BLE001
        print(f"[text_guard] 推理失败（{exc}），本页跳过文字保护")
        return None
    seg = None
    for out in outputs:
        if out.ndim == 4 and out.shape[1] == 1:
            seg = out[0, 0]
            break
    if seg is None:
        return None
    if float(seg.min()) < -0.01 or float(seg.max()) > 1.01:
        seg = 1.0 / (1.0 + np.exp(-seg))
    mask = (seg >= float(threshold)).astype(np.uint8)
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        mask = cv2.dilate(mask, kernel)
    mask_f = cv2.resize(mask.astype(np.float32), (w, h),
                        interpolation=cv2.INTER_LINEAR)
    if feather_px > 0:
        mask_f = cv2.GaussianBlur(mask_f, (0, 0), feather_px)
    mask_f = np.clip(mask_f, 0.0, 1.0).astype(np.float32)

    if len(_mask_cache) > 8:
        _mask_cache.clear()
    _mask_cache[key] = mask_f
    return mask_f


def _extreme_tone_gate(source_bw_bgr: np.ndarray, *,
                       black_max: int = 92, white_min: int = 210,
                       soften_px: int = 18) -> np.ndarray:
    """Return a 0..1 gate that only keeps clear bubble/text tones.

    False-positive text masks are most harmful when they restore mid-tone art
    (hair, clothes, screentones) back to grayscale.  Real dialogue protection
    only needs the original page's near-black ink and near-white bubble paper,
    so we explicitly suppress mid-tones here.
    """
    if source_bw_bgr.ndim == 2:
        gray = source_bw_bgr
    else:
        gray = cv2.cvtColor(source_bw_bgr, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.float32)
    soften = max(1.0, float(soften_px))
    black = np.clip((float(black_max) + soften - g) / soften, 0.0, 1.0)
    white = np.clip((g - (float(white_min) - soften)) / soften, 0.0, 1.0)
    return np.maximum(black, white).astype(np.float32)


def protect_text_regions(result_bgr: np.ndarray,
                         source_bw_bgr: np.ndarray) -> np.ndarray:
    """Composite the original page back over detected text pixels.

    Protection is intentionally limited to near-black ink and near-white bubble
    paper.  This prevents occasional detector false positives from restoring
    arbitrary artwork regions to grayscale.
    """
    if result_bgr is None or source_bw_bgr is None:
        return result_bgr
    mask = text_mask(source_bw_bgr)
    if mask is None or float(mask.max()) <= 0.0:
        return result_bgr
    h, w = result_bgr.shape[:2]
    src = source_bw_bgr
    if src.shape[:2] != (h, w):
        src = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)
    if src.ndim == 2:
        src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    gate = _extreme_tone_gate(src)
    alpha_2d = np.clip(mask.astype(np.float32) * gate, 0.0, 1.0)
    if float(alpha_2d.max()) <= 0.0:
        return result_bgr
    alpha = alpha_2d[..., None]
    blended = (result_bgr.astype(np.float32) * (1.0 - alpha)
               + src.astype(np.float32) * alpha)
    return np.clip(blended + 0.5, 0, 255).astype(np.uint8)
