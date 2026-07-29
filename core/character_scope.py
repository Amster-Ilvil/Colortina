"""Character foreground likelihood via SkyTNT anime-segmentation (ISNet).

Replaces the old edge-density + centre-prior heuristic behind the custom
colour bias "影响范围 = 角色 / 背景" options with a real segmentation
model. Model: ``isnetis.onnx`` (skytnt/anime-seg, Apache-2.0). Optional:
when the file is missing, callers fall back to the heuristic.
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
    return getattr(Config, "CHAR_SEG_PATH",
                   os.path.join(Config.WEIGHTS_DIR, "isnetis.onnx"))


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
            print("[character_scope] 未找到 isnetis.onnx，"
                  "角色范围回退到启发式估计")
            return None
        try:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.log_severity_level = 3
            _session = ort.InferenceSession(
                path, sess_options=so, providers=["CPUExecutionProvider"])
            print("[character_scope] 角色分割模型已就绪（本地 ONNX）")
        except Exception as exc:  # noqa: BLE001 — optional feature
            print(f"[character_scope] 角色分割模型加载失败（{exc}），回退启发式")
            _session = None
    return _session


def _fingerprint(image_bgr: np.ndarray) -> tuple:
    h, w = image_bgr.shape[:2]
    sy, sx = max(1, h // 24), max(1, w // 24)
    sample = np.ascontiguousarray(image_bgr[::sy, ::sx][:24, :24])
    return (id(image_bgr), h, w, int(zlib.adler32(sample.tobytes())))


def character_likelihood(image_bgr: np.ndarray) -> np.ndarray | None:
    """Return float32 0..1 character-foreground map at page size, or None."""
    session = _ensure_session()
    if session is None or image_bgr is None:
        return None
    key = _fingerprint(image_bgr)
    cached = _mask_cache.get(key)
    if cached is not None:
        return cached

    h, w = image_bgr.shape[:2]
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # Keep aspect ratio; pad to square with zeros like the reference impl.
    scale = _INPUT_SIZE / max(h, w)
    rh, rw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((_INPUT_SIZE, _INPUT_SIZE, 3), np.float32)
    canvas[:rh, :rw] = resized.astype(np.float32) / 255.0
    inp = canvas.transpose(2, 0, 1)[None]
    try:
        (out,) = session.run(None, {session.get_inputs()[0].name: inp})
    except Exception as exc:  # noqa: BLE001
        print(f"[character_scope] 推理失败（{exc}），回退启发式")
        return None
    alpha = np.clip(out[0, 0][:rh, :rw], 0.0, 1.0)
    mask = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = np.clip(mask, 0.0, 1.0).astype(np.float32)

    if len(_mask_cache) > 8:
        _mask_cache.clear()
    _mask_cache[key] = mask
    return mask
