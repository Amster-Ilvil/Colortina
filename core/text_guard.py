"""Text and speech-overlay protection for manga pages.

The optional comic-text-detector ONNX model remains the preferred source.  V5.6.1
also exposes a conservative local heuristic used only around an active semantic
selection, so hair/object completion can treat lettering as an *occluder* rather
than a hard terminal boundary even when the optional model is absent.

The two concepts are deliberately separate:

* ``text_occlusion_mask`` returns a logical obstacle/occluder mask that semantic
  selection may cross.
* ``protect_text_regions`` composites the original monochrome glyph/paper tones
  back at the very end, keeping lettering visually above the recoloured object.
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
                  "文字气泡保护使用局部保守回退")
            return None
        try:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.log_severity_level = 3
            _session = ort.InferenceSession(
                path, sess_options=so, providers=["CPUExecutionProvider"])
            print("[text_guard] 文字检测模型已就绪（本地 ONNX）")
        except Exception as exc:  # noqa: BLE001 — optional feature, never fatal
            print(f"[text_guard] 文字检测模型加载失败（{exc}），使用局部保守回退")
            _session = None
    return _session


def _fingerprint(image_bgr: np.ndarray) -> tuple:
    h, w = image_bgr.shape[:2]
    sy, sx = max(1, h // 24), max(1, w // 24)
    sample = np.ascontiguousarray(image_bgr[::sy, ::sx][:24, :24])
    return (id(image_bgr), h, w, int(zlib.adler32(sample.tobytes())))


def _binary_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.zeros(shape, np.uint8)
    work = np.asarray(mask)
    if work.ndim == 3:
        work = np.max(work, axis=2)
    if work.shape != shape:
        work = cv2.resize(work.astype(np.uint8), (shape[1], shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    return np.where(work > 0, 255, 0).astype(np.uint8)


def text_mask(image_bgr: np.ndarray, *, threshold: float = 0.3,
              dilate_px: int = 3, feather_px: int = 2) -> np.ndarray | None:
    """Return a float32 0..1 detector mask at page resolution, or ``None``."""
    session = _ensure_session()
    if session is None or image_bgr is None:
        return None
    key = _fingerprint(image_bgr) + (round(threshold, 3), dilate_px, feather_px)
    cached = _mask_cache.get(key)
    if cached is not None:
        return cached

    h, w = image_bgr.shape[:2]
    source = np.asarray(image_bgr)
    if source.ndim == 2:
        rgb = cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    inp = cv2.resize(rgb, (_INPUT_SIZE, _INPUT_SIZE),
                     interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    inp = inp.transpose(2, 0, 1)[None]
    try:
        outputs = session.run(None, {session.get_inputs()[0].name: inp})
    except Exception as exc:  # noqa: BLE001
        print(f"[text_guard] 推理失败（{exc}），本页改用局部保守回退")
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


def _heuristic_text_mask(image_bgr: np.ndarray,
                         vicinity_mask: np.ndarray | None = None) -> np.ndarray:
    """Conservative glyph grouping used only near an active selection.

    This is intentionally *not* a page-wide OCR substitute.  It keeps compact,
    ink-like connected components that form horizontal or vertical groups and
    discards panel rules, gutters and large filled artwork.  When a vicinity is
    supplied, only groups near that semantic object survive.
    """
    source = np.asarray(image_bgr)
    if source.ndim == 2:
        gray = source.astype(np.uint8, copy=False)
    else:
        gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    page_area = max(1, h * w)
    dark = (gray <= 92).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    glyph = np.zeros((h, w), np.uint8)
    kept_labels: list[int] = []
    max_area = max(128, int(page_area * 0.040))
    max_w = max(18, int(w * 0.20))
    max_h = max(18, int(h * 0.20))
    for label in range(1, count):
        x, y, bw, bh, area = [int(v) for v in stats[label]]
        if area < 5 or area > max_area or bw < 1 or bh < 1:
            continue
        aspect = bw / max(1.0, float(bh))
        fill = area / max(1.0, float(bw * bh))
        # Long, almost one-dimensional structures are panel rules or speed lines.
        if ((bw > max_w and bh <= 5) or (bh > max_h and bw <= 5) or
                aspect > 18.0 or aspect < 1.0 / 18.0):
            continue
        if fill < 0.025:
            continue
        glyph[labels == label] = 1
        kept_labels.append(label)

    if not kept_labels:
        return np.zeros((h, w), np.uint8)

    # Japanese manga text can run in either direction.  Join close glyphs with
    # direction-specific kernels, then retain clusters containing multiple
    # components (or one genuinely large sound-effect glyph).
    scale = max(1.0, max(h, w) / 1000.0)
    long_gap = max(5, int(round(11 * scale))) | 1
    short_gap = max(3, int(round(3 * scale))) | 1
    grouped = cv2.dilate(glyph, np.ones((short_gap, long_gap), np.uint8))
    grouped |= cv2.dilate(glyph, np.ones((long_gap, short_gap), np.uint8))
    group_count, group_labels, group_stats, _ = cv2.connectedComponentsWithStats(
        grouped, connectivity=8)
    result = np.zeros((h, w), np.uint8)
    for group_id in range(1, group_count):
        gx, gy, gw, gh, group_area = [int(v) for v in group_stats[group_id]]
        if group_area <= 0:
            continue
        group_pixels = group_labels == group_id
        members = np.unique(labels[group_pixels & (glyph > 0)])
        members = members[members > 0]
        original_ink = int(np.count_nonzero(glyph & group_pixels.astype(np.uint8)))
        large_single = original_ink >= max(40, int(page_area * 0.00025))
        if len(members) < 2 and not large_single:
            continue
        # Vertical Japanese text may legitimately be tall and horizontal
        # sound effects may be wide. Reject only clusters that occupy a large
        # span in *both* axes (typical artwork/panel mass), or an implausibly
        # large share of the page.
        if ((gw > int(w * 0.45) and gh > int(h * 0.45)) or
                group_area > int(page_area * 0.22)):
            continue
        ink_pixels = glyph.astype(bool) & group_pixels
        yy, xx = np.where(ink_pixels)
        if xx.size >= 3:
            points = np.column_stack([xx, yy]).astype(np.int32)
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(result, hull, 255)
        else:
            result[ink_pixels] = 255

    if vicinity_mask is not None and np.any(result):
        vicinity = _binary_mask(vicinity_mask, (h, w))
        radius = max(12, int(round(max(h, w) * 0.055)))
        distance = cv2.distanceTransform((vicinity == 0).astype(np.uint8),
                                         cv2.DIST_L2, 5)
        result[distance > float(radius)] = 0
    return result


def text_occlusion_mask(image_bgr: np.ndarray, *, threshold: float = 0.26,
                         dilate_px: int = 3,
                         vicinity_mask: np.ndarray | None = None,
                         use_fallback: bool = True) -> np.ndarray | None:
    """Return a uint8 logical occlusion mask for selection completion.

    Detector output is preferred.  The fallback is intentionally enabled only
    by callers that provide/accept a conservative local heuristic.
    """
    if image_bgr is None:
        return None
    h, w = np.asarray(image_bgr).shape[:2]
    detected = text_mask(image_bgr, threshold=threshold, dilate_px=0, feather_px=0)
    if detected is not None and float(detected.max()) > 0.0:
        result = np.where(detected >= 0.20, 255, 0).astype(np.uint8)
    elif use_fallback:
        result = _heuristic_text_mask(image_bgr, vicinity_mask=vicinity_mask)
    else:
        return None

    if vicinity_mask is not None and np.any(result):
        vicinity = _binary_mask(vicinity_mask, (h, w))
        radius = max(16, int(round(max(h, w) * 0.065)))
        distance = cv2.distanceTransform((vicinity == 0).astype(np.uint8),
                                         cv2.DIST_L2, 5)
        result[distance > float(radius)] = 0
    if dilate_px > 0 and np.any(result):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        result = cv2.dilate(result, kernel, iterations=1)
    return np.where(result > 0, 255, 0).astype(np.uint8)


def _extreme_tone_gate(source_bw_bgr: np.ndarray, *,
                       black_max: int = 92, white_min: int = 210,
                       soften_px: int = 18) -> np.ndarray:
    """Return a 0..1 gate that only keeps clear bubble/text tones."""
    if source_bw_bgr.ndim == 2:
        gray = source_bw_bgr
    else:
        gray = cv2.cvtColor(source_bw_bgr, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.float32)
    soften = max(1.0, float(soften_px))
    black = np.clip((float(black_max) + soften - g) / soften, 0.0, 1.0)
    white = np.clip((g - (float(white_min) - soften)) / soften, 0.0, 1.0)
    return np.maximum(black, white).astype(np.float32)


def protect_text_regions(result_bgr: np.ndarray, source_bw_bgr: np.ndarray, *,
                         mask_override: np.ndarray | None = None,
                         use_fallback: bool = False,
                         vicinity_mask: np.ndarray | None = None) -> np.ndarray:
    """Composite original lettering/paper tones over the final colour layer.

    ``vicinity_mask`` is useful for local recolour: the conservative fallback is
    then restricted to the active semantic region, avoiding page-wide false
    positives while still restoring text after the post-style colour lock.
    """
    if result_bgr is None or source_bw_bgr is None:
        return result_bgr
    h, w = result_bgr.shape[:2]
    if mask_override is not None:
        mask = _binary_mask(mask_override, (h, w)).astype(np.float32) / 255.0
    elif use_fallback:
        logical = text_occlusion_mask(
            source_bw_bgr, dilate_px=3, vicinity_mask=vicinity_mask,
            use_fallback=True)
        if logical is None:
            return result_bgr
        mask = logical.astype(np.float32) / 255.0
    else:
        mask = text_mask(source_bw_bgr)
    if mask is None or float(mask.max()) <= 0.0:
        return result_bgr

    src = np.asarray(source_bw_bgr)
    if src.shape[:2] != (h, w):
        src = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)
    if src.ndim == 2:
        src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    if vicinity_mask is not None:
        vicinity = _binary_mask(vicinity_mask, (h, w))
        # Restore only text that overlaps or immediately borders the logical
        # selection; the actual artwork outside remains byte-identical.
        vicinity = cv2.dilate(vicinity, np.ones((7, 7), np.uint8), iterations=1)
        mask = mask * (vicinity.astype(np.float32) / 255.0)
    gate = _extreme_tone_gate(src)
    alpha_2d = np.clip(mask.astype(np.float32) * gate, 0.0, 1.0)
    if float(alpha_2d.max()) <= 0.0:
        return result_bgr
    alpha = alpha_2d[..., None]
    blended = (result_bgr.astype(np.float32) * (1.0 - alpha)
               + src.astype(np.float32) * alpha)
    return np.clip(blended + 0.5, 0, 255).astype(np.uint8)
