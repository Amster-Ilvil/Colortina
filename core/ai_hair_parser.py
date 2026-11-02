"""AI hair parsing with real upstream backend formats.

Backends:
- StyleAnime / Danbooru-Parsing: PyTorch BiSeNet-19 (main)
- Anime-Face-Segmentation: PyTorch MobileNetV2 U-Net-7 (fallback)
- SkyTNT anime-segmentation: ONNX character foreground (anti-leak)
- yakhyo face-parsing: ONNX BiSeNet-19 (skin/eyeglass/cloth auxiliary)

All models are optional and downloaded only after user confirmation.  Parsing is
performed per detected character crop rather than on a whole comic page, which
is essential for multi-panel / multi-character pages.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import importlib.util
import threading
import time
import zlib

import cv2
import numpy as np


@dataclass
class HairParsingResult:
    hair_mask: np.ndarray | None
    eye_mask: np.ndarray | None = None
    skin_mask: np.ndarray | None = None
    cloth_mask: np.ndarray | None = None
    eyeglass_mask: np.ndarray | None = None
    face_mask: np.ndarray | None = None
    foreground_mask: np.ndarray | None = None
    diagnostics: dict | None = None


_TORCH_MODEL_CACHE: dict[tuple[str, str, str], object] = {}
_ORT_SESSION_CACHE: dict[str, object] = {}
_ANIMESEG_PIPELINE_CACHE: dict[str, object] = {}
_PARSE_CACHE: dict[tuple, HairParsingResult] = {}
_CACHE_LOCK = threading.RLock()
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _model_path(name: str) -> str:
    from config import Config
    paths = {
        "styleanime": getattr(Config, "STYLEANIME_PARSING_PATH", ""),
        "anime_face_segmentation": getattr(Config, "ANIME_FACE_SEG_PATH", ""),
        "yakhyo_face_parsing": getattr(Config, "YAKHYO_FACE_PARSING_ONNX_PATH", ""),
        "skytnt_foreground": getattr(Config, "CHAR_SEG_PATH", ""),
    }
    return str(paths.get(name, ""))


def _valid_model(path: str, min_bytes: int = 1024 * 1024) -> bool:
    try:
        return bool(path) and os.path.isfile(path) and os.path.getsize(path) >= min_bytes
    except OSError:
        return False


def available_backends() -> dict[str, bool]:
    animeseg_ready = bool(
        importlib.util.find_spec("anime_seg") is not None and
        os.environ.get("COLORTINA_ANIMESEG_ENABLE", "0") == "1")
    try:
        from core.hair_model_manager import model_status
        rows = model_status()
        result = {key: bool(rows[key]["ready"]) for key in rows}
        result["animeseg"] = animeseg_ready
        return result
    except Exception:
        return {
            "animeseg": animeseg_ready,
            "styleanime": _valid_model(_model_path("styleanime"), 8 * 1024 * 1024),
            "anime_face_segmentation": _valid_model(
                _model_path("anime_face_segmentation"), 5 * 1024 * 1024),
            "skytnt_foreground": _valid_model(
                _model_path("skytnt_foreground"), 150 * 1024 * 1024),
            "yakhyo_face_parsing": _valid_model(
                _model_path("yakhyo_face_parsing"), 35 * 1024 * 1024),
        }


def _fingerprint(image_bgr: np.ndarray, backend: str, paths: tuple[str, ...]) -> tuple:
    h, w = image_bgr.shape[:2]
    sy, sx = max(1, h // 32), max(1, w // 32)
    sample = np.ascontiguousarray(image_bgr[::sy, ::sx][:32, :32])
    mtimes = []
    for path in paths:
        try:
            stat = os.stat(path)
            mtimes.append((stat.st_size, stat.st_mtime_ns))
        except OSError:
            mtimes.append((0, 0))
    return (backend, h, w, int(zlib.adler32(sample.tobytes())), tuple(mtimes))


def clear_cache() -> None:
    with _CACHE_LOCK:
        _PARSE_CACHE.clear()


def _device():
    import torch
    try:
        from config import Config
        from core.device_runtime import resolve_torch_device
        device, report = resolve_torch_device(
            getattr(Config, "ML_DEVICE", "auto"), torch_module=torch,
            smoke_test=False)
        return device, report.selected
    except Exception:
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps"), "mps"
        return torch.device("cpu"), "cpu"


def _unwrap_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "network"):
            value = obj.get(key)
            if isinstance(value, dict):
                obj = value
                break
    if not isinstance(obj, dict):
        raise RuntimeError("checkpoint does not contain a state_dict")
    cleaned = {}
    for key, value in obj.items():
        name = str(key)
        for prefix in ("module.", "model.", "net."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        cleaned[name] = value
    return cleaned


def _load_torch_model(backend: str):
    import torch
    path = _model_path(backend)
    if not _valid_model(path):
        return None, "missing"
    device, device_name = _device()
    cache_key = (backend, path, device_name)
    with _CACHE_LOCK:
        cached = _TORCH_MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached, device_name

        if backend == "styleanime":
            from core.third_party.hair_parsing.bisenet19 import build_bisenet19
            model = build_bisenet19(19)
        elif backend == "anime_face_segmentation":
            from core.third_party.hair_parsing.anime_face_unet import build_anime_face_unet
            model = build_anime_face_unet()
        else:
            raise ValueError(f"unsupported torch backend: {backend}")

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
        state = _unwrap_state_dict(checkpoint)
        missing, unexpected = model.load_state_dict(state, strict=False)
        # A few wrapper prefixes / auxiliary heads are acceptable. A mostly
        # incompatible checkpoint is not.
        total = max(1, len(model.state_dict()))
        if len(missing) > total * 0.35:
            raise RuntimeError(
                f"{backend} checkpoint incompatible: missing={len(missing)}, "
                f"unexpected={len(unexpected)}")
        model.eval().to(device)
        _TORCH_MODEL_CACHE[cache_key] = model
        return model, device_name


def _load_ort_session(path: str):
    with _CACHE_LOCK:
        cached = _ORT_SESSION_CACHE.get(path)
        if cached is not None:
            return cached
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        providers = ["CPUExecutionProvider"]
        session = ort.InferenceSession(path, sess_options=so, providers=providers)
        _ORT_SESSION_CACHE[path] = session
        return session


def _clip_box(box, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    x, y, bw, bh = [int(round(v)) for v in box]
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    bw = max(1, min(w - x, bw))
    bh = max(1, min(h - y, bh))
    return x, y, bw, bh


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _character_crops(source_bgr: np.ndarray) -> tuple[list[tuple[int, int, int, int]], dict]:
    h, w = source_bgr.shape[:2]
    faces = []
    detector = "none"
    try:
        from core.anime_face_detector import detect_anime_faces
        faces = detect_anime_faces(source_bgr, max_faces=24, allow_download=False)
        if faces:
            detector = "anime_lbp"
    except Exception:
        faces = []
    if not faces:
        try:
            from core.anime_face_detector import _generic_haar_faces
            faces = _generic_haar_faces(source_bgr, 24)
            if faces:
                detector = "generic_haar_strict"
        except Exception:
            faces = []

    boxes: list[tuple[int, int, int, int]] = []
    for face in faces:
        fx, fy, fw, fh = [int(v) for v in face.bbox]
        hx, hy, hw, hh = [int(v) for v in getattr(face, "head_bbox", face.bbox)]
        bx, by, bw, bh = [int(v) for v in getattr(face, "body_bbox", (hx, hy, hw, hh))]
        # Preserve context above/around the head and a long vertical corridor for
        # long hair, while avoiding an entire comic page crop for tiny faces.
        x0 = min(hx - int(0.28 * hw), bx + int(0.04 * bw), fx - int(0.85 * fw))
        x1 = max(hx + int(1.28 * hw), bx + int(0.96 * bw), fx + int(1.85 * fw))
        y0 = min(hy - int(0.14 * hh), fy - int(0.42 * fh))
        y1 = max(hy + int(3.85 * hh), by + int(0.96 * bh), fy + int(5.1 * fh))
        box = _clip_box((x0, y0, x1 - x0, y1 - y0), (h, w))
        if box[2] * box[3] < h * w * 0.0008:
            continue
        if any(_iou(box, old) > 0.72 for old in boxes):
            continue
        boxes.append(box)

    # No reliable face: whole-page parsing remains a functional fallback. Also
    # include a whole page crop for single-portrait pages, where it preserves
    # long hair outside the detector's body box.
    if not boxes:
        boxes = [(0, 0, w, h)]
        detector = "whole_page_fallback"
    elif len(boxes) == 1 and boxes[0][2] * boxes[0][3] > h * w * 0.28:
        boxes = [(0, 0, w, h)]
        detector += "+whole_page"
    return boxes, {"face_count": len(faces), "crop_count": len(boxes),
                   "crop_detector": detector,
                   "crop_boxes": [tuple(int(v) for v in box) for box in boxes]}


def _torch_labels(crop_bgr: np.ndarray, backend: str, size: int | None = None) -> tuple[np.ndarray | None, dict]:
    import torch
    model, device_name = _load_torch_model(backend)
    if size is None:
        from config import Config
        size = int(getattr(
            Config,
            "STYLEANIME_INPUT_SIZE" if backend == "styleanime"
            else "ANIME_FACE_SEG_INPUT_SIZE",
            512))
    size = max(64, int(size))
    if model is None:
        return None, {"backend": backend, "reason": "model_missing"}
    device = next(model.parameters()).device
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    tensor = resized.astype(np.float32) / 255.0
    if backend == "styleanime":
        tensor = (tensor - _IMAGENET_MEAN) / _IMAGENET_STD
    tensor = torch.from_numpy(tensor.transpose(2, 0, 1)).unsqueeze(0).to(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        labels = output.squeeze(0).argmax(0).detach().cpu().numpy().astype(np.uint8)
    labels = cv2.resize(labels, (crop_bgr.shape[1], crop_bgr.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
    return labels, {"backend": backend, "device": device_name,
                    "inference_ms": round((time.perf_counter() - started) * 1000, 2)}


def _yakhyo_labels(crop_bgr: np.ndarray, size: int = 512) -> tuple[np.ndarray | None, dict]:
    path = _model_path("yakhyo_face_parsing")
    if not _valid_model(path):
        return None, {"backend": "yakhyo_face_parsing", "reason": "model_missing"}
    try:
        session = _load_ort_session(path)
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        image = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = (image - _IMAGENET_MEAN) / _IMAGENET_STD
        inp = image.transpose(2, 0, 1)[None].astype(np.float32)
        started = time.perf_counter()
        outputs = session.run(None, {session.get_inputs()[0].name: inp})
        logits = np.asarray(outputs[0])
        labels = logits.squeeze(0).argmax(0).astype(np.uint8)
        labels = cv2.resize(labels, (crop_bgr.shape[1], crop_bgr.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        return labels, {"backend": "yakhyo_face_parsing", "device": "onnx-cpu",
                        "inference_ms": round((time.perf_counter() - started) * 1000, 2)}
    except Exception as exc:  # noqa: BLE001
        return None, {"backend": "yakhyo_face_parsing",
                      "reason": f"inference_error:{type(exc).__name__}:{exc}"}


def _mask(labels: np.ndarray | None, ids) -> np.ndarray | None:
    if labels is None:
        return None
    return np.where(np.isin(labels, np.asarray(tuple(ids), dtype=np.uint8)),
                    255, 0).astype(np.uint8)


def _paste_union(destination: np.ndarray, crop_mask: np.ndarray | None,
                 box: tuple[int, int, int, int]) -> None:
    if crop_mask is None:
        return
    x, y, w, h = box
    if crop_mask.shape[:2] != (h, w):
        crop_mask = cv2.resize(crop_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    destination[y:y + h, x:x + w] = cv2.bitwise_or(
        destination[y:y + h, x:x + w],
        np.where(crop_mask > 0, 255, 0).astype(np.uint8))


def _parse_main(image_bgr: np.ndarray, source_bgr: np.ndarray,
                backend: str) -> HairParsingResult | None:
    h, w = image_bgr.shape[:2]
    boxes, crop_diag = _character_crops(source_bgr)
    hair = np.zeros((h, w), np.uint8)
    skin = np.zeros((h, w), np.uint8)
    cloth = np.zeros((h, w), np.uint8)
    eyeglass = np.zeros((h, w), np.uint8)
    eye = np.zeros((h, w), np.uint8)
    face = np.zeros((h, w), np.uint8)
    rows = []
    for box in boxes:
        x, y, bw, bh = box
        crop = image_bgr[y:y + bh, x:x + bw]
        labels, diag = _torch_labels(crop, backend)
        diag["box"] = box
        rows.append(diag)
        if labels is None:
            continue
        if backend == "styleanime":
            _paste_union(hair, _mask(labels, (17,)), box)
            _paste_union(skin, _mask(labels, (1,)), box)
            _paste_union(cloth, _mask(labels, (16,)), box)
            _paste_union(eyeglass, _mask(labels, (6,)), box)
            _paste_union(eye, _mask(labels, (4, 5)), box)
            _paste_union(face, _mask(labels, range(1, 16)), box)
        else:  # Anime-Face-Segmentation label order is explicit in upstream util.py
            _paste_union(hair, _mask(labels, (1,)), box)
            _paste_union(eye, _mask(labels, (2,)), box)
            _paste_union(face, _mask(labels, (4,)), box)
            _paste_union(skin, _mask(labels, (5,)), box)
            _paste_union(cloth, _mask(labels, (6,)), box)
    if not any(np.any(mask) for mask in (hair, eye, skin, cloth, eyeglass, face)):
        return None
    return HairParsingResult(
        hair_mask=hair,
        eye_mask=eye,
        skin_mask=skin,
        cloth_mask=cloth,
        eyeglass_mask=eyeglass,
        face_mask=face,
        diagnostics={"backend": backend, **crop_diag, "crop_inference": rows},
    )


def _parse_yakhyo_aux(image_bgr: np.ndarray, source_bgr: np.ndarray) -> HairParsingResult | None:
    h, w = image_bgr.shape[:2]
    boxes, crop_diag = _character_crops(source_bgr)
    hair = np.zeros((h, w), np.uint8)
    skin = np.zeros((h, w), np.uint8)
    cloth = np.zeros((h, w), np.uint8)
    eyeglass = np.zeros((h, w), np.uint8)
    eye = np.zeros((h, w), np.uint8)
    face = np.zeros((h, w), np.uint8)
    rows = []
    for box in boxes:
        x, y, bw, bh = box
        labels, diag = _yakhyo_labels(image_bgr[y:y + bh, x:x + bw])
        diag["box"] = box
        rows.append(diag)
        if labels is None:
            continue
        _paste_union(hair, _mask(labels, (17,)), box)
        _paste_union(skin, _mask(labels, (1,)), box)
        _paste_union(cloth, _mask(labels, (16,)), box)
        _paste_union(eyeglass, _mask(labels, (6,)), box)
        _paste_union(eye, _mask(labels, (4, 5)), box)
        _paste_union(face, _mask(labels, [i for i in range(1, 19)
                                         if i not in (16, 17, 18)]), box)
    if not any(np.any(mask) for mask in (hair, eye, skin, cloth, eyeglass, face)):
        return None
    return HairParsingResult(
        hair_mask=hair, eye_mask=eye, skin_mask=skin, cloth_mask=cloth,
        eyeglass_mask=eyeglass, face_mask=face,
        diagnostics={"backend": "yakhyo_face_parsing", **crop_diag,
                     "crop_inference": rows},
    )


def _skytnt_foreground(image_bgr: np.ndarray) -> np.ndarray | None:
    try:
        from core.character_scope import character_likelihood
        probability = character_likelihood(image_bgr)
        if probability is None:
            return None
        return np.where(probability >= 0.28, 255, 0).astype(np.uint8)
    except Exception:
        return None


def _main_order(preferred: str) -> list[str]:
    key = str(preferred or "auto").strip().lower()
    if key in {"animeseg", "anime_seg", "mask2former"}:
        return ["animeseg"]
    if key in {"styleanime", "danbooru-parsing", "danbooru_parsing"}:
        return ["styleanime"]
    if key in {"anime-face-segmentation", "anime_face_segmentation"}:
        return ["anime_face_segmentation"]
    if key in {"legacy", "none", "off"}:
        return []
    return ["animeseg", "anime_face_segmentation", "styleanime"]


def _animeseg_explicit(preferred: str) -> bool:
    return str(preferred or "").strip().lower() in {"animeseg", "anime_seg", "mask2former"}


def _semantic_backend_ready(backend: str, preferred: str, available: dict[str, bool]) -> bool:
    if backend != "animeseg":
        return bool(available.get(backend, False))
    # Auto mode stays opt-in through COLORTINA_ANIMESEG_ENABLE to avoid a
    # surprise heavy backend. Explicit UI selection, however, is itself user
    # intent and must work whenever the package is installed.
    return bool(available.get("animeseg", False) or
                (_animeseg_explicit(preferred) and importlib.util.find_spec("anime_seg") is not None))


def _parse_animeseg(image_bgr: np.ndarray, *, allow_download: bool = False) -> HairParsingResult | None:
    """Optional AnimeSeg Mask2Former adapter.

    The dependency/model is never enabled silently. Set COLORTINA_ANIMESEG_ENABLE=1
    (or select the backend through an integration that does so) after installing
    ``anime_seg``. This prevents a surprise Hugging Face download in offline Mac jobs.
    """
    if importlib.util.find_spec("anime_seg") is None:
        return None
    try:
        from anime_seg import AnimeSegPipeline
        key = "mask2former"
        pipe = _ANIMESEG_PIPELINE_CACHE.get(key)
        if pipe is None:
            # AnimeSeg resolves weights through Hugging Face.  Offline mode is
            # intentionally scoped to this construction attempt; do not leave
            # process-wide HF/Transformers flags behind because another Colortina
            # model downloader may legitimately run later in the same session.
            old_hf_offline = os.environ.get("HF_HUB_OFFLINE")
            old_tf_offline = os.environ.get("TRANSFORMERS_OFFLINE")
            try:
                if not allow_download:
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    os.environ["TRANSFORMERS_OFFLINE"] = "1"
                pipe = AnimeSegPipeline.from_mask2former()
            finally:
                if old_hf_offline is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = old_hf_offline
                if old_tf_offline is None:
                    os.environ.pop("TRANSFORMERS_OFFLINE", None)
                else:
                    os.environ["TRANSFORMERS_OFFLINE"] = old_tf_offline
            device, _name = _device()
            pipe = pipe.to(str(device))
            _ANIMESEG_PIPELINE_CACHE[key] = pipe
        rgb = cv2.cvtColor(image_bgr.astype(np.uint8), cv2.COLOR_BGR2RGB)
        from PIL import Image
        result = pipe(Image.fromarray(rgb), width=image_bgr.shape[1], height=image_bgr.shape[0])
        arr = np.asarray(result)
        if arr.ndim == 2:
            labels = arr.astype(np.uint8)
        else:
            # Published AnimeSeg Mask2Former palette.
            palette = {
                0: (0, 0, 0), 1: (255, 220, 180), 2: (100, 150, 255),
                3: (255, 0, 0), 4: (0, 255, 255), 5: (255, 255, 0),
                6: (150, 255, 0), 7: (0, 255, 100), 8: (255, 140, 0),
                9: (255, 0, 150), 10: (180, 0, 255), 11: (128, 128, 0),
            }
            labels = np.zeros(arr.shape[:2], np.uint8)
            rgb_arr = arr[..., :3].astype(np.int16)
            for idx, color in palette.items():
                dist = np.max(np.abs(rgb_arr - np.asarray(color, np.int16)), axis=2)
                labels[dist <= 3] = int(idx)
        hair = np.where(labels == 3, 255, 0).astype(np.uint8)
        eye = np.where((labels == 4) | (labels == 5), 255, 0).astype(np.uint8)
        skin = np.where(labels == 1, 255, 0).astype(np.uint8)
        face = np.where(labels == 2, 255, 0).astype(np.uint8)
        cloth = np.where(labels == 10, 255, 0).astype(np.uint8)
        foreground = np.where(labels != 0, 255, 0).astype(np.uint8)
        if not any(np.any(x) for x in (hair, eye, skin, face, cloth)):
            return None
        return HairParsingResult(
            hair_mask=hair, eye_mask=eye, skin_mask=skin, face_mask=face,
            cloth_mask=cloth, foreground_mask=foreground,
            diagnostics={"backend": "animeseg_mask2former", "classes": 12})
    except Exception as exc:
        return HairParsingResult(hair_mask=None, eye_mask=None,
                                 diagnostics={"backend": "animeseg_mask2former",
                                              "error": f"{type(exc).__name__}: {exc}"})


def _candidate_quality(candidate: HairParsingResult, source_bgr: np.ndarray,
                       semantic_bgr: np.ndarray) -> dict:
    """Score a parser mask against the actual mc-v2 warm-hair failure signal."""
    hair = None if candidate is None else candidate.hair_mask
    if hair is None or not np.any(hair):
        return {"score": -1.0, "reason": "empty"}
    mask = hair > 0
    gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(semantic_bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    warm = (((hue <= 24) | (hue >= 170)) & (sat >= 72) & (val >= 24))
    hair_pixels = int(mask.sum())
    warm_pixels = int(warm.sum())
    overlap = int(np.count_nonzero(mask & warm))
    warm_recall = overlap / max(1, warm_pixels)
    warm_precision = overlap / max(1, hair_pixels)
    dark_ratio = float(np.mean(gray[mask] < 160)) if hair_pixels else 0.0
    mean_gray = float(np.mean(gray[mask])) if hair_pixels else 255.0
    fraction = hair_pixels / max(1, gray.size)
    size_score = 1.0 if 0.00008 <= fraction <= 0.30 else max(0.0, 1.0 - abs(fraction - 0.12) * 4.0)
    score = (0.40 * min(1.0, warm_recall * 2.4) +
             0.30 * min(1.0, warm_precision * 5.0) +
             0.20 * dark_ratio + 0.10 * size_score)
    if fraction > 0.42:
        score -= 0.35
    if mean_gray > 215:
        score -= 0.18
    return {
        "score": round(float(score), 6),
        "hair_pixels": hair_pixels,
        "hair_fraction": round(float(fraction), 6),
        "warm_overlap_pixels": overlap,
        "warm_recall": round(float(warm_recall), 6),
        "warm_precision": round(float(warm_precision), 6),
        "dark_ratio": round(float(dark_ratio), 6),
        "mean_source_gray": round(float(mean_gray), 3),
    }


def _refine(hair: np.ndarray, main: HairParsingResult,
            aux: HairParsingResult | None,
            foreground: np.ndarray | None) -> np.ndarray:
    result = np.where(hair > 0, 255, 0).astype(np.uint8)
    # yakhyo hair is used only as local support around the anime parser's result,
    # never as an independent whole-page hair detector.
    if aux is not None and aux.hair_mask is not None and np.any(result):
        radius = max(9, min(41, int(round(min(result.shape) * 0.018))))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (radius * 2 + 1, radius * 2 + 1))
        nearby = cv2.dilate((result > 0).astype(np.uint8), kernel, 1) > 0
        support = (aux.hair_mask > 0) & nearby
        result[support] = 255

    exclusion = np.zeros_like(result)
    for candidate in (
            main.skin_mask, main.cloth_mask, main.eyeglass_mask,
            aux.skin_mask if aux else None,
            aux.cloth_mask if aux else None,
            aux.eyeglass_mask if aux else None):
        if candidate is not None:
            exclusion = cv2.bitwise_or(
                exclusion, np.where(candidate > 0, 255, 0).astype(np.uint8))
    if np.any(exclusion):
        # Preserve a tiny border at hair/face contacts; aggressive subtraction
        # otherwise cuts off bangs.
        exclusion = cv2.erode(exclusion, np.ones((3, 3), np.uint8), 1)
        result = cv2.bitwise_and(result, cv2.bitwise_not(exclusion))
    if foreground is not None:
        foreground_soft = cv2.dilate((foreground > 0).astype(np.uint8),
                                     np.ones((5, 5), np.uint8), 1) * 255
        result = cv2.bitwise_and(result, foreground_soft)

    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE,
                              np.ones((7, 7), np.uint8), iterations=1)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN,
                              np.ones((3, 3), np.uint8), iterations=1)
    page_area = max(1, result.shape[0] * result.shape[1])
    binary = (result > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    kept = np.zeros_like(result)
    minimum = max(36, int(page_area * 0.000035))
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum:
            kept[labels == label] = 255
    return kept


def parse_hair_regions(image_bgr: np.ndarray,
                       *, source_bgr: np.ndarray | None = None,
                       preferred_backend: str = "auto",
                       allow_download: bool = False,
                       callback=None) -> HairParsingResult:
    """Return a merged hair mask and auxiliary exclusion masks.

    When ``allow_download`` is true this function may prepare the requested
    models, but normal GUI use performs that step in a dedicated download thread
    after explicit confirmation.
    """
    image = np.asarray(image_bgr)
    source = image if source_bgr is None else np.asarray(source_bgr)
    if image.ndim == 2:
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if source.ndim == 2:
        source = cv2.cvtColor(source.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if source.shape[:2] != image.shape[:2]:
        source = cv2.resize(source, (image.shape[1], image.shape[0]),
                            interpolation=cv2.INTER_AREA)

    requested_order = _main_order(preferred_backend)
    if allow_download and requested_order:
        try:
            from core.hair_model_manager import ensure_models
            keys = [k for k in requested_order if k != "animeseg"]
            keys += ["skytnt_foreground", "yakhyo_face_parsing"]
            if keys:
                ensure_models(keys, callback=callback)
        except Exception as exc:  # fallback remains available
            if callback:
                callback(f"AI 发色模型准备失败，将回退：{exc}")

    available = available_backends()
    paths = tuple(_model_path(name) for name in (
        "styleanime", "anime_face_segmentation",
        "skytnt_foreground", "yakhyo_face_parsing"))
    key = _fingerprint(image, str(preferred_backend), paths)
    with _CACHE_LOCK:
        cached = _PARSE_CACHE.get(key)
        if cached is not None:
            diagnostics = dict(cached.diagnostics or {})
            diagnostics["cache_hit"] = True
            return HairParsingResult(
                hair_mask=None if cached.hair_mask is None else cached.hair_mask.copy(),
                eye_mask=None if cached.eye_mask is None else cached.eye_mask.copy(),
                skin_mask=None if cached.skin_mask is None else cached.skin_mask.copy(),
                cloth_mask=None if cached.cloth_mask is None else cached.cloth_mask.copy(),
                eyeglass_mask=None if cached.eyeglass_mask is None else cached.eyeglass_mask.copy(),
                face_mask=None if cached.face_mask is None else cached.face_mask.copy(),
                foreground_mask=(None if cached.foreground_mask is None
                                 else cached.foreground_mask.copy()),
                diagnostics=diagnostics,
            )

    started = time.perf_counter()
    diagnostics: dict = {
        "requested_backend": str(preferred_backend or "auto"),
        "available_backends": dict(available),
        "selected_main_backend": None,
        "attempted_main_backends": [],
        "used_foreground_backend": False,
        "used_aux_backend": False,
        "cache_hit": False,
    }

    main = None
    main_backend = None
    parsed_candidates: list[tuple[str, HairParsingResult, dict]] = []
    auto_mode = str(preferred_backend or "auto").strip().lower() == "auto"
    for backend in requested_order:
        if not _semantic_backend_ready(backend, preferred_backend, available):
            continue
        diagnostics["attempted_main_backends"].append(backend)
        try:
            candidate = (_parse_animeseg(image, allow_download=allow_download)
                         if backend == "animeseg" else _parse_main(image, source, backend))
        except Exception as exc:  # noqa: BLE001
            diagnostics[f"{backend}_error"] = f"{type(exc).__name__}: {exc}"
            continue
        if candidate is None or candidate.hair_mask is None or not np.any(candidate.hair_mask):
            continue
        quality = _candidate_quality(candidate, source, image)
        parsed_candidates.append((backend, candidate, quality))
        diagnostics.setdefault("backend_quality", {})[backend] = quality
        # Anime-Face-Segmentation is the default manga-oriented backend. When
        # it already agrees well with the warm mc-v2 hair signal, skip the
        # slower StyleAnime pass instead of letting a broad StyleAnime mask
        # override a cleaner result.
        if auto_mode and backend == "anime_face_segmentation":
            fraction = float(quality.get("hair_fraction", 1.0))
            # Prefer the manga-specific lightweight parser whenever its mask is
            # not catastrophically broad. StyleAnime is retained only as a true
            # fallback, because its wide semantic masks can include glasses or
            # clothing on black-and-white manga.
            if fraction < 0.45 and quality.get("score", -1) > -0.10:
                break

    if parsed_candidates:
        selected_row = None
        if auto_mode:
            for row in parsed_candidates:
                if row[0] == "anime_face_segmentation":
                    fraction = float(row[2].get("hair_fraction", 1.0))
                    if fraction < 0.45 and row[2].get("score", -1) > -0.10:
                        selected_row = row
                        break
        if selected_row is None:
            parsed_candidates.sort(key=lambda row: row[2].get("score", -1), reverse=True)
            selected_row = parsed_candidates[0]
        main_backend, main, selected_quality = selected_row
        diagnostics["selected_main_backend"] = main_backend
        diagnostics["selected_backend_quality"] = selected_quality
        diagnostics["main_backend_diagnostics"] = main.diagnostics

    aux = None
    if available.get("yakhyo_face_parsing", False):
        try:
            aux = _parse_yakhyo_aux(image, source)
            diagnostics["used_aux_backend"] = aux is not None
            if aux is not None:
                diagnostics["aux_backend_diagnostics"] = aux.diagnostics
        except Exception as exc:  # noqa: BLE001
            diagnostics["aux_backend_error"] = f"{type(exc).__name__}: {exc}"

    foreground = None
    if available.get("skytnt_foreground", False):
        foreground = _skytnt_foreground(image)
        diagnostics["used_foreground_backend"] = foreground is not None

    if main is None:
        diagnostics.update({
            "reason": "no_main_backend_result",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        })
        result = HairParsingResult(
            hair_mask=None,
            eye_mask=aux.eye_mask if aux else None,
            skin_mask=aux.skin_mask if aux else None,
            cloth_mask=aux.cloth_mask if aux else None,
            eyeglass_mask=aux.eyeglass_mask if aux else None,
            face_mask=aux.face_mask if aux else None,
            foreground_mask=foreground,
            diagnostics=diagnostics,
        )
    else:
        hair = _refine(main.hair_mask, main, aux, foreground)
        diagnostics.update({
            "reason": "ai_hair_mask_ready" if np.any(hair) else "ai_hair_mask_empty",
            "raw_hair_pixels": int(np.count_nonzero(main.hair_mask)),
            "refined_hair_pixels": int(np.count_nonzero(hair)),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        })
        result = HairParsingResult(
            hair_mask=hair,
            eye_mask=main.eye_mask if main.eye_mask is not None else (aux.eye_mask if aux else None),
            skin_mask=main.skin_mask if main.skin_mask is not None else (aux.skin_mask if aux else None),
            cloth_mask=main.cloth_mask if main.cloth_mask is not None else (aux.cloth_mask if aux else None),
            eyeglass_mask=(main.eyeglass_mask if main.eyeglass_mask is not None
                           else (aux.eyeglass_mask if aux else None)),
            face_mask=main.face_mask if main.face_mask is not None else (aux.face_mask if aux else None),
            foreground_mask=foreground,
            diagnostics=diagnostics,
        )

    with _CACHE_LOCK:
        if len(_PARSE_CACHE) >= 8:
            _PARSE_CACHE.clear()
        _PARSE_CACHE[key] = result
    return result



def parse_semantic_regions(image_bgr: np.ndarray,
                           *, source_bgr: np.ndarray | None = None,
                           preferred_backend: str = "auto",
                           allow_download: bool = False,
                           callback=None) -> HairParsingResult:
    """Return general anime semantic masks without the hair-quality gate.

    ``parse_hair_regions`` intentionally ranks backends using the historical
    warm-hair failure signal.  That is correct for the legacy hair guard but
    inappropriate for semantic-mask colour layers: an eye/face result must not
    be discarded merely because no usable hair was detected.  This function
    therefore selects the first available requested semantic backend directly
    and preserves its eye mask even on hairless/close-up crops.
    """
    image = np.asarray(image_bgr)
    source = image if source_bgr is None else np.asarray(source_bgr)
    if image.ndim == 2:
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if source.ndim == 2:
        source = cv2.cvtColor(source.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if source.shape[:2] != image.shape[:2]:
        source = cv2.resize(source, (image.shape[1], image.shape[0]),
                            interpolation=cv2.INTER_AREA)

    order = _main_order(preferred_backend)
    if allow_download and order:
        try:
            from core.hair_model_manager import ensure_models
            managed = [k for k in order if k != "animeseg"]
            if managed:
                ensure_models(managed, callback=callback)
        except Exception as exc:
            if callback:
                callback(f"AI 语义模型准备失败，将回退：{exc}")

    available = available_backends()
    diagnostics = {
        "requested_backend": str(preferred_backend or "auto"),
        "available_backends": dict(available),
        "semantic_mode": True,
        "selected_main_backend": None,
    }
    main = None
    for backend in order:
        if not _semantic_backend_ready(backend, preferred_backend, available):
            continue
        try:
            candidate = (_parse_animeseg(image, allow_download=allow_download)
                         if backend == "animeseg" else _parse_main(image, source, backend))
        except Exception as exc:  # noqa: BLE001
            diagnostics[f"{backend}_error"] = f"{type(exc).__name__}: {exc}"
            continue
        if candidate is not None:
            main = candidate
            diagnostics["selected_main_backend"] = backend
            break

    aux = None
    if available.get("yakhyo_face_parsing", False):
        try:
            aux = _parse_yakhyo_aux(image, source)
        except Exception as exc:  # noqa: BLE001
            diagnostics["aux_error"] = f"{type(exc).__name__}: {exc}"
    foreground = None
    if available.get("skytnt_foreground", False):
        foreground = _skytnt_foreground(image)

    if main is None and aux is None:
        diagnostics["reason"] = "no_semantic_backend_result"
        return HairParsingResult(
            hair_mask=None, eye_mask=None, foreground_mask=foreground,
            diagnostics=diagnostics)

    primary = main if main is not None else aux

    def choose(name):
        first = getattr(primary, name, None) if primary is not None else None
        if first is not None and np.any(first):
            return first
        second = getattr(aux, name, None) if aux is not None and aux is not primary else None
        return second if second is not None and np.any(second) else first

    hair = choose("hair_mask")
    if hair is not None and np.any(hair):
        refine_owner = primary if getattr(primary, "hair_mask", None) is hair else aux
        hair = _refine(hair, refine_owner, aux if refine_owner is primary else None, foreground)
    eye = choose("eye_mask")
    diagnostics["reason"] = "semantic_masks_ready"
    diagnostics["hair_pixels"] = int(np.count_nonzero(hair)) if hair is not None else 0
    diagnostics["eye_pixels"] = int(np.count_nonzero(eye)) if eye is not None else 0
    return HairParsingResult(
        hair_mask=hair,
        eye_mask=eye,
        skin_mask=choose("skin_mask"),
        cloth_mask=choose("cloth_mask"),
        eyeglass_mask=choose("eyeglass_mask"),
        face_mask=choose("face_mask"),
        foreground_mask=foreground,
        diagnostics=diagnostics,
    )
