"""High-fidelity finishing pass for the illustrative ink-wash preset.

The stock manga-colorization-v2 generator often produces bright, flat anime
colours.  This deterministic post-pass keeps the generator's semantic colour
choices while moving the rendering toward a restrained hand-painted manga
page: source-paper luminance, softened chroma, warm natural skin, crisp ink and
light paper grain.  It deliberately does *not* decide a character's hair/eye
colour.  Identity colours are supplied by CharacterLibrary / manual hints or a
region command, and region locks run after this pass.
"""
from __future__ import annotations

import copy
import hashlib
import os
import threading
from collections import OrderedDict

import cv2
import numpy as np


_AUTO_HAIR_RESULT_CACHE: "OrderedDict[tuple, tuple[np.ndarray, dict, bool]]" = OrderedDict()
_FACE_RESULT_CACHE: "OrderedDict[tuple, list]" = OrderedDict()
_AUTO_HAIR_CACHE_LOCK = threading.RLock()
_AUTO_HAIR_RESULT_CACHE_MAX = 3
_FACE_RESULT_CACHE_MAX = 8


def _trim_lru(cache: OrderedDict, maximum: int) -> None:
    while len(cache) > int(maximum):
        cache.popitem(last=False)


def _exact_array_key(array: np.ndarray | None) -> tuple:
    """Return a collision-resistant key without changing analysis semantics."""
    if array is None:
        return ("none",)
    arr = np.asarray(array)
    if arr.size == 0:
        return ("empty", tuple(int(v) for v in arr.shape), str(arr.dtype))
    contiguous = arr if arr.flags.c_contiguous else np.ascontiguousarray(arr)
    digest = hashlib.blake2b(memoryview(contiguous), digest_size=16).hexdigest()
    return (tuple(int(v) for v in arr.shape), str(arr.dtype), digest)


def _file_signature(path: str | os.PathLike | None) -> tuple:
    try:
        value = os.fspath(path) if path else ""
        stat = os.stat(value)
        return (value, int(stat.st_size), int(stat.st_mtime_ns))
    except (OSError, TypeError, ValueError):
        return (str(path or ""), 0, 0)


def _hair_runtime_signature() -> tuple:
    """Invalidate result caches when optional parser/detector files change."""
    model_rows = []
    try:
        from config import Config
        for name in (
                "STYLEANIME_PARSING_PATH",
                "ANIME_FACE_SEG_PATH",
                "YAKHYO_FACE_PARSING_ONNX_PATH",
                "CHAR_SEG_PATH"):
            model_rows.append((name, _file_signature(getattr(Config, name, ""))))
        for name in (
                "ML_DEVICE",
                "STYLEANIME_INPUT_SIZE",
                "ANIME_FACE_SEG_INPUT_SIZE",
                "YAKHYO_FACE_PARSING_INPUT_SIZE"):
            model_rows.append((name, str(getattr(Config, name, ""))))
    except Exception:
        model_rows = []
    cascade_row = ("anime_cascade", "", 0, 0)
    try:
        from core.anime_face_detector import ensure_anime_face_cascade
        cascade_path = ensure_anime_face_cascade(allow_download=False)
        cascade_row = ("anime_cascade",) + _file_signature(cascade_path)
    except Exception:
        pass
    return (tuple(model_rows), cascade_row, os.environ.get("COLORTINA_ALLOW_GENERIC_HAAR", ""))


def _strip_preview_layers(diagnostics: dict) -> dict:
    result = copy.deepcopy(diagnostics)
    result.pop("_preview_layers", None)
    ai = result.get("ai_hair_parser")
    if isinstance(ai, dict):
        ai.pop("_preview_layers", None)
    return result


def clear_auto_hair_result_cache() -> None:
    """Clear only deterministic post-analysis caches; model caches stay loaded."""
    with _AUTO_HAIR_CACHE_LOCK:
        _AUTO_HAIR_RESULT_CACHE.clear()
        _FACE_RESULT_CACHE.clear()


def _cached_detect_anime_faces(source_bgr: np.ndarray, *, max_faces: int = 32) -> list:
    source = _ensure_bgr(source_bgr)
    key = (_exact_array_key(source), int(max_faces), _hair_runtime_signature())
    with _AUTO_HAIR_CACHE_LOCK:
        cached = _FACE_RESULT_CACHE.get(key)
        if cached is not None:
            _FACE_RESULT_CACHE.move_to_end(key)
            return list(cached)
    try:
        from core.anime_face_detector import detect_anime_faces
        faces = detect_anime_faces(source, max_faces=max_faces, allow_download=False)
    except Exception:
        faces = []
    with _AUTO_HAIR_CACHE_LOCK:
        _FACE_RESULT_CACHE[key] = list(faces)
        _trim_lru(_FACE_RESULT_CACHE, _FACE_RESULT_CACHE_MAX)
    return list(faces)


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGRA2BGR)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _normalize_limit_mask_shape(limit_mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if limit_mask is None:
        return None
    arr = np.asarray(limit_mask)
    if arr.size == 0:
        return None
    if arr.ndim == 3:
        arr = np.max(arr, axis=2)
    if arr.shape[:2] != tuple(shape):
        arr = cv2.resize(arr.astype(np.uint8), (int(shape[1]), int(shape[0])),
                         interpolation=cv2.INTER_NEAREST)
    arr = np.where(arr > 0, 255, 0).astype(np.uint8)
    if not np.any(arr):
        return None
    return arr


def _crop_auto_hair_outputs_to_limit(
        mask: np.ndarray,
        diagnostics: dict,
        limit_mask: np.ndarray | None) -> tuple[np.ndarray, dict]:
    limit = _normalize_limit_mask_shape(limit_mask, mask.shape[:2])
    final_mask = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
    if limit is None:
        diagnostics.setdefault("selection_limit_strategy", "none")
        diagnostics["selected_pixels"] = int(np.count_nonzero(final_mask))
        return final_mask, diagnostics

    pre_crop_pixels = int(np.count_nonzero(final_mask))
    cropped = cv2.bitwise_and(final_mask, limit)
    diagnostics["selection_limit_strategy"] = "detect_full_page_then_crop_final_mask"
    diagnostics["selection_limit_pixels"] = int(np.count_nonzero(limit))
    diagnostics["selected_pixels_before_limit_crop"] = pre_crop_pixels
    diagnostics["selected_pixels"] = int(np.count_nonzero(cropped))
    diagnostics["limit_masked"] = True
    diagnostics["selection_limit_removed_pixels"] = pre_crop_pixels - diagnostics["selected_pixels"]
    if diagnostics["selected_pixels"] == 0 and pre_crop_pixels > 0:
        diagnostics["reason"] = "selection_limit_cropped_all_candidates"

    preview_layers = diagnostics.get("_preview_layers")
    if isinstance(preview_layers, dict):
        for key in ("primary", "supplemental", "rejected_thin"):
            layer = preview_layers.get(key)
            if layer is None:
                continue
            layer_u8 = np.where(np.asarray(layer) > 0, 255, 0).astype(np.uint8)
            preview_layers[key] = cv2.bitwise_and(layer_u8, limit)
    return cropped, diagnostics


def _large_component_mask(mask: np.ndarray, *, min_area: int) -> np.ndarray:
    work = np.where(mask, 1, 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(work, 8)
    kept = np.zeros_like(work)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= int(min_area):
            kept[labels == label] = 1
    return kept




def _normalize_auto_black_hair_guard(mode) -> tuple[bool, str]:
    """Normalize GUI / legacy config values for automatic black-hair guard.

    Accepted values:
    - False / "off" / "disabled": disabled
    - True / "on" / "normal" / "standard": enabled, standard thresholds
    - "conservative" / "strict": enabled, stricter thresholds
    """
    if isinstance(mode, str):
        key = mode.strip().lower()
        if key in {"", "0", "false", "off", "disabled", "none"}:
            return False, "off"
        if key in {"strict", "conservative", "conservative_black"}:
            return True, "conservative"
        return True, "normal"
    return (bool(mode), "normal" if bool(mode) else "off")




def _normalize_auto_black_hair_scope(scope) -> str:
    if isinstance(scope, str):
        key = scope.strip().lower()
        if key in {"face", "near_face", "character", "character_only", "人物附近"}:
            return "near_face"
        if key in {"selection", "mask", "selected", "selected_region", "蓝框", "蓝色选区"}:
            return "selection"
        return "page"
    return "page"


def _bbox_overlap_or_close(a: tuple[int, int, int, int],
                           b: tuple[int, int, int, int],
                           *, pad_ratio: float = 0.30) -> bool:
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    pad = max(bw, bh) * float(max(0.0, pad_ratio))
    bx1, by1 = bx - pad, by - pad
    bx2, by2 = bx + bw + pad, by + bh + pad
    ax1, ay1 = ax, ay
    ax2, ay2 = ax + aw, ay + ah
    return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)






def _merge_bbox_list(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    xs0 = [int(b[0]) for b in boxes]
    ys0 = [int(b[1]) for b in boxes]
    xs1 = [int(b[0] + b[2]) for b in boxes]
    ys1 = [int(b[1] + b[3]) for b in boxes]
    x0, y0 = min(xs0), min(ys0)
    x1, y1 = max(xs1), max(ys1)
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def _cluster_warm_hair_candidates(candidates: list[dict]) -> list[dict]:
    """Group nearby warm fragments into hairstyle-level clusters.

    Root cause of the left/right alternation: the detector seeded completion from a
    single warm fragment at a time. One hairstyle with a large connected fragment
    filled well, while another hairstyle split into several red islands got only
    a partial local window. Clustering nearby fragments first lets each hairstyle
    build one shared local window and one shared visible seed mask.
    """
    if not candidates:
        return []
    n = len(candidates)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    def same_face(a: dict, b: dict) -> bool:
        fa, fb = a.get('face_bbox'), b.get('face_bbox')
        return fa is not None and fb is not None and tuple(int(v) for v in fa) == tuple(int(v) for v in fb)

    for i in range(n):
        ai = candidates[i]
        box_i = ai['bbox']
        for j in range(i + 1, n):
            aj = candidates[j]
            box_j = aj['bbox']
            should_merge = False
            if same_face(ai, aj):
                should_merge = _bbox_overlap_or_close(box_i, box_j, pad_ratio=1.05)
            elif _bbox_overlap_or_close(box_i, box_j, pad_ratio=0.72):
                should_merge = True
            else:
                # Fallback distance rule for highlight-split bangs / side strands.
                ix, iy, iw, ih = [float(v) for v in box_i]
                jx, jy, jw, jh = [float(v) for v in box_j]
                icx, icy = ix + iw * 0.5, iy + ih * 0.5
                jcx, jcy = jx + jw * 0.5, jy + jh * 0.5
                dx, dy = abs(icx - jcx), abs(icy - jcy)
                max_side = max(iw, ih, jw, jh)
                if dx <= max_side * 1.08 and dy <= max_side * 0.96:
                    should_merge = True
            if should_merge:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for idx in range(n):
        groups.setdefault(find(idx), []).append(idx)

    clustered = []
    for member_ids in groups.values():
        members = [candidates[idx] for idx in member_ids]
        boxes = [m['bbox'] for m in members]
        merged_bbox = _merge_bbox_list(boxes)
        face_members = [m for m in members if m.get('face_bbox') is not None]
        face_matched = any(bool(m.get('face_matched', False)) for m in members)
        face_bbox = None
        if face_members:
            face_bbox = min((m['face_bbox'] for m in face_members), key=lambda b: b[2] * b[3])
        clustered.append({
            'members': members,
            'member_labels': [int(m['label']) for m in members],
            'bbox': merged_bbox,
            'face_bbox': face_bbox,
            'face_matched': face_matched,
            'area': int(sum(int(m['area']) for m in members)),
            'area_fraction': float(sum(float(m['area_fraction']) for m in members)),
            'score': float(max(float(m['score']) for m in members) + 0.18 * len(members) + 0.22 * (1 if face_matched else 0)),
            'member_count': len(members),
        })
    clustered.sort(key=lambda item: (bool(item.get('face_matched', False)), item['score'], item['area']), reverse=True)
    return clustered



def _component_face_match_score(box: tuple[int, int, int, int], face) -> float:
    """Score whether a warm component likely belongs to one detected face/head."""
    head = tuple(int(v) for v in getattr(face, 'head_bbox', getattr(face, 'bbox', box)))
    body = tuple(int(v) for v in getattr(face, 'body_bbox', head))
    x, y, w, h = [float(v) for v in box]
    cx, cy = x + w * 0.5, y + h * 0.5
    hx, hy, hw, hh = [float(v) for v in head]
    bx, by, bw, bh = [float(v) for v in body]
    score = 0.0
    if _bbox_overlap_or_close(box, head, pad_ratio=0.95):
        score += 3.0
    if _bbox_overlap_or_close(box, body, pad_ratio=0.18):
        score += 0.95
    body_left = bx - bw * 0.08
    body_right = bx + bw * 1.08
    body_top = hy - hh * 0.22
    body_bottom = by + bh
    if body_left <= cx <= body_right and body_top <= cy <= body_bottom:
        score += 0.8
    dx = abs(cx - (hx + hw * 0.5)) / max(1.0, bw * 0.60)
    dy = max(0.0, cy - (hy + hh * 0.45)) / max(1.0, bh * 0.82)
    score -= 0.55 * dx
    score -= 0.20 * dy
    return float(score)


def _build_face_guided_candidate_groups(candidates: list[dict], faces: list, page_shape: tuple[int, int]) -> tuple[list[dict], set[int]]:
    """Merge warm fragments at the *hairstyle/person* level instead of per-island.

    The old failure mode (left side complete, right side incomplete or vice versa)
    came from picking one warm island at a time. Here we first associate all warm
    fragments to a detected face/body and then rebuild the hair object from that
    grouped visible evidence.
    """
    if not candidates or not faces:
        return [], set()
    height, width = page_shape
    assigned: dict[int, list[dict]] = {idx: [] for idx in range(len(faces))}
    assigned_labels: set[int] = set()
    for item in candidates:
        best_idx = None
        best_score = -1e9
        for idx, face in enumerate(faces):
            score = _component_face_match_score(item['bbox'], face)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is not None and best_score >= 0.85:
            assigned[best_idx].append(item)
            assigned_labels.update([int(v) for v in item.get('member_labels', [item.get('label', -1)]) if int(v) >= 0])
    groups: list[dict] = []
    for idx, members in assigned.items():
        if not members:
            continue
        face = faces[idx]
        member_boxes = [tuple(int(v) for v in m['bbox']) for m in members]
        merged_bbox = _merge_bbox_list(member_boxes)
        head = tuple(int(v) for v in getattr(face, 'head_bbox', getattr(face, 'bbox', merged_bbox)))
        body = tuple(int(v) for v in getattr(face, 'body_bbox', head))
        # Extend bbox downward when a close-up has long straight hair.
        mx, my, mw, mh = merged_bbox
        bx, by, bw, bh = body
        hx, hy, hw, hh = head
        x0 = min(mx, max(0, int(hx - 0.40 * hw)), max(0, int(bx + 0.14 * bw)))
        x1 = max(mx + mw, min(width, int(hx + 1.40 * hw)), min(width, int(bx + 0.86 * bw)))
        y0 = min(my, max(0, int(hy - 0.16 * hh)))
        y1 = max(my + mh, min(height, int(hy + 3.45 * hh)), min(height, int(by + 0.95 * bh)))
        merged_bbox = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))
        member_labels = []
        for m in members:
            vals = m.get('member_labels', [m.get('label')])
            member_labels.extend(int(v) for v in vals if v is not None)
        area = int(sum(int(m['area']) for m in members))
        area_fraction = float(sum(float(m['area_fraction']) for m in members))
        score = float(max(float(m['score']) for m in members) + 1.6 + 0.12 * len(members))
        groups.append({
            'members': members,
            'member_labels': sorted(set(member_labels)),
            'bbox': merged_bbox,
            'raw_bbox': _merge_bbox_list(member_boxes),
            'head_bbox': head,
            'body_bbox': body,
            'face_bbox': tuple(int(v) for v in getattr(face, 'bbox', head)),
            'face_matched': True,
            'area': area,
            'area_fraction': area_fraction,
            'score': score,
            'member_count': len(members),
            'source': 'face_group',
        })
    groups.sort(key=lambda item: (item['score'], item['area']), reverse=True)
    return groups, assigned_labels


def _build_local_limit_mask(height: int, width: int, item: dict, lm: np.ndarray | None) -> np.ndarray:
    x, y, bw, bh = [int(v) for v in item['bbox']]
    pad_x = max(28, int(round(bw * 0.22 if item.get('source') == 'face_group' else bw * 0.85)))
    pad_y = max(36, int(round(bh * 0.16 if item.get('source') == 'face_group' else bh * 0.95)))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(width, x + bw + pad_x)
    y1 = min(height, y + bh + pad_y)
    local_limit = np.zeros((height, width), np.uint8)
    local_limit[y0:y1, x0:x1] = 255

    if item.get('head_bbox') is not None:
        hx, hy, hw, hh = [int(v) for v in item['head_bbox']]
        head_x0 = max(0, int(hx - 0.24 * hw))
        head_y0 = max(0, int(hy - 0.16 * hh))
        head_x1 = min(width, int(hx + 1.24 * hw))
        head_y1 = min(height, int(hy + 1.55 * hh))
        local_limit[head_y0:head_y1, head_x0:head_x1] = 255

        cx = int(round(hx + hw * 0.5))
        corridor_half = max(18, int(round(hw * 0.72)))
        corr_x0 = max(0, cx - corridor_half)
        corr_x1 = min(width, cx + corridor_half)
        corr_y0 = max(0, int(hy + 0.06 * hh))
        corr_y1 = min(height, int(hy + 3.65 * hh))
        local_limit[corr_y0:corr_y1, corr_x0:corr_x1] = 255

    if item.get('body_bbox') is not None:
        bx, by, bw2, bh2 = [int(v) for v in item['body_bbox']]
        bx0 = max(0, int(bx + 0.12 * bw2))
        bx1 = min(width, int(bx + 0.88 * bw2))
        by0 = max(0, int(by - 0.04 * bh2))
        by1 = min(height, int(by + 0.92 * bh2))
        local_limit[by0:by1, bx0:bx1] = 255

    if item.get('face_bbox') is not None:
        fx, fy, fw, fh = [int(v) for v in item['face_bbox']]
        fpad_x = max(16, int(round(fw * 0.48)))
        fpad_y = max(18, int(round(fh * 0.58)))
        fx0, fy0 = max(0, fx - fpad_x), max(0, fy - fpad_y)
        fx1, fy1 = min(width, fx + fw + fpad_x), min(height, fy + fh + fpad_y)
        local_limit[fy0:fy1, fx0:fx1] = 255

    if lm is not None:
        local_limit = cv2.bitwise_and(local_limit, lm)
    return local_limit

def _target_rgb_to_hsv(target_rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    rgb = np.asarray([[list(target_rgb)]], dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[0, 0]
    return float(hsv[0]), float(hsv[1]), float(hsv[2])


def _rebuild_hair_target_from_source(source_gray: np.ndarray,
                                     target_rgb: tuple[int, int, int]) -> tuple[np.ndarray, str]:
    h, s, v = _target_rgb_to_hsv(target_rgb)
    # Near-black / near-neutral colours use the mature ink reconstruction path.
    if v < 72 or s < 24:
        target_luma = 18.0 + 0.08 * source_gray
        highlight = (np.clip(source_gray - 165.0, 0.0, 90.0) / 90.0) ** 2 * 105.0
        target_luma = np.minimum(source_gray, target_luma + highlight)
        target_luma = np.clip(target_luma, 0.0, 145.0).astype(np.uint8)
        return cv2.cvtColor(target_luma, cv2.COLOR_GRAY2BGR), "ink_neutral"

    source_norm = np.clip(source_gray / 255.0, 0.0, 1.0)
    highlight = (np.clip(source_gray - 155.0, 0.0, 95.0) / 95.0) ** 2
    new_v = (14.0 + source_gray * (0.36 + 0.36 * (v / 255.0)) +
             highlight * (38.0 + 0.22 * v))
    new_v = np.minimum(source_gray, new_v)
    new_v = np.clip(new_v, 0.0, max(110.0, float(v))).astype(np.uint8)
    new_s = np.clip(s * (0.98 - 0.58 * source_norm) + 8.0, 0.0, 255.0).astype(np.uint8)
    new_h = np.full(source_gray.shape, h, dtype=np.uint8)
    hsv = np.dstack([new_h, new_s, new_v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), "color_from_source"


def apply_illustrative_inkwash_finish(
        graded_bgr: np.ndarray,
        source_bw_bgr: np.ndarray,
        *,
        semantic_color_bgr: np.ndarray | None = None,
        wash_strength: float = 1.0,
        skin_warmth: float = 1.0,
        paper_grain: float = 0.08,
        ink_restore: float = 0.56) -> np.ndarray:
    """Turn a normal pale grade into an illustration-like ink-wash page.

    Parameters are intentionally narrow and deterministic.  Character colour
    locks are not performed here because a global "black hair" assumption would
    break books containing several characters.  Existing region commands and
    the CharacterLibrary remain authoritative and are applied later.
    """
    color = _ensure_bgr(graded_bgr)
    source = _ensure_bgr(source_bw_bgr)
    semantic_color = (_ensure_bgr(semantic_color_bgr)
                      if semantic_color_bgr is not None else color)
    if source.shape[:2] != color.shape[:2]:
        source = cv2.resize(source, (color.shape[1], color.shape[0]),
                            interpolation=cv2.INTER_AREA)
    if semantic_color.shape[:2] != color.shape[:2]:
        semantic_color = cv2.resize(
            semantic_color, (color.shape[1], color.shape[0]),
            interpolation=cv2.INTER_AREA)

    strength = float(np.clip(wash_strength, 0.0, 1.5))
    skin_warmth = float(np.clip(skin_warmth, 0.0, 1.5))
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Preserve the printed page's tonal structure while retaining enough model
    # shading for skin and fabric to read as painted rather than flat fill.
    lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB).astype(np.float32)
    light, a, b = cv2.split(lab)
    drawable = ((source_gray > 14) & (source_gray < 248)).astype(np.float32)
    generated_luma = np.clip(0.48 + 0.10 * drawable, 0.0, 1.0)
    generated_luma = 1.0 - (1.0 - generated_luma) * min(1.0, strength)
    light = source_gray * (1.0 - generated_luma) + light * generated_luma

    # Watercolour is a soft chroma wash over sharp ink.  Bilateral filtering
    # smooths colour within regions without smearing panel lines as a Gaussian
    # RGB blur would.
    a = cv2.bilateralFilter(a.astype(np.float32), 7, 17.0, 7.0)
    b = cv2.bilateralFilter(b.astype(np.float32), 7, 17.0, 7.0)

    # Page-white / sparse-background areas receive less chroma; dense drawable
    # regions keep more.  This gives airy backgrounds without desaturating the
    # main figure to grayscale.
    ink_density = (255.0 - source_gray) / 255.0
    chroma_keep = np.clip(0.67 + 0.27 * ink_density, 0.62, 0.96)
    chroma_keep = 1.0 - (1.0 - chroma_keep) * min(1.0, strength)
    a = 128.0 + (a - 128.0) * chroma_keep
    b = 128.0 + (b - 128.0) * chroma_keep
    out = cv2.cvtColor(
        np.dstack([np.clip(light, 0, 255), np.clip(a, 0, 255),
                   np.clip(b, 0, 255)]).astype(np.uint8),
        cv2.COLOR_LAB2BGR)

    # Conservative skin harmonisation.  It only touches sizeable components
    # already classified as skin-like by the generated page, so wood/paper is
    # not globally painted peach.  Blush and lips survive because the blend is
    # deliberately modest.
    ycrcb = cv2.cvtColor(semantic_color, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    skin_like = ((y > 70) & (y < 251) &
                 (cr > 127) & (cr < 190) &
                 (cb > 67) & (cb < 146) &
                 (source_gray > 42) & (source_gray < 248))
    skin_like = _large_component_mask(
        skin_like, min_area=max(32, int(out.shape[0] * out.shape[1] * 0.000018)))
    skin_soft = cv2.GaussianBlur(skin_like.astype(np.float32), (7, 7), 0)
    if np.any(skin_soft > 0.01) and skin_warmth > 0:
        # BGR warm ivory target with source-dependent lightness.
        target = np.empty_like(out, dtype=np.float32)
        target[:] = (184.0, 214.0, 242.0)
        shade = np.clip(source_gray / 218.0, 0.34, 1.08)[..., None]
        target = np.clip(target * shade + 255.0 * (1.0 - shade) * 0.08,
                         0, 255)
        amount = (skin_soft * (0.34 * skin_warmth))[..., None]
        out = np.clip(out.astype(np.float32) * (1.0 - amount) +
                      target * amount, 0, 255).astype(np.uint8)

    # Reuse the scan's own micro-texture as a subtle paper grain.  No random
    # noise is introduced, so export is reproducible and batch pages do not
    # shimmer when viewed in sequence.
    if paper_grain > 0:
        low = cv2.GaussianBlur(source_gray, (0, 0), 3.0)
        grain = (source_gray - low)[..., None]
        out = np.clip(out.astype(np.float32) +
                      grain * float(np.clip(paper_grain, 0.0, 0.25)),
                      0, 255).astype(np.uint8)

    # Restore dense original ink after all soft colour operations.  This keeps
    # text, eyelashes, hatching and panel borders crisp without erasing colour
    # from ordinary gray screentones.
    if ink_restore > 0:
        line = np.clip((84.0 - source_gray) / 44.0, 0.0, 1.0)
        line = cv2.GaussianBlur(line, (3, 3), 0)[..., None]
        source_bgr = cv2.cvtColor(source_gray.astype(np.uint8),
                                  cv2.COLOR_GRAY2BGR)
        amount = line * float(np.clip(ink_restore, 0.0, 1.0))
        out = np.clip(out.astype(np.float32) * (1.0 - amount) +
                      source_bgr.astype(np.float32) * amount,
                      0, 255).astype(np.uint8)
    return out





def _crop_arrays_to_limit(source: np.ndarray,
                          colour: np.ndarray,
                          visible: np.ndarray,
                          warm: np.ndarray,
                          local_limit: np.ndarray,
                          seed_xy: tuple[int, int],
                          *,
                          pad: int = 8):
    """Crop page-wide arrays to the active local limit ROI.

    This keeps the V5.8.2 candidate logic intact while making the expensive
    hair-object completion run only on the relevant neighbourhood. The crop is
    intentionally conservative so the resulting mask stays behaviorally close to
    the original full-page implementation.
    """
    ys, xs = np.nonzero(local_limit > 0)
    if xs.size == 0 or ys.size == 0:
        h, w = local_limit.shape[:2]
        return (source, colour, visible, warm, local_limit, seed_xy, (0, 0, w, h))
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(local_limit.shape[1], int(xs.max()) + 1 + pad)
    y1 = min(local_limit.shape[0], int(ys.max()) + 1 + pad)
    sx, sy = int(seed_xy[0]), int(seed_xy[1])
    sx = min(max(sx, x0), x1 - 1)
    sy = min(max(sy, y0), y1 - 1)
    roi = (x0, y0, x1, y1)
    return (source[y0:y1, x0:x1],
            colour[y0:y1, x0:x1],
            visible[y0:y1, x0:x1],
            warm[y0:y1, x0:x1],
            local_limit[y0:y1, x0:x1],
            (sx - x0, sy - y0),
            roi)

def _warm_visible_core_mask(source_bw_bgr: np.ndarray,
                            semantic_color_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the narrow warm visible seed used by the legacy detector."""
    source = _ensure_bgr(source_bw_bgr)
    colour = _ensure_bgr(semantic_color_bgr)
    if colour.shape[:2] != source.shape[:2]:
        colour = cv2.resize(colour, (source.shape[1], source.shape[0]),
                            interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(colour, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    warm = (((hue <= 20) | (hue >= 172)) &
            (sat >= 118) & (val >= 32) & (gray < 248))
    warm = cv2.morphologyEx(
        warm.astype(np.uint8), cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8), iterations=1)
    warm = cv2.morphologyEx(
        warm, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return warm, gray, hsv


def _face_guided_ai_hair_prior(source_bgr: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """Build head/side-hair support and a conservative face-centre exclusion."""
    source = _ensure_bgr(source_bgr)
    h, w = source.shape[:2]
    faces = _cached_detect_anime_faces(source, max_faces=32)
    if not faces:
        return None, None, 0
    allowed = np.zeros((h, w), np.uint8)
    exclude = np.zeros((h, w), np.uint8)
    for face in faces:
        fx, fy, fw, fh = [int(v) for v in face.bbox]
        hx, hy, hw, hh = [int(v) for v in getattr(face, "head_bbox", face.bbox)]
        bx, by, bw, bh = [int(v) for v in getattr(face, "body_bbox", (hx, hy, hw, hh))]

        # Broad head zone includes bangs and short hair.
        x0 = max(0, int(hx - 0.32 * hw))
        x1 = min(w, int(hx + 1.32 * hw))
        y0 = max(0, int(hy - 0.18 * hh))
        y1 = min(h, int(hy + 1.55 * hh))
        allowed[y0:y1, x0:x1] = 255

        # Long hair is normally beside the torso, not across its entire centre.
        corridor_top = max(0, int(fy + 0.05 * fh))
        corridor_bottom = min(h, max(int(fy + 5.2 * fh), int(by + 0.96 * bh)))
        left0 = max(0, int(fx - 1.20 * fw))
        left1 = min(w, int(fx + 0.26 * fw))
        right0 = max(0, int(fx + 0.74 * fw))
        right1 = min(w, int(fx + 2.20 * fw))
        if left1 > left0:
            allowed[corridor_top:corridor_bottom, left0:left1] = 255
        if right1 > right0:
            allowed[corridor_top:corridor_bottom, right0:right1] = 255

        # Remove the central eye/cheek/mouth area. Hair completion may bridge
        # across this hole from true bangs, but glasses and blush cannot start
        # an independent recolour object here.
        center = np.zeros_like(exclude)
        cx = int(round(fx + fw * 0.50))
        cy = int(round(fy + fh * 0.59))
        rx = max(4, int(round(fw * 0.37)))
        ry = max(4, int(round(fh * 0.34)))
        cv2.ellipse(center, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
        exclude = cv2.bitwise_or(exclude, center)

    allowed = cv2.morphologyEx(allowed, cv2.MORPH_CLOSE,
                               np.ones((7, 7), np.uint8), iterations=1)
    return allowed, exclude, len(faces)


def _component_outline_perimeter(component: np.ndarray) -> float:
    """Return a stable outer-outline perimeter for one binary component."""
    contours, _ = cv2.findContours(
        np.where(component, 255, 0).astype(np.uint8),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return float(sum(cv2.arcLength(contour, True) for contour in contours))


def _suppress_thin_supplemental_hair(
        supplemental_mask: np.ndarray,
        primary_mask: np.ndarray,
        *,
        face_exclusion: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Remove wire-like false positives from *supplemental* hair recovery.

    The AI-anchored primary mask is intentionally left untouched.  This pass is
    applied only to the legacy remainder which V5.12.2 adds back after the main
    hairstyle has already been recovered.  It rejects isolated line/shell
    components and prunes narrow branches pulled out of otherwise valid missing
    hair blocks.

    The measurements are resolution-aware and deliberately combine several
    weak cues instead of relying on one brittle area threshold:

    * equivalent local thickness (``2 * area / perimeter``),
    * distance-transform core radius/fraction,
    * bounding-box elongation,
    * overlap with a thin shell around the primary hair object,
    * overlap with the stricter eye/cheek/mouth neighbourhood.

    Returns ``(kept, rejected, diagnostics_rows)``.  ``rejected`` is useful for
    the GUI preview but is never recoloured.
    """
    candidate = np.where(np.asarray(supplemental_mask) > 0, 255, 0).astype(np.uint8)
    primary = np.where(np.asarray(primary_mask) > 0, 255, 0).astype(np.uint8)
    if candidate.shape[:2] != primary.shape[:2]:
        primary = cv2.resize(primary, (candidate.shape[1], candidate.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
    if not np.any(candidate):
        return candidate, np.zeros_like(candidate), []

    height, width = candidate.shape[:2]
    page_area = max(1, height * width)
    scale = float(np.clip(min(height, width) / 1000.0, 0.72, 2.40))

    # A narrow outside band catches parallel warm shells along the completed
    # long-hair boundary without treating a separate character as attached.
    shell_radius = max(7, min(25, int(round(11.0 * scale))))
    shell_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (shell_radius * 2 + 1, shell_radius * 2 + 1))
    primary_shell = cv2.dilate(
        (primary > 0).astype(np.uint8), shell_kernel, iterations=1) > 0
    primary_shell &= ~(primary > 0)

    face_near = None
    if face_exclusion is not None:
        face_arr = np.asarray(face_exclusion)
        if face_arr.ndim == 3:
            face_arr = np.max(face_arr, axis=2)
        if face_arr.shape[:2] != candidate.shape[:2]:
            face_arr = cv2.resize(face_arr.astype(np.uint8),
                                  (width, height), interpolation=cv2.INTER_NEAREST)
        if np.any(face_arr):
            face_radius = max(7, min(31, int(round(14.0 * scale))))
            face_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (face_radius * 2 + 1, face_radius * 2 + 1))
            face_near = cv2.dilate((face_arr > 0).astype(np.uint8),
                                   face_kernel, iterations=1) > 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (candidate > 0).astype(np.uint8), connectivity=8)
    kept = np.zeros_like(candidate)
    rejected = np.zeros_like(candidate)
    rows: list[dict] = []

    base_line_width = 2.75 * scale
    base_core_radius = max(1.55, 1.72 * scale)
    large_block_area = max(1500, int(page_area * 0.00085))

    for label in range(1, count):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        short_side = max(1, min(bw, bh))
        long_side = max(bw, bh)
        elongation = float(long_side / float(short_side))

        component_u8 = np.where(component, 1, 0).astype(np.uint8)
        distance = cv2.distanceTransform(component_u8, cv2.DIST_L2, 5)
        distance_values = distance[component]
        max_radius = float(distance_values.max()) if distance_values.size else 0.0
        core_fraction = float(np.mean(distance_values >= base_core_radius)) if distance_values.size else 0.0
        perimeter = _component_outline_perimeter(component)
        equivalent_width = float((2.0 * area) / max(1.0, perimeter))
        shell_fraction = float(np.mean(primary_shell[component])) if area else 0.0
        face_fraction = (float(np.mean(face_near[component]))
                         if face_near is not None and area else 0.0)

        substantial_block = (
            area >= large_block_area and
            max_radius >= 2.35 * scale and
            equivalent_width >= 2.85 * scale and
            (max_radius >= 4.50 * scale or elongation < 4.0))
        reason = "kept"

        # Entire-component rejection: isolated highlight/outline wires normally
        # have no meaningful distance-transform core even when their total area
        # is large because the line runs for hundreds of pixels.
        if not substantial_block:
            if (max_radius < base_core_radius and
                    (elongation >= 2.35 or shell_fraction >= 0.28 or face_fraction >= 0.08)):
                reason = "no_thick_core"
            elif (equivalent_width < base_line_width and
                  (elongation >= 3.0 or shell_fraction >= 0.34 or face_fraction >= 0.08)):
                reason = "equivalent_width_too_low"
            elif core_fraction < 0.012 and elongation >= 5.0:
                reason = "extreme_thin_elongation"
            elif (elongation >= 7.0 and
                  equivalent_width < 6.20 * scale and
                  max_radius < 3.55 * scale):
                reason = "long_wire_component"
            elif (shell_fraction >= 0.56 and
                  equivalent_width < 6.10 * scale and
                  max_radius < 3.65 * scale):
                reason = "primary_edge_shell"
            elif (face_fraction >= 0.12 and
                  equivalent_width < 6.25 * scale and
                  max_radius < 3.70 * scale):
                reason = "face_near_thin_line"

        if reason != "kept":
            rejected[component] = 255
            rows.append({
                "label": int(label), "kept": False, "reason": reason,
                "area": area, "bbox": (x, y, bw, bh),
                "elongation": round(elongation, 4),
                "equivalent_width": round(equivalent_width, 3),
                "max_core_radius": round(max_radius, 3),
                "core_fraction": round(core_fraction, 5),
                "primary_shell_fraction": round(shell_fraction, 5),
                "face_near_fraction": round(face_fraction, 5),
                "pruned_pixels": area,
            })
            continue

        # Branch pruning: preserve the body of a valid supplemental hair block,
        # but stop a 1-4 px warm line from being dragged far out of that body.
        # Dilation restores the ordinary object boundary after opening; only a
        # long narrow appendage remains outside the support and is removed.
        stricter = shell_fraction >= 0.30 or face_fraction >= 0.10
        open_radius = max(
            2, int(np.ceil((2.80 if stricter else 2.42) * scale)))
        open_radius = min(open_radius, max(1, short_side // 3))
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_radius * 2 + 1, open_radius * 2 + 1))
        opened = cv2.morphologyEx(component_u8, cv2.MORPH_OPEN,
                                  open_kernel, iterations=1)
        pruned_component = component.copy()
        if np.any(opened):
            restore_radius = max(2, int(round((2.25 if stricter else 2.70) * scale)))
            restore_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (restore_radius * 2 + 1, restore_radius * 2 + 1))
            body_support = cv2.dilate(opened, restore_kernel, iterations=1) > 0
            pruned_component = component & body_support

            # Do not leave tiny detached crumbs after branch removal.
            if np.any(pruned_component):
                pc_count, pc_labels, pc_stats, _ = cv2.connectedComponentsWithStats(
                    pruned_component.astype(np.uint8), connectivity=8)
                min_piece = max(28, int(round(42 * scale * scale)))
                cleaned = np.zeros_like(pruned_component)
                for pc_label in range(1, pc_count):
                    if int(pc_stats[pc_label, cv2.CC_STAT_AREA]) >= min_piece:
                        cleaned[pc_labels == pc_label] = True
                if np.any(cleaned):
                    pruned_component = cleaned

        removed_branch = component & ~pruned_component
        kept_pixels = int(np.count_nonzero(pruned_component))
        pruned_pixels = int(np.count_nonzero(removed_branch))
        # A component whose body disappears under the minimum-thickness opening
        # is itself only a line, even when the earlier aggregate metrics were
        # marginal.  Keep large real blocks as a final safety valve.
        if kept_pixels == 0 or (kept_pixels < max(55, int(area * 0.16)) and not substantial_block):
            rejected[component] = 255
            reason = "body_removed_by_thickness_opening"
            kept_pixels = 0
            pruned_pixels = area
        else:
            kept[pruned_component] = 255
            rejected[removed_branch] = 255
            if pruned_pixels > 0:
                reason = "kept_body_pruned_thin_branches"

        rows.append({
            "label": int(label), "kept": bool(kept_pixels > 0), "reason": reason,
            "area": area, "bbox": (x, y, bw, bh),
            "elongation": round(elongation, 4),
            "equivalent_width": round(equivalent_width, 3),
            "max_core_radius": round(max_radius, 3),
            "core_fraction": round(core_fraction, 5),
            "primary_shell_fraction": round(shell_fraction, 5),
            "face_near_fraction": round(face_fraction, 5),
            "kept_pixels": kept_pixels,
            "pruned_pixels": pruned_pixels,
        })

    return kept, rejected, rows


def _ai_hair_mask_for_recolor(
        source_bw_bgr: np.ndarray, semantic_color_bgr: np.ndarray,
        *, parser_backend: str = "auto",
        parser_allow_download: bool = False,
        limit_mask: np.ndarray | None = None,
        return_preview_layers: bool = False) -> tuple[np.ndarray | None, dict]:
    diagnostics = {
        "enabled": True,
        "backend_request": str(parser_backend or "auto"),
    }
    try:
        from core.ai_hair_parser import parse_hair_regions
        parsed = parse_hair_regions(
            semantic_color_bgr, source_bgr=source_bw_bgr,
            preferred_backend=parser_backend,
            allow_download=parser_allow_download)
    except Exception as exc:  # noqa: BLE001
        diagnostics.update({
            "available": False,
            "reason": f"ai_parser_error:{type(exc).__name__}:{exc}",
        })
        return None, diagnostics
    diagnostics.update(dict(parsed.diagnostics or {}))
    hair_mask = parsed.hair_mask
    if hair_mask is None or not np.any(hair_mask):
        diagnostics["available"] = False
        diagnostics.setdefault("reason", "ai_parser_no_hair_mask")
        return None, diagnostics

    hair_mask = np.where(hair_mask > 0, 255, 0).astype(np.uint8)
    if limit_mask is not None:
        lm = np.asarray(limit_mask)
        if lm.ndim == 3:
            lm = np.max(lm, axis=2)
        if lm.shape[:2] != hair_mask.shape[:2]:
            lm = cv2.resize(lm.astype(np.uint8),
                            (hair_mask.shape[1], hair_mask.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        hair_mask = cv2.bitwise_and(
            hair_mask, np.where(lm > 0, 255, 0).astype(np.uint8))
    if not np.any(hair_mask):
        diagnostics.update({"available": False, "reason": "ai_mask_outside_limit"})
        return None, diagnostics

    face_prior, face_exclusion, face_count = _face_guided_ai_hair_prior(source_bw_bgr)
    if face_prior is not None and np.any(face_prior):
        hair_mask = cv2.bitwise_and(hair_mask, face_prior)
        if face_exclusion is not None and np.any(face_exclusion):
            # Keep a small dilated warm seed even inside the face centre so real
            # bangs are not cut merely because they overlap the forehead/eyes.
            warm_seed_pre, _gray_pre, _hsv_pre = _warm_visible_core_mask(
                source_bw_bgr, semantic_color_bgr)
            preserved_bangs = cv2.dilate((warm_seed_pre > 0).astype(np.uint8),
                                         np.ones((5, 5), np.uint8), iterations=1) * 255
            removable = cv2.bitwise_and(face_exclusion,
                                        cv2.bitwise_not(preserved_bangs))
            hair_mask = cv2.bitwise_and(hair_mask, cv2.bitwise_not(removable))
        diagnostics["face_guided_prior"] = True
        diagnostics["face_count"] = int(face_count)
        diagnostics["face_prior_pixels"] = int(np.count_nonzero(face_prior))
        diagnostics["face_exclusion_pixels"] = int(np.count_nonzero(face_exclusion)) if face_exclusion is not None else 0
    else:
        diagnostics["face_guided_prior"] = False
        diagnostics["face_count"] = 0
    if not np.any(hair_mask):
        diagnostics.update({"available": False, "reason": "ai_mask_removed_by_face_prior"})
        return None, diagnostics

    page_area = max(1, hair_mask.shape[0] * hair_mask.shape[1])
    warm_visible, source_gray, hsv = _warm_visible_core_mask(
        source_bw_bgr, semantic_color_bgr)
    warm_overlap = cv2.bitwise_and(hair_mask, warm_visible)
    warm_pixels = int(np.count_nonzero(warm_overlap))
    hair_pixels = int(np.count_nonzero(hair_mask))
    within = hair_mask > 0
    hue, sat, val = cv2.split(hsv)
    warm_semantic = (((hue <= 24) | (hue >= 170)) &
                     (sat >= 72) & (val >= 24))
    warm_ratio = float(np.mean(warm_semantic[within])) if np.any(within) else 0.0
    mean_gray = float(source_gray[within].mean()) if np.any(within) else 255.0
    diagnostics.update({
        "available": True,
        "hair_pixels": hair_pixels,
        "warm_overlap_pixels": warm_pixels,
        "warm_semantic_ratio": round(warm_ratio, 6),
        "mean_source_gray": round(mean_gray, 3),
    })

    # AI parsing says where hair is; the warm-overlap gate decides whether this
    # automatic correction should act on it. This preserves intentionally dyed
    # characters while fixing mc-v2's orange/red false-black response.
    minimum_overlap = max(24, int(round(hair_pixels * 0.0008)))
    if warm_pixels < minimum_overlap and not (warm_ratio >= 0.12 and mean_gray < 176.0):
        diagnostics.update({
            "available": False,
            "reason": "ai_hair_not_warm_enough_for_recolor",
        })
        return None, diagnostics

    # Never apply the raw parser mask directly. StyleAnime in particular can
    # label glasses, clothing shadows or neighbouring black shapes as hair on
    # manga pages. Keep only parser components anchored to the actual warm mc-v2
    # error, then let the proven hair-object completion rebuild inside that AI
    # semantic prior.
    support_radius = max(7, min(31, int(round(min(hair_mask.shape) * 0.012))))
    support_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (support_radius * 2 + 1, support_radius * 2 + 1))
    warm_support = cv2.dilate((warm_visible > 0).astype(np.uint8),
                              support_kernel, iterations=1) > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (hair_mask > 0).astype(np.uint8), connectivity=8)
    anchored = np.zeros_like(hair_mask)
    kept_components = 0
    for label in range(1, count):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        overlap = int(np.count_nonzero(component & warm_support))
        direct = int(np.count_nonzero(component & (warm_visible > 0)))
        if direct >= max(6, int(area * 0.00035)) or overlap >= max(18, int(area * 0.0025)):
            anchored[component] = 255
            kept_components += 1
    if not np.any(anchored):
        diagnostics.update({
            "available": False,
            "reason": "ai_hair_has_no_warm_anchored_component",
            "ai_component_count": int(max(0, count - 1)),
        })
        return None, diagnostics

    # The AI mask is a semantic boundary, not the final recolour mask. Allow a
    # modest dark-line corridor around it so split highlights and long strands
    # can be reconstructed, while still excluding page-wide false positives.
    limit_radius = max(11, min(49, int(round(min(hair_mask.shape) * 0.020))))
    limit_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (limit_radius * 2 + 1, limit_radius * 2 + 1))
    ai_limit = cv2.dilate((anchored > 0).astype(np.uint8),
                          limit_kernel, iterations=1) * 255
    warm_near = cv2.bitwise_and(warm_visible, ai_limit)
    ai_limit = cv2.bitwise_or(
        ai_limit,
        cv2.dilate((warm_near > 0).astype(np.uint8), limit_kernel, iterations=1) * 255)
    if limit_mask is not None:
        lm = np.asarray(limit_mask)
        if lm.ndim == 3:
            lm = np.max(lm, axis=2)
        if lm.shape[:2] != ai_limit.shape[:2]:
            lm = cv2.resize(lm.astype(np.uint8),
                            (ai_limit.shape[1], ai_limit.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        ai_limit = cv2.bitwise_and(ai_limit, np.where(lm > 0, 255, 0).astype(np.uint8))

    rebuilt, rebuild_diag = _dominant_warm_ink_hair_mask(
        source_bw_bgr, semantic_color_bgr, mode="normal",
        scope="page", limit_mask=ai_limit)
    if rebuilt is None or not np.any(rebuilt):
        diagnostics.update({
            "available": False,
            "reason": "ai_anchored_rebuild_empty",
            "ai_component_count": int(max(0, count - 1)),
            "ai_kept_components": kept_components,
            "anchored_pixels": int(np.count_nonzero(anchored)),
            "rebuild": rebuild_diag,
        })
        return None, diagnostics

    # Hybrid recovery: if the AI-guided rebuild misses another strong legacy
    # warm-hair object elsewhere on the page (for example an upside-down head or
    # a small side character that the parser/face prior misses), recover only
    # those legacy components that still look like real false-hair objects.
    primary_rebuilt = np.where(rebuilt > 0, 255, 0).astype(np.uint8)
    hybrid_candidate = np.zeros_like(primary_rebuilt)
    hybrid_extra = np.zeros_like(primary_rebuilt)
    hybrid_rejected_thin = np.zeros_like(primary_rebuilt)
    legacy_page_mask, legacy_page_diag = _dominant_warm_ink_hair_mask(
        source_bw_bgr, semantic_color_bgr, mode="normal",
        scope="page", limit_mask=limit_mask)
    supplemental_rows = []
    thin_filter_rows = []
    if legacy_page_mask is not None and np.any(legacy_page_mask):
        # IMPORTANT: subtract the AI result pixel-by-pixel *before* component
        # analysis. The previous v5.12.1 logic skipped an entire legacy
        # component whenever any part overlapped the AI mask. On multi-person
        # pages the legacy rebuild can merge several hairstyles into one large
        # component, so a small overlap on one character incorrectly discarded
        # the still-missing red hair of another character.
        rebuilt_dilated_u8 = cv2.dilate(
            (rebuilt > 0).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            iterations=1)
        legacy_remainder = cv2.bitwise_and(
            np.where(legacy_page_mask > 0, 255, 0).astype(np.uint8),
            cv2.bitwise_not(rebuilt_dilated_u8 * 255))
        legacy_remainder = cv2.morphologyEx(
            legacy_remainder, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        legacy_remainder = cv2.morphologyEx(
            legacy_remainder, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

        cc_count, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(
            (legacy_remainder > 0).astype(np.uint8), 8)
        warm_bool = warm_visible > 0
        source_gray_u8 = source_gray.astype(np.uint8)
        face_prior_bool = (face_prior > 0) if face_prior is not None else None
        foreground_bool = (parsed.foreground_mask > 0) if getattr(parsed, "foreground_mask", None) is not None else None
        for cc_label in range(1, cc_count):
            area = int(cc_stats[cc_label, cv2.CC_STAT_AREA])
            if area < max(220, int(page_area * 0.00018)):
                continue
            comp = cc_labels == cc_label
            warm_comp = int(np.count_nonzero(comp & warm_bool))
            warm_fraction = warm_comp / max(1.0, float(area))
            if warm_comp < max(16, int(area * 0.012)):
                continue
            gray_vals = source_gray_u8[comp]
            if gray_vals.size == 0:
                continue
            mean_comp = float(gray_vals.mean())
            dark_ratio_comp = float(np.mean(gray_vals < 178))
            if mean_comp > 214.0 or dark_ratio_comp < 0.075:
                continue
            near_face = int(np.count_nonzero(comp & face_prior_bool)) if face_prior_bool is not None and np.any(face_prior_bool) else 0
            foreground_support = int(np.count_nonzero(comp & foreground_bool)) if foreground_bool is not None and np.any(foreground_bool) else 0
            supported = (
                near_face >= max(10, int(area * 0.012)) or
                (foreground_support >= max(24, int(area * 0.10))
                 and warm_fraction >= 0.06 and dark_ratio_comp >= 0.14) or
                # Strong, large warm false-hair objects should survive even if
                # the face detector misses an upside-down / profile / close-up
                # face, but only when the parser still marks them foreground.
                # Without this requirement a large warm garment can satisfy the
                # colour/tone gates by itself and be recoloured as hair.
                (area >= max(900, int(page_area * 0.0012)) and warm_fraction >= 0.05
                 and dark_ratio_comp >= 0.12
                 and foreground_support >= max(18, int(area * 0.055)))
            )
            if not supported:
                continue
            hybrid_candidate[comp] = 255
            supplemental_rows.append({
                "label": int(cc_label),
                "area": area,
                "warm_pixels": warm_comp,
                "warm_fraction": round(float(warm_fraction), 6),
                "mean_gray": round(mean_comp, 3),
                "dark_ratio": round(dark_ratio_comp, 4),
                "near_face_pixels": near_face,
                "foreground_support_pixels": foreground_support,
            })
    if np.any(hybrid_candidate):
        hybrid_extra, hybrid_rejected_thin, thin_filter_rows = (
            _suppress_thin_supplemental_hair(
                hybrid_candidate, primary_rebuilt,
                face_exclusion=face_exclusion))
    if np.any(hybrid_extra):
        rebuilt = cv2.bitwise_or(primary_rebuilt, hybrid_extra.astype(np.uint8))
    else:
        rebuilt = primary_rebuilt

    diagnostics.update({
        "reason": "ai_anchored_hair_rebuild_ready",
        "ai_component_count": int(max(0, count - 1)),
        "ai_kept_components": kept_components,
        "anchored_pixels": int(np.count_nonzero(anchored)),
        "ai_limit_pixels": int(np.count_nonzero(ai_limit)),
        "refined_pixels": int(np.count_nonzero(rebuilt)),
        "rebuild": rebuild_diag,
        "hybrid_legacy_page_rebuild": legacy_page_diag,
        "hybrid_legacy_remainder_pixels": int(np.count_nonzero(legacy_remainder)) if legacy_page_mask is not None and np.any(legacy_page_mask) else 0,
        "hybrid_semantic_candidate_pixels": int(np.count_nonzero(hybrid_candidate)),
        "hybrid_semantic_candidate_components": len(supplemental_rows),
        "hybrid_extra_pixels": int(np.count_nonzero(hybrid_extra)),
        "hybrid_extra_components": int(sum(1 for row in thin_filter_rows if row.get("kept"))),
        "hybrid_rejected_thin_pixels": int(np.count_nonzero(hybrid_rejected_thin)),
        "hybrid_rejected_thin_components": int(sum(1 for row in thin_filter_rows if not row.get("kept"))),
        "hybrid_extra_rebuilds": supplemental_rows,
        "hybrid_thin_filter": thin_filter_rows,
    })
    if return_preview_layers:
        diagnostics["_preview_layers"] = {
            "primary": primary_rebuilt,
            "supplemental": hybrid_extra,
            "rejected_thin": hybrid_rejected_thin,
        }
    return np.where(rebuilt > 0, 255, 0).astype(np.uint8), diagnostics


def _dominant_warm_ink_hair_mask(
        source_bw_bgr: np.ndarray,
        semantic_color_bgr: np.ndarray,
        *,
        mode: str = "normal",
        scope: str = "page",
        limit_mask: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Detect one or more inked hair objects rendered as warm red/orange.

    Earlier versions selected only one *dominant* component. On pages with
    several characters, two or more similarly sized orange hair components
    caused the dominance gate to reject the whole page, so the target colour
    was never applied. This version supports multiple candidate hair objects.

    The detector still uses the warm mc-v2 error colour as a seed, but the final
    object masks are rebuilt from the original monochrome structure. In
    ``near_face`` mode, face adjacency is preferred. If the optional face model
    is unavailable or finds no face, the function falls back to conservative
    page-local component rules instead of silently disabling correction.
    """
    source = _ensure_bgr(source_bw_bgr)
    colour = _ensure_bgr(semantic_color_bgr)
    if colour.shape[:2] != source.shape[:2]:
        colour = cv2.resize(colour, (source.shape[1], source.shape[0]),
                            interpolation=cv2.INTER_AREA)
    height, width = source.shape[:2]
    page_area = max(1, height * width)
    _guard_enabled, normalized_mode = _normalize_auto_black_hair_guard(mode)
    normalized_scope = _normalize_auto_black_hair_scope(scope)
    conservative = normalized_mode == "conservative"
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(colour, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    # Stock mc-v2's false-black response is usually vivid red/orange. A slightly
    # wider hue/saturation gate catches softened style output while still
    # excluding most skin and beige paper.
    warm = (((hue <= 16) | (hue >= 176)) &
            (sat >= (176 if conservative else 150)) &
            (val >= 38) & (gray < 246))
    warm = cv2.morphologyEx(
        warm.astype(np.uint8), cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8), iterations=1)
    warm = cv2.morphologyEx(
        warm, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    lm = None
    if limit_mask is not None:
        lm = np.asarray(limit_mask)
        if lm.ndim == 3:
            lm = np.max(lm, axis=2)
        if lm.shape[:2] != warm.shape[:2]:
            lm = cv2.resize(lm.astype(np.uint8),
                            (warm.shape[1], warm.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        lm = np.where(lm > 0, 255, 0).astype(np.uint8)

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        warm, connectivity=8)
    minimum_pixels = max(220 if not conservative else 420,
                         int(round(page_area * (0.00012 if not conservative else 0.00022))))
    candidates: list[dict] = []
    selection_overlap_by_label: dict[int, int] = {}
    for label in range(1, count):
        x, y, box_w, box_h, area = [int(v) for v in stats[label]]
        if area < minimum_pixels or box_w < 8 or box_h < 8:
            continue
        component = labels == label
        source_pixels = gray[component]
        if source_pixels.size == 0:
            continue
        area_fraction = area / page_area
        dark_ratio = float(np.mean(source_pixels < 135))
        very_dark_ratio = float(np.mean(source_pixels < 82))
        mean_gray = float(np.mean(source_pixels))
        long_aspect = max(box_w / max(1.0, box_h),
                          box_h / max(1.0, box_w))
        box_height_fraction = box_h / max(1.0, height)
        box_width_fraction = box_w / max(1.0, width)

        if long_aspect > 7.5 and area_fraction < 0.018:
            continue
        min_dark_ratio = 0.15 if conservative else 0.085
        max_mean_gray = 174.0 if conservative else 196.0
        if mean_gray > max_mean_gray or dark_ratio < min_dark_ratio:
            continue
        if box_height_fraction < 0.026 or box_width_fraction < 0.016:
            continue

        selection_overlap = int(np.count_nonzero(component & (lm > 0))) if lm is not None else 0
        selection_overlap_by_label[int(label)] = selection_overlap
        score = (np.log1p(area) + 2.5 * dark_ratio +
                 1.0 * very_dark_ratio +
                 1.2 * min(1.0, box_height_fraction * 3.2) +
                 0.8 * min(1.0, box_width_fraction * 3.2))
        candidates.append({
            "label": int(label),
            "area": int(area),
            "area_fraction": float(area_fraction),
            "bbox": (x, y, box_w, box_h),
            "mean_gray": mean_gray,
            "dark_ratio": dark_ratio,
            "very_dark_ratio": very_dark_ratio,
            "score": float(score),
            "face_bbox": None,
            "selection_overlap_pixels": selection_overlap,
        })

    if not candidates:
        return np.zeros((height, width), np.uint8), {
            "enabled": True, "detected": False,
            "reason": "no_warm_ink_hair_component",
            "candidate_count": 0,
            "mode": normalized_mode,
            "scope": normalized_scope,
            "limit_masked": bool(lm is not None),
        }

    faces = []
    face_fallback = False
    matched_face_candidates = 0
    faces = _cached_detect_anime_faces(source, max_faces=32)
    if not faces and normalized_scope == "near_face":
        face_fallback = True
    for item in candidates:
        item["face_matched"] = False

    # First build stable hairstyle/person groups from detected faces. This is the
    # direct fix for the old left-versus-right alternation.
    face_groups, assigned_labels = _build_face_guided_candidate_groups(candidates, faces, (height, width))
    matched_face_candidates = int(sum(len(item.get('member_labels', [])) for item in face_groups))

    # Fallback / supplementary global grouping: keep unassigned valid components so
    # side-profile or tiny heads can still be corrected if face detection misses.
    candidates.sort(key=lambda item: (item["score"], item["area"]), reverse=True)
    grouped_candidates = _cluster_warm_hair_candidates(candidates)
    leftover_grouped_candidates = []
    for item in grouped_candidates:
        labels_in_item = [int(v) for v in item.get('member_labels', [])]
        if assigned_labels.intersection(labels_in_item):
            continue
        leftover_grouped_candidates.append(item)

    def _group_selection_overlap(item: dict) -> int:
        vals = item.get('member_labels', []) or [item.get('label', -1)]
        return int(sum(int(selection_overlap_by_label.get(int(v), 0)) for v in vals if int(v) >= 0))

    for group in face_groups:
        group['selection_overlap_pixels'] = _group_selection_overlap(group)
    for group in grouped_candidates:
        group['selection_overlap_pixels'] = _group_selection_overlap(group)
    for group in leftover_grouped_candidates:
        group['selection_overlap_pixels'] = _group_selection_overlap(group)

    min_fraction = 0.0012 if conservative else 0.00055
    selected_candidates = [item for item in face_groups if item["area_fraction"] >= min_fraction * 0.55]
    selected_candidates.extend([
        item for item in leftover_grouped_candidates
        if item["area_fraction"] >= min_fraction
    ])
    if not selected_candidates:
        selected_candidates = (face_groups[:1] or leftover_grouped_candidates[:1] or grouped_candidates[:1])
    selected_candidates.sort(key=lambda item: (bool(item.get("face_matched", False)), item["score"], item["area"]), reverse=True)
    selected_candidates = selected_candidates[:16]
    if lm is not None:
        overlapping_candidates = [item for item in selected_candidates
                                  if int(item.get('selection_overlap_pixels', 0)) > 0]
        if not overlapping_candidates:
            return np.zeros((height, width), np.uint8), {
                "enabled": True, "detected": False,
                "reason": "no_warm_hair_candidate_inside_selection",
                "candidate_count": int(len(selected_candidates)),
                "mode": normalized_mode,
                "scope": normalized_scope,
                "limit_masked": True,
            }
        selected_candidates = overlapping_candidates
    if normalized_scope == "near_face" and not face_groups:
        face_fallback = True

    combined = np.zeros((height, width), np.uint8)
    rebuild_rows = []
    visible_total = 0
    for item in selected_candidates:
        member_labels = [int(v) for v in item.get("member_labels", [])]
        visible = np.where(np.isin(labels, member_labels), 255, 0).astype(np.uint8)
        if not np.any(visible):
            continue
        visible_total += int(np.count_nonzero(visible))
        distance = cv2.distanceTransform((visible > 0).astype(np.uint8),
                                         cv2.DIST_L2, 5)
        seed_y, seed_x = np.unravel_index(int(np.argmax(distance)), distance.shape)

        # Constrain each rebuild to a per-hairstyle / per-character neighbourhood.
        # Face-guided candidates receive a long-hair corridor so close-ups with
        # straight hair can fill both left and right strands in one pass.
        x, y, bw, bh = item["bbox"]
        local_limit = _build_local_limit_mask(height, width, item, lm)

        try:
            from core.mcv2_semantic_selection import build_complete_hair_object_mask

            # V5.8.2 behaviour is preserved, but the heavy completion work runs on
            # the candidate's local ROI instead of the full page. This keeps the
            # same local_limit geometry while greatly reducing runtime.
            (src_roi, col_roi, vis_roi, warm_roi, limit_roi, seed_roi, roi) =                 _crop_arrays_to_limit(source, colour, visible, warm, local_limit,
                                      (int(seed_x), int(seed_y)), pad=8)
            complete_roi, rebuild = build_complete_hair_object_mask(
                src_roi, col_roi, vis_roi, seed_roi,
                limit_mask=limit_roi, use_line_regions=True, max_ratio=2.35)

            # Second pass: original V5.8.2 nearby-warm recovery, unchanged in
            # spirit. It catches fragments split by highlights and line art.
            radius = max(11, min(55, int(round(max(bw, bh) * 0.26))))
            near_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
            near_complete = cv2.dilate((complete_roi > 0).astype(np.uint8),
                                       near_kernel, iterations=1) > 0
            local_warm = (warm_roi > 0) & (limit_roi > 0)
            neighbour_warm = local_warm & near_complete
            expanded_visible = cv2.bitwise_or(
                vis_roi, np.where(neighbour_warm, 255, 0).astype(np.uint8))
            added_warm = int(np.count_nonzero(expanded_visible) -
                             np.count_nonzero(vis_roi))
            second_added = 0
            third_added = 0
            if added_warm > 0:
                complete2_roi, rebuild2 = build_complete_hair_object_mask(
                    src_roi, col_roi, expanded_visible, seed_roi,
                    limit_mask=limit_roi, use_line_regions=True, max_ratio=2.75)
                complete_roi = cv2.bitwise_or(complete_roi.astype(np.uint8),
                                              complete2_roi.astype(np.uint8))

                # Third pass: only additive. V5.8.2 already fills the lower-left
                # hairstyle well; this extra pass specifically tries to recover
                # remaining warm fragments near the current completed hair so the
                # right-side hairstyle can fill more completely too.
                radius2 = max(radius + 10, min(88, int(round(max(bw, bh) * 0.42))))
                near_kernel2 = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (radius2 * 2 + 1, radius2 * 2 + 1))
                near_complete2 = cv2.dilate((complete_roi > 0).astype(np.uint8),
                                            near_kernel2, iterations=1) > 0
                neighbour_warm2 = local_warm & near_complete2
                expanded_visible2 = cv2.bitwise_or(
                    expanded_visible, np.where(neighbour_warm2, 255, 0).astype(np.uint8))
                second_added = int(np.count_nonzero(expanded_visible2) -
                                   np.count_nonzero(expanded_visible))
                if second_added > 0:
                    complete3_roi, rebuild3 = build_complete_hair_object_mask(
                        src_roi, col_roi, expanded_visible2, seed_roi,
                        limit_mask=limit_roi, use_line_regions=True, max_ratio=3.10)
                    complete_roi = cv2.bitwise_or(complete_roi.astype(np.uint8),
                                                  complete3_roi.astype(np.uint8))
                    third_added = second_added
                    rebuild = {
                        "first": rebuild,
                        "second": rebuild2,
                        "third": rebuild3,
                        "nearby_warm_added_pixels": added_warm,
                        "second_nearby_warm_added_pixels": second_added,
                    }
                else:
                    rebuild = {
                        "first": rebuild,
                        "second": rebuild2,
                        "nearby_warm_added_pixels": added_warm,
                        "second_nearby_warm_added_pixels": 0,
                    }
            else:
                rebuild = {
                    "first": rebuild,
                    "nearby_warm_added_pixels": 0,
                    "second_nearby_warm_added_pixels": 0,
                }

            x0, y0, x1, y1 = roi
            complete = np.zeros((height, width), np.uint8)
            complete[y0:y1, x0:x1] = np.where(complete_roi > 0, 255, 0).astype(np.uint8)
        except Exception as exc:
            complete = cv2.bitwise_and(visible, local_limit)
            rebuild = {"fallback": True,
                       "error": f"{type(exc).__name__}: {exc}"}
        combined = cv2.bitwise_or(combined, complete.astype(np.uint8))
        rebuild_rows.append({
            "bbox": tuple(int(v) for v in item["bbox"]),
            "face_matched": bool(item.get("face_matched", False)),
            "member_count": int(item.get("member_count", 1)),
            "visible_pixels": int(np.count_nonzero(visible)),
            "selected_pixels": int(np.count_nonzero(complete)),
            "rebuild": rebuild,
        })

    if lm is not None and np.any(combined):
        combined = cv2.bitwise_and(combined, lm)

    if not np.any(combined):
        return combined, {
            "enabled": True, "detected": False,
            "reason": "hair_object_rebuild_empty",
            "candidate_count": len(candidates),
            "selected_candidate_count": len(selected_candidates),
            "cluster_candidate_count": len(grouped_candidates),
            "mode": normalized_mode,
            "scope": normalized_scope,
            "face_fallback": bool(face_fallback),
            "limit_masked": bool(lm is not None),
        }

    diagnostics = {
        "enabled": True,
        "detected": True,
        "candidate_count": len(candidates),
        "selected_candidate_count": len(selected_candidates),
        "cluster_candidate_count": len(grouped_candidates),
        "visible_pixels": int(visible_total),
        "selected_pixels": int(np.count_nonzero(combined)),
        "mode": normalized_mode,
        "scope": normalized_scope,
        "face_count": int(len(faces)),
        "face_matched_candidates": int(matched_face_candidates),
        "face_fallback": bool(face_fallback),
        "limit_masked": bool(lm is not None),
        "rebuilds": rebuild_rows,
    }
    return combined, diagnostics


def build_auto_hair_recolor_mask(
        source_bw_bgr: np.ndarray,
        semantic_color_bgr: np.ndarray,
        *,
        mode: str | bool = "normal",
        scope: str = "page",
        limit_mask: np.ndarray | None = None,
        parser_enabled: bool = True,
        parser_backend: str = "auto",
        parser_allow_download: bool = False,
        return_preview_layers: bool = False) -> tuple[np.ndarray, dict]:
    """Build the exact logical mask used by automatic hair recolouring.

    The preview button and final pipeline share this function, preventing the UI
    from showing a legacy-only mask while the actual export uses an AI parser.
    """
    source = _ensure_bgr(source_bw_bgr)
    semantic = _ensure_bgr(semantic_color_bgr)
    if semantic.shape[:2] != source.shape[:2]:
        semantic = cv2.resize(semantic, (source.shape[1], source.shape[0]),
                              interpolation=cv2.INTER_AREA)
    _enabled, normalized_mode = _normalize_auto_black_hair_guard(mode)
    normalized_scope = _normalize_auto_black_hair_scope(scope)
    cache_key = (
        _exact_array_key(source),
        _exact_array_key(semantic),
        _exact_array_key(limit_mask),
        normalized_mode,
        normalized_scope,
        bool(parser_enabled),
        str(parser_backend or "auto").strip().lower(),
        bool(parser_allow_download),
        _hair_runtime_signature(),
    )
    use_result_cache = not bool(parser_allow_download)
    if use_result_cache:
        with _AUTO_HAIR_CACHE_LOCK:
            cached = _AUTO_HAIR_RESULT_CACHE.get(cache_key)
            if cached is not None:
                cached_mask, cached_diagnostics, cached_has_preview = cached
                if cached_has_preview or not return_preview_layers:
                    _AUTO_HAIR_RESULT_CACHE.move_to_end(cache_key)
                    diagnostics = (copy.deepcopy(cached_diagnostics)
                                   if return_preview_layers
                                   else _strip_preview_layers(cached_diagnostics))
                    diagnostics["session_cache_hit"] = True
                    return cached_mask.copy(), diagnostics
    diagnostics: dict = {"session_cache_hit": False}
    mask = None
    normalized_limit_mask = _normalize_limit_mask_shape(limit_mask, source.shape[:2])
    ai_limit_mask = None
    legacy_limit_mask = normalized_limit_mask
    if parser_enabled and str(parser_backend or "auto").lower() not in {
            "legacy", "off", "none"}:
        mask, ai_diagnostics = _ai_hair_mask_for_recolor(
            source, semantic, parser_backend=parser_backend,
            parser_allow_download=parser_allow_download,
            limit_mask=ai_limit_mask,
            return_preview_layers=return_preview_layers)
        diagnostics["ai_hair_parser"] = dict(ai_diagnostics or {})
        if return_preview_layers and isinstance(ai_diagnostics, dict):
            preview_layers = ai_diagnostics.get("_preview_layers")
            if isinstance(preview_layers, dict):
                diagnostics["_preview_layers"] = preview_layers
    if mask is None or not np.any(mask):
        legacy_mask, legacy_diagnostics = _dominant_warm_ink_hair_mask(
            source, semantic, mode=normalized_mode,
            scope=normalized_scope, limit_mask=legacy_limit_mask)
        mask = legacy_mask
        diagnostics["legacy_detector"] = dict(legacy_diagnostics or {})
        diagnostics.update(dict(legacy_diagnostics or {}))
        diagnostics["selected_detector"] = "legacy"
        if return_preview_layers:
            diagnostics["_preview_layers"] = {
                "primary": np.where(mask > 0, 255, 0).astype(np.uint8),
                "supplemental": np.zeros(source.shape[:2], np.uint8),
                "rejected_thin": np.zeros(source.shape[:2], np.uint8),
            }
    else:
        diagnostics.update({
            "enabled": True,
            "detected": True,
            "selected_pixels": int(np.count_nonzero(mask)),
            "selected_detector": "ai_hair_parser",
            "reason": "ai_hair_mask_ready",
        })
    if mask is None:
        mask = np.zeros(source.shape[:2], np.uint8)
    final_mask, diagnostics = _crop_auto_hair_outputs_to_limit(
        mask, diagnostics, normalized_limit_mask)
    if use_result_cache:
        with _AUTO_HAIR_CACHE_LOCK:
            existing = _AUTO_HAIR_RESULT_CACHE.get(cache_key)
            # A preview-rich entry is never replaced by a poorer non-preview entry.
            if existing is None or return_preview_layers or not existing[2]:
                _AUTO_HAIR_RESULT_CACHE[cache_key] = (
                    final_mask.copy(), copy.deepcopy(diagnostics),
                    bool(return_preview_layers))
                _AUTO_HAIR_RESULT_CACHE.move_to_end(cache_key)
                _trim_lru(_AUTO_HAIR_RESULT_CACHE, _AUTO_HAIR_RESULT_CACHE_MAX)
    return final_mask, diagnostics


def apply_auto_ink_black_hair_guard(
        finished_bgr: np.ndarray,
        source_bw_bgr: np.ndarray,
        *,
        semantic_color_bgr: np.ndarray,
        strength: float = 1.0,
        mode: str | bool = "normal",
        scope: str = "page",
        limit_mask: np.ndarray | None = None,
        target_rgb: tuple[int, int, int] | None = None,
        parser_enabled: bool = True,
        parser_backend: str = "auto",
        parser_allow_download: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    """Neutralise the dominant orange false-hair object in automatic mode.

    This is a final identity safeguard, not a global desaturation filter.  Only
    the automatically recovered hair object is changed.  Its luminance is
    rebuilt from the original manga tones: dense ink becomes near-black while
    white highlight strands remain visible.  All pixels outside the mask stay
    byte-identical.
    """
    result = _ensure_bgr(finished_bgr)
    source = _ensure_bgr(source_bw_bgr)
    semantic = _ensure_bgr(semantic_color_bgr)
    if source.shape[:2] != result.shape[:2]:
        source = cv2.resize(source, (result.shape[1], result.shape[0]),
                            interpolation=cv2.INTER_AREA)
    if semantic.shape[:2] != result.shape[:2]:
        semantic = cv2.resize(semantic, (result.shape[1], result.shape[0]),
                              interpolation=cv2.INTER_AREA)

    _enabled, normalized_mode = _normalize_auto_black_hair_guard(mode)
    normalized_scope = _normalize_auto_black_hair_scope(scope)
    mask, diagnostics = build_auto_hair_recolor_mask(
        source, semantic, mode=normalized_mode, scope=normalized_scope, limit_mask=limit_mask,
        parser_enabled=parser_enabled, parser_backend=parser_backend,
        parser_allow_download=parser_allow_download)
    if not np.any(mask):
        diagnostics["applied"] = False
        return result, np.zeros(result.shape[:2], np.uint8), diagnostics

    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(np.float32)
    target_rgb = tuple(int(np.clip(v, 0, 255)) for v in (target_rgb or (24, 24, 24)))
    rebuilt_target, recolor_mode = _rebuild_hair_target_from_source(source_gray, target_rgb)

    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0,
                             (5, 5), 0)
    alpha *= float(np.clip(strength, 0.0, 1.0))
    alpha = alpha[..., None]
    output = np.clip(result.astype(np.float32) * (1.0 - alpha) +
                     rebuilt_target.astype(np.float32) * alpha,
                     0, 255).astype(np.uint8)
    # Guarantee exact preservation away from the logical object.  The blurred
    # edge is allowed only within a tiny dilation so no page-wide pixels drift.
    vicinity = cv2.dilate((mask > 0).astype(np.uint8),
                          np.ones((5, 5), np.uint8), iterations=1) > 0
    output[~vicinity] = result[~vicinity]

    selected = mask > 0
    if np.any(selected):
        spread = (np.max(output[selected], axis=1).astype(np.float32) -
                  np.min(output[selected], axis=1).astype(np.float32))
        diagnostics["mean_selected_channel_spread"] = round(float(spread.mean()), 3)
        diagnostics["mean_selected_luma"] = round(float(
            cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)[selected].mean()), 3)
    diagnostics["applied"] = True
    diagnostics["strength"] = round(float(np.clip(strength, 0.0, 1.0)), 3)
    diagnostics["mode"] = normalized_mode
    diagnostics["scope"] = normalized_scope
    diagnostics["target_rgb"] = tuple(int(v) for v in target_rgb)
    diagnostics["recolor_mode"] = recolor_mode
    return output, mask, diagnostics
