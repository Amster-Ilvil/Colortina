"""Colortina v4 unified colour pipeline.

The pipeline keeps three concepts independent:

- StyleDescriptor: relative rendering language only;
- CharacterLibrary: absolute, confidence-gated identity colours;
- ScenePalette: optional absolute environment colours.

Structured HintSpec objects retain source, strength, semantic label, character
identity and confidence until a soft region-clipped model mask is rasterized.
Low-confidence identity matches are not locked.  A hint-blob detector can run
one degraded retry, and tiled mode uses a whole-page low-resolution chroma
prior to reduce colour drift between tiles.
"""

from __future__ import annotations

from dataclasses import replace
from collections import OrderedDict
import threading
import time
import zlib
import os
import hashlib

import numpy as np
import cv2

from config import Config
from core.hint_manager import HintManager
from core.lineart_fill import label_regions
from core.ml_colorizer import MangaColorizer
from core.presets import get_quality, get_style
from core.style_post import apply_style_grade
from core.image_filter import apply_image_filter
from core.cancellation import check_cancelled
from core.mcv2_region_control import region_control_signature

_colorizer: MangaColorizer | None = None  # legacy/current mc-v2 cache
_colorizer_load_lock = threading.Lock()  # guards engine construction
_guided = None
_job_guided_cache: dict = {}
_raw_result_cache: OrderedDict = OrderedDict()
_raw_result_cache_bytes: int = 0
_raw_result_cache_lock = threading.RLock()
_RAW_RESULT_CACHE_MAX_ITEMS = 8
_RAW_RESULT_CACHE_MAX_BYTES = 256 * 1024 * 1024
_builtin_style_profile_cache: dict[str, object] = {}
_page_context_cache: OrderedDict = OrderedDict()
_page_context_cache_lock = threading.RLock()
_PAGE_CONTEXT_CACHE_MAX_ITEMS = 16


def _image_fingerprint(image_bgr: np.ndarray) -> tuple:
    """Collision-resistant fingerprint for an immutable source page.

    The old key combined ``id(image)`` with a tiny 24x24 sample. During long
    batches Python can reuse an array id after the previous page is released,
    while manga pages often share almost-identical white-border samples. That
    allowed a raw mc-v2 result from page N to be reused for page N+1, which
    looked like mc-v2 and ink-wash styles being mixed on a few pages. Hashing
    the complete source buffer is still tiny compared with model inference and
    makes cross-page reuse impossible unless the pages are byte-identical.
    """
    contiguous = np.ascontiguousarray(image_bgr)
    h, w = contiguous.shape[:2]
    digest = hashlib.blake2b(
        memoryview(contiguous).cast("B"), digest_size=16).digest()
    return (h, w, str(contiguous.dtype), digest)


def _hint_signature(specs: list) -> tuple:
    return tuple((
        round(float(s.x_norm), 5), round(float(s.y_norm), 5), tuple(map(int, s.rgb)),
        round(float(s.radius_norm), 5), round(float(s.strength), 3),
        round(float(s.confidence), 3), str(s.source), int(s.region_id or 0),
        str(s.semantic or ''), int(s.character_id or -1),
    ) for s in specs)


def _label_map_signature(label_map) -> tuple | None:
    if label_map is None:
        return None
    labels = getattr(label_map, 'labels', None)
    if labels is None:
        return (getattr(label_map, 'gap_close', None), getattr(label_map, 'line_low', None))
    h, w = labels.shape[:2]
    sy = max(1, h // 16)
    sx = max(1, w // 16)
    sample = np.ascontiguousarray(labels[::sy, ::sx][:16, :16])
    return (h, w, int(getattr(label_map, 'gap_close', 0)),
            int(getattr(label_map, 'line_low', 0)), int(zlib.adler32(sample.tobytes())))


def _raw_cache_key(image_bgr: np.ndarray, quality, denoise_sigma: int,
                   model_specs: list, label_map, region_control=None,
                   hint_render_mode: str = "mixed") -> tuple:
    return (
        _image_fingerprint(image_bgr),
        int(quality.model_size), bool(quality.tiled_inference),
        int(quality.tile_size), int(quality.tile_overlap), bool(quality.per_panel),
        int(denoise_sigma),
        _hint_signature(model_specs), _label_map_signature(label_map),
        region_control_signature(region_control),
        str(hint_render_mode or "").strip().lower(),
    )


def _raw_cache_get(key: tuple) -> np.ndarray | None:
    with _raw_result_cache_lock:
        value = _raw_result_cache.get(key)
        if value is None:
            return None
        _raw_result_cache.move_to_end(key)
        return value.copy()


def _raw_cache_put(key: tuple, result: np.ndarray) -> None:
    global _raw_result_cache_bytes
    value = np.ascontiguousarray(result).copy()
    size = int(value.nbytes)
    if size > _RAW_RESULT_CACHE_MAX_BYTES // 2:
        return
    with _raw_result_cache_lock:
        old = _raw_result_cache.pop(key, None)
        if old is not None:
            _raw_result_cache_bytes -= int(old.nbytes)
        _raw_result_cache[key] = value
        _raw_result_cache_bytes += size
        while (_raw_result_cache and
               (len(_raw_result_cache) > _RAW_RESULT_CACHE_MAX_ITEMS or
                _raw_result_cache_bytes > _RAW_RESULT_CACHE_MAX_BYTES)):
            _, evicted = _raw_result_cache.popitem(last=False)
            _raw_result_cache_bytes -= int(evicted.nbytes)


def clear_raw_result_cache() -> None:
    global _raw_result_cache_bytes
    with _raw_result_cache_lock:
        _raw_result_cache.clear()
        _raw_result_cache_bytes = 0


def clear_page_context_cache() -> None:
    """Drop page-sized segmentation and semantic analysis objects."""
    with _page_context_cache_lock:
        _page_context_cache.clear()


def clear_transient_page_caches() -> None:
    """Release page pixel caches while keeping loaded neural model weights."""
    clear_raw_result_cache()
    clear_page_context_cache()
    try:
        from core.ai_hair_parser import clear_cache as clear_ai_hair_cache
        clear_ai_hair_cache()
    except Exception:
        pass

def _page_context_cache_get(key: tuple):
    with _page_context_cache_lock:
        value = _page_context_cache.get(key)
        if value is None:
            return None
        _page_context_cache.move_to_end(key)
        return value


def _page_context_cache_put(key: tuple, value) -> None:
    with _page_context_cache_lock:
        _page_context_cache[key] = value
        _page_context_cache.move_to_end(key)
        while len(_page_context_cache) > _PAGE_CONTEXT_CACHE_MAX_ITEMS:
            _page_context_cache.popitem(last=False)


def _need_semantic_context(style) -> bool:
    if style is None:
        return False
    return (str(getattr(style, 'semantic_mode', 'all')) != 'all' or
            bool(getattr(style, 'force_environment_grayscale', False)))


def _need_guided_context(style, *, effective_profile=None, character_memories=None,
                         character_library=None, scene_palette=None,
                         reference_strength: float = 0.0, has_manual_region: bool = False) -> bool:
    if _need_semantic_context(style):
        return True
    if has_manual_region:
        return True
    if effective_profile is not None:
        return True
    if scene_palette is not None:
        return True
    if character_memories:
        return True
    if (character_library is not None and
            bool(getattr(character_library, 'characters', None)) and
            reference_strength > 0.001):
        return True
    return False



def _memory_signature(character_memories: dict | None) -> tuple:
    rows = []
    for key, memory in sorted((character_memories or {}).items()):
        slots = getattr(memory, 'slots', []) or []
        rows.append((key, tuple((getattr(s, 'slot_id', -1), round(float(getattr(s, 'tone', 0.0)), 2),
                                 str(getattr(s, 'color_hex', '')), bool(getattr(s, 'locked', False)),
                                 int(getattr(s, 'hits', 0))) for s in slots)))
    return tuple(rows)


def _page_context_key(image_bgr: np.ndarray, style_profile, character_memories,
                      character_library, scene_palette, style_strength: float,
                      reference_strength: float, forced_matches: dict | None) -> tuple:
    descriptor = getattr(style_profile, '_descriptor', None)
    return (
        _image_fingerprint(image_bgr), id(style_profile), getattr(descriptor, 'revision', 0),
        id(character_library), getattr(character_library, 'revision', 0),
        id(scene_palette), getattr(scene_palette, 'revision', 0),
        _memory_signature(character_memories), round(float(style_strength), 3),
        round(float(reference_strength), 3), tuple(sorted((forced_matches or {}).items())),
    )


def _guided_cache_key(style_profile, character_memories, character_library=None,
                      scene_palette=None, style_strength: float = 1.0,
                      reference_strength: float = 1.0):
    cm_key = tuple(sorted((k, id(v)) for k, v in (character_memories or {}).items()))
    descriptor = getattr(style_profile, "_descriptor", None)
    style_revision = getattr(descriptor, "revision", 0)
    library_revision = getattr(character_library, "revision", 0)
    scene_revision = getattr(scene_palette, "revision", 0)
    return (id(style_profile), style_revision, cm_key,
            id(character_library), library_revision,
            id(scene_palette), scene_revision,
            round(float(style_strength), 3), round(float(reference_strength), 3))




def get_colorizer(cfg: Config = Config, *, ensure_weights: bool = True,
                  status_callback=None) -> MangaColorizer:
    """Return the cached mc-v2 instance.

    ``ensure_weights`` lets the background worker perform the download once
    with visible UI status, then load the model without repeating downloader
    work. Construction is lock-guarded so a UI preload thread and a colorize
    worker never build the model twice.
    """
    global _colorizer
    if _colorizer is not None:
        return _colorizer
    with _colorizer_load_lock:
        if _colorizer is not None:
            return _colorizer
        if ensure_weights:
            from core.model_downloader import ensure_models_downloaded
            ensure_models_downloaded(
                cfg.WEIGHTS_DIR, callback=status_callback or print)
        if status_callback:
            status_callback("正在初始化 mc-v2 网络...")
        _colorizer = MangaColorizer(
            device=cfg.ML_DEVICE,
            generator_path=cfg.GENERATOR_WEIGHTS_PATH,
            extractor_path=cfg.EXTRACTOR_WEIGHTS_PATH,
            denoiser_weights_dir=cfg.DENOISER_WEIGHTS_DIR,
            region_adapter_path=getattr(cfg, "REGION_ADAPTER_WEIGHTS_PATH", ""),
            tunnel2_adapter_path=(
                getattr(cfg, "TUNNEL2_SEMANTIC_ADAPTER_PATH", "")
                if getattr(cfg, "TUNNEL2_SEMANTIC_ADAPTER_ENABLED", True)
                else ""),
        )
        detail = _colorizer.device_name
        if _colorizer.device_name == "mps":
            detail += "（mc-v2 生成器；FFDNet 降噪在 CPU）"
        print(f"[pipeline] mc-v2 loaded on device: {detail}")
        if status_callback:
            if _colorizer.device_warning:
                status_callback(_colorizer.device_warning)
            else:
                status_callback(f"mc-v2 已加载：{detail}")
        return _colorizer


def get_guided_colorist(cfg: Config = Config, style_profile=None,
                        character_memories: dict | None = None,
                        character_library=None, scene_palette=None,
                        style_strength: float = 1.0,
                        reference_strength: float = 1.0):
    """Return a revision-aware GuidedColorist for this job."""
    global _guided
    key = _guided_cache_key(
        style_profile, character_memories, character_library, scene_palette,
        style_strength=style_strength,
        reference_strength=reference_strength)
    if (style_profile is None and not character_memories
            and character_library is None and scene_palette is None
            and style_strength == 1.0 and reference_strength == 1.0):
        if _guided is None:
            from core.guided_colorist import GuidedColorist
            _guided = GuidedColorist(cfg)
        return _guided
    guided = _job_guided_cache.get(key)
    if guided is None:
        from core.guided_colorist import GuidedColorist
        descriptor = getattr(style_profile, "_descriptor", None)
        guided = GuidedColorist(
            cfg, style_descriptor=descriptor, style_profile=style_profile,
            character_memories=character_memories,
            character_library=character_library, scene_palette=scene_palette,
            style_strength=style_strength,
            reference_strength=reference_strength)
        _job_guided_cache[key] = guided
    return guided


def build_page_context(image_bgr: np.ndarray, cfg: Config = Config,
                       style_profile=None, character_memories: dict | None = None,
                       character_library=None, scene_palette=None,
                       style_strength: float = 1.0,
                       reference_strength: float = 1.0,
                       forced_matches: dict[int, int] | None = None):
    """Build a lightweight cached semantic/identity context when needed."""
    key = _page_context_key(
        image_bgr, style_profile, character_memories, character_library, scene_palette,
        style_strength, reference_strength, forced_matches)
    cached = _page_context_cache_get(key)
    if cached is not None:
        return cached

    guided = get_guided_colorist(
        cfg, style_profile=style_profile, character_memories=character_memories,
        character_library=character_library, scene_palette=scene_palette,
        style_strength=style_strength, reference_strength=reference_strength)
    if not guided.available and not character_memories and character_library is None and scene_palette is None:
        from core.page_color_context import PageColorContext
        context = PageColorContext(diagnostics={"guided_unavailable": True, "fast_mode": True})
    else:
        context = guided.analyze_page(image_bgr, forced_matches=forced_matches)
        diagnostics = dict(getattr(context, 'diagnostics', {}) or {})
        diagnostics['fast_mode'] = True
        context.diagnostics = diagnostics
    _page_context_cache_put(key, context)
    return context


def build_auto_hints(image_bgr: np.ndarray, cfg: Config = Config,
                     style_profile=None, character_memories: dict | None = None,
                     character_library=None, scene_palette=None,
                     style_strength: float = 1.0,
                     reference_strength: float = 1.0,
                     forced_matches: dict[int, int] | None = None) -> list:
    context = build_page_context(
        image_bgr, cfg=cfg, style_profile=style_profile,
        character_memories=character_memories, character_library=character_library,
        scene_palette=scene_palette, style_strength=style_strength,
        reference_strength=reference_strength, forced_matches=forced_matches)
    return list(getattr(context, 'hints', []) or [])


def _effective_job_quality(style, quality, *, needs_guided: bool, model_specs: list):
    """Honor the user-selected 32-aligned inference resolution.

    Older builds exposed a quality value but forced 576 for every job.  Keep
    tiling/per-panel disabled for stable checkpoint-compatible inference,
    without misreporting larger resolution requests as completed.
    """
    size = int(getattr(quality, "model_size", 576))
    if size <= 0 or size % 32:
        raise ValueError(f"mc-v2 inference size must be divisible by 32: {size}")
    effective_quality = replace(quality, tiled_inference=False, per_panel=False)
    effective_denoise = 15
    optimizations = [f"model_resolution_{size}", 'disable_per_panel',
                     'disable_tiled_inference']
    if not needs_guided:
        optimizations.append('skip_guided_analysis')
    if not model_specs:
        optimizations.append('skip_hint_retry_no_model_hints')
    return effective_quality, effective_denoise, optimizations




def _is_default_style_tuning(tuning: dict | None) -> bool:
    if not tuning:
        return True
    # Only controls visibly shared by all styles decide whether original mc-v2
    # remains untouched. Hidden Light Wash 3 / legacy semantic values must not
    # accidentally activate original-style grading after a style switch.
    default_map = {
        "color_strength": 100,
        "brightness": 100,
        "warmth": 100,
        "highlight_preserve": 100,
        "softness": 100,
        "flatten": 100,
    }
    for key, default in default_map.items():
        try:
            value = int(round(float(tuning.get(key, default))))
        except Exception:
            value = default
        if value != default:
            return False
    return True


def _adjustable_original_style():
    """Synthetic baseline for fine-tuning the original mc-v2 output.

    At default slider positions this style is not used; the raw mc-v2 output is
    returned unchanged. Once the user moves a style-fine-tuning control while
    the original preset is selected, we grade from this near-neutral baseline
    so the controls have visible effect without depending on another preset.
    """
    from core.presets import StylePreset
    return StylePreset(
        key="none_tuned",
        label="MC v2（细调）",
        description="Adjustable baseline for original mc-v2 fine tuning.",
        saturation_boost=1.0,
        white_threshold=255,
        black_threshold=0,
        neutral_transition=64,
        l_blend_alpha=1.0,
        guided_filter_radius=2,
        guided_filter_eps=0.01,
        chroma_warm_shift=0.0,
        chroma_red_shift=0.0,
        l_gamma=1.0,
        cel_flatten=0.0,
        neutral_fade_floor=1.0,
        denoise_sigma=15,
        diffusion_steps=16,
    )

def _apply_pastel_tuning(style, tuning: dict | None):
    """Return a tuned copy of a style preset.

    This now handles both unified monochrome pastel and the two light-wash
    styles. The controls expose the factors that visibly change the final page:
    colour amount, brightness, warmth, highlight preservation, wash softness,
    flattening, character colour, and scene tint.
    """
    if not tuning or style is None:
        return style
    style_key = str(getattr(style, "key", "") or "")

    def factor(key: str, default: float = 100.0, lo: float = 0.0, hi: float = 200.0) -> float:
        try:
            return float(np.clip(float(tuning.get(key, default)), lo, hi)) / 100.0
        except Exception:
            return default / 100.0

    color = factor("color_strength")
    brightness = factor("brightness")
    warmth = factor("warmth")
    softness = factor("softness")
    flatten = factor("flatten")
    highlight = factor("highlight_preserve")

    def interp01(value: float, lo: float, mid: float, hi: float) -> float:
        if value <= 1.0:
            t = np.clip(value, 0.0, 1.0)
            return float(lo * (1.0 - t) + mid * t)
        t = np.clip(value - 1.0, 0.0, 1.0)
        return float(mid * (1.0 - t) + hi * t)

    sat = float(np.clip(style.saturation_boost * interp01(color, 0.24, 1.0, 1.95), 0.04, 2.8))
    gamma = float(np.clip(style.l_gamma * interp01(brightness, 1.34, 1.0, 0.72), 0.68, 1.58))
    # Dedicated continuous strength for the 淡彩水墨3 family.  At 0 the page
    # is an extremely faint wash; 100 preserves the preset; 200 increases
    # visible colour while remaining substantially softer than ordinary mc-v2.
    if str(getattr(style, "key", "")).startswith("light3"):
        light3_intensity = factor("light3_intensity", 100.0, 0.0, 200.0)
        sat *= interp01(light3_intensity, 0.18, 1.0, 1.72)
        gamma *= interp01(light3_intensity, 0.96, 1.0, 1.04)
    warm_shift = float(style.chroma_warm_shift + (interp01(warmth, -1.0, 0.0, 1.0) * 11.0))
    red_shift = float(style.chroma_red_shift + (interp01(warmth, -1.0, 0.0, 1.0) * 2.8))
    radius = int(np.clip(round(max(1.0, float(style.guided_filter_radius)) * interp01(softness, 0.55, 1.0, 1.95)), 1, 11))
    eps = float(np.clip(float(style.guided_filter_eps) * interp01(softness, 0.55, 1.0, 2.15), 0.003, 0.11))
    cel = float(np.clip(float(style.cel_flatten) + interp01(flatten, -0.22, 0.0, 0.58), 0.0, 0.90))
    fade_floor = float(np.clip(float(style.neutral_fade_floor) * interp01(highlight, 0.55, 1.0, 2.25), 0.02, 0.98))
    if str(getattr(style, "key", "")).startswith("light3"):
        light3_intensity = factor("light3_intensity", 100.0, 0.0, 200.0)
        fade_floor *= interp01(light3_intensity, 0.55, 1.0, 1.30)
        fade_floor = float(np.clip(fade_floor, 0.015, 0.45))

    updates = dict(
        saturation_boost=sat,
        l_gamma=gamma,
        chroma_warm_shift=warm_shift,
        chroma_red_shift=red_shift,
        guided_filter_radius=radius,
        guided_filter_eps=eps,
        cel_flatten=cel,
        neutral_fade_floor=fade_floor,
    )

    semantic = str(getattr(style, "semantic_mode", "all"))
    if semantic != "all":
        overall = factor("person_strength")
        hair = factor("hair_strength")
        skin = factor("skin_strength")
        eyes = factor("eye_strength")
        clothing = factor("clothing_strength")
        env = factor("environment_strength", 100.0, 0.0, 180.0)
        skin_warm = factor("skin_warmth", 100.0, 50.0, 170.0)

        base_person = max(0.18, float(style.person_chroma_scale))
        base_env = max(0.0, float(style.environment_chroma_scale))
        base_unknown = max(0.0, float(style.unknown_chroma_scale))
        person_gain = interp01(overall, 0.20, 1.0, 2.20)
        hair_gain = interp01(hair, 0.18, 1.0, 2.35)
        skin_gain = interp01(skin, 0.22, 1.0, 2.05)
        eye_gain = interp01(eyes, 0.20, 1.0, 2.40)
        clothing_gain = interp01(clothing, 0.18, 1.0, 2.10)
        env_gain = interp01(env, 0.0, 1.0, 2.55)
        env_scale = float(np.clip(base_env * env_gain, 0.0, 0.42))
        unknown_scale = float(np.clip(base_unknown * env_gain, 0.0, 0.24))
        force_gray = env_scale < 0.006

        updates.update(
            person_chroma_scale=float(np.clip(base_person * person_gain, 0.0, 2.0)),
            hair_chroma_scale=float(np.clip(style.hair_chroma_scale * person_gain * hair_gain, 0.0, 2.4)),
            skin_chroma_scale=float(np.clip(style.skin_chroma_scale * person_gain * skin_gain, 0.0, 2.1)),
            eye_chroma_scale=float(np.clip(style.eye_chroma_scale * person_gain * eye_gain, 0.0, 2.6)),
            clothing_chroma_scale=float(np.clip(style.clothing_chroma_scale * person_gain * clothing_gain, 0.0, 2.2)),
            environment_chroma_scale=env_scale,
            unknown_chroma_scale=unknown_scale,
            force_environment_grayscale=force_gray,
            skin_target_b=float(np.clip(style.skin_target_b + (interp01(skin_warm, -1.0, 0.0, 1.0)) * 8.5, 3.0, 19.0)),
            skin_target_a=float(np.clip(style.skin_target_a + (interp01(skin_warm, -1.0, 0.0, 1.0)) * 2.6, -5.0, 8.5)),
            skin_neutralize=float(np.clip(style.skin_neutralize * interp01(skin_warm, 0.78, 1.0, 1.22), 0.0, 1.0)),
        )
    return replace(style, **updates)



def _post_only_context(hint_manager: HintManager, diagnostics: dict):
    """Reuse lightweight post-style state for one page across style switches."""
    from core.page_color_context import PageColorContext
    context = getattr(hint_manager, '_post_only_page_context', None)
    if context is None:
        context = PageColorContext()
        hint_manager._post_only_page_context = context
    context.diagnostics = dict(diagnostics)
    return context




def _normalize_auto_black_hair_guard_setting(mode) -> tuple[bool, str]:
    if isinstance(mode, str):
        key = mode.strip().lower()
        if key in {"", "0", "false", "off", "disabled", "none"}:
            return False, "off"
        if key in {"strict", "conservative", "conservative_black"}:
            return True, "conservative"
        return True, "normal"
    return (bool(mode), "normal" if bool(mode) else "off")




def _normalize_auto_black_hair_guard_scope(scope) -> str:
    if isinstance(scope, str):
        key = scope.strip().lower()
        if key in {"face", "near_face", "character", "character_only"}:
            return "near_face"
        if key in {"selection", "mask", "selected", "selected_region"}:
            return "selection"
    return "page"


def colorize_page(image_bgr: np.ndarray,
                   hint_manager: HintManager | None = None,
                   cfg: Config = Config,
                   regenerate_auto: bool = True,
                   style_key: str | None = None,
                   quality_key: str | None = None,
                   character_memories: dict | None = None,
                   character_library=None,
                   scene_palette=None,
                   style_strength: float = 1.0,
                   reference_strength: float = 1.0,
                   manual_strength: float = 1.0,
                   pastel_tuning: dict | None = None,
                   filter_tuning: dict | None = None,
                   custom_color_bias: dict | None = None,
                   forced_matches: dict[int, int] | None = None,
                   preserve_empty_auto_hints: bool = False,
                   learn_identity: bool = True,
                   return_filter_base: bool = False,
                   hint_render_mode: str = "mixed",
                   protect_text: bool = True,
                   region_control=None,
                   two_stage_identity: bool | None = None,
                   suppress_non_character_purple: bool | None = None,
                   auto_black_hair_guard: bool | str = False,
                   auto_black_hair_guard_scope: str = "page",
                   auto_black_hair_guard_limit_mask: np.ndarray | None = None,
                   auto_black_hair_guard_target_rgb: tuple[int, int, int] | None = None,
                   ai_hair_parser_enabled: bool = True,
                   ai_hair_parser_backend: str = "auto",
                   ai_hair_parser_allow_download: bool = False,
                   semantic_mask_controls: dict | None = None,
                   ink_fidelity: bool = False,
                   cancel_check=None):
    """Run the full Auto pipeline on one page.

    """
    check_cancelled(cancel_check)
    if hint_manager is None:
        hint_manager = HintManager()
    style_strength = float(np.clip(style_strength, 0.0, 1.0))
    auto_black_hair_guard_enabled, auto_black_hair_guard_mode = _normalize_auto_black_hair_guard_setting(auto_black_hair_guard)
    auto_black_hair_guard_scope = _normalize_auto_black_hair_guard_scope(auto_black_hair_guard_scope)
    semantic_mask_controls = dict(semantic_mask_controls or {})
    semantic_mask_enabled = bool(semantic_mask_controls.get("enabled", False))
    semantic_hair_authoritative = bool(
        semantic_mask_enabled and
        dict(semantic_mask_controls.get("hair") or {}).get("enabled", False))
    reference_strength = float(np.clip(reference_strength, 0.0, 1.0))
    manual_strength = float(np.clip(manual_strength, 0.0, 1.0))
    if two_stage_identity is None:
        two_stage_identity = bool(getattr(cfg, "ENABLE_TWO_STAGE_IDENTITY", True))
    if suppress_non_character_purple is None:
        suppress_non_character_purple = bool(
            getattr(cfg, "SUPPRESS_NON_CHARACTER_PURPLE", True))
    # Region maps are built lazily only when a model hint actually needs them.

    style_key_normalized = (style_key or "").lower()
    original_style_selected = style_key_normalized == "none"
    default_style_tuning = _is_default_style_tuning(pastel_tuning)
    # Plain original mc-v2: no style processing at all.
    is_none_style = original_style_selected and default_style_tuning

    # Only enable semantic/identity analysis when it materially changes the
    # result. This keeps ordinary "raw mc-v2" jobs fast, while letting
    # monochrome pastel masks, character identity colours and manual region
    # instructions actually function again.
    has_manual_region = any(getattr(h, 'source', '') == 'manual_region'
                            for h in getattr(hint_manager, 'manual_hints', []) or [])
    effective_profile = None
    needs_guided = False
    if not is_none_style and has_manual_region:
        effective_profile = _builtin_style_profile(style_key or "neutral")

    style = (_adjustable_original_style() if (original_style_selected and not default_style_tuning)
             else get_style(style_key))
    style = _apply_pastel_tuning(style, pastel_tuning)
    # The ink-wash renderer is a final, hue-faithful treatment of the native
    # generator output, not a second colourizer. Do not grade/desaturate the
    # same colours first using the legacy heuristic ink-wash style engine.
    native_wash_style = str(getattr(style, "key", "")) in {
        "light2", "light2_soft", "light3"}
    # Git legacy: the original light2/light3 grade rather than the new final
    # Generator-first ink-wash. They remain independently selectable.
    from core.presets import GIT_LEGACY_STYLE_KEYS
    git_legacy_wash_style = str(getattr(style, "key", "")) in GIT_LEGACY_STYLE_KEYS
    quality = get_quality(quality_key)

    needs_guided = _need_guided_context(
        style, effective_profile=effective_profile,
        character_memories=character_memories, character_library=character_library,
        scene_palette=scene_palette, reference_strength=reference_strength,
        has_manual_region=has_manual_region)

    if needs_guided:
        check_cancelled(cancel_check)
        page_context = build_page_context(
            image_bgr, cfg=cfg, style_profile=effective_profile,
            character_memories=character_memories, character_library=character_library,
            scene_palette=scene_palette, style_strength=style_strength,
            reference_strength=reference_strength, forced_matches=forced_matches)
        check_cancelled(cancel_check)
        if regenerate_auto or (not hint_manager.auto_hints and not preserve_empty_auto_hints):
            hint_manager.set_auto_hints(build_auto_hints(
                image_bgr, cfg=cfg, style_profile=effective_profile,
                character_memories=character_memories, character_library=character_library,
                scene_palette=scene_palette, style_strength=style_strength,
                reference_strength=reference_strength, forced_matches=forced_matches))
            check_cancelled(cancel_check)
    else:
        page_context = _post_only_context(hint_manager, {
            "plain_auto": bool(is_none_style),
            "post_only_style": None if is_none_style else (style_key or "custom"),
            "fast_mode": True,
        })
        if regenerate_auto or (not hint_manager.auto_hints and not preserve_empty_auto_hints):
            hint_manager.set_auto_hints([])

    check_cancelled(cancel_check)
    descriptor = getattr(effective_profile, "_descriptor", None) if effective_profile is not None else None
    merged_specs = hint_manager.merge_specs(
        image_bgr=image_bgr, style_descriptor=descriptor,
        style_strength=style_strength, manual_strength=manual_strength)
    # Manual dabs ARE model instructions: with the "mixed" hint renderer
    # they fill exactly the enclosed line-art region they were placed in
    # (bounded, WYSIWYG), so the historic fear of a tiny hint flooding a
    # whole connected face no longer applies.
    #
    # ``manual_paint`` is the brush's OWN independent channel: strokes
    # recorded with that source are painted onto the result as local
    # dots/marks after generation and are never sent to the model.
    local_manual_specs = [h for h in merged_specs
                          if h.source == "manual_paint"]
    model_specs = [h for h in merged_specs
                   if h.source != "manual_paint"]

    label_map = None
    if model_specs:
        check_cancelled(cancel_check)
        region_map = hint_manager.region_map
        if region_map is None:
            hint_manager.bind_source_image(image_bgr)
            region_map = hint_manager.region_map
        label_map = region_map if region_map is not None else label_regions(image_bgr)

    effective_quality, effective_denoise, job_optimizations = _effective_job_quality(
        style, quality, needs_guided=needs_guided, model_specs=model_specs)

    check_cancelled(cancel_check)
    model_started = time.perf_counter()
    raw_key = _raw_cache_key(
        image_bgr, effective_quality, effective_denoise, model_specs, label_map,
        region_control=region_control, hint_render_mode=hint_render_mode)
    raw_result = _raw_cache_get(raw_key)
    raw_cache_hit = raw_result is not None
    if raw_cache_hit:
        job_optimizations.append('reuse_cached_raw_mc_v2')
    else:
        colorizer = get_colorizer(cfg)
        raw_result = colorizer.colorize(
            image_bgr,
            size=effective_quality.model_size,
            denoise_sigma=effective_denoise,
            tiled=effective_quality.tiled_inference,
            tile_size=effective_quality.tile_size,
            tile_overlap=effective_quality.tile_overlap,
            per_panel=effective_quality.per_panel,
            hint_points=model_specs or None,
            label_map=label_map,
            hint_render_mode=hint_render_mode,
            region_control=region_control,
            cancel_check=cancel_check,
        )
        check_cancelled(cancel_check)
        _raw_cache_put(raw_key, raw_result)
    check_cancelled(cancel_check)
    model_seconds = time.perf_counter() - model_started

    # A single automatic degradation retry is cheaper and safer than adding
    # more hints when coloured dots survive around hint centres. When there
    # are no model hints at all, the retry path is skipped entirely.
    retried = False
    if model_specs:
        from core.hint_artifact import detect_hint_blobs
        artifact_report = detect_hint_blobs(raw_result, model_specs)
        if artifact_report.should_retry:
            from core.hint_composer import degrade_for_retry
            retry_specs = degrade_for_retry(model_specs)
            if retry_specs != model_specs:
                colorizer = get_colorizer(cfg)
                check_cancelled(cancel_check)
                raw_result = colorizer.colorize(
                    image_bgr,
                    size=effective_quality.model_size,
                    denoise_sigma=effective_denoise,
                    tiled=effective_quality.tiled_inference,
                    tile_size=effective_quality.tile_size,
                    tile_overlap=effective_quality.tile_overlap,
                    per_panel=effective_quality.per_panel,
                    hint_points=retry_specs or None,
                    label_map=label_map,
                    hint_render_mode=hint_render_mode,
                    region_control=region_control,
                    cancel_check=cancel_check,
                )
                check_cancelled(cancel_check)
                model_specs = retry_specs
                retry_key = _raw_cache_key(
                    image_bgr, effective_quality, effective_denoise, model_specs, label_map,
                    region_control=region_control, hint_render_mode=hint_render_mode)
                _raw_cache_put(retry_key, raw_result)
                artifact_report = detect_hint_blobs(raw_result, model_specs)
                retried = True
    else:
        from types import SimpleNamespace
        artifact_report = SimpleNamespace(score=0.0, suspicious=False, should_retry=False)

    check_cancelled(cancel_check)

    # Stage 1: make the reference palette the input to style grading.  This
    # prevents the style pass from learning a different hair/eye/skin colour.
    identity_base_result = raw_result
    identity_base_lock_applied = False
    check_cancelled(cancel_check)
    if (two_stage_identity and character_library is not None and
            reference_strength > 0.0 and page_context is not None):
        from core.character_consistency import apply_character_palette_lock
        check_cancelled(cancel_check)
        identity_base_result = apply_character_palette_lock(
            raw_result, image_bgr, character_library,
            strength=reference_strength,
            assignments=page_context.identity_assignments,
            segmentation=page_context.segmentation)
        identity_base_lock_applied = identity_base_result is not raw_result

    grade_preset = style
    grade_seconds = 0.0
    if is_none_style or native_wash_style:
        result_bgr = identity_base_result
    else:
        grade_preset = style
        grade_started = time.perf_counter()
        check_cancelled(cancel_check)
        graded = apply_style_grade(identity_base_result, image_bgr, grade_preset, context=page_context)
        check_cancelled(cancel_check)
        grade_seconds = time.perf_counter() - grade_started
        if style_strength >= 0.999:
            result_bgr = graded
        elif style_strength <= 0.001:
            result_bgr = identity_base_result
        else:
            result_bgr = np.clip(
                identity_base_result.astype(np.float32) * (1.0 - style_strength) +
                graded.astype(np.float32) * style_strength,
                0, 255).astype(np.uint8)

    # Hints guide the model; this deterministic lock guarantees identity
    # colours across pages for matched hair, skin, eyes and clothing regions.
    post_style_identity_lock_applied = False
    if (character_library is not None and reference_strength > 0.0 and
            page_context is not None):
        from core.character_consistency import apply_character_palette_lock
        result_bgr = apply_character_palette_lock(
            result_bgr, image_bgr, character_library,
            strength=reference_strength,
            assignments=page_context.identity_assignments,
            segmentation=page_context.segmentation)
        post_style_identity_lock_applied = True

    if (not is_none_style and
            bool(getattr(grade_preset, "force_environment_grayscale", False))):
        from core.style_post import enforce_character_only_color
        result_bgr = enforce_character_only_color(
            result_bgr, image_bgr, context=page_context)

    check_cancelled(cancel_check)
    if local_manual_specs:
        from core.local_brush import apply_local_brush_recolor
        region_map = hint_manager.region_map
        h, w = result_bgr.shape[:2]
        for spec in local_manual_specs:
            check_cancelled(cancel_check)
            ix = min(w - 1, max(0, int(round(spec.x_norm * (w - 1)))))
            iy = min(h - 1, max(0, int(round(spec.y_norm * (h - 1)))))
            radius_px = max(1, int(round(spec.radius_norm * w)))
            result_bgr, _mask = apply_local_brush_recolor(
                image_bgr, result_bgr, ix, iy, radius_px, spec.rgb,
                opacity=min(0.9, 0.9 * spec.effective_strength),
                region_map=region_map,
                gap_close=getattr(region_map, "gap_close", 4) if region_map is not None else 4)

    check_cancelled(cancel_check)
    if custom_color_bias and custom_color_bias.get("enabled"):
        from core.custom_color_bias import apply_global_color_bias
        bias_rgb = tuple(custom_color_bias.get("rgb") or (255, 160, 160))
        bias_strength = float(custom_color_bias.get("strength", 35)) / 100.0
        bias_scope = str(custom_color_bias.get("scope", "page"))
        bias_tone_range = str(custom_color_bias.get("tone_range", "all"))
        result_bgr = apply_global_color_bias(
            result_bgr, image_bgr, bias_rgb, bias_strength, bias_scope,
            bias_tone_range,
            protect_skin=bool(custom_color_bias.get("protect_skin", True)),
            protect_lineart=bool(custom_color_bias.get("protect_lineart", True)),
            protect_saturated=bool(custom_color_bias.get("protect_saturated", True)),
        )

    check_cancelled(cancel_check)
    if protect_text:
        # 文字气泡保护：把原始黑白页合成回检测到的文字像素，
        # 对话框保持纸白、字保持墨黑。模型缺失时自动跳过。
        from core.text_guard import protect_text_regions
        result_bgr = protect_text_regions(result_bgr, image_bgr)

    check_cancelled(cancel_check)
    filter_base_bgr = result_bgr.copy()
    result_bgr = apply_image_filter(
        result_bgr, filter_tuning,
        style_strength=style_strength,
        is_styled=not is_none_style,
        source_bw_bgr=image_bgr)

    # Filters are allowed to soften the page, but must not undo the palette
    # contract established by the two-stage pass.
    post_filter_identity_lock_applied = False
    if (two_stage_identity and character_library is not None and
            reference_strength > 0.0 and page_context is not None):
        from core.character_consistency import apply_character_palette_lock
        result_bgr = apply_character_palette_lock(
            result_bgr, image_bgr, character_library,
            strength=reference_strength,
            assignments=page_context.identity_assignments,
            segmentation=page_context.segmentation)
        post_filter_identity_lock_applied = True

    purple_guard_applied = False
    check_cancelled(cancel_check)
    if (suppress_non_character_purple and not is_none_style
            and not native_wash_style and not git_legacy_wash_style):
        from core.style_post import suppress_non_character_purple
        result_bgr = suppress_non_character_purple(
            result_bgr, image_bgr, context=page_context,
            strength=float(getattr(cfg, "PURPLE_GUARD_STRENGTH", 0.62)))
        purple_guard_applied = True

    water_harmonized = False
    check_cancelled(cancel_check)
    if not is_none_style and not native_wash_style and not git_legacy_wash_style:
        from core.style_post import harmonize_water_color
        water_result = harmonize_water_color(
            result_bgr, image_bgr, context=page_context, strength=0.82)
        water_harmonized = water_result is not result_bgr
        result_bgr = water_result

    # Automatic mode can misread a large ink-filled hairstyle as orange/red.
    # Run this after the global style, identity, purple and water passes so
    # later processing cannot warm the corrected hair again. Explicit character
    # and region identities remain authoritative inside the guard itself.
    auto_black_hair_mask = None
    auto_black_hair_diagnostics = {
        "enabled": bool(auto_black_hair_guard_enabled), "applied": False,
        "mode": auto_black_hair_guard_mode,
        "reason": "style_or_identity_gate",
    }
    effective_style_name = str(getattr(grade_preset, "key", style_key or "none"))
    ink_guard_styles = {
        "light2", "light2_soft", "light3",
    }
    explicit_region_identity = bool(
        region_control and region_control.get("enabled", False))
    explicit_library_identity = bool(
        character_library is not None and page_context is not None and
        getattr(page_context, "identity_assignments", None))
    check_cancelled(cancel_check)
    if (bool(auto_black_hair_guard_enabled) and
            not semantic_hair_authoritative and
            effective_style_name in ink_guard_styles):
        from core.illustrative_inkwash import apply_auto_ink_black_hair_guard
        result_bgr, auto_black_hair_mask, auto_black_hair_diagnostics = (
            apply_auto_ink_black_hair_guard(
                result_bgr, image_bgr, semantic_color_bgr=raw_result,
                strength=1.0, mode=auto_black_hair_guard_mode,
                scope=auto_black_hair_guard_scope,
                limit_mask=auto_black_hair_guard_limit_mask,
                target_rgb=auto_black_hair_guard_target_rgb,
                parser_enabled=bool(ai_hair_parser_enabled),
                parser_backend=str(ai_hair_parser_backend or "auto"),
                parser_allow_download=bool(ai_hair_parser_allow_download)))
        if protect_text and auto_black_hair_mask is not None and np.any(auto_black_hair_mask):
            from core.text_guard import protect_text_regions
            result_bgr = protect_text_regions(
                result_bgr, image_bgr, use_fallback=True,
                vicinity_mask=auto_black_hair_mask)

    # Build semantic masks at source resolution before optional upscale, then
    # apply them at the very end as authoritative binary replacements.  This
    # keeps mc-v2/style/upscale free to do their normal work while guaranteeing
    # that the final hair/iris/background pixels obey the user's mask layer.
    semantic_masks = None
    semantic_mask_diagnostics = {
        "enabled": bool(semantic_mask_enabled), "applied": False,
        "composition": "hard_replace",
    }
    if semantic_mask_enabled:
        check_cancelled(cancel_check)
        try:
            from core.semantic_mask_layers import build_semantic_masks
            fusion_cfg = dict(semantic_mask_controls.get("fusion") or {})
            use_generator_features = bool(fusion_cfg.get("use_generator_features", True))
            generator_feature_provider = None
            if use_generator_features:
                def generator_feature_provider(region_labels):
                    return get_colorizer(cfg).extract_region_embeddings(
                        image_bgr, region_labels, size=effective_quality.model_size)

            use_generator_semantic_head = bool(
                fusion_cfg.get("use_generator_semantic_head", True) and
                getattr(cfg, "GENERATOR_SEMANTIC_HEAD_ENABLED", True))
            generator_semantic_provider = None
            if use_generator_semantic_head:
                semantic_head_path = str(getattr(
                    cfg, "GENERATOR_SEMANTIC_HEAD_PATH", "") or "")
                semantic_head_size = int(getattr(
                    cfg, "GENERATOR_SEMANTIC_HEAD_SIZE", effective_quality.model_size))
                semantic_head_conf = float(getattr(
                    cfg, "GENERATOR_SEMANTIC_HEAD_CONFIDENCE", 0.45))

                def generator_semantic_provider(semantic_source):
                    return get_colorizer(cfg).predict_generator_semantics(
                        semantic_source, weights_path=semantic_head_path,
                        size=semantic_head_size,
                        confidence_threshold=semantic_head_conf)

                try:
                    semantic_mtime = (os.path.getmtime(semantic_head_path)
                                      if semantic_head_path else 0.0)
                except OSError:
                    semantic_mtime = 0.0
                generator_semantic_provider.cache_token = (
                    semantic_head_path, round(float(semantic_mtime), 3),
                    int(semantic_head_size), round(float(semantic_head_conf), 4))
            semantic_masks = build_semantic_masks(
                result_bgr, image_bgr,
                parser_backend=str(ai_hair_parser_backend or "auto"),
                parser_allow_download=bool(ai_hair_parser_allow_download),
                manual_masks=dict(semantic_mask_controls.get("manual_masks") or {}),
                generator_feature_provider=generator_feature_provider,
                generator_semantic_provider=generator_semantic_provider,
                use_region_fusion=bool(fusion_cfg.get("enabled", True)),
                use_trappedball=bool(fusion_cfg.get("use_trappedball", True)),
                use_clip_regions=bool(fusion_cfg.get("use_clip_regions", True)),
                clip_allow_download=bool(fusion_cfg.get("clip_allow_download", False)),
                use_cache=bool(fusion_cfg.get("use_cache", True)))
        except Exception as exc:
            semantic_mask_diagnostics = {
                "enabled": True, "applied": False,
                "composition": "hard_replace",
                "reason": f"mask_build_error:{type(exc).__name__}:{exc}",
            }

    check_cancelled(cancel_check)
    if quality.use_upscale:
        from core.upscaler import upscale
        result_bgr = upscale(result_bgr, scale=4,
                             weights_path=getattr(cfg, "ESRGAN_MODEL_PATH", None))

    if semantic_mask_enabled and semantic_masks is not None:
        check_cancelled(cancel_check)
        try:
            from core.semantic_mask_layers import (
                apply_semantic_mask_overrides, build_character_semantic_overrides)
            character_semantic_overrides = None
            if (bool(semantic_mask_controls.get("use_character_library_colors", False)) and
                    character_library is not None and page_context is not None):
                character_semantic_overrides = build_character_semantic_overrides(
                    page_context, character_library, semantic_masks, result_bgr.shape[:2])
            result_bgr, semantic_masks, semantic_mask_diagnostics = apply_semantic_mask_overrides(
                result_bgr, image_bgr, semantic_mask_controls,
                parser_backend=str(ai_hair_parser_backend or "auto"),
                parser_allow_download=False, masks=semantic_masks,
                character_overrides=character_semantic_overrides)
            # Text/balloon restoration stays above every colour layer.
            if protect_text and semantic_mask_diagnostics.get("applied", False):
                from core.text_guard import protect_text_regions
                source_for_guard = image_bgr
                if source_for_guard.shape[:2] != result_bgr.shape[:2]:
                    source_for_guard = cv2.resize(
                        source_for_guard, (result_bgr.shape[1], result_bgr.shape[0]),
                        interpolation=cv2.INTER_CUBIC)
                result_bgr = protect_text_regions(result_bgr, source_for_guard)
        except Exception as exc:
            semantic_mask_diagnostics = {
                "enabled": True, "applied": False,
                "composition": "hard_replace",
                "reason": f"mask_apply_error:{type(exc).__name__}:{exc}",
            }

    check_cancelled(cancel_check)
    learned_identity_updates = 0
    if (learn_identity and character_library is not None and page_context is not None
            and reference_strength > 0.0):
        try:
            learned_identity_updates = int(character_library.learn_from_colorized_page(
                page_context, result_bgr, strength=reference_strength))
        except Exception:
            learned_identity_updates = 0

    identity_drift_alerts = 0
    identity_drift_max = 0.0
    if character_library is not None and page_context is not None:
        try:
            rows = character_library.diagnostic_rows(
                page_context, result_bgr=result_bgr, max_rows=128)
            identity_drift_alerts = sum(len(row.get("drift_alerts", [])) for row in rows)
            identity_drift_max = max(
                [float(row.get("max_delta_e", 0.0)) for row in rows] or [0.0])
        except Exception:
            identity_drift_alerts = 0
            identity_drift_max = 0.0

    check_cancelled(cancel_check)
    native_model = getattr(getattr(_colorizer, "_model", None), "colorizer", None)
    native_generator = getattr(native_model, "generator", None)
    native_adapter = getattr(getattr(_colorizer, "_model", None),
                             "tunnel2_semantic_adapter", None)
    native_head_diag = dict(
        dict(semantic_mask_diagnostics or {}).get("generator_semantic_head") or {})
    native_hair_pixels = int(
        dict(semantic_mask_diagnostics or {}).get("hair_pixels", 0) or 0)
    native_control_enabled = bool(
        region_control and region_control.get("enabled", False) and
        str(region_control.get("generator_control_mode", "direct_tunnel2")).lower()
        in {"direct_tunnel2", "direct", "tunnel2"})
    diagnostics = dict(getattr(page_context, "diagnostics", {}) or {})
    diagnostics.update({
        "composed_hint_count": len(merged_specs),
        "model_hint_count": len(model_specs),
        "local_manual_edit_count": len(local_manual_specs),
        "hint_blob_score": artifact_report.score,
        "hint_blob_suspicious": artifact_report.suspicious,
        "hint_retry": retried,
        "effective_model_size": int(effective_quality.model_size),
        "effective_per_panel": bool(effective_quality.per_panel),
        "effective_tiled": bool(effective_quality.tiled_inference),
        "effective_denoise_sigma": int(effective_denoise),
        "effective_style_key": str(getattr(grade_preset, "key", style_key or "none")),
        "git_legacy_inkwash": bool(git_legacy_wash_style),
        "effective_style_strength": round(float(style_strength), 3),
        "job_optimizations": list(job_optimizations),
        "raw_cache_hit": bool(raw_cache_hit),
        "model_seconds": round(float(model_seconds), 3),
        "grade_seconds": round(float(grade_seconds), 3),
        "identity_drift_alerts": int(identity_drift_alerts),
        "identity_drift_max_delta_e": round(float(identity_drift_max), 1),
        "two_stage_identity": bool(two_stage_identity),
        "identity_base_lock_applied": bool(identity_base_lock_applied),
        "post_style_identity_lock_applied": bool(post_style_identity_lock_applied),
        "post_filter_identity_lock_applied": bool(post_filter_identity_lock_applied),
        "non_character_purple_guard": bool(purple_guard_applied),
        "water_color_harmonized": bool(water_harmonized),
        "auto_black_hair_guard_enabled": bool(auto_black_hair_guard_enabled),
        "auto_black_hair_guard_mode": auto_black_hair_guard_mode,
        "auto_black_hair_guard_scope": auto_black_hair_guard_scope,
        "auto_black_hair_guard_limit_mask": bool(auto_black_hair_guard_limit_mask is not None and np.any(auto_black_hair_guard_limit_mask)),
        "auto_black_hair_guard_target_rgb": (tuple(int(v) for v in auto_black_hair_guard_target_rgb) if auto_black_hair_guard_target_rgb is not None else (24, 24, 24)),
        "ai_hair_parser_enabled": bool(ai_hair_parser_enabled),
        "ai_hair_parser_backend": str(ai_hair_parser_backend or "auto"),
        "ai_hair_parser_allow_download": bool(ai_hair_parser_allow_download),
        "semantic_mask_layers_enabled": bool(semantic_mask_enabled),
        "semantic_mask_hair_authoritative": bool(semantic_hair_authoritative),
        "semantic_mask_layers": dict(semantic_mask_diagnostics),
        "auto_hair_recolor_overrode_region_identity": bool(auto_black_hair_guard_enabled and explicit_region_identity),
        "auto_hair_recolor_overrode_library_identity": bool(auto_black_hair_guard_enabled and explicit_library_identity),
        "auto_black_hair_guard_applied": bool(
            auto_black_hair_diagnostics.get("applied", False)),
        "auto_black_hair_guard": dict(auto_black_hair_diagnostics),
        "region_control_enabled": bool(region_control and region_control.get("enabled", False)),
        "region_control_target_rgb": tuple(region_control.get("target_rgb", ())) if region_control else (),
        "region_control_dual_route": bool(region_control and region_control.get("enabled", False)),
        "region_control_feature_gain": float(region_control.get("feature_gain", 0.0)) if region_control else 0.0,
        "region_control_adapter_requested": bool(region_control and region_control.get("use_adapter", False)),
        "generator_native_control": {
            "enabled": native_control_enabled,
            "semantic_backend": native_head_diag.get("backend", "legacy_or_unavailable"),
            "control_backend": "tunnel2_adapter" if native_adapter is not None else "tunnel2_residual",
            "injection_point": "tunnel2" if native_control_enabled else None,
            "adapter_loaded": bool(native_adapter is not None),
            "adapter_load_error": getattr(
                getattr(_colorizer, "_model", None),
                "tunnel2_semantic_adapter_error", None),
            "encoder_forward_count": (int(getattr(native_generator,
                                                    "encoder_forward_count", 0))
                                       if native_generator is not None else None),
            "hair_pixels": native_hair_pixels,
            "hair_color_coverage": dict(semantic_mask_diagnostics or {}).get(
                "hair_color_coverage"),
            "outside_changed_pixels": dict(semantic_mask_diagnostics or {}).get(
                "outside_changed_pixels"),
        },
    })
    check_cancelled(cancel_check)
    if ink_fidelity and not native_wash_style and not git_legacy_wash_style:
        from core.ink_fidelity import restore_ink_fidelity
        check_cancelled(cancel_check)
        result_bgr = restore_ink_fidelity(image_bgr, result_bgr)
        if filter_base_bgr is not None and return_filter_base:
            filter_base_bgr = restore_ink_fidelity(image_bgr, filter_base_bgr)
        diagnostics["ink_fidelity_applied"] = True
    else:
        diagnostics["ink_fidelity_applied"] = False
    # Ink-wash is now the primary render path for the three pale manga presets.
    # The original mc-v2 mode ('none') never passes through this renderer.
    ink_style = str(getattr(grade_preset, "key", style_key or "none"))
    if ink_style in {"light2", "light2_soft", "light3"}:
        from core.inkwash_render import render_inkwash
        ink_strength = float(np.clip(
            float((pastel_tuning or {}).get("inkwash_strength", 78)) / 100.0,
            0.0, 1.0))
        check_cancelled(cancel_check)
        ink_light3_intensity = float(np.clip(
            float((pastel_tuning or {}).get("light3_intensity", 100)) / 100.0,
            0.0, 2.0))
        if ink_strength > 0:
            # Explicit semantic/character recolours have higher priority than
            # paper-wash aesthetics. Preserve these exact edited pixels.
            preserve_mask = None
            if semantic_mask_enabled and semantic_masks is not None and semantic_mask_diagnostics.get("applied"):
                preserve_mask = np.zeros(result_bgr.shape[:2], dtype=np.uint8)
                for name, attr in (("hair", "hair_mask"), ("iris", "iris_mask"),
                                   ("garment", "cloth_mask"),
                                   ("background", "background_mask")):
                    layer = dict(semantic_mask_controls.get(name) or {})
                    if not layer.get("enabled", False):
                        continue
                    semantic_mask = getattr(semantic_masks, attr, None)
                    if semantic_mask is None:
                        continue
                    if semantic_mask.shape[:2] != preserve_mask.shape:
                        semantic_mask = cv2.resize(
                            semantic_mask.astype(np.uint8),
                            (preserve_mask.shape[1], preserve_mask.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
                    preserve_mask[np.asarray(semantic_mask) > 0] = 255
            src_ink = image_bgr
            if src_ink.shape[:2] != result_bgr.shape[:2]:
                src_ink = cv2.resize(src_ink, (result_bgr.shape[1], result_bgr.shape[0]),
                                     interpolation=cv2.INTER_CUBIC)
            # Work with the native, ungraded generator result PLUS any explicit
            # later user recolours/filters. Previous code used the early
            # identity_base_result, silently discarding manual edits here.
            inkwash_base_bgr = result_bgr
            if inkwash_base_bgr.shape[:2] != result_bgr.shape[:2]:
                inkwash_base_bgr = cv2.resize(
                    inkwash_base_bgr, (result_bgr.shape[1], result_bgr.shape[0]),
                    interpolation=cv2.INTER_CUBIC)
            result_bgr = render_inkwash(src_ink, inkwash_base_bgr,
                                        preset=ink_style, strength=ink_strength,
                                        preserve_mask=preserve_mask,
                                        light3_intensity=ink_light3_intensity)
            if return_filter_base and filter_base_bgr is not None:
                filter_base_bgr = render_inkwash(
                    src_ink, filter_base_bgr, preset=ink_style, strength=ink_strength,
                    preserve_mask=preserve_mask,
                    light3_intensity=ink_light3_intensity)
            diagnostics["inkwash_base"] = "native_generator_plus_explicit_edits"
        diagnostics["inkwash_strength"] = round(ink_strength, 3)
        if ink_style == "light3":
            diagnostics["light3_intensity_effective"] = round(ink_light3_intensity, 3)
        diagnostics["inkwash_render_applied"] = ink_strength > 0
    else:
        diagnostics["inkwash_render_applied"] = False
    hint_manager.last_diagnostics = diagnostics
    hint_manager.last_page_context = page_context
    if return_filter_base:
        return result_bgr, filter_base_bgr
    return result_bgr


# ── Built-in descriptor injection ────────────────────────────────────────────
# When a built-in style key is selected (shonen, seinen, …), we wrap it in
# a synthetic StyleProfile that carries the matching StyleDescriptor so the
# GuidedColorist gets layered hints even without a user-provided reference.
#
# Cached in a module-level dict so each style is only wrapped once.
_builtin_profile_cache: dict = {}


def _builtin_style_profile(style_key: str):
    """Return a synthetic StyleProfile carrying the built-in StyleDescriptor
    for `style_key`, or None if the key has no descriptor (e.g. 'none')."""
    from core.builtin_descriptors import BUILTIN_DESCRIPTORS
    if style_key not in BUILTIN_DESCRIPTORS:
        return None
    if style_key in _builtin_profile_cache:
        return _builtin_profile_cache[style_key]
    desc = BUILTIN_DESCRIPTORS[style_key]
    from core.style_engine import StyleProfile, _descriptor_to_profile
    profile = _descriptor_to_profile(desc)
    profile._descriptor = desc
    _builtin_profile_cache[style_key] = profile
    return profile
