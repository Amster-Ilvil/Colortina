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

import numpy as np

from config import Config
from core.hint_manager import HintManager
from core.lineart_fill import label_regions
from core.ml_colorizer import MangaColorizer
from core.presets import get_quality, get_style
from core.style_post import apply_style_grade
from core.image_filter import apply_image_filter

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
    """Cheap, stable fingerprint for an immutable source page."""
    h, w = image_bgr.shape[:2]
    sy = max(1, h // 24)
    sx = max(1, w // 24)
    sample = np.ascontiguousarray(image_bgr[::sy, ::sx][:24, :24])
    return (id(image_bgr), h, w, str(image_bgr.dtype), int(zlib.adler32(sample.tobytes())))


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
                   model_specs: list, label_map) -> tuple:
    return (
        _image_fingerprint(image_bgr),
        int(quality.model_size), bool(quality.tiled_inference),
        int(quality.tile_size), int(quality.tile_overlap), bool(quality.per_panel),
        int(denoise_sigma),
        _hint_signature(model_specs), _label_map_signature(label_map),
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
        )
        print(f"[pipeline] mc-v2 loaded on device: {_colorizer.device_name}")
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
    """Always force the fastest generation mode."""
    effective_quality = replace(quality, key='draft', label='Fast', model_size=576,
                                tiled_inference=False, per_panel=False,
                                tile_size=576, tile_overlap=64)
    effective_denoise = 15
    optimizations = ['force_fastest_quality', 'disable_guided_analysis',
                     'disable_per_panel', 'disable_tiled_inference']
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
                   protect_text: bool = True):
    """Run the full Auto pipeline on one page.

    """
    if hint_manager is None:
        hint_manager = HintManager()
    style_strength = float(np.clip(style_strength, 0.0, 1.0))
    reference_strength = float(np.clip(reference_strength, 0.0, 1.0))
    manual_strength = float(np.clip(manual_strength, 0.0, 1.0))
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
    quality = get_quality(quality_key)

    needs_guided = _need_guided_context(
        style, effective_profile=effective_profile,
        character_memories=character_memories, character_library=character_library,
        scene_palette=scene_palette, reference_strength=reference_strength,
        has_manual_region=has_manual_region)

    if needs_guided:
        page_context = build_page_context(
            image_bgr, cfg=cfg, style_profile=effective_profile,
            character_memories=character_memories, character_library=character_library,
            scene_palette=scene_palette, style_strength=style_strength,
            reference_strength=reference_strength, forced_matches=forced_matches)
        if regenerate_auto or (not hint_manager.auto_hints and not preserve_empty_auto_hints):
            hint_manager.set_auto_hints(build_auto_hints(
                image_bgr, cfg=cfg, style_profile=effective_profile,
                character_memories=character_memories, character_library=character_library,
                scene_palette=scene_palette, style_strength=style_strength,
                reference_strength=reference_strength, forced_matches=forced_matches))
    else:
        page_context = _post_only_context(hint_manager, {
            "plain_auto": bool(is_none_style),
            "post_only_style": None if is_none_style else (style_key or "custom"),
            "fast_mode": True,
        })
        if regenerate_auto or (not hint_manager.auto_hints and not preserve_empty_auto_hints):
            hint_manager.set_auto_hints([])

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
        region_map = hint_manager.region_map
        if region_map is None:
            hint_manager.bind_source_image(image_bgr)
            region_map = hint_manager.region_map
        label_map = region_map if region_map is not None else label_regions(image_bgr)

    effective_quality, effective_denoise, job_optimizations = _effective_job_quality(
        style, quality, needs_guided=needs_guided, model_specs=model_specs)

    model_started = time.perf_counter()
    raw_key = _raw_cache_key(
        image_bgr, effective_quality, effective_denoise, model_specs, label_map)
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
        )
        _raw_cache_put(raw_key, raw_result)
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
                )
                model_specs = retry_specs
                retry_key = _raw_cache_key(
                    image_bgr, effective_quality, effective_denoise, model_specs, label_map)
                _raw_cache_put(retry_key, raw_result)
                artifact_report = detect_hint_blobs(raw_result, model_specs)
                retried = True
    else:
        from types import SimpleNamespace
        artifact_report = SimpleNamespace(score=0.0, suspicious=False, should_retry=False)

    grade_preset = style
    grade_seconds = 0.0
    if is_none_style:
        result_bgr = raw_result
    else:
        grade_preset = style
        grade_started = time.perf_counter()
        graded = apply_style_grade(raw_result, image_bgr, grade_preset, context=page_context)
        grade_seconds = time.perf_counter() - grade_started
        if style_strength >= 0.999:
            result_bgr = graded
        elif style_strength <= 0.001:
            result_bgr = raw_result
        else:
            result_bgr = np.clip(
                raw_result.astype(np.float32) * (1.0 - style_strength) +
                graded.astype(np.float32) * style_strength,
                0, 255).astype(np.uint8)

    # Hints guide the model; this deterministic lock guarantees identity
    # colours across pages for matched hair, skin, eyes and clothing regions.
    if character_library is not None and reference_strength > 0.0 and page_context is not None:
        from core.character_consistency import apply_character_palette_lock
        result_bgr = apply_character_palette_lock(
            result_bgr, image_bgr, character_library,
            strength=reference_strength,
            assignments=page_context.identity_assignments,
            segmentation=page_context.segmentation)

    if (not is_none_style and
            bool(getattr(grade_preset, "force_environment_grayscale", False))):
        from core.style_post import enforce_character_only_color
        result_bgr = enforce_character_only_color(
            result_bgr, image_bgr, context=page_context)

    if local_manual_specs:
        from core.local_brush import apply_local_brush_recolor
        region_map = hint_manager.region_map
        h, w = result_bgr.shape[:2]
        for spec in local_manual_specs:
            ix = min(w - 1, max(0, int(round(spec.x_norm * (w - 1)))))
            iy = min(h - 1, max(0, int(round(spec.y_norm * (h - 1)))))
            radius_px = max(1, int(round(spec.radius_norm * w)))
            result_bgr, _mask = apply_local_brush_recolor(
                image_bgr, result_bgr, ix, iy, radius_px, spec.rgb,
                opacity=min(0.9, 0.9 * spec.effective_strength),
                region_map=region_map,
                gap_close=getattr(region_map, "gap_close", 4) if region_map is not None else 4)

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

    if protect_text:
        # 文字气泡保护：把原始黑白页合成回检测到的文字像素，
        # 对话框保持纸白、字保持墨黑。模型缺失时自动跳过。
        from core.text_guard import protect_text_regions
        result_bgr = protect_text_regions(result_bgr, image_bgr)

    filter_base_bgr = result_bgr.copy()
    result_bgr = apply_image_filter(
        result_bgr, filter_tuning,
        style_strength=style_strength,
        is_styled=not is_none_style,
        source_bw_bgr=image_bgr)

    if quality.use_upscale:
        from core.upscaler import upscale
        result_bgr = upscale(result_bgr, scale=4,
                             weights_path=getattr(cfg, "ESRGAN_MODEL_PATH", None))

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
        "job_optimizations": list(job_optimizations),
        "raw_cache_hit": bool(raw_cache_hit),
        "model_seconds": round(float(model_seconds), 3),
        "grade_seconds": round(float(grade_seconds), 3),
        "identity_drift_alerts": int(identity_drift_alerts),
        "identity_drift_max_delta_e": round(float(identity_drift_max), 1),
    })
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
