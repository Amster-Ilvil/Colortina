"""Colortina — unified Auto pipeline.

    image
      |
      v
  auto hints (GuidedColorist: segment -> CLIP label ->
              StyleDescriptor -> TieredColor [hi/mid/shadow per region]
              -> layered HintPoints placed on tier-matched pixels)
      |
      v
  HintManager.merge(manual hints)
      |
      v
  mc-v2 colorize (with merged hint_points, QualityPreset knobs)
      |
      v
  style grading (core.style_post, StylePreset knobs)
      |
      v
  optional upscale (Ultra quality)
      |
      v
  output image

The key improvement over the old pipeline: hints are now LAYERED.
Each semantic region (hair, skin, sky…) generates 3 hint colors
(highlight / mid-tone / shadow) derived from the StyleDescriptor's
color language, placed on pixels whose luminance matches that tier.
mc-v2 sees a realistic hint gradient and can interpolate naturally,
making style differences visible in the actual AI colorization output
rather than only in post-processing.
"""

from __future__ import annotations

import numpy as np

from config import Config
from core.hint_manager import HintManager
from core.lineart_fill import label_regions
from core.ml_colorizer import MangaColorizer
from core.presets import get_quality, get_style
from core.style_post import apply_style_grade

_colorizer: MangaColorizer | None = None
_guided = None
_job_guided_cache: dict = {}


def _guided_cache_key(style_profile, character_memories):
    cm_key = tuple(sorted((k, id(v)) for k, v in (character_memories or {}).items()))
    return (id(style_profile), cm_key)


def get_colorizer(cfg: Config = Config) -> MangaColorizer:
    global _colorizer
    if _colorizer is None:
        from core.model_downloader import ensure_models_downloaded
        ensure_models_downloaded(cfg.WEIGHTS_DIR, callback=print)
        _colorizer = MangaColorizer(
            device=cfg.ML_DEVICE,
            generator_path=cfg.GENERATOR_WEIGHTS_PATH,
            extractor_path=cfg.EXTRACTOR_WEIGHTS_PATH,
            denoiser_weights_dir=cfg.DENOISER_WEIGHTS_DIR,
        )
        print(f"[pipeline] mc-v2 loaded on device: {_colorizer.device_name}")
    return _colorizer


def get_guided_colorist(cfg: Config = Config, style_profile=None,
                        character_memories: dict | None = None):
    """Return a GuidedColorist for this job.

    If style_profile has an attached StyleDescriptor (v2 extraction or
    built-in descriptor injection), it drives layered hint generation.
    """
    global _guided
    key = _guided_cache_key(style_profile, character_memories)

    if style_profile is None and not character_memories:
        # Pure singleton: no style, no character memory
        if _guided is None:
            from core.guided_colorist import GuidedColorist
            _guided = GuidedColorist(cfg)
        return _guided

    guided = _job_guided_cache.get(key)
    if guided is None:
        from core.guided_colorist import GuidedColorist
        descriptor = getattr(style_profile, "_descriptor", None)
        guided = GuidedColorist(
            cfg,
            style_descriptor=descriptor,
            style_profile=style_profile,
            character_memories=character_memories)
        _job_guided_cache[key] = guided
    return guided


def build_auto_hints(image_bgr: np.ndarray, cfg: Config = Config,
                     style_profile=None, character_memories: dict | None = None) -> list:
    if not cfg.USE_GUIDED_HINTS:
        return []
    guided = get_guided_colorist(cfg, style_profile=style_profile,
                                 character_memories=character_memories)
    if not guided.available:
        print("[pipeline] CLIP classifier unavailable — falling back to "
              "plain mc-v2 auto-colorize (no guided hints)")
        return []
    return guided.hints_for_page(image_bgr)


def colorize_page(image_bgr: np.ndarray,
                   hint_manager: HintManager | None = None,
                   cfg: Config = Config,
                   regenerate_auto: bool = True,
                   style_key: str | None = None,
                   quality_key: str | None = None,
                   style_profile=None,
                   character_memories: dict | None = None) -> np.ndarray:
    """Run the full Auto pipeline on one page.

    `style_profile` may be a legacy StyleProfile OR a new v2 profile that
    carries a StyleDescriptor — the pipeline handles both.  When a
    StyleDescriptor is present, GuidedColorist generates layered
    (hi/mid/shadow) hints per region, making the style actually visible
    in the model's colorization output rather than only in post-processing.
    """
    if hint_manager is None:
        hint_manager = HintManager()

    # "none" — raw mc-v2, no guided hints, no grading
    is_none_style = style_profile is None and (style_key or "").lower() == "none"

    # If no custom profile is given, inject the built-in StyleDescriptor for
    # the chosen preset key so hint generation uses layered colour language
    # even for built-in presets (this is the core fix: style now affects the
    # model's colorization, not only post-processing).
    effective_profile = style_profile
    if effective_profile is None and not is_none_style:
        effective_profile = _builtin_style_profile(style_key or "neutral")

    style   = (effective_profile.to_style_preset() if effective_profile is not None
               else get_style(style_key))
    quality = get_quality(quality_key)

    if regenerate_auto or not hint_manager.auto_hints:
        if is_none_style:
            auto_points = []
        else:
            auto_points = build_auto_hints(
                image_bgr, cfg,
                style_profile=effective_profile,
                character_memories=character_memories)
        hint_manager.set_auto_hints(auto_points)

    merged_points = hint_manager.merge()

    label_map = None
    if merged_points:
        label_map = label_regions(image_bgr)

    colorizer = get_colorizer(cfg)
    raw_result = colorizer.colorize(
        image_bgr,
        size=quality.model_size,
        denoise_sigma=style.denoise_sigma,
        tiled=quality.tiled_inference,
        tile_size=quality.tile_size,
        tile_overlap=quality.tile_overlap,
        per_panel=quality.per_panel,
        hint_points=merged_points or None,
        label_map=label_map,
    )

    if is_none_style:
        result_bgr = raw_result
    else:
        # Prefer descriptor-derived grade preset when available
        desc = getattr(effective_profile, "_descriptor", None) if effective_profile else None
        grade_preset = desc.to_style_preset() if desc is not None else style
        result_bgr = apply_style_grade(raw_result, image_bgr, grade_preset)

    if quality.use_upscale:
        from core.upscaler import upscale
        result_bgr = upscale(result_bgr, scale=4,
                             weights_path=getattr(cfg, "ESRGAN_MODEL_PATH", None))

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
