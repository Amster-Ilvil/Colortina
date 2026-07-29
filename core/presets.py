"""Style and quality presets.

Style presets bundle post-processing tunables (saturation curve, neutral
preservation, guided-filter sharpness) that match a target aesthetic.

Quality presets bundle resolution / tiling / output-format flags that
trade time for fidelity.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


# ── Style presets ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StylePreset:
    """Bundle of post-processing knobs for a target aesthetic."""

    key: str
    label: str
    description: str
    # Saturation: peak multiplier for low-chroma pixels (vibrance).
    saturation_boost: float = 1.4
    # Neutral preservation thresholds.
    white_threshold: int = 220
    black_threshold: int = 30
    neutral_transition: int = 30
    # Chroma-aware L-blend: 0 = full original L (current behavior),
    # 1 = full colorized L. Used inside the line-mask falloff.
    l_blend_alpha: float = 0.0
    # Guided filter feel.
    guided_filter_radius: int = 2
    guided_filter_eps: float = 0.01
    # Optional global tone curve applied as gain on A/B chroma after sat boost.
    chroma_warm_shift: float = 0.0   # +shifts B channel (yellow), -shifts blue
    chroma_red_shift: float = 0.0    # +shifts A channel (red), -shifts green
    # Optional global luminance gamma (applied to L channel only outside ink).
    l_gamma: float = 1.0
    # Hint to the colorizer (denoise sigma / inference steps).
    denoise_sigma: int = 15
    diffusion_steps: int = 16
    # Cel flattening: snap each lineart-bounded region's chroma toward the
    # region mean (0 = off, 1 = fully flat fills).  This is what makes a page
    # read as hand-colored manhwa instead of one gradient wash.
    cel_flatten: float = 0.0
    # Floor for the soft neutral fade: bright surfaces (bedding, highlights,
    # light hair) always keep at least this fraction of their chroma.  Only
    # the hard masks (bubbles / gutters / ink) may force true zero.
    neutral_fade_floor: float = 0.35
    # Semantic chroma policy used by monochrome pastel modes.
    semantic_mode: str = "all"
    person_chroma_scale: float = 1.0
    skin_chroma_scale: float = 1.0
    hair_chroma_scale: float = 1.0
    eye_chroma_scale: float = 1.0
    clothing_chroma_scale: float = 1.0
    environment_chroma_scale: float = 1.0
    unknown_chroma_scale: float = 1.0
    # Monochrome-pastel skin hue guard.  Pull low-chroma skin toward a warm
    # ivory base while preserving high-chroma blush/lips.
    skin_neutralize: float = 0.0
    skin_target_a: float = 1.0   # centred OpenCV LAB a (0 = neutral)
    skin_target_b: float = 9.0   # centred OpenCV LAB b (+ = warm/yellow)
    # When true, all pixels outside the detected person mask are restored to
    # the original page grayscale after grading. Used by character-only pastel.
    force_environment_grayscale: bool = False


STYLE_PRESETS: dict[str, StylePreset] = {
    "none": StylePreset(
        key="none",
        label="MC v2",
        description=("No style processing at all - skips guided/CLIP hint "
                     "generation and post-processing color grading, "
                     "returning the original manga-colorization-v2 output "
                     "unmodified."),
        denoise_sigma=15,
        diffusion_steps=16,
    ),
    "light3": StylePreset(
        key="light3",
        label="淡彩水墨（极淡）",
        description=("Very pale variant of 淡彩水墨: keeps the same soft "
                     "watercolor / ink-wash feel, but pushes the "
                     "palette noticeably lighter and lower-saturation for an "
                     "even more delicate wash."),
        saturation_boost=0.26,
        l_blend_alpha=0.0,
        l_gamma=0.90,
        guided_filter_radius=4,
        guided_filter_eps=0.032,
        cel_flatten=0.0,
        neutral_fade_floor=0.09,
        denoise_sigma=20,
        diffusion_steps=16,
    ),
    "light2": StylePreset(
        key="light2",
        label="淡彩水墨",
        description=("Soft watercolor preset with clearer visible color than the "
                     "extremely pale variant, while retaining gentle warmth, "
                     "light gradients and restrained saturation."),
        saturation_boost=0.85,
        white_threshold=221,
        black_threshold=24,
        neutral_transition=28,
        l_blend_alpha=0.08,
        l_gamma=1.02,
        guided_filter_radius=3,
        guided_filter_eps=0.025,
        chroma_warm_shift=0.76,
        cel_flatten=0.047,
        neutral_fade_floor=0.45,
        denoise_sigma=15,
        diffusion_steps=16,
    ),

}


def get_style(key: Optional[str]) -> StylePreset:
    """Return a style preset by key, falling back to neutral.

    Legacy compatibility: the removed old "light" preset now resolves to
    the built-in 淡彩水墨 style (light2).
    """
    if not key:
        return STYLE_PRESETS["none"]
    key_norm = key.lower()
    if key_norm == "light":
        key_norm = "light2"
    return STYLE_PRESETS.get(key_norm, STYLE_PRESETS["none"])


# ── Quality presets ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class QualityPreset:
    """Bundle of resolution / tiling / output flags trading speed for fidelity."""

    key: str
    label: str
    description: str
    # Internal model resolution (mc-v2 working size, must be /32).
    model_size: int = 768
    # Tiled colorization at native page resolution.
    tiled_inference: bool = False
    tile_size: int = 768
    tile_overlap: int = 96
    # Per-panel colorization (uses panel_detector).
    per_panel: bool = False
    # Diffusion steps multiplier (applied to style preset's base).
    diffusion_step_mult: float = 1.0
    # Real-ESRGAN upscale.
    use_upscale: bool = False
    # Output format: "jpg" or "png".
    output_format: str = "jpg"
    jpeg_quality: int = 92
    # Multi-pass refinement (reference mode only).
    refine_pass: bool = False
    # Estimated seconds per page (rough hint for UI).
    seconds_per_page_estimate: float = 4.0


QUALITY_PRESETS: dict[str, QualityPreset] = {
    "draft": QualityPreset(
        key="draft",
        label="Fast",
        description="Fastest mode — low internal resolution, no tiling, no per-panel pass.",
        model_size=576,
        tiled_inference=False,
        per_panel=False,
        diffusion_step_mult=0.7,
        use_upscale=False,
        output_format="jpg",
        jpeg_quality=85,
        seconds_per_page_estimate=2.0,
    ),
}


def get_quality(key: Optional[str]) -> QualityPreset:
    """Return the only supported quality preset: the fastest draft mode."""
    return QUALITY_PRESETS["draft"]


def all_styles_json() -> list[dict]:
    """Return all style presets in a UI-friendly form."""
    return [asdict(p) for p in STYLE_PRESETS.values()]


def all_qualities_json() -> list[dict]:
    """Return all quality presets in a UI-friendly form."""
    return [asdict(p) for p in QUALITY_PRESETS.values()]
