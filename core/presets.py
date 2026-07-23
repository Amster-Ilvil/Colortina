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
        label="MC v2 (\u539f\u59cb mc-v2)",
        description=("No style processing at all - skips guided/CLIP hint "
                     "generation and post-processing color grading, "
                     "returning the original manga-colorization-v2 output "
                     "unmodified."),
        denoise_sigma=15,
        diffusion_steps=16,
    ),
    "light": StylePreset(
        key="light",
        label="\u6de1\u5f69\u6c34\u58a8 (Light Wash)",
        description=("Pale watercolor / ink-wash rendering: chroma pulled "
                     "well below mc-v2's saturated defaults, lightness "
                     "lifted, edges soft - colors read as translucent "
                     "washes over the ink lines rather than solid fills."),
        saturation_boost=0.42,
        l_blend_alpha=0.0,
        l_gamma=0.88,
        guided_filter_radius=4,
        guided_filter_eps=0.03,
        cel_flatten=0.0,
        neutral_fade_floor=0.12,
        denoise_sigma=20,
        diffusion_steps=16,
    ),
    "light2": StylePreset(
        key="light2",
        label="淡彩水墨2（参考风格）",
        description=("Bundled reference style extracted from 10 color pages. "
                     "Keeps a soft watercolor palette with stronger visible color "
                     "than black-and-white pastel."),
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
    "monochrome": StylePreset(
        key="monochrome",
        label="黑白淡彩（统一）",
        description=("Unified monochrome pastel mode. Character colour, scene tint, "
                     "highlight retention, warmth and flattening can be tuned in detail "
                     "without switching between separate people/full-page presets."),
        saturation_boost=1.05,
        l_blend_alpha=0.0,
        l_gamma=1.0,
        cel_flatten=0.02,
        neutral_fade_floor=0.13,
        semantic_mode="page_pastel",
        person_chroma_scale=0.36,
        skin_chroma_scale=0.60,
        hair_chroma_scale=0.86,
        eye_chroma_scale=1.00,
        clothing_chroma_scale=0.64,
        environment_chroma_scale=0.08,
        unknown_chroma_scale=0.04,
        skin_neutralize=0.62,
        skin_target_a=2.0,
        skin_target_b=8.7,
        denoise_sigma=12,
        diffusion_steps=16,
    ),
    "monochrome_people": StylePreset(
        key="monochrome_people",
        label="黑白淡彩·人物 (人物轻着色)",
        description=("Preserve original manga luminance; keep visibly coloured skin, "
                     "hair, eyes and clothing while the environment stays almost monochrome."),
        saturation_boost=1.06,
        l_blend_alpha=0.0,
        l_gamma=1.0,
        cel_flatten=0.0,
        neutral_fade_floor=0.10,
        semantic_mode="people_pastel",
        person_chroma_scale=0.34,
        skin_chroma_scale=0.58,
        hair_chroma_scale=0.84,
        eye_chroma_scale=1.00,
        clothing_chroma_scale=0.62,
        environment_chroma_scale=0.0,
        unknown_chroma_scale=0.0,
        skin_neutralize=0.62,
        skin_target_a=2.0,
        skin_target_b=9.0,
        force_environment_grayscale=True,
        denoise_sigma=12,
        diffusion_steps=16,
    ),
    "monochrome_page": StylePreset(
        key="monochrome_page",
        label="黑白淡彩·全页 (环境微着色)",
        description=("Characters receive clear but restrained colour and the environment keeps "
                     "a faint atmosphere over the black-and-white page."),
        saturation_boost=1.04,
        l_blend_alpha=0.0,
        l_gamma=1.0,
        cel_flatten=0.0,
        neutral_fade_floor=0.11,
        semantic_mode="page_pastel",
        person_chroma_scale=0.38,
        skin_chroma_scale=0.62,
        hair_chroma_scale=0.88,
        eye_chroma_scale=1.00,
        clothing_chroma_scale=0.66,
        environment_chroma_scale=0.13,
        unknown_chroma_scale=0.08,
        skin_neutralize=0.62,
        skin_target_a=2.0,
        skin_target_b=8.5,
        denoise_sigma=12,
        diffusion_steps=16,
    ),
}


def get_style(key: Optional[str]) -> StylePreset:
    """Return a style preset by key, falling back to neutral."""
    if not key:
        return STYLE_PRESETS["none"]
    return STYLE_PRESETS.get(key.lower(), STYLE_PRESETS["none"])


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
