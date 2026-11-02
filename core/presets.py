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
    # Ink-wash stabilization for manga pages with heavy screentones / dark fills.
    preserve_screentones: float = 0.0
    background_neutral_guard: float = 0.0
    background_chroma_cap: float = 0.0
    hue_stabilize: float = 0.0
    panel_aware_environment: bool = False
    # Additional cast suppression for inkwash backgrounds.
    green_cast_suppression: float = 0.0
    cyan_cast_suppression: float = 0.0
    background_target_a: float = 0.0
    background_target_b: float = 0.0
    low_detail_bg_neutralize: float = 0.0
    low_detail_bg_chroma_cap: float = 0.0


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
        preserve_screentones=0.94,
        background_neutral_guard=0.90,
        background_chroma_cap=5.2,
        hue_stabilize=0.54,
        panel_aware_environment=True,
        semantic_mode="inkwash",
        person_chroma_scale=0.78,
        skin_chroma_scale=0.90,
        hair_chroma_scale=0.90,
        eye_chroma_scale=0.94,
        clothing_chroma_scale=0.84,
        environment_chroma_scale=0.06,
        unknown_chroma_scale=0.03,
        green_cast_suppression=0.92,
        cyan_cast_suppression=0.82,
        background_target_a=1.0,
        background_target_b=1.9,
        low_detail_bg_neutralize=0.74,
        low_detail_bg_chroma_cap=4.2,
    ),
    "light2_soft": StylePreset(
        key="light2_soft",
        label="淡彩水墨（淡）",
        description=("Intermediate pale watercolor / ink-wash preset between "
                     "淡彩水墨 and 淡彩水墨（极淡）. It keeps visible, coherent "
                     "colour across the page while using a lighter, quieter wash."),
        saturation_boost=0.52,
        white_threshold=220,
        black_threshold=27,
        neutral_transition=29,
        l_blend_alpha=0.04,
        l_gamma=0.96,
        guided_filter_radius=4,
        guided_filter_eps=0.029,
        chroma_warm_shift=0.36,
        cel_flatten=0.020,
        neutral_fade_floor=0.24,
        denoise_sigma=18,
        diffusion_steps=16,
        preserve_screentones=0.90,
        background_neutral_guard=0.80,
        background_chroma_cap=6.6,
        hue_stabilize=0.48,
        panel_aware_environment=True,
        semantic_mode="inkwash",
        person_chroma_scale=0.84,
        skin_chroma_scale=0.92,
        hair_chroma_scale=0.92,
        eye_chroma_scale=0.95,
        clothing_chroma_scale=0.86,
        environment_chroma_scale=0.08,
        unknown_chroma_scale=0.04,
        green_cast_suppression=0.86,
        cyan_cast_suppression=0.76,
        background_target_a=1.2,
        background_target_b=2.2,
        low_detail_bg_neutralize=0.64,
        low_detail_bg_chroma_cap=5.0,
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
        preserve_screentones=0.84,
        background_neutral_guard=0.70,
        background_chroma_cap=7.8,
        hue_stabilize=0.42,
        panel_aware_environment=True,
        semantic_mode="inkwash",
        person_chroma_scale=0.88,
        skin_chroma_scale=0.94,
        hair_chroma_scale=0.94,
        eye_chroma_scale=0.97,
        clothing_chroma_scale=0.88,
        environment_chroma_scale=0.10,
        unknown_chroma_scale=0.05,
        green_cast_suppression=0.80,
        cyan_cast_suppression=0.68,
        background_target_a=1.4,
        background_target_b=2.5,
        low_detail_bg_neutralize=0.56,
        low_detail_bg_chroma_cap=5.8,
    ),

    # Original GitHub light2/light3 style presets, isolated from the current
    # Generator-first ink-wash renderer. See GIT_LEGACY_STYLE_KEYS below.
    "light2_git_legacy": StylePreset(
        key="light2_git_legacy",
        label="淡彩水墨（Git 旧版）",
        description="Git main 原版 light2 参数，使用旧 apply_style_grade 调色路径。",
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
    "light3_git_legacy": StylePreset(
        key="light3_git_legacy",
        label="淡彩水墨（极淡·Git 旧版）",
        description="Git main 原版 light3 参数，使用旧 apply_style_grade 调色路径。",
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
}

# These keys always run the original Git grading code. Never remap legacy
# keys to the new light2/light3 independent renderer.
GIT_LEGACY_STYLE_KEYS = frozenset({"light2_git_legacy", "light3_git_legacy"})


def get_style(key: Optional[str]) -> StylePreset:
    """Return a style preset by key, falling back to neutral.

    Legacy compatibility: removed style keys resolve to the built-in
    淡彩水墨 style (light2).
    """
    if not key:
        return STYLE_PRESETS["none"]
    key_norm = key.lower()
    if key_norm in {"light", "light2_illustrative"}:
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
    "standard": QualityPreset(
        key="standard", label="精细 · 768",
        description="Higher internal resolution for intricate manga line work.",
        model_size=768, tiled_inference=False, per_panel=False,
        output_format="png", seconds_per_page_estimate=6.0,
    ),
    "detail": QualityPreset(
        key="detail", label="高清 · 960（实验）",
        description="More internal detail at higher CPU/RAM cost.",
        model_size=960, tiled_inference=False, per_panel=False,
        output_format="png", seconds_per_page_estimate=12.0,
    ),
}


def get_quality(key: Optional[str]) -> QualityPreset:
    """Return an explicitly selected model input resolution (32-aligned)."""
    return QUALITY_PRESETS.get(str(key or "draft"), QUALITY_PRESETS["draft"])


def all_styles_json() -> list[dict]:
    """Return all style presets in a UI-friendly form."""
    return [asdict(p) for p in STYLE_PRESETS.values()]


def all_qualities_json() -> list[dict]:
    """Return all quality presets in a UI-friendly form."""
    return [asdict(p) for p in QUALITY_PRESETS.values()]
