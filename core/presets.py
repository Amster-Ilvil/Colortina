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
        label="Draft",
        description="Fast preview — lower internal resolution, JPEG output.",
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
        key="standard",
        label="Standard",
        description="Balanced — 768 internal, full post-processing.",
        model_size=768,
        tiled_inference=False,
        per_panel=True,
        diffusion_step_mult=1.0,
        use_upscale=False,
        output_format="jpg",
        jpeg_quality=95,
        seconds_per_page_estimate=5.0,
    ),
    "ultra": QualityPreset(
        key="ultra",
        label="Ultra",
        description="Tiled native resolution + per-panel + 4x upscale + refine pass.",
        model_size=768,
        tiled_inference=True,
        tile_size=768,
        tile_overlap=128,
        per_panel=True,
        diffusion_step_mult=1.3,
        use_upscale=True,
        # JPEG q95 4:4:4 — lossless PNG of continuous-tone colorized art is
        # 100-400 MB/page after 4x upscale and buys nothing visually
        output_format="jpg",
        jpeg_quality=95,
        refine_pass=True,
        seconds_per_page_estimate=30.0,
    ),
}


def get_quality(key: Optional[str]) -> QualityPreset:
    """Return a quality preset by key, falling back to standard."""
    if not key:
        return QUALITY_PRESETS["standard"]
    return QUALITY_PRESETS.get(key.lower(), QUALITY_PRESETS["standard"])


def all_styles_json() -> list[dict]:
    """Return all style presets in a UI-friendly form."""
    return [asdict(p) for p in STYLE_PRESETS.values()]


def all_qualities_json() -> list[dict]:
    """Return all quality presets in a UI-friendly form."""
    return [asdict(p) for p in QUALITY_PRESETS.values()]
