"""Built-in StyleDescriptors for every named preset.

These are the same presets that exist in core/presets.py (Shonen, Seinen,
Webtoon, etc.), but now expressed as StyleDescriptors so they drive the
Hint Generator and actually change what mc-v2 produces — not just the
post-processing layer.

Convention:
  - global_warm_cool:  + = warm overall, - = cool overall
  - warm_bias:  per-region warming (LAB b shift on mid-tones)
  - shadow_bias: LAB b shift on shadows relative to mid-tone
  - shadow_hue_rotate: LAB a shift on shadows (+ = more red/magenta)
  - highlight_bias: LAB b shift on highlights
  - cel_flatten: 0 = full gradient, 0.8+ = flat cel-fills
"""

from core.style_descriptor import StyleDescriptor, RegionDescriptor

# ── Shonen Vibrant ────────────────────────────────────────────────────────────
# Punchy primaries, warm skin, rich shadow on hair, high-energy look.
SHONEN = StyleDescriptor(
    name="Shonen Vibrant",
    description="Jump-style: punchy primaries, warm skin, deep blue-shadow hair",
    source="builtin",
    global_warm_cool=3.0,
    global_saturation=1.6,
    global_contrast=1.4,
    global_shadow_lift=0.0,
    cel_flatten=0.1,
    hair=RegionDescriptor(
        warm_bias=1.0, shadow_bias=-8.0, shadow_hue_rotate=-4.0,
        shadow_desat=0.1, highlight_bias=5.0, saturation_scale=1.5,
        gradient=0.7, contrast=1.4, hint_layers=3),
    skin=RegionDescriptor(
        warm_bias=5.0, shadow_bias=2.0, shadow_hue_rotate=3.0,
        shadow_desat=0.15, highlight_bias=1.0, saturation_scale=1.0,
        gradient=0.6, contrast=1.2, hint_layers=3),
    sky=RegionDescriptor(
        warm_bias=-6.0, gradient=0.8, saturation_scale=1.2,
        highlight_bias=4.0, hint_layers=3),
    foliage=RegionDescriptor(warm_bias=3.0, saturation_scale=1.1, hint_layers=2),
    clothing_primary=RegionDescriptor(
        warm_bias=2.0, shadow_bias=-4.0, saturation_scale=1.4, hint_layers=3),
    clothing_secondary=RegionDescriptor(
        warm_bias=-2.0, saturation_scale=1.2, hint_layers=3),
    clothing_accent=RegionDescriptor(
        warm_bias=5.0, saturation_scale=1.5, hint_layers=2),
    background=RegionDescriptor(warm_bias=1.0, saturation_scale=0.75, hint_layers=1),
    saturation=0.92, contrast=0.85, temperature="warm",
    shadow_strength=0.65, gradient=0.6,
)

# ── Seinen Muted ──────────────────────────────────────────────────────────────
# Cool greys, restrained chroma, cinematic shadow structure.
SEINEN = StyleDescriptor(
    name="Seinen Muted",
    description="Cool greys, restrained palette, cinematic shadows",
    source="builtin",
    global_warm_cool=-3.0,
    global_saturation=0.75,
    global_contrast=1.3,
    global_shadow_lift=0.1,
    cel_flatten=0.2,
    hair=RegionDescriptor(
        warm_bias=-2.0, shadow_bias=-5.0, shadow_hue_rotate=-2.0,
        shadow_desat=0.35, highlight_desat=0.2, saturation_scale=0.7,
        gradient=0.65, contrast=1.5, hint_layers=3),
    skin=RegionDescriptor(
        warm_bias=2.0, shadow_bias=-3.0, shadow_desat=0.25,
        saturation_scale=0.75, gradient=0.5, contrast=1.3, hint_layers=3),
    sky=RegionDescriptor(
        warm_bias=-8.0, saturation_scale=0.65, gradient=0.6, hint_layers=2),
    foliage=RegionDescriptor(warm_bias=-1.0, saturation_scale=0.65, hint_layers=2),
    clothing_primary=RegionDescriptor(
        warm_bias=-3.0, shadow_desat=0.3, saturation_scale=0.7, hint_layers=3),
    clothing_secondary=RegionDescriptor(saturation_scale=0.65, hint_layers=2),
    clothing_accent=RegionDescriptor(warm_bias=-1.0, saturation_scale=0.6, hint_layers=2),
    background=RegionDescriptor(warm_bias=-2.0, saturation_scale=0.45, hint_layers=1),
    saturation=0.65, contrast=0.88, temperature="cool",
    shadow_strength=0.72, gradient=0.6,
)

# ── Webtoon ───────────────────────────────────────────────────────────────────
# Flat cel-shaded, bright, hard edges, clean colors.
WEBTOON = StyleDescriptor(
    name="Webtoon",
    description="Flat cel-shaded, bright, clean colors, hard edges",
    source="builtin",
    global_warm_cool=2.0,
    global_saturation=1.45,
    global_contrast=1.15,
    global_shadow_lift=0.05,
    cel_flatten=0.75,
    hair=RegionDescriptor(
        warm_bias=0.0, shadow_bias=-6.0, shadow_desat=0.2,
        highlight_bias=8.0, highlight_desat=0.3, saturation_scale=1.3,
        gradient=0.2, contrast=1.6, hint_layers=3),
    skin=RegionDescriptor(
        warm_bias=4.0, shadow_bias=1.0, shadow_hue_rotate=2.0,
        highlight_desat=0.3, saturation_scale=0.9, gradient=0.2, hint_layers=2),
    sky=RegionDescriptor(
        warm_bias=-5.0, highlight_bias=6.0, saturation_scale=1.1,
        gradient=0.3, hint_layers=2),
    foliage=RegionDescriptor(warm_bias=2.0, saturation_scale=1.0, gradient=0.2, hint_layers=2),
    clothing_primary=RegionDescriptor(
        shadow_bias=-5.0, saturation_scale=1.35, gradient=0.15, hint_layers=3),
    clothing_secondary=RegionDescriptor(saturation_scale=1.2, gradient=0.15, hint_layers=2),
    clothing_accent=RegionDescriptor(saturation_scale=1.4, gradient=0.1, hint_layers=2),
    background=RegionDescriptor(saturation_scale=0.8, gradient=0.2, hint_layers=1),
    saturation=0.88, contrast=0.80, temperature="warm",
    shadow_strength=0.55, gradient=0.25,
)

# ── Manhwa Flat Color ─────────────────────────────────────────────────────────
MANHWA = StyleDescriptor(
    name="Manhwa Flat Color",
    description="Distinct flat fills per region — the hand-colored manhwa look",
    source="builtin",
    global_warm_cool=1.0,
    global_saturation=1.35,
    global_contrast=1.1,
    global_shadow_lift=0.08,
    cel_flatten=0.65,
    hair=RegionDescriptor(
        warm_bias=1.0, shadow_bias=-4.0, shadow_desat=0.15,
        highlight_bias=6.0, highlight_desat=0.25, saturation_scale=1.2,
        gradient=0.25, contrast=1.5, hint_layers=3),
    skin=RegionDescriptor(
        warm_bias=4.5, shadow_bias=1.5, shadow_hue_rotate=2.5,
        saturation_scale=0.88, gradient=0.2, hint_layers=2),
    sky=RegionDescriptor(warm_bias=-4.0, saturation_scale=1.0, gradient=0.35, hint_layers=2),
    foliage=RegionDescriptor(warm_bias=3.0, saturation_scale=0.95, gradient=0.2, hint_layers=2),
    clothing_primary=RegionDescriptor(
        shadow_bias=-4.0, saturation_scale=1.25, gradient=0.15, hint_layers=3),
    clothing_secondary=RegionDescriptor(saturation_scale=1.1, gradient=0.15, hint_layers=2),
    clothing_accent=RegionDescriptor(saturation_scale=1.3, gradient=0.1, hint_layers=2),
    background=RegionDescriptor(saturation_scale=0.7, gradient=0.2, hint_layers=1),
    saturation=0.85, contrast=0.78, temperature="warm",
    shadow_strength=0.58, gradient=0.28,
)

# ── Watercolor ────────────────────────────────────────────────────────────────
WATERCOLOR = StyleDescriptor(
    name="Watercolor",
    description="Soft chroma falloff, gentle saturation, washed feel",
    source="builtin",
    global_warm_cool=1.0,
    global_saturation=0.85,
    global_contrast=0.9,
    global_shadow_lift=0.15,
    cel_flatten=0.0,
    hair=RegionDescriptor(
        warm_bias=2.0, shadow_bias=-2.0, shadow_desat=0.4,
        highlight_bias=3.0, highlight_desat=0.5, saturation_scale=0.8,
        gradient=0.9, contrast=0.9, hint_layers=3),
    skin=RegionDescriptor(
        warm_bias=3.0, shadow_bias=1.0, shadow_desat=0.3,
        highlight_desat=0.6, saturation_scale=0.75, gradient=0.85, hint_layers=3),
    sky=RegionDescriptor(
        warm_bias=-3.0, highlight_desat=0.5, saturation_scale=0.7,
        gradient=0.9, hint_layers=3),
    foliage=RegionDescriptor(warm_bias=2.0, saturation_scale=0.7, gradient=0.85, hint_layers=2),
    clothing_primary=RegionDescriptor(
        shadow_desat=0.35, saturation_scale=0.8, gradient=0.8, hint_layers=3),
    clothing_secondary=RegionDescriptor(saturation_scale=0.75, gradient=0.8, hint_layers=2),
    clothing_accent=RegionDescriptor(saturation_scale=0.85, gradient=0.75, hint_layers=2),
    background=RegionDescriptor(saturation_scale=0.5, gradient=0.8, hint_layers=1),
    saturation=0.70, contrast=0.65, temperature="warm",
    shadow_strength=0.45, gradient=0.85,
)

# ── Marvel/DC ─────────────────────────────────────────────────────────────────
MARVEL = StyleDescriptor(
    name="Marvel/DC",
    description="Saturated primaries, heavy shadow, comic-press feel",
    source="builtin",
    global_warm_cool=3.0,
    global_saturation=1.7,
    global_contrast=1.6,
    global_shadow_lift=0.0,
    cel_flatten=0.05,
    hair=RegionDescriptor(
        warm_bias=2.0, shadow_bias=-10.0, shadow_hue_rotate=-3.0,
        shadow_desat=0.0, highlight_bias=6.0, saturation_scale=1.6,
        gradient=0.65, contrast=1.7, hint_layers=3),
    skin=RegionDescriptor(
        warm_bias=6.0, shadow_bias=3.0, shadow_hue_rotate=4.0,
        saturation_scale=1.1, gradient=0.5, contrast=1.5, hint_layers=3),
    sky=RegionDescriptor(
        warm_bias=-7.0, saturation_scale=1.4, gradient=0.7, hint_layers=3),
    foliage=RegionDescriptor(warm_bias=3.0, saturation_scale=1.3, hint_layers=2),
    clothing_primary=RegionDescriptor(
        warm_bias=3.0, shadow_bias=-6.0, saturation_scale=1.6, hint_layers=3),
    clothing_secondary=RegionDescriptor(
        warm_bias=-2.0, saturation_scale=1.4, hint_layers=3),
    clothing_accent=RegionDescriptor(warm_bias=6.0, saturation_scale=1.7, hint_layers=2),
    background=RegionDescriptor(warm_bias=2.0, saturation_scale=0.85, hint_layers=1),
    saturation=0.95, contrast=0.92, temperature="warm",
    shadow_strength=0.75, gradient=0.55,
)

# ── 90s Pulp ──────────────────────────────────────────────────────────────────
PULP = StyleDescriptor(
    name="90s Pulp",
    description="Limited palette, halftone-friendly, slightly faded",
    source="builtin",
    global_warm_cool=5.0,
    global_saturation=0.95,
    global_contrast=1.1,
    global_shadow_lift=0.05,
    cel_flatten=0.35,
    hair=RegionDescriptor(
        warm_bias=3.0, shadow_bias=-3.0, shadow_desat=0.2,
        highlight_bias=4.0, highlight_desat=0.3, saturation_scale=0.9,
        gradient=0.45, contrast=1.2, hint_layers=3),
    skin=RegionDescriptor(
        warm_bias=6.0, shadow_bias=3.0, shadow_hue_rotate=3.0,
        saturation_scale=0.9, gradient=0.4, hint_layers=2),
    sky=RegionDescriptor(warm_bias=-4.0, saturation_scale=0.85, gradient=0.5, hint_layers=2),
    foliage=RegionDescriptor(warm_bias=2.0, saturation_scale=0.8, gradient=0.4, hint_layers=2),
    clothing_primary=RegionDescriptor(
        warm_bias=2.0, saturation_scale=0.95, gradient=0.4, hint_layers=3),
    clothing_secondary=RegionDescriptor(saturation_scale=0.85, gradient=0.4, hint_layers=2),
    clothing_accent=RegionDescriptor(warm_bias=4.0, saturation_scale=1.0, gradient=0.3, hint_layers=2),
    background=RegionDescriptor(warm_bias=4.0, saturation_scale=0.65, gradient=0.4, hint_layers=1),
    saturation=0.75, contrast=0.78, temperature="warm",
    shadow_strength=0.60, gradient=0.42,
)

# ── Neutral ───────────────────────────────────────────────────────────────────
# The default — moderate shifts, balanced output.
NEUTRAL = StyleDescriptor(
    name="Neutral (default)",
    description="Faithful to model output, balanced post-processing",
    source="builtin",
    global_warm_cool=0.5,
    global_saturation=1.15,
    global_contrast=1.1,
    global_shadow_lift=0.05,
    cel_flatten=0.3,
    hair=RegionDescriptor(
        warm_bias=0.5, shadow_bias=-3.0, shadow_desat=0.1,
        highlight_bias=2.0, saturation_scale=1.1,
        gradient=0.55, contrast=1.2, hint_layers=3),
    skin=RegionDescriptor(
        warm_bias=3.0, shadow_bias=1.0, shadow_hue_rotate=1.5,
        saturation_scale=0.95, gradient=0.5, hint_layers=2),
    sky=RegionDescriptor(warm_bias=-4.0, saturation_scale=0.9, gradient=0.65, hint_layers=2),
    foliage=RegionDescriptor(warm_bias=2.0, saturation_scale=0.85, gradient=0.5, hint_layers=2),
    clothing_primary=RegionDescriptor(saturation_scale=1.1, gradient=0.5, hint_layers=3),
    clothing_secondary=RegionDescriptor(saturation_scale=1.0, gradient=0.5, hint_layers=2),
    clothing_accent=RegionDescriptor(saturation_scale=1.1, gradient=0.45, hint_layers=2),
    background=RegionDescriptor(saturation_scale=0.7, gradient=0.45, hint_layers=1),
    saturation=0.82, contrast=0.75, temperature="neutral",
    shadow_strength=0.60, gradient=0.5,
)

# Registry: preset key -> StyleDescriptor
BUILTIN_DESCRIPTORS: dict[str, StyleDescriptor] = {
    "shonen":    SHONEN,
    "seinen":    SEINEN,
    "webtoon":   WEBTOON,
    "manhwa":    MANHWA,
    "watercolor": WATERCOLOR,
    "marvel":    MARVEL,
    "pulp":      PULP,
    "neutral":   NEUTRAL,
    # "none" deliberately absent — no-op path bypasses descriptor entirely
}
