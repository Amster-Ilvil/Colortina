"""StyleDirector — converts a StyleDescriptor into concrete hint colors.

This is the bridge between "color language" (StyleDescriptor) and the
actual RGB triples that feed mc-v2's hint channel.

Design:
  - We NEVER invent an absolute color.  We start from the DEFAULT_PALETTE
    seed (the palette's mid-tone for a given region class) and SHIFT it
    in LAB space according to the StyleDescriptor's RegionDescriptor.
  - Three output tiers per region: highlight / mid / shadow.
    Each tier sits at a different LAB L level and carries the style's
    warm/cool/hue character for that tier.
  - The caller (GuidedColorist) places these tier colors on pixels whose
    actual grayscale value matches that tier, so mc-v2 sees a natural
    highlight-mid-shadow gradient instead of a single flat color.

Key guarantee:
  The mc-v2 model is free to decide that character A has black hair and
  character B has blond hair — we never override that.  We only control
  HOW the hair is shaded: how warm the shadow is, how the highlight
  catches light, etc.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.color_director import DEFAULT_PALETTE, hex_to_bgr
from core.style_descriptor import StyleDescriptor, RegionDescriptor


@dataclass
class TieredColor:
    """Three hint color tiers for one region + how many layers to use."""
    highlight_rgb: tuple[int, int, int]
    mid_rgb:       tuple[int, int, int]
    shadow_rgb:    tuple[int, int, int]
    hint_layers:   int = 3


def _bgr_to_lab_px(bgr: tuple[int, int, int]) -> np.ndarray:
    """Convert a single BGR tuple to a 1×1 LAB pixel (L 0-255, A/B 0-255)."""
    px = np.array([[[bgr[0], bgr[1], bgr[2]]]], dtype=np.uint8)
    lab = cv2.cvtColor(px, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab[0, 0]   # [L, A, B]


def _lab_px_to_bgr(lab: np.ndarray) -> tuple[int, int, int]:
    """Convert a LAB triple (float, centred A/B at 128) to BGR uint8."""
    px = np.array([[[np.clip(lab[0], 0, 255),
                     np.clip(lab[1], 0, 255),
                     np.clip(lab[2], 0, 255)]]], dtype=np.uint8)
    bgr = cv2.cvtColor(px, cv2.COLOR_LAB2BGR)
    b, g, r = int(bgr[0, 0, 0]), int(bgr[0, 0, 1]), int(bgr[0, 0, 2])
    return (b, g, r)


def _apply_regional_style(seed_bgr: tuple[int, int, int],
                          rd: RegionDescriptor,
                          global_warm_cool: float = 0.0,
                          global_saturation: float = 1.0) -> TieredColor:
    """Apply a RegionDescriptor's language onto a seed BGR color.

    Returns three tiers (highlight / mid / shadow) in BGR.

    The seed is the DEFAULT_PALETTE mid-tone for this region class.
    We only shift in LAB — never replace.
    """
    lab = _bgr_to_lab_px(seed_bgr)
    L0, A0, B0 = float(lab[0]), float(lab[1]), float(lab[2])

    # Global saturation applied as chroma scale around gray axis (A=B=128)
    def sat_scale(A, B, scale):
        a = (A - 128.0) * scale + 128.0
        b = (B - 128.0) * scale + 128.0
        return a, b

    total_warm = B0 + rd.warm_bias + global_warm_cool
    A_mid, B_mid = sat_scale(A0, total_warm, rd.saturation_scale * global_saturation)

    # --- Mid-tone ---
    mid_lab = np.array([L0, A_mid, B_mid])

    # --- Highlight ---  brighter, style's hi_bias on top of mid
    L_hi = float(np.clip(L0 + 30.0, 0, 255))
    A_hi = A_mid + rd.highlight_hue_rotate
    B_hi = B_mid + rd.highlight_bias
    if rd.highlight_desat > 0:
        A_hi, B_hi = sat_scale(A_hi, B_hi, 1.0 - rd.highlight_desat)
    hi_lab = np.array([L_hi, np.clip(A_hi, 0, 255), np.clip(B_hi, 0, 255)])

    # --- Shadow ---  darker, style's shadow shifts
    L_sh = float(np.clip(L0 - 35.0, 0, 255))
    A_sh = A_mid + rd.shadow_hue_rotate
    B_sh = B_mid + rd.shadow_bias
    if rd.shadow_desat > 0:
        A_sh, B_sh = sat_scale(A_sh, B_sh, 1.0 - rd.shadow_desat)
    sh_lab = np.array([L_sh, np.clip(A_sh, 0, 255), np.clip(B_sh, 0, 255)])

    return TieredColor(
        highlight_rgb=_bgr_to_rgb(_lab_px_to_bgr(hi_lab)),
        mid_rgb=      _bgr_to_rgb(_lab_px_to_bgr(mid_lab)),
        shadow_rgb=   _bgr_to_rgb(_lab_px_to_bgr(sh_lab)),
        hint_layers=rd.hint_layers,
    )


def _bgr_to_rgb(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    return (bgr[2], bgr[1], bgr[0])


class StyleDirector:
    """Converts a StyleDescriptor + palette key into TieredColor hints.

    One instance per job; reuses cached results for the same (descriptor,
    palette_key) pair so we don't recompute every page.
    """

    def __init__(self, descriptor: StyleDescriptor, palette: dict | None = None):
        self._desc    = descriptor
        self._palette = {**DEFAULT_PALETTE, **(palette or {})}
        self._cache: dict[str, TieredColor] = {}

    def get_tiered(self, palette_key: str) -> TieredColor:
        """Return the TieredColor for a palette key.

        The seed color comes from the palette (DEFAULT or overridden).
        We then apply the StyleDescriptor's language for that region.
        """
        if palette_key in self._cache:
            return self._cache[palette_key]

        seed_hex = self._palette.get(palette_key) or DEFAULT_PALETTE.get(
            palette_key, "#888888")
        seed_bgr = hex_to_bgr(seed_hex)   # (b, g, r)

        rd = self._desc.region(palette_key)
        tc = _apply_regional_style(
            seed_bgr, rd,
            global_warm_cool=self._desc.global_warm_cool,
            global_saturation=self._desc.global_saturation,
        )
        self._cache[palette_key] = tc
        return tc
