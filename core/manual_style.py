"""Style-aware manual recolor helpers.

These functions remap a user-selected RGB colour so manual brush / bucket edits
sit more naturally inside the currently selected rendering style.
"""
from __future__ import annotations

import cv2
import numpy as np


def adapt_rgb_to_style(rgb: tuple[int, int, int], style) -> tuple[int, int, int]:
    """Return a style-adjusted RGB target for manual recolor tools.

    The chosen hue remains recognisable, but low-saturation / wash styles make
    the target lighter and less saturated so brush and bucket edits do not pop
    out more strongly than the surrounding AI result.
    """
    if rgb is None:
        return 0, 0, 0
    r, g, b = [int(np.clip(v, 0, 255)) for v in rgb]
    if style is None or getattr(style, 'key', 'none') == 'none':
        return r, g, b

    bgr = np.array([[[b, g, r]]], dtype=np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    sat_boost = float(np.clip(getattr(style, 'saturation_boost', 1.0), 0.05, 3.0))
    if sat_boost < 1.0:
        sat_scale = float(np.clip(0.10 + 0.90 * sat_boost, 0.08, 1.0))
    else:
        sat_scale = float(np.clip(0.94 + (sat_boost - 1.0) * 0.10, 0.90, 1.18))
    hsv[0, 0, 1] = np.clip(hsv[0, 0, 1] * sat_scale, 0.0, 255.0)

    gamma = float(np.clip(getattr(style, 'l_gamma', 1.0), 0.65, 1.6))
    L = float(lab[0, 0, 0]) / 255.0
    L = float(np.clip(np.power(max(L, 1e-4), gamma), 0.0, 1.0))
    lab[0, 0, 0] = L * 255.0

    pale_amount = float(np.clip((1.0 - sat_boost) / 0.92, 0.0, 1.0))
    if pale_amount > 0.0:
        hsv[0, 0, 2] = np.clip(hsv[0, 0, 2] + (255.0 - hsv[0, 0, 2]) * (0.10 + pale_amount * 0.16), 0.0, 255.0)

    temp_shift = float(getattr(style, 'chroma_warm_shift', 0.0))
    red_shift = float(getattr(style, 'chroma_red_shift', 0.0))
    lab[0, 0, 1] = np.clip(lab[0, 0, 1] + red_shift * 0.55, 0.0, 255.0)
    lab[0, 0, 2] = np.clip(lab[0, 0, 2] + temp_shift * 0.75, 0.0, 255.0)

    hsv_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    lab_bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    blend = 0.62 if sat_boost < 1.0 else 0.40
    out = hsv_bgr * blend + lab_bgr * (1.0 - blend)
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)[0, 0]
    return int(out[2]), int(out[1]), int(out[0])
