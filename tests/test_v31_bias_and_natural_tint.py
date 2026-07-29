from pathlib import Path

import cv2
import numpy as np

from core.custom_color_bias import apply_global_color_bias
from core.image_filter import apply_image_filter

ROOT = Path(__file__).resolve().parents[1]


def _textured_neutral_page(size: int = 128):
    source = np.full((size, size, 3), 255, np.uint8)
    cv2.rectangle(source, (8, 8), (size - 9, size - 9), (0, 0, 0), 2)
    yy, xx = np.mgrid[:size, :size]
    texture = (((xx * 3 + yy * 5) % 13) - 6).astype(np.int16) * 2
    base = np.clip(184 + texture, 0, 255).astype(np.uint8)
    image = np.stack([
        np.clip(base.astype(np.int16) + 5, 0, 255),
        base,
        np.clip(base.astype(np.int16) - 5, 0, 255),
    ], axis=2).astype(np.uint8)
    return source, image


def test_custom_bias_200_percent_is_stronger_than_100_percent():
    source, image = _textured_neutral_page()
    target = (235, 72, 118)
    normal = apply_global_color_bias(
        image, source, target, 1.0, protect_lineart=True)
    strong = apply_global_color_bias(
        image, source, target, 2.0, protect_lineart=True)

    roi = np.s_[20:108, 20:108]
    normal_change = float(np.abs(normal[roi].astype(np.int16) - image[roi]).mean())
    strong_change = float(np.abs(strong[roi].astype(np.int16) - image[roi]).mean())
    # V5.4.x 重新标定：100% 即接近完整效果，200% 消耗剩余余量。
    assert strong_change > normal_change * 1.25

    # Black border remains protected even at 200%.
    assert float(np.abs(strong[8, 40].astype(np.int16) - image[8, 40]).mean()) < 3.0


def test_natural_filter_preserves_luminance_texture():
    source, image = _textured_neutral_page()
    out = apply_image_filter(image, {
        'color_filter_enabled': True,
        'color_filter_strength': 120,
        'color_filter_color': '#4f83d8',
    }, source_bw_bgr=source)

    before_l = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[20:108, 20:108, 0].astype(np.float32)
    after_l = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[20:108, 20:108, 0].astype(np.float32)
    before_detail = before_l - cv2.GaussianBlur(before_l, (0, 0), 1.4)
    after_detail = after_l - cv2.GaussianBlur(after_l, (0, 0), 1.4)
    ratio = float(after_detail.std() / max(before_detail.std(), 1e-5))
    assert 0.72 < ratio < 1.30

    # It must still create a visible colour tendency rather than being a no-op.
    assert float(np.abs(out[20:108, 20:108].astype(np.int16)
                        - image[20:108, 20:108]).mean()) > 5.0


def test_ui_exposes_extended_safe_strength_ranges():
    main = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
    assert 'self._custom_color_bias_slider.setRange(0, 200)' in main
    assert 'self._custom_color_bias_spin.setRange(0, 200)' in main
    assert 'self._filter_color_strength_slider.setRange(0, 150)' in main
    assert 'self._filter_color_strength_spin.setRange(0, 150)' in main
