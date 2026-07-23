import unittest
from pathlib import Path

import cv2
import numpy as np

from core.image_filter import apply_image_filter
from core.presets import get_style
from pipeline import _apply_pastel_tuning

ROOT = Path(__file__).resolve().parents[1]


class TestV748FilterAndLight2(unittest.TestCase):
    def test_light2_supports_style_fine_tuning(self):
        style = get_style('light2')
        tuned = _apply_pastel_tuning(style, {
            'color_strength': 160, 'brightness': 70, 'warmth': 140,
            'highlight_preserve': 120, 'softness': 130, 'flatten': 80,
        })
        self.assertNotEqual(tuned.saturation_boost, style.saturation_boost)
        self.assertNotEqual(tuned.l_gamma, style.l_gamma)
        self.assertNotEqual(tuned.guided_filter_radius, style.guided_filter_radius)

    def test_filter_changes_image(self):
        base = np.full((24, 24, 3), (120, 140, 170), dtype=np.uint8)
        out = apply_image_filter(base, {
            'brightness': 130, 'contrast': 120, 'saturation': 140,
            'warmth': 120, 'shadow_lift': 110, 'highlight': 120,
        }, style_strength=1.0, is_styled=True)
        self.assertFalse(np.array_equal(base, out))

    def test_color_filters_are_gentler_at_low_style_strength(self):
        base = np.full((24, 24, 3), (90, 140, 200), dtype=np.uint8)
        low = apply_image_filter(base, {'saturation': 160, 'warmth': 160}, style_strength=0.1, is_styled=True)
        high = apply_image_filter(base, {'saturation': 160, 'warmth': 160}, style_strength=1.0, is_styled=True)
        hsv_base = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
        hsv_low = cv2.cvtColor(low, cv2.COLOR_BGR2HSV)
        hsv_high = cv2.cvtColor(high, cv2.COLOR_BGR2HSV)
        self.assertGreaterEqual(int(hsv_high[...,1].mean()) - int(hsv_base[...,1].mean()),
                                int(hsv_low[...,1].mean()) - int(hsv_base[...,1].mean()))

    def test_ui_contains_filter_controls_and_hint(self):
        text = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
        i18n = (ROOT / 'ui' / 'i18n.py').read_text(encoding='utf-8')
        self.assertIn('_filter_group', text)
        self.assertIn('_current_filter_tuning', text)
        self.assertIn('image_filter_group', i18n)
        self.assertIn('style_strength_hint', i18n)


if __name__ == '__main__':
    unittest.main()
