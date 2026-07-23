import unittest
from pathlib import Path
import cv2
import numpy as np

from core.manual_style import adapt_rgb_to_style
from core.presets import get_style
from pipeline import _apply_pastel_tuning

ROOT = Path(__file__).resolve().parents[1]


class TestV743ManualStyleMatch(unittest.TestCase):
    @staticmethod
    def _sat(rgb):
        arr = np.array([[[rgb[2], rgb[1], rgb[0]]]], dtype=np.uint8)
        hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)
        return int(hsv[0, 0, 1])

    def test_light3_makes_manual_target_paler(self):
        rgb = (230, 60, 70)
        style = _apply_pastel_tuning(get_style('light3'), {'color_strength': 100, 'brightness': 100, 'warmth': 100, 'softness': 100, 'flatten': 100, 'highlight_preserve': 100})
        out = adapt_rgb_to_style(rgb, style)
        self.assertLess(self._sat(out), self._sat(rgb))
        self.assertGreater(sum(out), sum(rgb) - 35)

    def test_none_style_keeps_manual_target(self):
        rgb = (120, 180, 220)
        self.assertEqual(adapt_rgb_to_style(rgb, get_style('none')), rgb)

    def test_ui_contains_manual_match_controls(self):
        text = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
        i18n = (ROOT / 'ui' / 'i18n.py').read_text(encoding='utf-8')
        self.assertIn('_chk_manual_match_style', text)
        self.assertIn('_manual_target_rgb', text)
        self.assertIn('manual_match_style_checkbox', i18n)


if __name__ == '__main__':
    unittest.main()
