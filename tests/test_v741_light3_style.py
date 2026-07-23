import unittest
from pathlib import Path

from core.presets import STYLE_PRESETS, get_style

ROOT = Path(__file__).resolve().parents[1]


class TestV741Light3Style(unittest.TestCase):
    def test_light3_preset_exists_and_is_paler_than_light2(self):
        self.assertIn('light3', STYLE_PRESETS)
        light = get_style('light2')
        light3 = get_style('light3')
        self.assertEqual(light3.key, 'light3')
        self.assertLess(light3.saturation_boost, light.saturation_boost)
        self.assertLessEqual(light3.l_gamma, light.l_gamma)

    def test_ui_lists_light3(self):
        text = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
        self.assertIn('visible_style_keys = ["none", "light2", "light3"]', text)


if __name__ == '__main__':
    unittest.main()
