import unittest
from pathlib import Path

from core.presets import STYLE_PRESETS, get_style

ROOT = Path(__file__).resolve().parents[1]


class TestV747Light2NoReferenceUI(unittest.TestCase):
    def test_light2_is_plain_builtin_preset(self):
        self.assertIn('light2', STYLE_PRESETS)
        style = get_style('light2')
        self.assertEqual(style.label, '淡彩水墨')
        self.assertGreater(style.saturation_boost, get_style('light3').saturation_boost)
        self.assertLess(style.saturation_boost, 1.0)

    def test_reference_tab_and_navigation_removed(self):
        main = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
        self.assertIn('visible_style_keys = ["none", "light2", "light3"]', main)
        self.assertNotIn('_nav_reference_btn', main)
        self.assertNotIn('addTab(reference_tab', main)
        self.assertNotIn('reference_tab, reference_layout', main)
        self.assertIn('_right_tabs.setCurrentIndex(1)', main)
        self.assertIn('_right_tabs.setCurrentIndex(2)', main)

    def test_reference_dialog_file_removed(self):
        self.assertFalse((ROOT / 'ui' / 'reference_match_dialog.py').exists())


if __name__ == '__main__':
    unittest.main()
