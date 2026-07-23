import unittest
from pathlib import Path

from core.presets import get_style, STYLE_PRESETS

ROOT = Path(__file__).resolve().parents[1]


class TestV26StyleUiAndAlias(unittest.TestCase):
    def test_removed_light_alias_maps_to_light2(self):
        self.assertNotIn('light', STYLE_PRESETS)
        self.assertEqual(get_style('light').key, 'light2')

    def test_style_labels_renamed(self):
        self.assertEqual(get_style('light2').label, '淡彩水墨')
        self.assertEqual(get_style('light3').label, '淡彩水墨（极淡）')

    def test_ui_uses_three_visible_styles_and_no_separate_style_tab(self):
        text = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
        self.assertIn('visible_style_keys = ["none", "light2", "light3"]', text)
        self.assertNotIn('_nav_style_fine_btn', text)
        self.assertIn('self._render_detail_tabs = render_detail_tabs', text)
        self.assertIn('render_detail_tabs.addTab(detail_fine_tab, tr("style_detail_tab_fine"))', text)


if __name__ == '__main__':
    unittest.main()
