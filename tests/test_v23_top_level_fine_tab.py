import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestV23TopLevelFineTab(unittest.TestCase):
    def test_fine_tuning_is_nested_back_inside_render_tab(self):
        text = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
        self.assertIn('self._render_detail_tabs = render_detail_tabs', text)
        self.assertIn('render_detail_tabs.addTab(detail_fine_tab, tr("style_detail_tab_fine"))', text)
        self.assertIn('render_detail_tabs.addTab(detail_filter_tab, tr("image_filter_group"))', text)
        self.assertNotIn('_nav_style_fine_btn', text)

    def test_reset_button_is_inside_visible_fine_group(self):
        text = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
        reset_pos = text.index('self._btn_reset_style_fine = QPushButton')
        add_pos = text.index('fine_layout.addWidget(self._btn_reset_style_fine)')
        detail_pos = text.index('detail_fine_layout.addWidget(self._style_fine_group, stretch=1)')
        self.assertLess(reset_pos, add_pos)
        self.assertLess(add_pos, detail_pos)

    def test_original_and_light2_use_same_visible_controls(self):
        text = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
        self.assertIn('visible_style_keys = ["none", "light2", "light3"]', text)
        self.assertIn('style_group.setVisible(True)', text)

    def test_translations_exist(self):
        text = (ROOT / 'ui' / 'i18n.py').read_text(encoding='utf-8')
        self.assertIn('style_detail_tab_fine', text)
        self.assertIn('reset_style_fine_btn', text)


if __name__ == '__main__':
    unittest.main()
