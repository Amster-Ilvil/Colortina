import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestV22VisibleStyleFineTabs(unittest.TestCase):
    def test_style_fine_is_nested_on_render_with_filter_neighbor(self):
        text = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
        self.assertIn('self._render_detail_tabs = render_detail_tabs', text)
        self.assertIn('detail_fine_layout.addWidget(self._style_fine_group', text)
        self.assertIn('detail_filter_layout.addWidget(self._filter_group', text)
        self.assertNotIn('style_layout.addWidget(self._filter_group, stretch=1)', text)

    def test_original_and_light2_are_reported_as_fine_tunable(self):
        text = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
        self.assertIn('style_group.setVisible(True)', text)
        self.assertIn('active_label.setText(tr("style_fine_active")', text)

    def test_reset_button_is_inside_fine_panel(self):
        text = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
        fine_start = text.index('self._style_fine_group = QGroupBox(tr("style_fine_controls_group"))')
        filter_start = text.index('self._filter_group = QGroupBox(tr("image_filter_group"))')
        fine_block = text[fine_start:filter_start]
        self.assertIn('_btn_reset_style_fine', fine_block)
        self.assertIn('fine_layout.addWidget(self._btn_reset_style_fine)', fine_block)


if __name__ == '__main__':
    unittest.main()
