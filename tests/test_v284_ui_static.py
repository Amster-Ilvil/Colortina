import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestV284UiStatic(unittest.TestCase):
    def test_main_window_contains_closed_only_and_scope(self):
        text = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
        self.assertIn('_chk_selection_closed_only', text)
        self.assertIn('_custom_color_bias_scope = QComboBox()', text)
        self.assertIn('custom_color_bias_scope_page', text)
        self.assertIn('selection_fill_no_closed', text)

    def test_i18n_contains_new_keys(self):
        text = (ROOT / 'ui' / 'i18n.py').read_text(encoding='utf-8')
        for key in [
            'selection_closed_only', 'selection_closed_only_hint',
            'custom_color_bias_scope', 'custom_color_bias_scope_page',
            'custom_color_bias_scope_characters', 'custom_color_bias_scope_background'
        ]:
            self.assertIn(key, text)


if __name__ == '__main__':
    unittest.main()
