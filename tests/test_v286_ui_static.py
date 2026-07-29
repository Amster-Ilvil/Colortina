import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestV286UiStatic(unittest.TestCase):
    def test_ui_contains_feather_and_protection_controls(self):
        text = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
        for token in ['_selection_feather_slider', '_selection_feather_spin', '_chk_custom_bias_protect_skin', '_chk_custom_bias_protect_lineart', '_chk_custom_bias_protect_saturated', 'selection_feather']:
            self.assertIn(token, text)

    def test_language_rebuild_preserves_new_control_values(self):
        text = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
        for token in [
            'prev_custom_bias_enabled', 'prev_custom_bias_color',
            'prev_custom_bias_protect_skin', 'prev_selection_feather',
            'prev_selection_closed_only', '_update_custom_bias_controls_enabled'
        ]:
            self.assertIn(token, text)

    def test_i18n_contains_all_new_keys(self):
        text = (ROOT / 'ui' / 'i18n.py').read_text(encoding='utf-8')
        for key in ['selection_feather', 'selection_feather_hint', 'custom_color_bias_protect_skin', 'custom_color_bias_protect_lineart', 'custom_color_bias_protect_saturated', 'custom_color_bias_protection_hint']:
            self.assertIn(key, text)


if __name__ == '__main__':
    unittest.main()
