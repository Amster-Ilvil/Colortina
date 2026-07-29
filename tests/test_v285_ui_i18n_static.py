import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestV285UiI18nStatic(unittest.TestCase):
    def test_i18n_contains_new_selection_mode_and_tone_range_keys(self):
        text = (ROOT / 'ui' / 'i18n.py').read_text(encoding='utf-8')
        for key in [
            'selection_mode', 'selection_mode_replace', 'selection_mode_add', 'selection_mode_subtract',
            'custom_color_bias_tone_range', 'custom_color_bias_tone_all',
            'custom_color_bias_tone_highlights', 'custom_color_bias_tone_midtones', 'custom_color_bias_tone_shadows'
        ]:
            self.assertIn(key, text)


if __name__ == '__main__':
    unittest.main()
