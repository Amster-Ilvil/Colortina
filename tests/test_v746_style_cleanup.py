import unittest
from pathlib import Path

from core.presets import STYLE_PRESETS

ROOT = Path(__file__).resolve().parents[1]

class TestV746StyleCleanup(unittest.TestCase):
    def test_only_requested_builtin_styles_remain(self):
        self.assertEqual(list(STYLE_PRESETS), ['none', 'light3', 'light2'])

    def test_reference_style_ui_removed(self):
        main = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
        self.assertIn('visible_style_keys = ["none", "light2", "light3"]', main)
        for token in ('_custom_style_combo', '_new_style_from_reference', '_load_style_file', '_apply_saved_style', '_delete_saved_style'):
            self.assertNotIn(token, main)

    def test_bundled_reference_style_removed(self):
        self.assertFalse((ROOT / 'styles').exists())
        self.assertIn('light2', STYLE_PRESETS)
        pipe = (ROOT / 'pipeline.py').read_text(encoding='utf-8')
        self.assertNotIn('_builtin_reference_profile', pipe)

if __name__ == '__main__': unittest.main()
