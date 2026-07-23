import unittest
from pathlib import Path

from pipeline import _apply_pastel_tuning, _adjustable_original_style, _is_default_style_tuning
from core.presets import get_style

ROOT = Path(__file__).resolve().parents[1]


class TestV2FineTuneControls(unittest.TestCase):
    def test_light2_now_responds_to_style_fine_tuning(self):
        style = get_style('light2')
        tuned = _apply_pastel_tuning(style, {
            'color_strength': 160, 'brightness': 70, 'warmth': 140,
            'highlight_preserve': 120, 'softness': 130, 'flatten': 80,
        })
        self.assertNotEqual(tuned.saturation_boost, style.saturation_boost)
        self.assertNotEqual(tuned.l_gamma, style.l_gamma)

    def test_original_style_has_adjustable_baseline(self):
        base = _adjustable_original_style()
        tuned = _apply_pastel_tuning(base, {
            'color_strength': 140, 'brightness': 90, 'warmth': 120,
            'highlight_preserve': 100, 'softness': 110, 'flatten': 130,
        })
        self.assertEqual(base.key, 'none_tuned')
        self.assertNotEqual(tuned.saturation_boost, base.saturation_boost)
        self.assertNotEqual(tuned.cel_flatten, base.cel_flatten)

    def test_default_style_tuning_detection(self):
        self.assertTrue(_is_default_style_tuning({}))
        self.assertTrue(_is_default_style_tuning({'color_strength': 100, 'light3_intensity': 40}))
        self.assertFalse(_is_default_style_tuning({'brightness': 99}))

    def test_ui_contains_reset_button_and_visibility_logic(self):
        text = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
        i18n = (ROOT / 'ui' / 'i18n.py').read_text(encoding='utf-8')
        self.assertIn('_btn_reset_style_fine', text)
        self.assertIn('_reset_style_fine_tuning', text)
        self.assertIn('style_group.setVisible(True)', text)
        self.assertIn('self._render_detail_tabs = render_detail_tabs', text)
        self.assertIn('render_detail_tabs.addTab(detail_fine_tab, tr("style_detail_tab_fine"))', text)
        self.assertIn('style_fine_active', i18n)
        self.assertIn('reset_style_fine_btn', i18n)


if __name__ == '__main__':
    unittest.main()
