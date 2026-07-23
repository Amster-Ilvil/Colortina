import unittest
from pathlib import Path

from core.presets import get_style
from pipeline import _apply_pastel_tuning

ROOT = Path(__file__).resolve().parents[1]


class TestV744Light3IntensityPreview(unittest.TestCase):
    def test_light3_intensity_is_continuous(self):
        base = get_style('light3')
        faint = _apply_pastel_tuning(base, {'light3_intensity': 0})
        normal = _apply_pastel_tuning(base, {'light3_intensity': 100})
        stronger = _apply_pastel_tuning(base, {'light3_intensity': 200})
        self.assertLess(faint.saturation_boost, normal.saturation_boost)
        self.assertLess(normal.saturation_boost, stronger.saturation_boost)
        self.assertLess(stronger.saturation_boost, 0.50)

    def test_manual_match_checkbox_is_built(self):
        main = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
        self.assertIn('self._chk_manual_match_style = QCheckBox', main)
        self.assertIn('manual_color_preview_tooltip', main)
        self.assertIn('light3_intensity', main)

    def test_project_persists_new_controls(self):
        main = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
        self.assertIn('"light3_intensity": self._light3_intensity_slider.value()', main)
        self.assertIn('settings.get("light3_intensity", 100)', main)
        self.assertIn('settings.get("manual_match_style", True)', main)


if __name__ == '__main__':
    unittest.main()
