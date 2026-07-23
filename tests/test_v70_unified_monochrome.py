import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from core.presets import STYLE_PRESETS, get_style
from core.style_post import apply_style_grade
from pipeline import _apply_pastel_tuning

ROOT = Path(__file__).resolve().parents[1]


class _Region:
    def __init__(self, label_id):
        self.label_id = label_id


class TestV70UnifiedMonochrome(unittest.TestCase):
    @staticmethod
    def _chroma(img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        a = lab[..., 1] - 128.0
        b = lab[..., 2] - 128.0
        return np.sqrt(a * a + b * b)

    def _hair_context(self, h=24, w=24):
        labels = np.ones((h, w), np.int32)
        seg = SimpleNamespace(labels=labels, regions=[_Region(1)])
        return SimpleNamespace(
            segmentation=seg,
            semantic_labels=[('hair', 1.0)],
            identity_assignments={},
            character_instances=[SimpleNamespace(body_bbox=(0, 0, w, h), head_bbox=(0, 0, w, h))],
        )

    def test_unified_monochrome_preset_visible(self):
        self.assertIn('monochrome', STYLE_PRESETS)
        self.assertEqual(get_style('monochrome').label, '黑白淡彩（统一）')

    def test_environment_slider_continuously_changes_scene_chroma(self):
        base = get_style('monochrome')
        strict = _apply_pastel_tuning(base, {'environment_strength': 0})
        full = _apply_pastel_tuning(base, {'environment_strength': 150})
        self.assertEqual(strict.environment_chroma_scale, 0.0)
        self.assertTrue(strict.force_environment_grayscale)
        self.assertGreater(full.environment_chroma_scale, base.environment_chroma_scale)
        self.assertFalse(full.force_environment_grayscale)

    def test_global_style_tuning_changes_wash_style(self):
        base = get_style('light')
        tuned = _apply_pastel_tuning(base, {
            'color_strength': 150,
            'brightness': 130,
            'warmth': 140,
            'highlight_preserve': 150,
            'softness': 120,
            'flatten': 140,
        })
        self.assertGreater(tuned.saturation_boost, base.saturation_boost)
        self.assertLess(tuned.l_gamma, base.l_gamma)
        self.assertGreater(tuned.chroma_warm_shift, base.chroma_warm_shift)
        self.assertGreaterEqual(tuned.guided_filter_radius, base.guided_filter_radius)
        self.assertGreater(tuned.neutral_fade_floor, base.neutral_fade_floor)

    def test_hair_strength_still_visibly_changes_monochrome_output(self):
        h, w = 24, 24
        source = np.full((h, w, 3), 205, np.uint8)
        colorized = np.full((h, w, 3), (65, 115, 190), np.uint8)
        context = self._hair_context(h, w)

        low_style = _apply_pastel_tuning(get_style('monochrome'), {'hair_strength': 35})
        high_style = _apply_pastel_tuning(get_style('monochrome'), {'hair_strength': 165})
        low = apply_style_grade(colorized, source, low_style, context=context)
        high = apply_style_grade(colorized, source, high_style, context=context)
        self.assertGreater(float(self._chroma(high).mean()), float(self._chroma(low).mean()) * 1.20)

    def test_ui_strings_and_controls_exist(self):
        main = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
        i18n = (ROOT / 'ui' / 'i18n.py').read_text(encoding='utf-8')
        self.assertIn('visible_style_keys = ["none", "light", "light2", "monochrome"]', main)
        self.assertIn('_style_color_slider', main)
        self.assertIn('_pastel_environment_slider', main)
        self.assertIn('style_fine_controls_group', i18n)
        self.assertIn('"pastel_skin_warmth": "肤色偏暖"', i18n)


if __name__ == '__main__':
    unittest.main()
