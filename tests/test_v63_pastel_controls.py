import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from core.presets import get_style
from core.style_post import apply_style_grade
from pipeline import _apply_pastel_tuning


ROOT = Path(__file__).resolve().parents[1]


class _Region:
    def __init__(self, label_id):
        self.label_id = label_id


class TestV63PastelControls(unittest.TestCase):
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
            character_instances=[SimpleNamespace(
                body_bbox=(0, 0, w, h), head_bbox=(0, 0, w, h))],
        )

    def test_tuning_multiplies_character_attribute_scales(self):
        base = get_style('monochrome')
        tuned = _apply_pastel_tuning(base, {
            'person_strength': 120,
            'hair_strength': 150,
            'skin_strength': 80,
            'eye_strength': 110,
            'clothing_strength': 90,
            'environment_strength': 120,
        })
        self.assertGreater(tuned.hair_chroma_scale, base.hair_chroma_scale)
        self.assertLess(tuned.skin_chroma_scale, base.skin_chroma_scale * 1.2)
        self.assertGreater(tuned.environment_chroma_scale, 0.0)
        self.assertFalse(tuned.force_environment_grayscale)

    def test_strict_environment_can_override_full_page_pastel(self):
        base = get_style('monochrome')
        tuned = _apply_pastel_tuning(base, {'environment_strength': 0})
        self.assertEqual(tuned.environment_chroma_scale, 0.0)
        self.assertEqual(tuned.unknown_chroma_scale, 0.0)
        self.assertTrue(tuned.force_environment_grayscale)

    def test_hair_strength_slider_visibly_changes_hair_chroma(self):
        h, w = 24, 24
        source = np.full((h, w, 3), 205, np.uint8)
        colorized = np.full((h, w, 3), (65, 115, 190), np.uint8)
        context = self._hair_context(h, w)

        low_style = _apply_pastel_tuning(
            get_style('monochrome'),
            {'hair_strength': 35, 'person_strength': 100,
             'environment_strength': 120})
        high_style = _apply_pastel_tuning(
            get_style('monochrome'),
            {'hair_strength': 165, 'person_strength': 100,
             'environment_strength': 120})
        low = apply_style_grade(colorized, source, low_style, context=context)
        high = apply_style_grade(colorized, source, high_style, context=context)
        self.assertGreater(
            float(self._chroma(high).mean()),
            float(self._chroma(low).mean()) * 1.25)

    def test_ui_and_project_settings_include_pastel_controls(self):
        main = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
        worker = (ROOT / 'ui' / 'worker.py').read_text(encoding='utf-8')
        i18n = (ROOT / 'ui' / 'i18n.py').read_text(encoding='utf-8')
        self.assertIn('_pastel_controls_group', main)
        self.assertIn('_update_pastel_controls_visibility', main)
        self.assertIn('_pastel_environment_slider', main)
        self.assertIn('pastel_tuning=self._current_pastel_tuning()', main)
        self.assertIn('pastel_tuning=self._pastel_tuning', worker)
        self.assertIn('"pastel_hair_strength": "头发"', i18n)
        self.assertIn('"pastel_skin_warmth": "肤色偏暖"', i18n)


if __name__ == '__main__':
    unittest.main()
