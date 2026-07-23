import unittest
from pathlib import Path

import cv2
import numpy as np

import pipeline
from core.page_color_context import PageColorContext
from core.presets import STYLE_PRESETS, get_style
from core.style_engine import StyleProfile
from core.style_post import apply_style_grade
from pipeline import _apply_pastel_tuning

ROOT = Path(__file__).resolve().parents[1]


class TestV69LoadingAndStyles(unittest.TestCase):
    @staticmethod
    def chroma(image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        return np.sqrt((lab[..., 1] - 128.0) ** 2 + (lab[..., 2] - 128.0) ** 2)

    def test_import_is_lazy_and_does_not_build_region_map(self):
        text = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
        constructor = text[text.index('class PageState'):text.index('class MainWindow')]
        self.assertIn('self._original_bgr = original_bgr', constructor)
        self.assertNotIn('bind_source_image(original_bgr', constructor)
        self.assertIn('def _add_pages_fast', text)
        self.assertIn('PageState(path)', text)
        self.assertIn('self._pages[p]._original_bgr', text)


    def test_hint_manager_does_not_build_map_without_hints(self):
        text = (ROOT / 'core' / 'hint_manager.py').read_text(encoding='utf-8')
        self.assertIn('and (auto or manual)', text)

    def test_light2_is_bundled_and_loadable(self):
        self.assertIn('light2', STYLE_PRESETS)
        path = ROOT / 'styles' / '淡彩水墨2.ccstyle'
        self.assertTrue(path.is_file())
        profile = StyleProfile.load(str(path))
        self.assertEqual(profile.name, '淡彩水墨2')
        builtin = pipeline._builtin_reference_profile('light2')
        self.assertIsNotNone(builtin)
        self.assertEqual(builtin.name, '淡彩水墨2')

    def test_loaded_reference_style_changes_pixels(self):
        profile = StyleProfile.load(str(ROOT / 'styles' / '淡彩水墨2.ccstyle'))
        preset = profile.get_descriptor().to_style_preset(key='light2')
        source = np.full((80, 80, 3), 220, np.uint8)
        source[8:72, 8:72] = 170
        colored = np.full((80, 80, 3), (90, 170, 220), np.uint8)
        out = apply_style_grade(colored, source, preset, context=PageColorContext())
        self.assertGreater(float(np.abs(out.astype(np.int16) - colored.astype(np.int16)).mean()), 2.0)

    def test_monochrome_pastel_works_without_face_detection(self):
        source = np.full((120, 120, 3), 242, np.uint8)
        source[18:105, 30:92] = 175
        colorized = np.full((120, 120, 3), (175, 205, 135), np.uint8)  # border/background green
        colorized[18:105, 30:92] = (80, 120, 225)  # central character warm color
        ctx = PageColorContext()

        people = apply_style_grade(colorized, source, _apply_pastel_tuning(get_style('monochrome'), {'environment_strength': 0}), context=ctx)
        page = apply_style_grade(colorized, source, _apply_pastel_tuning(get_style('monochrome'), {'environment_strength': 130}), context=PageColorContext())
        pch = self.chroma(people)
        fch = self.chroma(page)
        self.assertGreater(float(pch[30:95, 40:82].mean()), 5.0)
        self.assertLess(float(pch[:12, :].mean()), 2.5)
        self.assertGreater(float(fch.mean()), 2.5)


if __name__ == '__main__':
    unittest.main()
