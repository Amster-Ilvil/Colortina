import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import pipeline
from core.hint_manager import HintManager
from core.page_color_context import PageColorContext
from core.presets import get_style
from core.style_post import apply_style_grade


class _FakeColorizer:
    def __init__(self):
        self.calls = 0
        self.device_name = "fake"

    def colorize(self, image_bgr, **kwargs):
        self.calls += 1
        out = image_bgr.copy()
        out[..., 2] = np.clip(out[..., 2].astype(np.int16) + 35, 0, 255).astype(np.uint8)
        return out


class TestV67PerformanceCache(unittest.TestCase):
    def setUp(self):
        pipeline.clear_raw_result_cache()

    def test_style_switch_reuses_same_raw_model_result(self):
        image = np.full((48, 48, 3), 210, np.uint8)
        hm = HintManager()
        fake = _FakeColorizer()
        with patch.object(pipeline, "get_colorizer", return_value=fake):
            first = pipeline.colorize_page(
                image, hint_manager=hm, style_key="light", quality_key="standard")
            second = pipeline.colorize_page(
                image, hint_manager=hm, style_key="monochrome_page", quality_key="standard")
        self.assertEqual(fake.calls, 1)
        self.assertEqual(first.shape, image.shape)
        self.assertEqual(second.shape, image.shape)
        self.assertTrue(hm.last_diagnostics.get("raw_cache_hit"))
        self.assertIn("reuse_cached_raw_mc_v2", hm.last_diagnostics.get("job_optimizations", []))
        self.assertFalse(hm.last_diagnostics.get("effective_per_panel"))
        self.assertLessEqual(hm.last_diagnostics.get("effective_model_size", 9999), 640)

    def test_reference_page_analysis_is_removed_for_speed(self):
        image = np.full((40, 40, 3), 200, np.uint8)
        hm = HintManager()
        fake = _FakeColorizer()
        library = SimpleNamespace(revision=3)
        with patch.object(pipeline, "get_colorizer", return_value=fake), \
             patch.object(pipeline, "build_page_context", wraps=pipeline.build_page_context) as build, \
             patch.object(pipeline, "get_guided_colorist") as guided:
            pipeline.colorize_page(
                image, hint_manager=hm, style_key="light", quality_key="draft",
                character_library=library, reference_strength=0.0)
            pipeline.colorize_page(
                image, hint_manager=hm, style_key="light", quality_key="draft",
                character_library=library, reference_strength=0.0)
        self.assertEqual(build.call_count, 0)
        self.assertEqual(guided.call_count, 0)

    def test_light_style_skips_character_detection(self):
        source = np.full((64, 64, 3), 220, np.uint8)
        colorized = source.copy()
        colorized[..., 2] = 245
        with patch("core.anime_face_detector.detect_anime_faces",
                   side_effect=AssertionError("light style should not detect faces")):
            out = apply_style_grade(colorized, source, get_style("light"), context=PageColorContext())
        self.assertEqual(out.shape, source.shape)

    def test_pastel_skips_face_detection_fallback_in_fast_mode(self):
        source = np.full((64, 64, 3), 220, np.uint8)
        colorized = source.copy()
        colorized[..., 2] = 245
        ctx = PageColorContext()
        with patch("core.anime_face_detector.detect_anime_faces",
                   side_effect=AssertionError("fast pastel should not detect faces")):
            out = apply_style_grade(
                colorized, source, get_style("monochrome_people"), context=ctx)
        self.assertEqual(out.shape, source.shape)


if __name__ == "__main__":
    unittest.main()
