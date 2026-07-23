from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from core.hint_manager import HintManager
from core.local_brush import apply_local_brush_recolor
from core.presets import get_style
from core.region_map import build_region_map
from core.region_segmenter import Region, Segmentation
from core.style_post import apply_style_grade


class V53LocalityAndSkinTests(unittest.TestCase):
    @staticmethod
    def face_page() -> np.ndarray:
        image = np.full((140, 180, 3), 248, np.uint8)
        cv2.rectangle(image, (15, 15), (164, 124), (0, 0, 0), 3)
        # A black eye/line inside the large face region.
        cv2.line(image, (55, 50), (95, 50), (0, 0, 0), 4)
        return image

    def test_local_brush_never_changes_the_rest_of_large_face_region(self):
        source = self.face_page()
        result = np.full_like(source, (225, 238, 248))  # warm ivory BGR
        before = result.copy()
        rm = build_region_map(source, gap_close=4)
        edited, alpha = apply_local_brush_recolor(
            source, result, 70, 78, 12, (245, 80, 120),
            opacity=0.85, region_map=rm)

        self.assertGreater(float(alpha.max()), 0.3)
        self.assertTrue(np.any(edited != before))
        # Hard locality contract: nothing beyond radius + small implementation
        # padding may change, even though the entire face is one connected area.
        yy, xx = np.nonzero(np.any(edited != before, axis=2))
        self.assertLessEqual(int(np.max(np.abs(xx - 70))), 14)
        self.assertLessEqual(int(np.max(np.abs(yy - 78))), 14)
        np.testing.assert_array_equal(edited[35, 130], before[35, 130])
        np.testing.assert_array_equal(edited[105, 35], before[105, 35])
        # Ink remains unchanged.
        np.testing.assert_array_equal(edited[50, 70], before[50, 70])

    def test_local_brush_fills_a_small_patch_not_just_a_single_dot(self):
        source = self.face_page()
        result = np.full_like(source, (225, 238, 248))
        rm = build_region_map(source, gap_close=4)
        edited, alpha = apply_local_brush_recolor(
            source, result, 70, 78, 12, (245, 80, 120),
            opacity=0.85, region_map=rm)
        changed = np.count_nonzero(alpha > 0.08)
        self.assertGreater(changed, 120)
        self.assertTrue(np.any(edited[72:85, 63:77] != result[72:85, 63:77]))


    def test_local_brush_respects_pale_separator_and_does_not_bleed(self):
        source = np.full((120, 140, 3), 250, np.uint8)
        cv2.rectangle(source, (10, 10), (130, 110), (0, 0, 0), 2)
        cv2.line(source, (70, 12), (70, 56), (118, 118, 118), 2)
        cv2.line(source, (70, 64), (70, 108), (118, 118, 118), 2)
        source[14:108, 14:68] = 188
        source[14:108, 72:126] = 226
        result = np.full_like(source, (230, 235, 242))
        before = result.copy()
        rm = build_region_map(source, gap_close=0)
        edited, alpha = apply_local_brush_recolor(
            source, result, 40, 60, 18, (245, 90, 110),
            opacity=0.85, region_map=rm, gap_close=0)
        self.assertGreater(float(alpha[:, :68].max()), 0.2)
        self.assertLess(float(alpha[:, 72:].max()), 0.05)
        left_delta = np.abs(edited[:, :68].astype(np.int16) - before[:, :68].astype(np.int16)).mean()
        right_delta = np.abs(edited[:, 72:].astype(np.int16) - before[:, 72:].astype(np.int16)).mean()
        self.assertGreater(float(left_delta), 1.0)
        self.assertLess(float(right_delta), 0.25)

    def test_manual_model_hint_no_longer_generates_region_tiers(self):
        source = self.face_page()
        hm = HintManager()
        hm.bind_source_image(source, gap_close=4)
        hm.add_manual_hint(70 / 180, 78 / 140, (245, 80, 120), 12 / 180)
        specs = hm.merge_specs(image_bgr=source)
        manual = [h for h in specs if h.source == "manual"]
        self.assertEqual(len(manual), 1)
        self.assertAlmostEqual(manual[0].x_norm, 70 / 180, places=4)


    def test_pipeline_withholds_local_manual_hint_from_model(self):
        from pipeline import colorize_page

        source = self.face_page()
        hm = HintManager()
        hm.bind_source_image(source, gap_close=4)
        hm.add_manual_hint(70 / 180, 78 / 140, (245, 80, 120), 12 / 180)

        class FakeColorizer:
            def __init__(self):
                self.received = None

            def colorize(self, image, **kwargs):
                self.received = kwargs.get("hint_points")
                return np.full_like(image, (225, 238, 248))

        fake = FakeColorizer()
        with patch("pipeline.get_colorizer", return_value=fake):
            out = colorize_page(
                source, hint_manager=hm, style_key="none", quality_key="draft")
        self.assertFalse(fake.received)
        # The local edit is still visible after the model result, but a far-away
        # point in the same connected face remains untouched.
        self.assertTrue(np.any(out[66:91, 58:83] != (225, 238, 248)))
        np.testing.assert_array_equal(out[35, 130], np.array([225, 238, 248], np.uint8))
        self.assertEqual(hm.last_diagnostics.get("local_manual_edit_count"), 1)
        self.assertEqual(hm.last_diagnostics.get("model_hint_count"), 0)

    def test_ui_connects_stroke_lifecycle_and_labels_brush_as_local(self):
        root = Path(__file__).resolve().parents[1]
        main = (root / "ui" / "main_window.py").read_text(encoding="utf-8")
        canvas = (root / "ui" / "canvas.py").read_text(encoding="utf-8")
        i18n = (root / "ui" / "i18n.py").read_text(encoding="utf-8")
        self.assertIn("brush_stroke_started.connect", main)
        self.assertIn("brush_stroke_finished.connect", main)
        self.assertIn("apply_brush_edit", main)
        self.assertIn("brush_stroke_started = Signal()", canvas)
        self.assertIn('"tool_brush": "区域画笔"', i18n)

    def test_picker_lightness_controls_are_present(self):
        root = Path(__file__).resolve().parents[1]
        main = (root / "ui" / "main_window.py").read_text(encoding="utf-8")
        i18n = (root / "ui" / "i18n.py").read_text(encoding="utf-8")
        self.assertIn("_picker_lightness_slider", main)
        self.assertIn("_adjust_picker_lightness", main)
        self.assertIn('"picker_lightness_label": "吸管深浅"', i18n)


if __name__ == "__main__":
    unittest.main()
