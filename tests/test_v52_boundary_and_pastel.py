from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from core.line_boundary import analyze_boundaries
from core.page_color_context import PageColorContext
from core.presets import get_style
from core.region_map import build_region_map
from core.region_segmenter import Region, Segmentation
from core.style_post import apply_style_grade


class V52BoundaryAndPastelTests(unittest.TestCase):
    @staticmethod
    def broken_box(gap: int = 6) -> np.ndarray:
        image = np.full((120, 160, 3), 255, np.uint8)
        left_end = 78 - gap // 2
        right_start = left_end + gap + 1
        cv2.line(image, (20, 20), (left_end, 20), (0, 0, 0), 2)
        cv2.line(image, (right_start, 20), (140, 20), (0, 0, 0), 2)
        cv2.line(image, (20, 20), (20, 100), (0, 0, 0), 2)
        cv2.line(image, (140, 20), (140, 100), (0, 0, 0), 2)
        cv2.line(image, (20, 100), (140, 100), (0, 0, 0), 2)
        return image

    def test_local_gap_repair_seals_box_and_is_monotonic(self):
        image = self.broken_box(6)
        open_map = build_region_map(image, gap_close=0)
        self.assertEqual(open_map.region_at(80, 60), open_map.region_at(5, 5))
        repaired_counts = []
        for gap in (4, 8, 12):
            region_map = build_region_map(image, gap_close=gap)
            self.assertNotEqual(region_map.region_at(80, 60),
                                region_map.region_at(5, 5))
            repaired_counts.append(int(np.count_nonzero(region_map.repaired)))
        self.assertEqual(repaired_counts, sorted(repaired_counts))

    def test_large_gap_value_does_not_globally_thicken_page(self):
        image = self.broken_box(6)
        # Add many unrelated lines.  A whole-page dilation would expand all of
        # them as the slider increases; local bridge repair must not.
        for y in range(30, 100, 10):
            cv2.line(image, (30, y), (65, y), (0, 0, 0), 1)
        base = analyze_boundaries(image, gap_close=0)
        strong = analyze_boundaries(image, gap_close=12)
        added = int(np.count_nonzero(strong.barrier) - np.count_nonzero(base.barrier))
        self.assertGreater(added, 0)
        self.assertLess(added, int(image.shape[0] * image.shape[1] * 0.015))

    def test_pale_gray_contour_is_still_a_boundary(self):
        image = np.full((80, 120, 3), 255, np.uint8)
        cv2.line(image, (60, 0), (60, 79), (115, 115, 115), 2)
        region_map = build_region_map(image, line_low=75, gap_close=0)
        self.assertNotEqual(region_map.region_at(20, 40),
                            region_map.region_at(100, 40))
        self.assertGreater(float(region_map.line_confidence[:, 60].mean()), 0.7)

    def test_people_pastel_keeps_character_colour_not_environment(self):
        source = np.full((80, 120, 3), 150, np.uint8)
        result = np.empty_like(source)
        result[:, :60] = (30, 50, 210)   # character, red-ish BGR
        result[:, 60:] = (40, 190, 50)   # environment, green-ish BGR
        labels = np.zeros((80, 120), np.int32)
        labels[:, :60] = 1
        labels[:, 60:] = 2
        regions = [
            Region(1, 4800, (0, 0, 60, 80), (30, 40), 150.0, 0.5),
            Region(2, 4800, (60, 0, 60, 80), (90, 40), 150.0, 0.5),
        ]
        context = PageColorContext(
            segmentation=Segmentation(regions, labels, 1.0),
            semantic_labels=[("hair", 0.95), ("background", 0.95)],
        )
        out = apply_style_grade(
            result, source, get_style("monochrome_people"), context=context)
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
        chroma = np.linalg.norm(lab[..., 1:3] - 128.0, axis=2)
        self.assertGreater(float(chroma[:, :60].mean()),
                           float(chroma[:, 60:].mean()) * 8.0)
        self.assertLess(float(chroma[:, 60:].mean()), 3.0)

    def test_page_pastel_keeps_faint_environment_tint(self):
        source = np.full((80, 120, 3), 150, np.uint8)
        result = np.empty_like(source)
        result[:, :60] = (30, 50, 210)
        result[:, 60:] = (40, 190, 50)
        labels = np.zeros((80, 120), np.int32)
        labels[:, :60] = 1
        labels[:, 60:] = 2
        regions = [
            Region(1, 4800, (0, 0, 60, 80), (30, 40), 150.0, 0.5),
            Region(2, 4800, (60, 0, 60, 80), (90, 40), 150.0, 0.5),
        ]
        context = PageColorContext(
            segmentation=Segmentation(regions, labels, 1.0),
            semantic_labels=[("skin", 0.95), ("background", 0.95)],
        )
        people = apply_style_grade(
            result, source, get_style("monochrome_people"), context=context)
        page = apply_style_grade(
            result, source, get_style("monochrome_page"), context=context)
        people_lab = cv2.cvtColor(people, cv2.COLOR_BGR2LAB).astype(np.float32)
        page_lab = cv2.cvtColor(page, cv2.COLOR_BGR2LAB).astype(np.float32)
        people_env = np.linalg.norm(people_lab[:, 60:, 1:3] - 128.0, axis=2).mean()
        page_env = np.linalg.norm(page_lab[:, 60:, 1:3] - 128.0, axis=2).mean()
        self.assertGreater(float(page_env), float(people_env) + 2.0)
        self.assertLess(float(page_env), 12.0)

    def test_people_pastel_character_colours_are_visible_not_too_faint(self):
        source = np.full((90, 90, 3), 190, np.uint8)
        result = np.full((90, 90, 3), (240, 240, 240), np.uint8)
        result[:, :30] = (170, 195, 235)   # skin-like
        result[:, 30:60] = (45, 95, 185)   # hair-like
        result[:, 60:] = (205, 205, 205)   # near-neutral background
        labels = np.zeros((90, 90), np.int32)
        labels[:, :30] = 1
        labels[:, 30:60] = 2
        labels[:, 60:] = 3
        regions = [
            Region(1, 2700, (0, 0, 30, 90), (15, 45), 190.0, 0.5),
            Region(2, 2700, (30, 0, 30, 90), (45, 45), 190.0, 0.5),
            Region(3, 2700, (60, 0, 30, 90), (75, 45), 190.0, 0.5),
        ]
        context = PageColorContext(
            segmentation=Segmentation(regions, labels, 1.0),
            semantic_labels=[("skin", 0.95), ("hair", 0.95), ("background", 0.95)],
        )
        out = apply_style_grade(result, source, get_style("monochrome_people"), context=context)
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
        chroma = np.linalg.norm(lab[..., 1:3] - 128.0, axis=2)
        self.assertGreater(float(chroma[:, :30].mean()), 7.5)
        self.assertGreater(float(chroma[:, 30:60].mean()), 11.0)
        self.assertLess(float(chroma[:, 60:].mean()), 4.0)


    def test_light_wash_and_monochrome_modes_are_visibly_different(self):
        source = np.full((90, 120, 3), 168, np.uint8)
        result = np.full((90, 120, 3), 240, np.uint8)
        # Character strip on left, background tint on right.
        result[:, :40] = (90, 110, 220)
        result[:, 40:80] = (150, 185, 105)
        result[:, 80:] = (210, 150, 100)
        labels = np.zeros((90, 120), np.int32)
        labels[:, :40] = 1
        labels[:, 40:80] = 2
        labels[:, 80:] = 3
        regions = [
            Region(1, 90 * 40, (0, 0, 40, 90), (20, 45), 180.0, 0.33),
            Region(2, 90 * 40, (40, 0, 40, 90), (60, 45), 180.0, 0.33),
            Region(3, 90 * 40, (80, 0, 40, 90), (100, 45), 180.0, 0.33),
        ]
        context = PageColorContext(
            segmentation=Segmentation(regions, labels, 1.0),
            semantic_labels=[("hair", 0.95), ("clothing", 0.92), ("background", 0.95)],
            identity_assignments={1: {"attribute": "hair", "lock_allowed": True},
                                  2: {"attribute": "clothing", "lock_allowed": True}},
            character_instances=[],
        )
        wash = apply_style_grade(result, source, get_style("light"), context=context)
        people = apply_style_grade(result, source, get_style("monochrome_people"), context=context)
        page = apply_style_grade(result, source, get_style("monochrome_page"), context=context)
        def chroma(img):
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
            a = lab[...,1] - 128.0
            b = lab[...,2] - 128.0
            return np.sqrt(a*a+b*b)
        wash_c = chroma(wash)
        people_c = chroma(people)
        page_c = chroma(page)
        self.assertLess(float(wash_c.mean()), float(people_c.mean()) * 0.78)
        self.assertGreater(float(page_c[:, 80:].mean()), float(people_c[:, 80:].mean()) * 1.3)
        self.assertGreater(float(people_c[:, :80].mean()), float(wash_c[:, :80].mean()) * 1.25)

    def test_post_only_pastel_uses_fast_simple_quality_path(self):
        from pipeline import colorize_page

        class FakeColorizer:
            def __init__(self):
                self.kwargs = None

            def colorize(self, image, **kwargs):
                self.kwargs = kwargs
                out = image.copy()
                out[..., 0] = 60
                out[..., 1] = 110
                out[..., 2] = 180
                return out

        image = np.full((96, 96, 3), 185, np.uint8)
        fake = FakeColorizer()
        with patch("pipeline.get_colorizer", return_value=fake), \
             patch("pipeline.build_page_context", side_effect=AssertionError("CLIP path called")):
            out = colorize_page(
                image, style_key="monochrome_people", quality_key="standard",
                character_memories=None, character_library=None,
                scene_palette=None)
        self.assertEqual(out.shape, image.shape)
        self.assertIsNotNone(fake.kwargs)
        self.assertFalse(fake.kwargs["per_panel"])
        self.assertFalse(fake.kwargs["tiled"])
        self.assertLessEqual(int(fake.kwargs["size"]), 640)
        self.assertLessEqual(int(fake.kwargs["denoise_sigma"]), 15)


    def test_post_only_pastel_does_not_start_guided_clip(self):
        from pipeline import colorize_page

        class FakeColorizer:
            def colorize(self, image, **_kwargs):
                out = image.copy()
                out[..., 0] = 80
                out[..., 1] = 130
                out[..., 2] = 190
                return out

        image = np.full((64, 64, 3), 180, np.uint8)
        with patch("pipeline.get_colorizer", return_value=FakeColorizer()),              patch("pipeline.build_page_context", side_effect=AssertionError("CLIP path called")):
            out = colorize_page(
                image, style_key="monochrome_people", quality_key="draft",
                character_memories=None, character_library=None,
                scene_palette=None)
        self.assertEqual(out.shape, image.shape)

if __name__ == "__main__":
    unittest.main()
