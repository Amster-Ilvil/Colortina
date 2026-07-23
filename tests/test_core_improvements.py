from __future__ import annotations

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from core.character_library import CharacterLibrary, CharacterProfile
from core.hint_manager import HintManager
from core.lineart_fill import lineart_region_recolor
from core.ml_colorizer import MangaColorizer
from core.project_store import load_project, save_project
from core.region_map import build_region_map
from core.style_analyzer import StyleAnalyzer
from core.style_descriptor import StyleDescriptor, RegionDescriptor
from core.style_engine import StyleEngine, _descriptor_to_profile


def boxed_page() -> np.ndarray:
    image = np.full((120, 180, 3), 255, np.uint8)
    cv2.rectangle(image, (10, 10), (80, 110), (0, 0, 0), 3)
    cv2.rectangle(image, (100, 10), (170, 110), (0, 0, 0), 3)
    # Add tone variation so manual tier selection has all three levels.
    image[15:45, 15:75] = 210
    image[45:78, 15:75] = 135
    image[78:105, 15:75] = 70
    return image


class CoreImprovementTests(unittest.TestCase):
    def test_manual_hint_stays_local_and_suppresses_only_nearby_auto(self):
        image = boxed_page()
        hm = HintManager()
        rm = hm.bind_source_image(image, gap_close=2)
        left = rm.region_at(30, 50)
        right = rm.region_at(130, 50)
        self.assertNotEqual(left, right)
        hm.set_auto_hints([
            (30 / 180, 50 / 120, (0, 0, 255), 0.01),
            (130 / 180, 50 / 120, (0, 255, 0), 0.01),
        ])
        hm.add_manual_hint(30 / 180, 50 / 120, (255, 0, 0), 0.02)
        specs = hm.merge_specs(image_bgr=image)
        colors = [p.rgb for p in specs]
        self.assertIn((0, 255, 0), colors)
        self.assertNotIn((0, 0, 255), colors)
        self.assertIn((255, 0, 0), colors)
        # A normal brush dab stays one local hint; it is no longer expanded
        # into highlight/mid/shadow points across the whole connected face.
        local = [p for p in specs if p.source == "manual"]
        self.assertEqual(len(local), 1)
        self.assertEqual(local[0].region_id, left)

    def test_manual_strength_zero_preserves_auto(self):
        image = boxed_page()
        hm = HintManager()
        hm.bind_source_image(image, gap_close=2)
        auto = [(30 / 180, 50 / 120, (12, 34, 56), 0.01)]
        hm.set_auto_hints(auto)
        hm.add_manual_hint(30 / 180, 50 / 120, (255, 0, 0), 0.02)
        points = hm.merge(image_bgr=image, manual_strength=0.0)
        self.assertEqual(points, auto)

    def test_region_fill_does_not_change_outside_mask(self):
        original = boxed_page()
        result = np.full_like(original, (160, 120, 80))
        before = result.copy()
        region_map = build_region_map(original, gap_close=2)
        changed, mask = lineart_region_recolor(
            original, result, 30, 55, "#e04080", gap_close=2,
            feather=7, region_map=region_map)
        outside = mask == 0
        self.assertTrue(np.array_equal(changed[outside], before[outside]))
        self.assertTrue(np.any(changed[mask > 0] != before[mask > 0]))


    def test_region_fill_refines_leaky_gap_connected_region(self):
        original = np.full((120, 140, 3), 255, np.uint8)
        cv2.rectangle(original, (10, 10), (130, 110), (0, 0, 0), 2)
        cv2.line(original, (70, 12), (70, 56), (118, 118, 118), 2)
        cv2.line(original, (70, 64), (70, 108), (118, 118, 118), 2)
        original[14:108, 14:68] = 180
        original[14:108, 72:126] = 225
        result = np.full_like(original, 235)
        result[:, :70] = (120, 150, 220)
        result[:, 70:] = (80, 200, 90)
        before = result.copy()
        region_map = build_region_map(original, gap_close=0)
        # The separator gap keeps the raw connected component suspiciously large.
        changed, mask = lineart_region_recolor(
            original, result, 35, 60, "#d04060", gap_close=0,
            feather=5, region_map=region_map)
        left_delta = np.abs(changed[:, :68].astype(np.int16) - before[:, :68].astype(np.int16)).mean()
        right_delta = np.abs(changed[:, 72:].astype(np.int16) - before[:, 72:].astype(np.int16)).mean()
        self.assertGreater(float(left_delta), 2.5)
        self.assertLess(float(right_delta), 0.8)

    def test_tiled_hint_remap(self):
        hints = [(0.75, 0.25, (1, 2, 3), 0.02)]
        remapped = MangaColorizer._hints_for_crop(
            hints, 100, 0, 100, 100, 200, 100)
        self.assertIsNotNone(remapped)
        x, y, color, radius = remapped[0]
        self.assertAlmostEqual(x, 0.5)
        self.assertAlmostEqual(y, 0.25)
        self.assertEqual(color, (1, 2, 3))
        self.assertAlmostEqual(radius, 0.04)

    def test_external_v2_style_load_keeps_descriptor(self):
        desc = StyleDescriptor(
            name="V2", global_warm_cool=7.5,
            hair=RegionDescriptor(shadow_hue_rotate=5.0),
            region_samples={"hair": 100})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "v2.ccstyle")
            desc.save(path)
            profile = StyleEngine(tmp).load_style(path)
            self.assertIsNotNone(profile._descriptor)
            self.assertEqual(profile._descriptor.global_warm_cool, 7.5)
            self.assertEqual(profile._descriptor.hair.shadow_hue_rotate, 5.0)

    def test_grayscale_reference_is_neutral(self):
        gray = np.tile(np.arange(128, dtype=np.uint8), (128, 1))
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        desc = StyleAnalyzer().analyze(bgr, name="gray", classifier=None)
        self.assertEqual(desc.temperature, "neutral")
        self.assertEqual(desc.saturation, 0.0)

    def test_project_roundtrip(self):
        image = boxed_page()
        hm = HintManager()
        hm.bind_source_image(image, gap_close=2)
        hm.add_manual_hint(0.2, 0.4, (10, 20, 30), 0.01)
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "page.png")
            cv2.imwrite(source, image)
            state = SimpleNamespace(
                path=source, hint_manager=hm,
                ai_result_bgr=image.copy(), result_bgr=image.copy(),
                forced_character_matches={7: -1},
                pipeline_diagnostics={"matched": 2, "hint_retry": True})
            desc = StyleDescriptor(name="ProjectStyle", global_warm_cool=2.0)
            profile = _descriptor_to_profile(desc)
            profile._descriptor = desc
            lib = CharacterLibrary()
            lib.characters.append(CharacterProfile(
                char_id=0, hair_tone=100, colors={"hair": "#112233"}))
            project = save_project(
                os.path.join(tmp, "book.ccproject"), pages=[state],
                style_profile=profile, character_library=lib,
                settings={"quality_key": "ultra"})
            loaded = load_project(project)
            self.assertEqual(len(loaded["pages"]), 1)
            self.assertEqual(len(loaded["pages"][0]["hint_manager"].manual_hints), 1)
            self.assertEqual(loaded["style_profile"]._descriptor.global_warm_cool, 2.0)
            self.assertEqual(loaded["character_library"].characters[0].colors["hair"], "#112233")
            self.assertEqual(loaded["settings"]["quality_key"], "ultra")
            self.assertEqual(loaded["pages"][0]["forced_character_matches"], {7: -1})
            self.assertEqual(loaded["pages"][0]["diagnostics"]["matched"], 2)
            self.assertTrue(loaded["pages"][0]["diagnostics_file"])

    def test_reference_point_sampling_ignores_white_paper(self):
        from core.reference_points import sample_reference_rgb
        image = np.full((40, 40, 3), 255, np.uint8)
        image[14:26, 14:26] = (30, 80, 180)  # BGR -> RGB about (180,80,30)
        rgb = sample_reference_rgb(image, 0.5, 0.5, radius_px=10)
        self.assertGreater(rgb[0], rgb[1])
        self.assertGreater(rgb[1], rgb[2])

    def test_quality_score_flags_colored_ink(self):
        from core.quality_score import assess_colorization
        original = np.full((80, 80, 3), 220, np.uint8)
        original[:, 38:42] = 0
        result = np.full_like(original, (160, 100, 80))
        result[:, 38:42] = (255, 0, 0)
        report = assess_colorization(original, result)
        self.assertTrue(report.line_bleed_detected)
        self.assertLess(report.score, 100)


if __name__ == "__main__":
    unittest.main()
