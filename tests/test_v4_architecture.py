from __future__ import annotations

import tempfile
import unittest

import cv2
import numpy as np

from core.character_consistency import apply_character_palette_lock
from core.character_library import CharacterLibrary, CharacterProfile
from core.hint_artifact import detect_hint_blobs
from core.evaluation import identity_delta_metrics, line_bleed_ratio
from core.hint_composer import HintComposer, degrade_for_retry
from core.hint_rasterizer import rasterize_hint_specs
from core.hint_spec import HintSpec
from core.ml_colorizer import MangaColorizer
from core.project_store import load_project
from core.region_map import build_region_map
from core.region_segmenter import Region, Segmentation
from core.style_analyzer import StyleAnalyzer


def boxed_page() -> np.ndarray:
    image = np.full((100, 140, 3), 255, np.uint8)
    cv2.rectangle(image, (10, 10), (65, 90), (0, 0, 0), 3)
    cv2.rectangle(image, (75, 10), (130, 90), (0, 0, 0), 3)
    image[14:88, 14:63] = 145
    image[14:88, 78:128] = 160
    return image


class V4ArchitectureTests(unittest.TestCase):
    def test_hint_spec_roundtrip_keeps_metadata(self):
        spec = HintSpec(0.2, 0.3, (1, 2, 3), 0.004, strength=0.6,
                        source="character_identity", region_id=8,
                        semantic="hair", character_id=4, confidence=0.8)
        loaded = HintSpec.from_dict(spec.to_dict())
        self.assertEqual(loaded.source, "character_identity")
        self.assertEqual(loaded.character_id, 4)
        self.assertAlmostEqual(loaded.effective_strength, 0.48)

    def test_style_only_and_low_conf_identity_are_dropped(self):
        hints = [
            HintSpec(0.2, 0.2, (255, 0, 0), source="style_only"),
            HintSpec(0.4, 0.4, (0, 255, 0), source="character_identity",
                     confidence=0.2),
            HintSpec(0.6, 0.6, (0, 0, 255), source="character_identity",
                     confidence=0.9),
        ]
        out = HintComposer().compose(hints, [])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].rgb, (0, 0, 255))

    def test_soft_rasterizer_has_fractional_mask_and_stays_in_region(self):
        image = boxed_page()
        rm = build_region_map(image, gap_close=2)
        spec = HintSpec(0.25, 0.5, (220, 30, 30), 0.015,
                        strength=0.55, source="character_identity",
                        confidence=0.9)
        hint, mask = rasterize_hint_specs(
            image.shape[0], image.shape[1], 128, [spec],
            label_map=rm, page_gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        self.assertTrue(np.any((mask > 0.0) & (mask < 1.0)))
        self.assertLessEqual(float(mask.max()), 0.55)
        self.assertEqual(hint.shape[:2], mask.shape)

    def test_low_confidence_lock_is_skipped(self):
        source = np.full((50, 50, 3), 140, np.uint8)
        result = np.full((50, 50, 3), (40, 180, 40), np.uint8)
        labels = np.zeros((50, 50), np.int32)
        labels[8:42, 8:42] = 3
        seg = Segmentation([], labels, 1.0)
        assignments = {3: {
            "rgb": (220, 30, 40), "attribute": "hair", "char_id": 1,
            "lock_allowed": False, "match_score": 0.8, "margin": 0.01,
            "semantic_confidence": 0.9,
        }}
        out = apply_character_palette_lock(
            result, source, strength=1.0,
            assignments=assignments, segmentation=seg)
        np.testing.assert_array_equal(out, result)

    def test_top2_ambiguity_disables_lock(self):
        lib = CharacterLibrary(match_threshold=0.9, min_margin=0.05)
        base = dict(hair_tone=110.0, hair_hist=[1 / 12] * 12,
                    hair_aspect=1.0, hair_area_frac=0.03,
                    appearance_embedding=[])
        lib.characters = [
            CharacterProfile(0, colors={"hair": "#aa3344"}, **base),
            CharacterProfile(1, colors={"hair": "#3355aa"}, **base),
        ]
        region = Region(1, 400, (20, 15, 30, 25), (35, 28), 110.0, 0.03)
        labels_map = np.zeros((80, 80), np.int32)
        labels_map[15:40, 20:50] = 1
        seg = Segmentation([region], labels_map, 1.0)
        gray = np.full((80, 80), 110, np.uint8)
        out = lib.assign_page_detailed(
            [region], [("hair", 0.9)], segmentation=seg, gray_page=gray)
        self.assertIn(1, out)
        self.assertFalse(out[1]["lock_allowed"])
        self.assertAlmostEqual(out[1]["margin"], 0.0)

    def test_page_rule_can_explicitly_disable_identity_lock(self):
        lib = CharacterLibrary(match_threshold=0.9)
        lib.characters = [CharacterProfile(
            0, hair_tone=110.0, colors={"hair": "#aa3344"},
            hair_hist=[1 / 12] * 12, hair_aspect=1.0,
            hair_area_frac=0.03)]
        region = Region(1, 400, (20, 15, 30, 25), (35, 28), 110.0, 0.03)
        labels_map = np.zeros((80, 80), np.int32)
        labels_map[15:40, 20:50] = 1
        seg = Segmentation([region], labels_map, 1.0)
        gray = np.full((80, 80), 110, np.uint8)
        out = lib.assign_page_detailed(
            [region], [("hair", 0.9)], segmentation=seg, gray_page=gray,
            forced_matches={1: -1})
        self.assertEqual(out, {})
        self.assertFalse(lib.last_instances[0].lock_allowed)

    def test_explicit_character_binding_overrides_ambiguity_gate(self):
        lib = CharacterLibrary(match_threshold=0.9, min_margin=0.5)
        base = dict(hair_tone=110.0, hair_hist=[1 / 12] * 12,
                    hair_aspect=1.0, hair_area_frac=0.03,
                    appearance_embedding=[])
        lib.characters = [
            CharacterProfile(0, colors={"hair": "#aa3344"}, **base),
            CharacterProfile(1, colors={"hair": "#3355aa"}, **base),
        ]
        region = Region(1, 400, (20, 15, 30, 25), (35, 28), 110.0, 0.03)
        labels_map = np.zeros((80, 80), np.int32)
        labels_map[15:40, 20:50] = 1
        seg = Segmentation([region], labels_map, 1.0)
        gray = np.full((80, 80), 110, np.uint8)
        out = lib.assign_page_detailed(
            [region], [("hair", 0.9)], segmentation=seg, gray_page=gray,
            forced_matches={1: 1})
        self.assertEqual(out[1]["char_id"], 1)
        self.assertTrue(out[1]["lock_allowed"])
        self.assertTrue(out[1]["forced"])

    def test_kmeans_fallback_does_not_invent_hair_semantics(self):
        image = np.zeros((80, 120, 3), np.uint8)
        image[:, :40] = (30, 30, 210)
        image[:, 40:80] = (30, 180, 30)
        image[:, 80:] = (210, 50, 30)
        desc = StyleAnalyzer().analyze(image, classifier=None)
        self.assertEqual(desc.region_samples, {})
        self.assertEqual(desc.hair.warm_bias, 0.0)
        self.assertEqual(desc.skin.warm_bias, 0.0)

    def test_structured_hint_crop_remap_keeps_source(self):
        spec = HintSpec(0.75, 0.25, (1, 2, 3), 0.02,
                        source="character_identity", character_id=2)
        remapped = MangaColorizer._hints_for_crop(
            [spec], 100, 0, 100, 100, 200, 100)
        self.assertEqual(len(remapped), 1)
        self.assertIsInstance(remapped[0], HintSpec)
        self.assertAlmostEqual(remapped[0].x_norm, 0.5)
        self.assertEqual(remapped[0].character_id, 2)

    def test_retry_policy_keeps_manual_and_identity_only(self):
        hints = [
            HintSpec(0.1, 0.1, (1, 2, 3), source="manual"),
            HintSpec(0.2, 0.2, (4, 5, 6), source="character_identity"),
            HintSpec(0.3, 0.3, (7, 8, 9), source="scene_palette"),
        ]
        out = degrade_for_retry(hints)
        self.assertEqual([h.source for h in out], ["manual", "character_identity"])
        self.assertLess(out[1].radius_norm, hints[1].radius_norm)

    def test_hint_blob_detector_finds_regular_color_chip(self):
        image = np.full((100, 100, 3), (120, 120, 120), np.uint8)
        cv2.circle(image, (50, 50), 5, (0, 0, 255), -1)
        spec = HintSpec(0.5, 0.5, (255, 0, 0), 0.05,
                        source="auto_instance")
        report = detect_hint_blobs(image, [spec], threshold=8.0)
        self.assertGreater(report.score, 8.0)
        self.assertGreaterEqual(report.suspicious, 1)

    def test_identity_metrics_measure_consistency_and_separation(self):
        report = identity_delta_metrics([
            {"character": "a", "attribute": "hair", "rgb": (200, 40, 40)},
            {"character": "a", "attribute": "hair", "rgb": (198, 43, 42)},
            {"character": "b", "attribute": "hair", "rgb": (40, 70, 200)},
        ])
        self.assertLess(report["same_character_delta_e_mean"], 4.0)
        self.assertGreater(report["different_character_delta_e_min"], 20.0)

    def test_line_bleed_ratio_detects_chromatic_ink(self):
        source = np.full((40, 40, 3), 220, np.uint8)
        source[:, 18:22] = 0
        clean = np.full((40, 40, 3), 180, np.uint8)
        dirty = clean.copy()
        dirty[:, 18:22] = (255, 0, 0)
        self.assertLess(line_bleed_ratio(source, clean),
                        line_bleed_ratio(source, dirty))

    def test_project_v1_migration(self):
        payload = '{"version":1,"pages":[],"settings":{}}'
        with tempfile.NamedTemporaryFile("w", suffix=".ccproject", delete=False,
                                         encoding="utf-8") as f:
            f.write(payload)
            path = f.name
        try:
            loaded = load_project(path)
            self.assertTrue(loaded["migration_log"])
            self.assertIsNone(loaded["scene_palette"])
        finally:
            import os
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
