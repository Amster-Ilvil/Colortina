import tempfile
import unittest

import cv2
import numpy as np

from core.character_consistency import apply_character_palette_lock
from core.character_library import CharacterLibrary, CharacterProfile
from core.reference_style import apply_reference_style
from core.region_segmenter import Region, Segmentation
from core.style_descriptor import StyleDescriptor
from core.style_engine import StyleEngine, _descriptor_to_profile


class ReferenceConsistencyTests(unittest.TestCase):
    def test_reference_style_changes_visible_chroma(self):
        source = np.full((80, 80, 3), 150, np.uint8)
        result = np.full((80, 80, 3), (70, 150, 70), np.uint8)  # green BGR
        desc = StyleDescriptor(
            name="Blue rendering",
            global_warm_cool=-8.0,
            style_scope={"character_rendering": True,
                         "background_rendering": True,
                         "global_ambience": 0.8},
            reference_palette=["#315fc8", "#89a9ef"],
        )
        out = apply_reference_style(result, source, desc, strength=1.0)
        # Reference ambience should still nudge the page in the blue direction
        # without forcing a heavy global recolour.
        self.assertGreaterEqual(float(out[..., 0].mean()), float(result[..., 0].mean()))
        self.assertGreater(float(np.abs(out.astype(np.int16) - result.astype(np.int16)).mean()), 0.05)

    def test_v3_reference_style_roundtrip_keeps_signature(self):
        desc = StyleDescriptor(
            name="Roundtrip", reference_lab_mean=[130.0, 140.0, 110.0],
            reference_lab_std=[25.0, 12.0, 15.0],
            reference_palette=["#aa3344", "#3355aa"],
        )
        profile = _descriptor_to_profile(desc)
        profile._descriptor = desc
        with tempfile.TemporaryDirectory() as tmp:
            engine = StyleEngine(tmp)
            path = engine.save_style(profile)
            loaded = engine.load_style(path)
            loaded_desc = loaded.get_descriptor()
        self.assertEqual(loaded_desc.reference_palette, desc.reference_palette)
        self.assertEqual(loaded_desc.reference_lab_mean, desc.reference_lab_mean)

    def test_character_palette_lock_enforces_identity_inside_only(self):
        source = np.full((60, 60, 3), 140, np.uint8)
        source[:5] = 255
        result = np.full((60, 60, 3), (60, 180, 60), np.uint8)  # green BGR
        labels = np.zeros((60, 60), np.int32)
        labels[10:50, 10:50] = 7
        seg = Segmentation([], labels, 1.0)
        lib = CharacterLibrary()
        lib._last_segmentation = seg
        lib.last_assignments = {
            7: {"rgb": (210, 45, 55), "attribute": "hair", "char_id": 0, "distance": 0.0}
        }
        out = apply_character_palette_lock(result, source, lib, strength=1.0)
        center = out[30, 30]
        outside = out[7, 7]
        self.assertGreater(int(center[2]), int(center[1]))  # red dominates green
        np.testing.assert_array_equal(outside, result[7, 7])


    def test_clothing_palette_lock_reduces_outfit_hue_drift(self):
        source = np.full((80, 80, 3), 150, np.uint8)
        result = np.full((80, 80, 3), 180, np.uint8)
        labels = np.zeros((80, 80), np.int32)
        labels[12:68, 18:62] = 5
        # Same outfit region with noticeably different upper/lower hues.
        result[12:40, 18:62] = (170, 180, 80)
        result[40:68, 18:62] = (70, 130, 190)
        seg = Segmentation([], labels, 1.0)
        lib = CharacterLibrary()
        lib._last_segmentation = seg
        lib.last_assignments = {
            5: {"rgb": (70, 120, 215), "attribute": "clothing", "char_id": 0, "distance": 0.0}
        }
        before_gap = float(np.linalg.norm(
            result[20:35, 24:56].mean(axis=(0, 1)).astype(np.float32) -
            result[46:61, 24:56].mean(axis=(0, 1)).astype(np.float32)))
        out = apply_character_palette_lock(result, source, lib, strength=1.0)
        after_gap = float(np.linalg.norm(
            out[20:35, 24:56].mean(axis=(0, 1)).astype(np.float32) -
            out[46:61, 24:56].mean(axis=(0, 1)).astype(np.float32)))
        self.assertLess(after_gap, before_gap * 0.55)


    def test_eye_palette_lock_tightens_two_iris_regions_toward_same_target(self):
        source = np.full((64, 96, 3), 145, np.uint8)
        result = np.full((64, 96, 3), 180, np.uint8)
        labels = np.zeros((64, 96), np.int32)
        labels[20:32, 20:32] = 3
        labels[20:32, 52:64] = 4
        result[20:32, 20:32] = (180, 70, 55)
        result[20:32, 52:64] = (65, 115, 180)
        seg = Segmentation([], labels, 1.0)
        lib = CharacterLibrary()
        lib._last_segmentation = seg
        lib.last_assignments = {
            3: {"rgb": (60, 150, 225), "attribute": "eyes", "char_id": 0,
                "lock_allowed": True, "semantic_confidence": 0.28, "match_score": 0.40, "margin": 0.03},
            4: {"rgb": (60, 150, 225), "attribute": "eyes", "char_id": 0,
                "lock_allowed": True, "semantic_confidence": 0.28, "match_score": 0.40, "margin": 0.03},
        }
        before_gap = float(np.linalg.norm(
            result[23:29, 23:29].mean(axis=(0, 1)).astype(np.float32) -
            result[23:29, 55:61].mean(axis=(0, 1)).astype(np.float32)))
        out = apply_character_palette_lock(result, source, lib, strength=1.0)
        after_gap = float(np.linalg.norm(
            out[23:29, 23:29].mean(axis=(0, 1)).astype(np.float32) -
            out[23:29, 55:61].mean(axis=(0, 1)).astype(np.float32)))
        self.assertLess(after_gap, before_gap * 0.55)


    def test_clothing_slot_palette_chooses_nearest_variant(self):
        source = np.full((72, 120, 3), 160, np.uint8)
        result = np.full((72, 120, 3), 170, np.uint8)
        labels = np.zeros((72, 120), np.int32)
        labels[18:56, 22:52] = 5
        # green-ish clothing patch should choose the green slot, not blue primary
        result[18:56, 22:52] = (70, 150, 80)
        seg = Segmentation([], labels, 1.0)
        lib = CharacterLibrary()
        lib._last_segmentation = seg
        lib.last_assignments = {
            5: {"rgb": (40, 90, 190), "slot_rgbs": [(40, 90, 190), (85, 165, 85)],
                "attribute": "clothing", "char_id": 0, "lock_allowed": True,
                "semantic_confidence": 0.55, "match_score": 0.72, "margin": 0.10}
        }
        out = apply_character_palette_lock(result, source, lib, strength=1.0)
        before = result[24:50, 28:46].mean(axis=(0, 1)).astype(np.float32)
        after = out[24:50, 28:46].mean(axis=(0, 1)).astype(np.float32)
        blue_target = np.array([190, 90, 40], np.float32)
        green_target = np.array([85, 165, 85], np.float32)
        self.assertLess(np.linalg.norm(after - green_target), np.linalg.norm(after - blue_target))
        self.assertGreater(after[1], max(after[0], after[2]))

    def test_duplicate_reference_profiles_merge(self):
        lib = CharacterLibrary(merge_threshold=0.3)
        base = CharacterProfile(
            -1, 100.0, {"hair": "#552233", "skin": "#deb39e"},
            hair_hist=[0.0, 0.1, 0.4, 0.5] + [0.0] * 8,
            hair_aspect=1.2, hair_area_frac=0.03,
            appearance_embedding=[1.0, 0.0, 0.0],
        )
        second = CharacterProfile(
            -1, 104.0, {"hair": "#5a2638", "eyes": "#4c79aa"},
            hair_hist=[0.0, 0.1, 0.42, 0.48] + [0.0] * 8,
            hair_aspect=1.18, hair_area_frac=0.031,
            appearance_embedding=[0.99, 0.02, 0.0],
        )
        self.assertTrue(lib._upsert_profile(base))
        self.assertFalse(lib._upsert_profile(second))
        self.assertEqual(len(lib.characters), 1)
        self.assertIn("eyes", lib.characters[0].colors)
        self.assertEqual(lib.characters[0].reference_samples, 2)

    def test_same_character_can_match_multiple_panels(self):
        lib = CharacterLibrary(match_threshold=0.9)
        lib.characters = [CharacterProfile(
            char_id=0, hair_tone=110.0, colors={"hair": "#7b3048"},
            hair_hist=[1 / 12] * 12, hair_aspect=1.0, hair_area_frac=0.02,
        )]
        r1 = Region(1, 100, (10, 10, 20, 20), (20, 20), 110.0, 0.02)
        r2 = Region(2, 100, (70, 10, 20, 20), (80, 20), 112.0, 0.02)
        labels_map = np.zeros((100, 100), np.int32)
        labels_map[10:30, 10:30] = 1
        labels_map[10:30, 70:90] = 2
        gray = np.full((100, 100), 110, np.uint8)
        seg = Segmentation([r1, r2], labels_map, 1.0)
        detailed = lib.assign_page_detailed(
            [r1, r2], [("hair", 0.9), ("hair", 0.9)],
            segmentation=seg, gray_page=gray)
        self.assertEqual(set(detailed), {1, 2})
        self.assertEqual(detailed[1]["char_id"], 0)
        self.assertEqual(detailed[2]["char_id"], 0)


    def test_reference_style_preserves_colour_separation(self):
        source = np.full((90, 120, 3), 150, np.uint8)
        result = np.full((90, 120, 3), 140, np.uint8)
        result[:, :40] = (40, 40, 200)   # red-ish in BGR
        result[:, 40:80] = (40, 180, 40) # green-ish
        result[:, 80:] = (200, 60, 40)   # blue-ish
        desc = StyleDescriptor(
            name="Warm ref",
            reference_lab_mean=[150.0, 140.0, 150.0],
            reference_lab_std=[22.0, 12.0, 14.0],
            reference_palette=["#d47a5a", "#e7b07c"],
        )
        out = apply_reference_style(result, source, desc, strength=1.0)
        means = [out[:, s:e].mean(axis=(0, 1)) for s, e in ((0, 40), (40, 80), (80, 120))]
        distances = [float(np.linalg.norm(means[i] - means[j])) for i in range(3) for j in range(i + 1, 3)]
        self.assertTrue(all(d > 20.0 for d in distances))

    def test_different_reference_characters_do_not_merge_by_shape_only(self):
        lib = CharacterLibrary(merge_threshold=0.3)
        a = CharacterProfile(
            -1, 100.0, {"hair": "#552233", "skin": "#deb39e"},
            hair_hist=[0.0, 0.1, 0.4, 0.5] + [0.0] * 8,
            hair_aspect=1.2, hair_area_frac=0.03,
            appearance_embedding=[1.0, 0.0, 0.0],
        )
        b = CharacterProfile(
            -1, 101.0, {"hair": "#3366cc", "skin": "#e0ba9f"},
            hair_hist=[0.0, 0.1, 0.39, 0.51] + [0.0] * 8,
            hair_aspect=1.18, hair_area_frac=0.031,
            appearance_embedding=[0.99, 0.01, 0.0],
        )
        self.assertTrue(lib._upsert_profile(a))
        self.assertTrue(lib._upsert_profile(b))
        self.assertEqual(len(lib.characters), 2)

if __name__ == "__main__":
    unittest.main()
