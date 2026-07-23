from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from core.character_consistency import apply_character_palette_lock
from core.character_library import (
    CharacterLibrary,
    CharacterProfile,
    _classify_clothing_part,
    _preferred_slot_index,
)
from core.page_color_context import PageColorContext
from core.region_segmenter import Region, Segmentation


class V59ClothingDiagnosticsTests(unittest.TestCase):
    def test_clothing_geometry_classifies_upper_lower_accessory(self):
        body = (10, 10, 100, 180)
        upper = Region(1, 800, (25, 45, 55, 35), (52, 62), 150.0, 0.04)
        lower = Region(2, 900, (24, 125, 58, 42), (53, 146), 150.0, 0.045)
        accessory = Region(3, 60, (50, 70, 6, 16), (53, 78), 150.0, 0.002)
        self.assertEqual(_classify_clothing_part(upper, body), "upper")
        self.assertEqual(_classify_clothing_part(lower, body), "lower")
        self.assertEqual(_classify_clothing_part(accessory, body), "accessory")

    def test_clothing_parts_map_to_distinct_slot_indices(self):
        slots = ["#223344", "#667788", "#aa3355"]
        self.assertEqual(_preferred_slot_index("upper", slots), 0)
        self.assertEqual(_preferred_slot_index("lower", slots), 1)
        self.assertEqual(_preferred_slot_index("accessory", slots), 2)
        self.assertEqual(_preferred_slot_index("accessory", slots[:2]), 1)


    def test_geometry_preferred_slot_overrides_nearest_colour(self):
        source = np.full((72, 100, 3), 155, np.uint8)
        result = np.full((72, 100, 3), 175, np.uint8)
        labels = np.zeros((72, 100), np.int32)
        labels[18:60, 20:78] = 7
        # Current patch starts reddish, but geometry marks it as lower clothing,
        # so slot 1 (blue) should win over nearest-colour slot 0 (red).
        result[18:60, 20:78] = (70, 75, 190)
        seg = Segmentation([], labels, 1.0)
        lib = CharacterLibrary()
        lib._last_segmentation = seg
        lib.last_assignments = {
            7: {
                "rgb": (190, 70, 70),
                "slot_rgbs": [(190, 70, 70), (60, 100, 210)],
                "preferred_slot_index": 1,
                "clothing_part": "lower",
                "attribute": "clothing",
                "char_id": 0,
                "lock_allowed": True,
                "semantic_confidence": 0.8,
                "match_score": 0.8,
                "margin": 0.12,
            }
        }
        out = apply_character_palette_lock(result, source, lib, strength=1.0)
        mean_bgr = out[28:50, 30:68].mean(axis=(0, 1)).astype(np.float32)
        # Blue target is RGB (60,100,210), i.e. BGR (210,100,60).
        blue_bgr = np.array([210, 100, 60], np.float32)
        red_bgr = np.array([70, 70, 190], np.float32)
        self.assertLess(np.linalg.norm(mean_bgr - blue_bgr),
                        np.linalg.norm(mean_bgr - red_bgr))

    def test_palette_schema_migrates_old_colors_into_slots(self):
        old = {
            "version": 4,
            "characters": [{
                "char_id": 0, "hair_tone": 100.0,
                "colors": {"hair": "#112233", "clothing": "#445566"},
            }],
        }
        lib = CharacterLibrary.from_dict(old)
        self.assertEqual(lib.to_dict()["version"], 5)
        self.assertEqual(lib.characters[0].color_slots["clothing"], ["#445566"])

    def test_diagnostic_rows_flags_actual_identity_drift(self):
        lib = CharacterLibrary()
        lib.characters = [CharacterProfile(
            0, 100.0,
            {"eyes": "#3366cc", "clothing": "#aa3344"},
            name="Alice",
            color_slots={"eyes": ["#3366cc"],
                         "clothing": ["#aa3344", "#224488"]},
        )]
        labels = np.zeros((80, 100), np.int32)
        labels[15:30, 18:34] = 2
        labels[36:68, 18:70] = 3
        seg = Segmentation([], labels, 1.0)
        context = PageColorContext(
            segmentation=seg,
            identity_assignments={
                2: {"rgb": (51, 102, 204), "attribute": "eyes",
                    "char_id": 0, "lock_allowed": True},
                3: {"rgb": (170, 51, 68), "slot_rgbs": [(170, 51, 68), (34, 68, 136)],
                    "preferred_slot_index": 1, "clothing_part": "lower",
                    "attribute": "clothing", "char_id": 0, "lock_allowed": True},
            },
        )
        result = np.full((80, 100, 3), 180, np.uint8)
        # Deliberately wrong colors: BGR pixels far from the target identities.
        result[15:30, 18:34] = (40, 210, 45)
        result[36:68, 18:70] = (30, 210, 210)
        rows = lib.diagnostic_rows(context, result_bgr=result)
        self.assertEqual(len(rows), 1)
        self.assertGreaterEqual(len(rows[0]["drift_alerts"]), 2)
        self.assertGreater(rows[0]["max_delta_e"], 20.0)
        self.assertEqual(rows[0]["part_counts"].get("lower"), 1)

    def test_ui_and_palette_manager_expose_swatches_alerts_and_slots(self):
        root = Path(__file__).resolve().parents[1]
        main = (root / "ui" / "main_window.py").read_text(encoding="utf-8")
        dialog = (root / "ui" / "character_dialog.py").read_text(encoding="utf-8")
        self.assertIn("background:{safe}", main)
        self.assertIn("character_diagnostics_alert", main)
        self.assertIn("Clothing Slots", dialog)
        self.assertIn("color_slots=slot_payload", dialog)


if __name__ == "__main__":
    unittest.main()
