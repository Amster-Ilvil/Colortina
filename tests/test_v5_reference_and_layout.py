from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from core.anime_face_detector import (
    _box_from_rotated,
    _box_to_rotated,
    detect_anime_faces,
)
from core.character_library import CharacterLibrary
from core.manual_reference import sample_hex_at


class V5ReferenceAndLayoutTests(unittest.TestCase):
    def _reference_image(self, hair_bgr=(35, 65, 125)):
        image = np.full((240, 200, 3), (230, 205, 190), np.uint8)
        # stylized head crop with colored hair ring and skin center
        cv2.ellipse(image, (100, 105), (72, 88), 0, 0, 360, hair_bgr, -1)
        cv2.ellipse(image, (100, 118), (44, 55), 0, 0, 360, (175, 205, 235), -1)
        cv2.circle(image, (82, 112), 7, (130, 80, 35), -1)
        cv2.circle(image, (118, 112), 7, (130, 80, 35), -1)
        return image

    def test_manual_reference_same_name_merges(self):
        lib = CharacterLibrary()
        image = self._reference_image()
        first = lib.add_manual_reference(
            image, (20, 12, 160, 205), name="Alice",
            colors={"hair": "#7d4123", "skin": "#edc9ae", "eyes": "#245a83"})
        second = lib.add_manual_reference(
            image, (18, 10, 164, 208), name="alice",
            colors={"hair": "#824729", "skin": "#ebc6aa", "eyes": "#285f88",
                    "clothing": "#3355aa"})
        self.assertEqual(first.char_id, second.char_id)
        self.assertEqual(len(lib.characters), 1)
        self.assertTrue(lib.characters[0].manual)
        self.assertEqual(lib.characters[0].reference_samples, 2)
        self.assertTrue(lib.characters[0].lineart_embedding)
        self.assertIn("hair", lib.characters[0].color_slots)
        self.assertIn("#3355aa", lib.characters[0].color_slots.get("clothing", []))

    def test_manual_reference_different_names_never_merge(self):
        lib = CharacterLibrary(merge_threshold=1.0)
        image = self._reference_image()
        lib.add_manual_reference(
            image, (20, 12, 160, 205), name="Alice",
            colors={"hair": "#7d4123"})
        lib.add_manual_reference(
            image, (20, 12, 160, 205), name="Beth",
            colors={"hair": "#7d4123"})
        self.assertEqual(len(lib.characters), 2)

    def test_manual_profile_roundtrip(self):
        lib = CharacterLibrary()
        image = self._reference_image()
        lib.add_manual_reference(
            image, (20, 12, 160, 205), name="Alice",
            colors={"hair": "#7d4123", "eyes": "#245a83"})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "people.ccpalette")
            lib.save(path)
            loaded = CharacterLibrary.load(path)
        self.assertEqual(loaded.characters[0].name, "Alice")
        self.assertTrue(loaded.characters[0].manual)
        self.assertEqual(loaded.characters[0].colors["eyes"], "#245a83")

    def test_robust_sample_hex(self):
        image = np.full((80, 80, 3), 255, np.uint8)
        image[25:55, 25:55] = (25, 75, 185)  # BGR -> reddish RGB
        value = sample_hex_at(image, 40, 40, radius_px=8)
        self.assertTrue(value.startswith("#"))
        r = int(value[1:3], 16)
        b = int(value[5:7], 16)
        self.assertGreater(r, b)

    def test_missing_anime_cascade_does_not_use_noisy_haar(self):
        page = np.full((200, 200, 3), 255, np.uint8)
        with patch("core.anime_face_detector.ensure_anime_face_cascade",
                   return_value=None), patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COLORTINA_ALLOW_GENERIC_HAAR", None)
            self.assertEqual(detect_anime_faces(page, allow_download=False), [])

    def test_rotation_box_roundtrip(self):
        shape = (300, 500)
        original = (42, 65, 120, 95)
        for rotation in (0, 90, 180, 270):
            rotated = _box_to_rotated(original, shape, rotation)
            restored = _box_from_rotated(rotated, shape, rotation)
            for a, b in zip(original, restored):
                self.assertLessEqual(abs(a - b), 1)

    def test_right_panel_is_tabbed_and_non_scrolling(self):
        source = (Path(__file__).resolve().parents[1] /
                  "ui" / "main_window.py").read_text(encoding="utf-8")
        start = source.index("    def _build_right_panel")
        end = source.index("\n    def ", start + 10)
        body = source[start:end]
        self.assertIn("QTabWidget", body)
        self.assertNotIn("QScrollArea", body)
        self.assertNotIn("setVerticalScrollBarPolicy", body)
        self.assertIn("_btn_manual_character", body)
        self.assertIn("right_tab_render", body)
        self.assertIn("right_tab_reference", body)
        self.assertIn("right_tab_edit", body)
        self.assertIn("right_tab_output", body)


    def test_reference_tab_has_character_diagnostics_panel(self):
        source = (Path(__file__).resolve().parents[1] /
                  "ui" / "main_window.py").read_text(encoding="utf-8")
        self.assertIn("character_diagnostics_group", source)
        self.assertIn("_character_diag_label", source)
        self.assertIn("_refresh_character_diagnostics", source)

    def test_responsive_density_has_no_scrollbars(self):
        source = (Path(__file__).resolve().parents[1] /
                  "ui" / "main_window.py").read_text(encoding="utf-8")
        self.assertIn("def _apply_responsive_density", source)
        self.assertIn("def resizeEvent", source)
        self.assertNotIn("QScrollArea", source.split("from PySide6.QtWidgets", 1)[1].split(")", 1)[0])


if __name__ == "__main__":
    unittest.main()
