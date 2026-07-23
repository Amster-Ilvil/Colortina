import unittest
import numpy as np
import cv2

from core.character_library import CharacterLibrary, CharacterProfile
from core.page_color_context import PageColorContext
from core.region_segmenter import Region, Segmentation


class IdentityLearningTests(unittest.TestCase):
    def test_library_learns_clothing_variant_from_colorized_page(self):
        lib = CharacterLibrary()
        lib.characters = [CharacterProfile(
            char_id=0,
            hair_tone=110.0,
            colors={"hair": "#663344", "clothing": "#4060c0"},
            color_slots={"hair": ["#663344"], "clothing": ["#4060c0"]},
        )]
        labels = np.zeros((50, 50), np.int32)
        labels[10:40, 10:40] = 5
        seg = Segmentation([Region(5, 1, (10, 10, 30, 30), (25, 25), 150.0, 0.3)], labels, 1.0)
        page = PageColorContext(
            segmentation=seg,
            semantic_labels=[("clothing", 0.88)],
            character_instances=[],
            identity_assignments={
                5: {
                    "char_id": 0,
                    "attribute": "clothing",
                    "lock_allowed": True,
                    "semantic_confidence": 0.88,
                    "match_score": 0.93,
                    "margin": 0.12,
                    "preferred_slot_index": 1,
                }
            },
            hints=[], diagnostics={}
        )
        result = np.full((50, 50, 3), 180, np.uint8)
        result[10:40, 10:40] = (70, 165, 85)  # green-ish BGR
        changed = lib.learn_from_colorized_page(page, result, strength=1.0)
        self.assertGreaterEqual(changed, 1)
        self.assertIn("#55a546", lib.characters[0].color_slots["clothing"])
        self.assertEqual(lib.characters[0].colors["clothing"], "#55a546")


if __name__ == "__main__":
    unittest.main()
