import unittest

import cv2
import numpy as np

from core.custom_color_bias import apply_global_color_bias
from core.manual_edit import (
    apply_selection_edit, build_polygon_selection_mask, build_rect_selection_mask,
)


class TestV29SelectionAndBias(unittest.TestCase):
    def setUp(self):
        self.source = np.full((80, 80, 3), 255, np.uint8)
        cv2.rectangle(self.source, (10, 10), (69, 69), (220, 220, 220), -1)
        cv2.rectangle(self.source, (10, 10), (69, 69), (0, 0, 0), 2)
        self.result = np.full((80, 80, 3), (210, 190, 170), np.uint8)

    def test_polygon_selection_mask_builds(self):
        mask = build_polygon_selection_mask((80, 80), [(20, 20), (60, 20), (60, 60), (20, 60)])
        self.assertEqual(mask.shape, (80, 80))
        self.assertGreater(int(np.count_nonzero(mask)), 1200)

    def test_rect_selection_mask_builds(self):
        mask = build_rect_selection_mask((80, 80), 20, 20, 60, 60)
        self.assertEqual(mask[30, 30], 255)
        self.assertEqual(mask[5, 5], 0)

    def test_selection_edit_never_changes_outside_mask(self):
        mask = build_rect_selection_mask((80, 80), 20, 20, 40, 40)
        edited, base, used_mask, changed = apply_selection_edit(
            self.source, self.result.copy(), self.result.copy(), mask, '#ff6060', feather=1)
        self.assertTrue(changed)
        outside = used_mask == 0
        self.assertTrue(np.array_equal(edited[outside], self.result[outside]))
        self.assertTrue(np.array_equal(base[outside], self.result[outside]))

    def test_global_color_bias_preserves_paper_and_lines(self):
        result = self.result.copy()
        result[:8, :] = 255
        result[:, :8] = 255
        cv2.line(result, (0, 0), (79, 79), (0, 0, 0), 2)
        biased = apply_global_color_bias(result, self.source, (255, 120, 120), 0.8)
        # genuinely neutral white paper should stay effectively unchanged
        self.assertLessEqual(int(np.abs(biased[2, 78].astype(int) - result[2, 78].astype(int)).max()), 2)
        # a colourized interior pixel should change even though the source page
        # was white there before mc-v2 assigned colour
        self.assertGreater(int(np.abs(biased[30, 30].astype(int) - result[30, 30].astype(int)).max()), 0)


if __name__ == '__main__':
    unittest.main()
