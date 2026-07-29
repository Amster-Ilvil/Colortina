import unittest

import cv2
import numpy as np

from core.custom_color_bias import apply_global_color_bias
from core.manual_edit import build_closed_region_selection_mask, build_rect_selection_mask


class TestV284ClosedSelectionAndScope(unittest.TestCase):
    def setUp(self):
        self.source = np.full((80, 80, 3), 255, np.uint8)
        cv2.rectangle(self.source, (10, 10), (69, 69), (0, 0, 0), 2)
        cv2.line(self.source, (40, 10), (40, 69), (0, 0, 0), 2)
        self.result = np.full((80, 80, 3), (170, 185, 210), np.uint8)

    def test_closed_region_selection_excludes_boundary_connected_area(self):
        sel = build_rect_selection_mask((80, 80), 5, 5, 50, 50)
        closed = build_closed_region_selection_mask(self.source, sel)
        self.assertEqual(closed[20, 20], 0)
        self.assertEqual(closed[20, 45], 0)

        sel2 = build_rect_selection_mask((80, 80), 10, 10, 40, 69)
        closed2 = build_closed_region_selection_mask(self.source, sel2)
        self.assertGreater(int(np.count_nonzero(closed2)), 200)
        self.assertEqual(closed2[20, 20], 255)

    def test_color_bias_scope_changes_where_effect_is_stronger(self):
        # create denser 'character-like' content in centre and plainer sides background
        src = np.full((120, 120, 3), 245, np.uint8)
        cv2.rectangle(src, (10, 10), (109, 109), (220, 220, 220), -1)
        for x in range(40, 81, 8):
            cv2.line(src, (x, 25), (x, 95), (0, 0, 0), 1)
        for y in range(25, 96, 8):
            cv2.line(src, (40, y), (80, y), (0, 0, 0), 1)
        result = np.full((120, 120, 3), (180, 190, 205), np.uint8)
        c_img = apply_global_color_bias(result, src, (255, 120, 120), 0.8, 'characters')
        b_img = apply_global_color_bias(result, src, (255, 120, 120), 0.8, 'background')
        # centre should shift more for character mode than background mode
        centre_char = int(np.abs(c_img[60, 60].astype(int) - result[60, 60].astype(int)).sum())
        centre_bg = int(np.abs(b_img[60, 60].astype(int) - result[60, 60].astype(int)).sum())
        side_char = int(np.abs(c_img[10, 10].astype(int) - result[10, 10].astype(int)).sum())
        side_bg = int(np.abs(b_img[10, 10].astype(int) - result[10, 10].astype(int)).sum())
        self.assertGreater(centre_char, centre_bg)
        self.assertGreater(side_bg, side_char)


if __name__ == '__main__':
    unittest.main()
