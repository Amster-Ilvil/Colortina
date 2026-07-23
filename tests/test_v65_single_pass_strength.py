import unittest

import cv2
import numpy as np

from core.local_brush import apply_local_brush_recolor
from core.lineart_fill import lineart_region_recolor
from core.region_map import build_region_map


class TestV65SinglePassStrength(unittest.TestCase):
    @staticmethod
    def _ab(rgb):
        bgr = np.array([[[rgb[2], rgb[1], rgb[0]]]], dtype=np.uint8)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0, 1:3]

    def test_local_brush_one_stroke_moves_centre_close_to_target(self):
        source = np.full((80, 80, 3), 230, np.uint8)
        current = np.full((80, 80, 3), (190, 190, 190), np.uint8)
        target_rgb = (220, 70, 100)
        rm = build_region_map(source, gap_close=6)
        edited, alpha = apply_local_brush_recolor(
            source, current, 40, 40, 12, target_rgb,
            opacity=1.0, region_map=rm, gap_close=6)
        self.assertGreater(float(alpha[40, 40]), 0.85)
        before_lab = cv2.cvtColor(current, cv2.COLOR_BGR2LAB).astype(np.float32)
        after_lab = cv2.cvtColor(edited, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_ab = self._ab(target_rgb)
        before_d = float(np.linalg.norm(before_lab[40, 40, 1:3] - target_ab))
        after_d = float(np.linalg.norm(after_lab[40, 40, 1:3] - target_ab))
        self.assertLess(after_d, before_d * 0.30)

    def test_region_fill_one_click_has_high_minimum_coverage(self):
        source = np.full((90, 120, 3), 255, np.uint8)
        cv2.rectangle(source, (20, 20), (100, 70), (0, 0, 0), 2)
        current = np.full_like(source, (185, 185, 185))
        target_hex = '#dc4664'
        region_map = build_region_map(source, gap_close=6)
        edited, mask = lineart_region_recolor(
            source, current.copy(), 55, 45, target_hex,
            gap_close=6, mode='shading', feather=2, region_map=region_map)
        self.assertTrue(mask.any())
        lab = cv2.cvtColor(edited, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_rgb = (220, 70, 100)
        target_ab = self._ab(target_rgb)
        before_lab = cv2.cvtColor(current, cv2.COLOR_BGR2LAB).astype(np.float32)
        before_d = float(np.linalg.norm(before_lab[45, 55, 1:3] - target_ab))
        center_d = float(np.linalg.norm(lab[45, 55, 1:3] - target_ab))
        edge_d = float(np.linalg.norm(lab[24, 24, 1:3] - target_ab))
        self.assertLess(center_d, before_d * 0.28)
        self.assertLess(edge_d, before_d * 0.32)


if __name__ == '__main__':
    unittest.main()
