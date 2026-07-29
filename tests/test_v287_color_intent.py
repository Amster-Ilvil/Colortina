import unittest

import cv2
import numpy as np

from core.custom_color_bias import apply_global_color_bias
from core.manual_edit import (
    apply_selection_edit,
    build_polygon_selection_mask,
    build_rect_selection_mask,
)


def _hue_degrees(bgr_pixel) -> float:
    px = np.asarray(bgr_pixel, dtype=np.float32).reshape(1, 1, 3) / 255.0
    return float(cv2.cvtColor(px, cv2.COLOR_BGR2HLS)[0, 0, 0])


def _hue_distance(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


class TestV287ColorIntent(unittest.TestCase):
    def setUp(self):
        self.source = np.full((120, 120, 3), 255, np.uint8)
        self.result = np.full((120, 120, 3), (180, 195, 210), np.uint8)

    def _assert_selection_follows_hue(self, mask, hex_color, expected_rgb):
        edited, _base, used, changed = apply_selection_edit(
            self.source, self.result, self.result.copy(), mask, hex_color, feather=0)
        self.assertTrue(changed)
        self.assertTrue(np.any(used))
        target_bgr = np.array(expected_rgb[::-1], np.uint8)
        target_hue = _hue_degrees(target_bgr)
        actual_hue = _hue_degrees(edited[60, 60])
        self.assertLessEqual(_hue_distance(actual_hue, target_hue), 1.5)
        self.assertTrue(np.array_equal(edited[5, 5], self.result[5, 5]))

    def test_rectangle_fill_follows_selected_red_instead_of_turning_orange(self):
        mask = build_rect_selection_mask((120, 120), 25, 25, 95, 95)
        self._assert_selection_follows_hue(mask, '#ff0000', (255, 0, 0))

    def test_lasso_fill_follows_selected_blue_instead_of_turning_purple(self):
        mask = build_polygon_selection_mask(
            (120, 120), [(20, 20), (100, 25), (95, 100), (25, 95)])
        self._assert_selection_follows_hue(mask, '#0000ff', (0, 0, 255))

    def test_selection_keeps_existing_lightness_gradient(self):
        gradient = np.zeros((120, 120, 3), np.uint8)
        for y in range(120):
            value = 70 + int(round(y * 150 / 119))
            gradient[y, :] = (value, value, value)
        mask = build_rect_selection_mask((120, 120), 15, 15, 105, 105)
        edited, _base, used, changed = apply_selection_edit(
            self.source, gradient, gradient.copy(), mask, '#ff3030', feather=0)
        self.assertTrue(changed)
        before_l = cv2.cvtColor(gradient.astype(np.float32) / 255.0, cv2.COLOR_BGR2HLS)[..., 1]
        after_l = cv2.cvtColor(edited.astype(np.float32) / 255.0, cv2.COLOR_BGR2HLS)[..., 1]
        self.assertLess(float(np.mean(np.abs(after_l[used > 0] - before_l[used > 0]))), 0.02)

    def test_custom_bias_reaches_colourized_white_source_regions_but_not_blank_paper(self):
        source = np.full((100, 100, 3), 255, np.uint8)
        result = np.full((100, 100, 3), 255, np.uint8)
        result[20:80, 20:80] = (180, 195, 210)
        before_hue = _hue_degrees(result[50, 50])
        target_hue = _hue_degrees(np.array([0, 0, 255], np.uint8))
        biased = apply_global_color_bias(
            result, source, (255, 0, 0), 0.70,
            protect_skin=False, protect_lineart=True, protect_saturated=False)
        after_hue = _hue_degrees(biased[50, 50])
        self.assertLess(_hue_distance(after_hue, target_hue),
                        _hue_distance(before_hue, target_hue))
        self.assertGreater(int(np.abs(biased[50, 50].astype(int) - result[50, 50].astype(int)).sum()), 12)
        self.assertLessEqual(int(np.abs(biased[5, 5].astype(int) - result[5, 5].astype(int)).max()), 1)


if __name__ == '__main__':
    unittest.main()
