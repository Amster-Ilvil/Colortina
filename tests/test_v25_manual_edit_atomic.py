import unittest

import cv2
import numpy as np

from core.manual_edit import apply_brush_edit, apply_region_edit


class TestV25ManualEditAtomic(unittest.TestCase):
    def setUp(self):
        self.source = np.full((180, 180, 3), 255, np.uint8)
        cv2.rectangle(self.source, (35, 35), (145, 145), (0, 0, 0), 3)
        self.result = np.full((180, 180, 3), (180, 190, 210), np.uint8)
        self.base = np.full((180, 180, 3), (170, 180, 205), np.uint8)

    def test_brush_updates_visible_and_filter_base(self):
        out, base, mask, changed = apply_brush_edit(
            self.source, self.result, self.base, 90, 90, 18, (245, 60, 80),
            opacity=1.0, gap_close=4)
        self.assertTrue(changed)
        self.assertGreater(float(mask.max()), 0.0)
        self.assertFalse(np.array_equal(out, self.result))
        self.assertFalse(np.array_equal(base, self.base))

    def test_region_updates_visible_and_filter_base(self):
        out, base, mask, changed = apply_region_edit(
            self.source, self.result, self.base, 90, 90, '#ff4060',
            gap_close=4, mode='shift', feather=2)
        self.assertTrue(mask.any())
        self.assertTrue(changed)
        self.assertFalse(np.array_equal(out, self.result))
        self.assertFalse(np.array_equal(base, self.base))

    def test_brush_fallback_never_behaves_as_dead_tool(self):
        # Dense cross-lines make the strict region/safe-interior logic difficult;
        # the conservative visible-radius fallback must still recolor non-ink pixels.
        source = np.full((80, 80, 3), 255, np.uint8)
        for p in range(20, 61, 5):
            cv2.line(source, (20, p), (60, p), (0, 0, 0), 1)
            cv2.line(source, (p, 20), (p, 60), (0, 0, 0), 1)
        result = np.full_like(source, (190, 195, 205))
        out, base, mask, changed = apply_brush_edit(
            source, result, result.copy(), 42, 42, 14, (40, 120, 245),
            opacity=1.0, gap_close=12)
        self.assertGreater(float(mask.max()), 0.0)
        self.assertTrue(changed)
        self.assertFalse(np.array_equal(out, result))
        self.assertFalse(np.array_equal(base, result))


if __name__ == '__main__':
    unittest.main()
