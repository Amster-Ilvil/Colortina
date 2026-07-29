import unittest
from pathlib import Path

import cv2
import numpy as np

from core.manual_edit import apply_selection_edit, combine_selection_masks

ROOT = Path(__file__).resolve().parents[1]


def _circular_hue_gap(a: float, b: float) -> float:
    return min(abs(a - b), 360.0 - abs(a - b))


class TestV289SelectionModesAndClothingFix(unittest.TestCase):
    def test_replace_add_subtract_have_independent_results(self):
        existing = np.zeros((12, 12), np.uint8)
        existing[1:7, 1:7] = 255
        incoming = np.zeros((12, 12), np.uint8)
        incoming[5:11, 5:11] = 255

        replaced = combine_selection_masks(existing, incoming, 'replace')
        added = combine_selection_masks(existing, incoming, 'add')
        subtracted = combine_selection_masks(existing, incoming, 'subtract')

        self.assertTrue(np.array_equal(replaced, incoming))
        self.assertEqual(np.count_nonzero(added), 68)
        self.assertEqual(np.count_nonzero(subtracted), 32)
        self.assertFalse(np.array_equal(replaced, added))
        self.assertFalse(np.array_equal(replaced, subtracted))
        self.assertFalse(np.array_equal(added, subtracted))

    def test_add_from_empty_starts_selection_and_subtract_from_empty_stays_empty(self):
        incoming = np.zeros((10, 10), np.uint8)
        incoming[2:8, 3:7] = 255
        self.assertTrue(np.array_equal(
            combine_selection_masks(None, incoming, 'add'), incoming))
        self.assertIsNone(combine_selection_masks(None, incoming, 'subtract'))

    def test_uniform_hue_mode_repairs_same_outfit_across_disconnected_pieces(self):
        source = np.full((140, 220, 3), 255, np.uint8)
        result = np.full((140, 220, 3), 245, np.uint8)
        # Same garment, but the model produced strongly different hues.
        result[25:115, 20:95] = (190, 65, 205)    # purple, BGR
        result[25:115, 125:200] = (60, 110, 220)  # orange/red, BGR
        mask = np.zeros((140, 220), np.uint8)
        mask[25:115, 20:95] = 255
        mask[25:115, 125:200] = 255

        out, _base, used_mask, changed = apply_selection_edit(
            source, result, result.copy(), mask, '#d04664',
            feather=2, closed_only=False, mode='shading')
        self.assertTrue(changed)
        self.assertEqual(np.count_nonzero(used_mask), np.count_nonzero(mask))

        hls = cv2.cvtColor(out.astype(np.float32) / 255.0, cv2.COLOR_BGR2HLS)
        left_h = float(np.median(hls[35:105, 30:85, 0]))
        right_h = float(np.median(hls[35:105, 135:190, 0]))
        left_s = float(np.median(hls[35:105, 30:85, 2]))
        right_s = float(np.median(hls[35:105, 135:190, 2]))
        self.assertLess(_circular_hue_gap(left_h, right_h), 1.2)
        self.assertLess(abs(left_s - right_s), 0.025)

        # The two pieces may keep different brightness/shading, but no longer
        # differ as purple versus red/orange.
        before_hls = cv2.cvtColor(result.astype(np.float32) / 255.0, cv2.COLOR_BGR2HLS)
        before_gap = _circular_hue_gap(
            float(np.median(before_hls[35:105, 30:85, 0])),
            float(np.median(before_hls[35:105, 135:190, 0])))
        self.assertGreater(before_gap, 20.0)

    def test_flat_mode_uses_exact_selected_color_inside_selection(self):
        source = np.full((80, 100, 3), 255, np.uint8)
        result = np.full((80, 100, 3), (190, 120, 70), np.uint8)
        mask = np.zeros((80, 100), np.uint8)
        mask[10:70, 15:85] = 255
        out, _base, _used, changed = apply_selection_edit(
            source, result, result.copy(), mask, '#204080',
            feather=3, mode='flat')
        self.assertTrue(changed)
        # Far enough from the feathered edge, RGB must match #204080.
        self.assertTrue(np.allclose(out[40, 50][::-1], (32, 64, 128), atol=1))

    def test_ui_has_exclusive_modes_and_preserves_accumulated_overlay(self):
        main = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
        canvas = (ROOT / 'ui' / 'canvas.py').read_text(encoding='utf-8')
        self.assertIn('combine_selection_masks,', main)
        self.assertIn('self._selection_mode_group.setExclusive(True)', main)
        self.assertIn('self._selection_mode_radios', main)
        self.assertIn('mode=fill_mode', main)
        self.assertIn('self._pending_selection_item', canvas)
        self.assertIn('def _clear_drawing_selection_overlay', canvas)
        self.assertIn('def set_selection_combine_mode', canvas)


if __name__ == '__main__':
    unittest.main()
