import unittest
from pathlib import Path

import cv2
import numpy as np

from core.image_filter import apply_image_filter
from core.manual_edit import apply_brush_edit, apply_selection_edit, build_rect_selection_mask

ROOT = Path(__file__).resolve().parents[1]


class TestV288SharedColorAndFilter(unittest.TestCase):
    def test_selection_fill_black_moves_region_close_to_black(self):
        source = np.full((140, 140, 3), 255, np.uint8)
        result = np.full((140, 140, 3), (198, 198, 198), np.uint8)
        selection = build_rect_selection_mask((140, 140), 25, 25, 115, 115)
        out, _base, mask, changed = apply_selection_edit(
            source, result, result.copy(), selection, '#000000', feather=2, closed_only=False)
        self.assertTrue(changed)
        self.assertTrue(np.any(mask > 0))
        region = out[mask > 0]
        self.assertLess(float(region.mean()), 70.0)
        self.assertLess(float(region.mean()), float(result[mask > 0].mean()) - 90.0)

    def test_natural_brush_black_darkens_without_flattening(self):
        source = np.full((180, 180, 3), 255, np.uint8)
        cv2.rectangle(source, (20, 20), (160, 160), (0, 0, 0), 3)
        base = np.full((180, 180, 3), (205, 205, 205), np.uint8)
        edited, _edited_base, mask, changed = apply_brush_edit(
            source, base.copy(), base.copy(), 90, 90, 22, (0, 0, 0), opacity=1.0, gap_close=4)
        self.assertTrue(changed)
        painted = edited[mask > 0.18]
        self.assertGreater(painted.size, 0)
        # V3 natural migration gives luminance only a small cue. Black should
        # visibly darken the patch while preserving the existing light field;
        # exact black remains available through the flat mode.
        self.assertLess(float(painted.mean()), 190.0)
        self.assertGreater(float(painted.mean()), 120.0)

    def test_color_filter_uses_shared_color_and_changes_page(self):
        source = np.full((120, 120, 3), 255, np.uint8)
        image = np.full((120, 120, 3), (165, 185, 210), np.uint8)
        tuning = {
            'brightness': 100,
            'contrast': 100,
            'saturation': 100,
            'warmth': 100,
            'shadow_lift': 100,
            'highlight': 100,
            'color_filter_enabled': True,
            'color_filter_strength': 70,
            'color_filter_rgb': (0, 0, 0),
            'color_filter_color': '#000000',
        }
        out = apply_image_filter(image, tuning, source_bw_bgr=source)
        self.assertLess(float(out.mean()), float(image.mean()) - 20.0)

    def test_ui_contains_shared_color_filter_controls(self):
        main = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
        self.assertIn('filter_color_enable', main)
        self.assertIn('color_filter_enabled', main)
        self.assertIn('_set_shared_selected_color', main)


if __name__ == '__main__':
    unittest.main()
