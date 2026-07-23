import unittest
from pathlib import Path

import cv2
import numpy as np

from core.image_filter import apply_image_filter
from core.manual_edit import apply_brush_edit, apply_region_edit

ROOT = Path(__file__).resolve().parents[1]


class TestV27ManualEditRuntime(unittest.TestCase):
    def test_region_fill_has_bounded_fallback_on_open_page(self):
        source = np.full((220, 220, 3), 255, np.uint8)
        # A broken frame deliberately cannot form a reliable enclosed region.
        cv2.line(source, (40, 40), (180, 40), (0, 0, 0), 3)
        cv2.line(source, (40, 40), (40, 180), (0, 0, 0), 3)
        cv2.line(source, (180, 40), (180, 130), (0, 0, 0), 3)
        result = np.full((220, 220, 3), (185, 190, 200), np.uint8)
        out, base, mask, changed = apply_region_edit(
            source, result, result.copy(), 110, 110, '#e65070',
            gap_close=0, mode='shift', feather=2)
        self.assertTrue(changed)
        self.assertTrue(mask.any())
        self.assertFalse(np.array_equal(out, result))
        self.assertFalse(np.array_equal(base, result))
        # Safety: fallback must not flood the whole page.
        self.assertLess(np.count_nonzero(mask), int(mask.size * 0.45))

    def test_manual_edit_survives_filter_reapply(self):
        source = np.full((180, 180, 3), 255, np.uint8)
        cv2.rectangle(source, (25, 25), (155, 155), (0, 0, 0), 3)
        base = np.full((180, 180, 3), (175, 185, 200), np.uint8)
        edited, edited_base, mask, changed = apply_brush_edit(
            source, base.copy(), base.copy(), 90, 90, 20, (235, 45, 75),
            opacity=1.0, gap_close=4)
        self.assertTrue(changed)
        tuning = {
            'brightness': 115, 'contrast': 108, 'saturation': 120,
            'warmth': 105, 'shadow_lift': 100, 'highlight': 100,
        }
        filtered_before = apply_image_filter(
            base, tuning, style_strength=1.0, is_styled=True,
            source_bw_bgr=source)
        filtered_after = apply_image_filter(
            edited_base, tuning, style_strength=1.0, is_styled=True,
            source_bw_bgr=source)
        changed_pixels = mask > 0.25
        self.assertTrue(np.any(changed_pixels))
        self.assertGreater(
            np.abs(filtered_after[changed_pixels].astype(int) -
                   filtered_before[changed_pixels].astype(int)).mean(), 2.0)

    def test_brush_refreshes_immediately_and_auto_button_is_bottom(self):
        main = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
        canvas = (ROOT / 'ui' / 'canvas.py').read_text(encoding='utf-8')
        self.assertIn('self._canvas.update_image_pixels(state.result_bgr)', main)
        self.assertIn('if not self._local_brush_stroke_active:', main)
        self.assertIn('def update_image_pixels', canvas)
        detail_pos = main.index('render_layout.addWidget(render_detail_tabs, stretch=4)')
        auto_pos = main.index('render_layout.addWidget(auto_group, stretch=1)')
        self.assertLess(detail_pos, auto_pos)


if __name__ == '__main__':
    unittest.main()
