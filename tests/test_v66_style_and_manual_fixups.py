import unittest
from pathlib import Path

import cv2
import numpy as np

from core.lineart_fill import lineart_mask_at_point, lineart_region_recolor
from core.local_brush import apply_local_brush_recolor
from core.region_map import build_region_map

ROOT = Path(__file__).resolve().parents[1]


class TestV66StyleAndManualFixups(unittest.TestCase):
    def test_pipeline_uses_selected_style_preset(self):
        text = (ROOT / 'pipeline.py').read_text(encoding='utf-8')
        self.assertIn('active_style_profile = style_profile or _builtin_reference_profile(style_key)', text)
        self.assertIn('style = get_style(style_key)', text)
        self.assertIn('grade_preset = style', text)

    def test_lineart_region_click_can_snap_to_nearby_small_region(self):
        img = np.full((80, 80), 255, dtype=np.uint8)
        cv2.rectangle(img, (8, 8), (72, 72), 0, 2)
        cv2.rectangle(img, (26, 22), (54, 58), 0, 2)
        region_map = build_region_map(img, gap_close=4)
        # Click almost on the inner border rather than in the clean interior.
        mask = lineart_mask_at_point(img, 27, 23, gap_close=4, region_map=region_map)
        area = int(np.count_nonzero(mask))
        self.assertGreater(area, 150)
        self.assertLess(area, 2000)

    def test_region_recolor_matches_target_hue_more_closely(self):
        src = np.full((60, 60), 255, dtype=np.uint8)
        cv2.rectangle(src, (5, 5), (55, 55), 0, 2)
        current = np.full((60, 60, 3), (235, 215, 205), dtype=np.uint8)
        recolored, mask = lineart_region_recolor(src, current.copy(), 30, 30, '#5bb8ff', mode='shift', gap_close=2, feather=2)
        target = np.array([255, 184, 91], dtype=np.float32)  # BGR for #5bb8ff
        pixels = recolored[mask > 0].astype(np.float32)
        before = current[mask > 0].astype(np.float32)
        after_dist = np.linalg.norm(pixels.mean(axis=0) - target)
        before_dist = np.linalg.norm(before.mean(axis=0) - target)
        self.assertLess(after_dist, before_dist)

    def test_local_brush_tracks_sampled_color_more_closely(self):
        source = np.full((40, 40), 255, dtype=np.uint8)
        current = np.full((40, 40, 3), (205, 225, 235), dtype=np.uint8)
        edited, alpha = apply_local_brush_recolor(source, current.copy(), 20, 20, 8, (80, 150, 240), opacity=1.0, gap_close=0)
        active = alpha > 0.4
        self.assertTrue(np.any(active))
        target = np.array([240, 150, 80], dtype=np.float32)  # BGR for RGB(80,150,240)
        before = np.linalg.norm(current[active].astype(np.float32).mean(axis=0) - target)
        after = np.linalg.norm(edited[active].astype(np.float32).mean(axis=0) - target)
        self.assertLess(after, before)

    def test_monochrome_people_strengthened_floors_present(self):
        text = (ROOT / 'core' / 'style_post.py').read_text(encoding='utf-8')
        self.assertIn('role_masks["hair"] * max(0.46', text)
        self.assertIn('keep_override = np.maximum(keep_override, role_masks["person"] * 0.28)', text)


if __name__ == '__main__':
    unittest.main()
