import unittest
from pathlib import Path

import cv2
import numpy as np

from core.perceptual_recolor import recolor_with_mode
from core.manual_edit import apply_brush_edit, apply_region_edit, apply_selection_edit

ROOT = Path(__file__).resolve().parents[1]


def _hls(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2HLS)


def _hue_gap(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


class TestV2811ReferenceModePort(unittest.TestCase):
    def test_low_saturation_source_is_not_a_noop_in_natural_or_uniform_mode(self):
        source = np.zeros((96, 128, 3), np.uint8)
        for y in range(source.shape[0]):
            value = 80 + int(140 * y / (source.shape[0] - 1))
            source[y, :] = (value, value, value)
        alpha = np.zeros(source.shape[:2], np.float32)
        alpha[8:88, 8:120] = 1.0
        target_rgb = (216, 54, 76)
        target_h = float(_hls(np.array(target_rgb[::-1], np.uint8).reshape(1, 1, 3))[0, 0, 0])

        for mode in ('shift', 'shading'):
            out = recolor_with_mode(source, target_rgb, alpha, mode=mode)
            selected = alpha > 0
            mean_delta = float(np.abs(out.astype(np.int16) - source.astype(np.int16))[selected].mean())
            self.assertGreater(mean_delta, 24.0, (mode, mean_delta))
            hls = _hls(out)
            colourful = selected & (hls[..., 2] > 0.20)
            self.assertGreater(int(np.count_nonzero(colourful)), 5000)
            centre_h = float(np.median(hls[..., 0][colourful]))
            self.assertLess(_hue_gap(centre_h, target_h), 4.0, (mode, centre_h, target_h))

    def test_natural_keeps_colour_texture_uniform_removes_it_flat_removes_shading(self):
        h, w = 120, 160
        image = np.zeros((h, w, 3), np.uint8)
        for y in range(h):
            f = 0.55 + 0.75 * y / (h - 1)
            image[y, :w // 2] = np.clip(np.array((190, 70, 145)) * f, 0, 255)
            image[y, w // 2:] = np.clip(np.array((65, 145, 220)) * f, 0, 255)
        alpha = np.ones((h, w), np.float32)
        rgb = (55, 128, 214)

        natural = recolor_with_mode(image, rgb, alpha, mode='shift')
        uniform = recolor_with_mode(image, rgb, alpha, mode='shading')
        flat = recolor_with_mode(image, rgb, alpha, mode='flat')
        natural_hls = _hls(natural)
        uniform_hls = _hls(uniform)

        # V3's restored LAB migration makes hue coherent but retains the model's
        # luminance texture and a small amount of saturation variation.
        self.assertGreater(float(np.std(natural_hls[..., 1])), 0.06)
        self.assertGreater(float(np.std(natural_hls[..., 2])), 0.005)
        self.assertLess(float(np.std(uniform_hls[..., 0])), 0.8)
        self.assertLess(float(np.std(uniform_hls[..., 2])), 0.01)
        self.assertGreater(float(np.std(uniform_hls[..., 1])), 0.03)
        self.assertTrue(np.all(flat == np.array(rgb[::-1], np.uint8)))

    def test_brush_bucket_and_selection_share_the_same_three_modes(self):
        source = np.full((128, 160, 3), 255, np.uint8)
        # Bucket regions above 35% of the page are intentionally rejected in V3.
        cv2.rectangle(source, (35, 25), (125, 103), (0, 0, 0), 3)
        result = np.zeros_like(source)
        for y in range(result.shape[0]):
            result[y, :] = (70 + y, 155 - y // 2, 205 - y // 3)
        selection = np.zeros(source.shape[:2], np.uint8)
        selection[16:112, 16:144] = 255
        rgb = (42, 170, 104)
        hex_colour = '#2aaa68'

        for mode in ('shift', 'shading', 'flat'):
            brush, _, brush_mask, brush_changed = apply_brush_edit(
                source, result, result.copy(), 80, 64, 28, rgb,
                opacity=1.0, mode=mode)
            bucket, _, bucket_mask, bucket_changed = apply_region_edit(
                source, result, result.copy(), 80, 64, hex_colour,
                mode=mode, feather=2)
            selected, _, selected_mask, selection_changed = apply_selection_edit(
                source, result, result.copy(), selection, hex_colour,
                mode=mode, feather=0)
            self.assertTrue(brush_changed, mode)
            self.assertTrue(bucket_changed, mode)
            self.assertTrue(selection_changed, mode)
            self.assertGreater(float(brush_mask.max()), 0.9)
            self.assertTrue(np.any(bucket_mask))
            self.assertTrue(np.any(selected_mask))
            if mode == 'flat':
                expected = np.array(rgb[::-1], np.uint8)
                self.assertTrue(np.array_equal(brush[64, 80], expected))
                self.assertTrue(np.array_equal(bucket[64, 80], expected))
                self.assertTrue(np.array_equal(selected[64, 80], expected))

    def test_mode_state_survives_ui_rebuild_and_project_roundtrip(self):
        source = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
        self.assertIn('prev_fill_mode = self._current_fill_mode()', source)
        self.assertIn('self._set_fill_mode(prev_fill_mode)', source)
        self.assertIn('"fill_mode": self._current_fill_mode()', source)
        self.assertIn('self._set_fill_mode(settings.get("fill_mode", "shift"))', source)
        self.assertEqual(source.count('mode=self._current_fill_mode()'), 1)
        self.assertGreaterEqual(source.count('fill_mode = self._current_fill_mode()'), 2)


if __name__ == '__main__':
    unittest.main()
