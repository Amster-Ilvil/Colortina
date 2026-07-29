import unittest
from pathlib import Path

import cv2
import numpy as np

from core.manual_edit import apply_brush_edit, apply_region_edit, apply_selection_edit

ROOT = Path(__file__).resolve().parents[1]
TARGET_HEX = '#2b78d0'
TARGET_RGB = (43, 120, 208)
TARGET_BGR = np.array(TARGET_RGB[::-1], dtype=np.uint8)


def _hls(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2HLS)


def _hue_gap(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _target_hue() -> float:
    return float(_hls(TARGET_BGR.reshape(1, 1, 3))[0, 0, 0])


def _textured_result(height: int = 140, width: int = 180) -> np.ndarray:
    """Two badly mismatched colour families plus a real luminance gradient."""
    image = np.zeros((height, width, 3), np.uint8)
    for y in range(height):
        factor = 0.65 + 0.60 * y / max(1, height - 1)
        left = np.clip(np.array((190, 80, 150), np.float32) * factor, 0, 255)
        right = np.clip(np.array((70, 120, 220), np.float32) * factor, 0, 255)
        image[y, :width // 2] = left
        image[y, width // 2:] = right
    return image


class TestV2811DistinctManualRecolorModes(unittest.TestCase):
    def setUp(self):
        self.source = np.full((140, 180, 3), 255, np.uint8)
        self.result = _textured_result()

    def test_selection_modes_are_visibly_and_mathematically_distinct(self):
        mask = np.zeros(self.source.shape[:2], np.uint8)
        mask[15:125, 15:165] = 255
        outputs = {}
        used_mask = None
        for mode in ('shift', 'shading', 'flat'):
            out, _base, used, changed = apply_selection_edit(
                self.source, self.result, self.result.copy(), mask, TARGET_HEX,
                feather=0, closed_only=False, mode=mode)
            self.assertTrue(changed)
            outputs[mode] = out
            used_mask = used

        selected = used_mask > 0
        shift_hls = _hls(outputs['shift'])
        shading_hls = _hls(outputs['shading'])
        flat_hls = _hls(outputs['flat'])

        # V3 natural migration uses the restored stable LAB path: the selected
        # colour becomes coherent while the mc-v2 luminance texture remains.
        self.assertGreater(float(np.std(shift_hls[..., 1][selected])), 0.05)
        self.assertGreater(float(np.std(shift_hls[..., 2][selected])), 0.005)

        # Uniform hue removes the purple/orange disagreement but keeps folds.
        self.assertLess(float(np.std(shading_hls[..., 0][selected])), 1.0)
        self.assertLess(float(np.std(shading_hls[..., 2][selected])), 0.01)
        self.assertGreater(float(np.std(shading_hls[..., 1][selected])), 0.035)

        # Pure colour is byte-exact and has no internal shading variation.
        self.assertTrue(np.all(outputs['flat'][selected] == TARGET_BGR))
        self.assertLess(float(np.std(flat_hls[..., 1][selected])), 0.001)

        for a, b in (('shift', 'shading'), ('shift', 'flat'), ('shading', 'flat')):
            mean_delta = float(np.abs(
                outputs[a].astype(np.int16) - outputs[b].astype(np.int16)
            )[selected].mean())
            self.assertGreater(mean_delta, 3.0, (a, b, mean_delta))

    def test_brush_uses_selected_mode_instead_of_one_fixed_algorithm(self):
        outputs = {}
        masks = {}
        for mode in ('shift', 'shading', 'flat'):
            out, _base, alpha, changed = apply_brush_edit(
                self.source, self.result, self.result.copy(),
                90, 70, 30, TARGET_RGB, opacity=1.0, gap_close=4, mode=mode)
            self.assertTrue(changed)
            outputs[mode] = out
            masks[mode] = alpha

        self.assertTrue(np.array_equal(outputs['flat'][70, 90], TARGET_BGR))
        self.assertFalse(np.array_equal(outputs['shift'][70, 90], outputs['shading'][70, 90]))

        shading_core = masks['shading'] >= 0.999
        flat_core = masks['flat'] >= 0.999
        self.assertGreater(int(np.count_nonzero(shading_core)), 30)
        self.assertGreater(int(np.count_nonzero(flat_core)), 30)

        shading_hls = _hls(outputs['shading'])
        flat_hls = _hls(outputs['flat'])
        self.assertLess(float(np.std(shading_hls[..., 0][shading_core])), 0.8)
        self.assertLess(float(np.std(shading_hls[..., 2][shading_core])), 0.01)
        self.assertGreater(float(np.std(shading_hls[..., 1][shading_core])), 0.01)
        self.assertTrue(np.all(outputs['flat'][flat_core] == TARGET_BGR))
        self.assertLess(float(np.std(flat_hls[..., 1][flat_core])), 0.001)

    def test_point_region_fill_uses_all_three_algorithms(self):
        source = np.full((140, 180, 3), 255, np.uint8)
        # Keep the enclosed block below V3's 35% page-area safety cap.
        cv2.rectangle(source, (38, 28), (142, 112), (0, 0, 0), 3)
        result = np.zeros_like(source)
        for y in range(result.shape[0]):
            result[y, :] = (100 + y // 2, 140 + y // 3, 180 - y // 4)

        outputs = {}
        masks = {}
        for mode in ('shift', 'shading', 'flat'):
            out, _base, used, changed = apply_region_edit(
                source, result, result.copy(), 80, 70, TARGET_HEX,
                gap_close=4, mode=mode, feather=2)
            self.assertTrue(changed)
            self.assertTrue(np.any(used))
            outputs[mode] = out
            masks[mode] = used

        self.assertTrue(np.array_equal(outputs['flat'][70, 80], TARGET_BGR))
        self.assertFalse(np.array_equal(outputs['shift'][70, 80], outputs['shading'][70, 80]))

        target_h = _target_hue()
        shading_h = float(_hls(outputs['shading'])[70, 80, 0])
        self.assertLess(_hue_gap(shading_h, target_h), 1.0)

        for a, b in (('shift', 'shading'), ('shift', 'flat'), ('shading', 'flat')):
            overlap = (masks[a] > 0) & (masks[b] > 0)
            self.assertGreater(int(np.count_nonzero(overlap)), 20)
            self.assertFalse(np.array_equal(outputs[a][overlap], outputs[b][overlap]))

    def test_black_is_really_dark_and_flat_black_is_exact(self):
        gradient = np.zeros((100, 100, 3), np.uint8)
        for y in range(100):
            value = 100 + y
            gradient[y, :] = (value, value, value)
        mask = np.zeros((100, 100), np.uint8)
        mask[10:90, 10:90] = 255

        outputs = {}
        used = None
        for mode in ('shift', 'shading', 'flat'):
            out, _base, used, changed = apply_selection_edit(
                np.full_like(gradient, 255), gradient, gradient.copy(), mask,
                '#000000', feather=0, mode=mode)
            self.assertTrue(changed)
            outputs[mode] = out

        selected = used > 0
        # Natural migration only gives luminance a small cue; black therefore
        # darkens the source without turning it into a flat near-black patch.
        self.assertLess(float(outputs['shift'][selected].mean()), 140.0)
        self.assertGreater(float(outputs['shift'][selected].mean()), 90.0)
        self.assertLess(float(outputs['shading'][selected].mean()), 55.0)
        self.assertGreater(float(_hls(outputs['shading'])[..., 1][selected].std()), 0.03)
        self.assertTrue(np.all(outputs['flat'][selected] == 0))

    def test_ui_passes_fill_mode_to_brush_region_and_selection(self):
        main = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
        self.assertIn('mode=self._current_fill_mode()', main)
        self.assertIn('fill_mode = self._current_fill_mode()', main)
        self.assertGreaterEqual(main.count('self._current_fill_mode()'), 5)
        self.assertIn('\"fill_mode\": self._current_fill_mode()', main)
        self.assertIn('self._set_fill_mode(prev_fill_mode)', main)


if __name__ == '__main__':
    unittest.main()
