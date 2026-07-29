import unittest
from pathlib import Path

import cv2
import numpy as np

from core.custom_color_bias import apply_global_color_bias
from core.manual_edit import apply_selection_edit, build_rect_selection_mask


class TestV286FeatherAndProtection(unittest.TestCase):
    def test_selection_feather_is_inward_and_never_crosses_mask(self):
        source = np.full((80, 80, 3), 210, np.uint8)
        result = np.full((80, 80, 3), (190, 180, 170), np.uint8)
        mask = build_rect_selection_mask((80, 80), 20, 20, 60, 60)
        hard, _, used, _ = apply_selection_edit(source, result, result, mask, '#ff6060', feather=0)
        soft, _, _, _ = apply_selection_edit(source, result, result, mask, '#ff6060', feather=10)
        outside = used == 0
        self.assertTrue(np.array_equal(soft[outside], result[outside]))
        edge_soft = int(np.abs(soft[21, 40].astype(int) - result[21, 40].astype(int)).sum())
        edge_hard = int(np.abs(hard[21, 40].astype(int) - result[21, 40].astype(int)).sum())
        center_soft = int(np.abs(soft[40, 40].astype(int) - result[40, 40].astype(int)).sum())
        self.assertLess(edge_soft, edge_hard)
        self.assertGreater(center_soft, edge_soft)

    def test_lineart_protection_reduces_change_near_ink(self):
        source = np.full((100, 100, 3), 210, np.uint8)
        cv2.line(source, (50, 10), (50, 90), (0, 0, 0), 3)
        result = np.full((100, 100, 3), (180, 195, 210), np.uint8)
        protected = apply_global_color_bias(result, source, (255, 110, 110), 0.95, protect_skin=False, protect_lineart=True, protect_saturated=False)
        unprotected = apply_global_color_bias(result, source, (255, 110, 110), 0.95, protect_skin=False, protect_lineart=False, protect_saturated=False)
        p = int(np.abs(protected[50, 47].astype(int) - result[50, 47].astype(int)).sum())
        u = int(np.abs(unprotected[50, 47].astype(int) - result[50, 47].astype(int)).sum())
        self.assertLess(p, u)

    def test_saturation_protection_reduces_change_on_vivid_patch(self):
        source = np.full((80, 80, 3), 210, np.uint8)
        result = np.full((80, 80, 3), (200, 180, 170), np.uint8)
        result[20:60, 20:60] = (20, 40, 240)
        protected = apply_global_color_bias(result, source, (120, 220, 255), 0.9, protect_skin=False, protect_lineart=False, protect_saturated=True)
        unprotected = apply_global_color_bias(result, source, (120, 220, 255), 0.9, protect_skin=False, protect_lineart=False, protect_saturated=False)
        p = int(np.abs(protected[40, 40].astype(int) - result[40, 40].astype(int)).sum())
        u = int(np.abs(unprotected[40, 40].astype(int) - result[40, 40].astype(int)).sum())
        self.assertLess(p, u)

    def test_skin_protection_reduces_change_on_skin_like_color(self):
        source = np.full((60, 60, 3), 210, np.uint8)
        result = np.full((60, 60, 3), (140, 170, 220), np.uint8)
        protected = apply_global_color_bias(
            result, source, (100, 180, 255), 0.95,
            protect_skin=True, protect_lineart=False, protect_saturated=False)
        unprotected = apply_global_color_bias(
            result, source, (100, 180, 255), 0.95,
            protect_skin=False, protect_lineart=False, protect_saturated=False)
        p = int(np.abs(protected[30, 30].astype(int) - result[30, 30].astype(int)).sum())
        u = int(np.abs(unprotected[30, 30].astype(int) - result[30, 30].astype(int)).sum())
        self.assertLess(p, u)

    def test_colorize_page_runtime_uses_custom_bias(self):
        from unittest.mock import patch
        import pipeline

        class FakeColorizer:
            def colorize(self, image_bgr, **_kwargs):
                return np.full_like(image_bgr, (180, 195, 210))

        source_a = np.full((32, 32, 3), 210, np.uint8)
        source_b = np.full((32, 32, 3), 211, np.uint8)
        with patch.object(pipeline, 'get_colorizer', return_value=FakeColorizer()):
            plain = pipeline.colorize_page(
                source_a, style_key='none', filter_tuning=None,
                custom_color_bias={'enabled': False})
        with patch.object(pipeline, 'get_colorizer', return_value=FakeColorizer()):
            biased = pipeline.colorize_page(
                source_b, style_key='none', filter_tuning=None,
                custom_color_bias={
                    'enabled': True, 'rgb': (255, 110, 110), 'strength': 90,
                    'protect_skin': False, 'protect_lineart': False,
                    'protect_saturated': False,
                })
        self.assertFalse(np.array_equal(plain, biased))

    def test_batch_worker_initializes_custom_bias_state(self):
        text = Path('ui/worker.py').read_text(encoding='utf-8')
        batch = text[text.index('class BatchColorizeWorker'):]
        self.assertIn('self._custom_color_bias = dict(custom_color_bias or {})', batch)
        single = text[:text.index('class BatchColorizeWorker')]
        self.assertEqual(single.count('self._custom_color_bias = dict(custom_color_bias or {})'), 1)

    def test_pipeline_actually_applies_custom_bias(self):
        text = Path('pipeline.py').read_text(encoding='utf-8')
        self.assertIn('apply_global_color_bias(', text)
        self.assertIn('protect_skin=bool(custom_color_bias.get', text)


if __name__ == '__main__':
    unittest.main()
