import re
import unittest
from pathlib import Path

import numpy as np

from core.local_brush import restore_local_brush_from_reference
from core.presets import get_style

ROOT = Path(__file__).resolve().parents[1]


class TestV60ManualToolsAndUi(unittest.TestCase):
    def test_source_has_no_brush_color_mode_and_picker_stays_put(self):
        main = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
        self.assertNotIn('_brush_color_mode_combo', main)
        self.assertNotIn('edit_grid.addWidget(QLabel(tr("brush_color_mode_label"))', main)
        body = re.search(r'def _on_color_picked\(self, rgb: tuple\):(.*?)(?:\n    def |\Z)', main, re.S)
        self.assertIsNotNone(body)
        body_text = body.group(1)
        self.assertNotIn('self._radio_brush.setChecked(True)', body_text)
        self.assertIn('picker_keep_tool_hint', body_text)

    def test_gap_ui_and_tooltip_reflect_bigger_means_smaller_region(self):
        main = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
        self.assertIn('_GAP_PX_MIN, _GAP_PX_MAX = 0, 24', main)
        i18n = (ROOT / 'ui' / 'i18n.py').read_text(encoding='utf-8')
        self.assertIn('调大 = 自动补更长的断线，区域更容易被封住，因此选择范围更小', i18n)

    def test_light_wash_name_cleanup_keeps_colored_builtin_style(self):
        style = get_style('light2')
        self.assertGreaterEqual(style.saturation_boost, 0.40)
        self.assertGreaterEqual(style.l_gamma, 0.88)
        self.assertLessEqual(style.guided_filter_radius, 4)
        self.assertGreaterEqual(style.neutral_fade_floor, 0.30)
        self.assertLessEqual(style.neutral_fade_floor, 0.60)

    def test_i18n_contains_picker_hint(self):
        i18n = (ROOT / 'ui' / 'i18n.py').read_text(encoding='utf-8')
        self.assertIn('"picker_keep_tool_hint": "吸管采色后保持当前工具，不会自动切回画笔。"', i18n)

    def test_restore_local_brush_from_reference_changes_only_local_area(self):
        source = np.full((40, 40, 3), 255, dtype=np.uint8)
        current = np.full((40, 40, 3), (20, 40, 200), dtype=np.uint8)
        restore = np.full((40, 40, 3), (140, 200, 40), dtype=np.uint8)
        edited, alpha = restore_local_brush_from_reference(
            source, current, restore, 20, 20, 6, opacity=1.0, gap_close=0)
        self.assertEqual(edited.shape, current.shape)
        self.assertEqual(alpha.shape[:2], current.shape[:2])
        self.assertGreater(float(alpha.max()), 0.0)
        changed = alpha > 1e-5
        self.assertTrue(np.any(changed))
        self.assertTrue(np.allclose(edited[~changed], current[~changed]))
        center = alpha > 0.85
        self.assertTrue(np.any(center))
        diff_restore = np.abs(edited[center].astype(int) - restore[center].astype(int)).mean()
        diff_current = np.abs(edited[center].astype(int) - current[center].astype(int)).mean()
        self.assertLess(diff_restore, diff_current)


if __name__ == '__main__':
    unittest.main()
