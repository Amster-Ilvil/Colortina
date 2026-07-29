import unittest

import cv2
import numpy as np
from pathlib import Path

from core.custom_color_bias import apply_global_color_bias


class TestV285SelectionAndToneRange(unittest.TestCase):
    def test_tone_range_bias_targets_highlights_and_shadows_differently(self):
        src = np.full((60, 60, 3), 220, np.uint8)
        result = np.zeros((60, 60, 3), np.uint8)
        # bright top half, dark bottom half, with colour so chroma gate stays active
        result[:30, :] = (210, 220, 235)
        result[30:, :] = (55, 70, 85)
        high = apply_global_color_bias(result, src, (255, 120, 120), 0.9, 'page', 'highlights')
        shadow = apply_global_color_bias(result, src, (255, 120, 120), 0.9, 'page', 'shadows')
        top_high = int(np.abs(high[10, 10].astype(int) - result[10, 10].astype(int)).sum())
        bot_high = int(np.abs(high[50, 10].astype(int) - result[50, 10].astype(int)).sum())
        top_shadow = int(np.abs(shadow[10, 10].astype(int) - result[10, 10].astype(int)).sum())
        bot_shadow = int(np.abs(shadow[50, 10].astype(int) - result[50, 10].astype(int)).sum())
        self.assertGreater(top_high, bot_high)
        self.assertGreater(bot_shadow, top_shadow)

    def test_source_contains_selection_combine_ui(self):
        text = Path('ui/main_window.py').read_text(encoding='utf-8')
        for token in ['_selection_mode_radios', 'selection_mode_replace', 'selection_mode_add', 'selection_mode_subtract', '_combine_selection_mask']:
            self.assertIn(token, text)


if __name__ == '__main__':
    unittest.main()
