import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestV29UiStatic(unittest.TestCase):
    def test_main_window_contains_new_fill_tools_and_custom_bias(self):
        text = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
        self.assertIn('_radio_lasso_bucket = QRadioButton', text)
        self.assertIn('_radio_rect_bucket = QRadioButton', text)
        self.assertIn('_current_custom_color_bias()', text)
        self.assertIn('_chk_custom_color_bias', text)
        self.assertIn('_on_polygon_fill_requested', text)
        self.assertIn('_on_rect_fill_requested', text)

    def test_canvas_contains_selection_tool_signals(self):
        text = (ROOT / 'ui' / 'canvas.py').read_text(encoding='utf-8')
        self.assertIn('polygon_fill_requested = Signal', text)
        self.assertIn('rect_fill_requested = Signal', text)
        self.assertIn('TOOL_LASSO_BUCKET', text)
        self.assertIn('TOOL_RECT_BUCKET', text)


if __name__ == '__main__':
    unittest.main()
