import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestV283SelectionConfirm(unittest.TestCase):
    def test_main_window_contains_selection_apply_cancel_flow(self):
        text = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
        for token in [
            '_btn_apply_selection', '_btn_cancel_selection',
            '_apply_pending_selection_fill', '_cancel_pending_selection_fill',
            "selection_pending_hint", 'Key_Return', 'Key_Escape'
        ]:
            self.assertIn(token, text)

    def test_canvas_keeps_selection_preview_until_cleared(self):
        text = (ROOT / 'ui' / 'canvas.py').read_text(encoding='utf-8')
        for token in [
            'selection_preview_active = Signal',
            'def clear_selection_preview',
            '_update_polygon_overlay(closed=True)',
            'self.selection_preview_active.emit(True)'
        ]:
            self.assertIn(token, text)


if __name__ == '__main__':
    unittest.main()
