import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestV25ManualEditUI(unittest.TestCase):
    def test_ui_uses_atomic_manual_edit_helpers(self):
        text = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
        self.assertIn('apply_brush_edit(', text)
        self.assertIn('apply_region_edit(', text)
        self.assertIn('state.filter_base_bgr = new_base', text)
        self.assertIn('self._radio_view_edited.setChecked(True)', text)


if __name__ == '__main__':
    unittest.main()
