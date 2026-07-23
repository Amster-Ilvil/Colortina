import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestV749FilterScope(unittest.TestCase):
    def test_ui_has_reset_and_scope_actions(self):
        text = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
        self.assertIn('_btn_filter_reset', text)
        self.assertIn('_filter_scope_combo', text)
        self.assertIn('_apply_filter_to_scope', text)
        self.assertIn('filter_base_bgr', text)

    def test_pipeline_can_return_filter_base(self):
        text = (ROOT / 'pipeline.py').read_text(encoding='utf-8')
        self.assertIn('return_filter_base: bool = False', text)
        self.assertIn('return result_bgr, filter_base_bgr', text)

    def test_snapshot_tracks_filter_base(self):
        text = (ROOT / 'core' / 'edit_snapshot.py').read_text(encoding='utf-8')
        self.assertIn('filter_base_bgr', text)


if __name__ == '__main__':
    unittest.main()
