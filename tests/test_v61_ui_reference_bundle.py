import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestV61UiReferenceBundle(unittest.TestCase):
    def test_ui_source_contains_book_reference_bundle_and_eyedropper_visibility(self):
        main = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
        self.assertIn('_build_book_reference_bundle', main)
        self.assertIn('book_reference_bundle_status', main)
        self.assertIn('_update_tool_specific_visibility', main)
        self.assertIn('_eyedropper_mode_point = QCheckBox', main)
        self.assertIn('_eyedropper_mode_region = QCheckBox', main)
        self.assertIn('_current_color_info = QLabel()', main)

    def test_i18n_contains_book_reference_bundle_labels(self):
        i18n = (ROOT / 'ui' / 'i18n.py').read_text(encoding='utf-8')
        self.assertIn('"book_reference_bundle": "整本参考建模…"', i18n)
        self.assertIn('"book_reference_bundle_hint": "会同时读取人物、衣服、瞳色、肤色与环境颜色，并在后续上色中尽量匹配。"', i18n)
        self.assertIn('"eyedropper_mode_point": "点采集（邻域中值）"', i18n)


if __name__ == '__main__':
    unittest.main()
