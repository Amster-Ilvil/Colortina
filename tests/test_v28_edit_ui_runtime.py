import os
import unittest
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    from PySide6.QtWidgets import QApplication
    from ui.main_window import MainWindow
except Exception:  # pragma: no cover
    QApplication = None
    MainWindow = None

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(QApplication is None or MainWindow is None, 'PySide6 runtime not available in test env')
class TestV28EditUiRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_edit_ui_uses_visible_eyedropper_checkboxes(self):
        win = MainWindow()
        try:
            self.assertTrue(hasattr(win, '_eyedropper_mode_point'))
            self.assertTrue(hasattr(win, '_eyedropper_mode_region'))
            self.assertEqual(win._current_eyedropper_mode(), 'point')
            win._set_eyedropper_mode('region')
            self.assertTrue(win._eyedropper_mode_region.isChecked())
            self.assertEqual(win._current_eyedropper_mode(), 'region')
        finally:
            win.close()

    def test_current_color_info_is_visible_and_updates(self):
        win = MainWindow()
        try:
            self.assertTrue(hasattr(win, '_current_color_info'))
            win._set_eyedropper_mode('point')
            win._apply_picked_color((120, 160, 210), remember_raw=True)
            text = win._current_color_info.text()
            self.assertIn('#', text)
            self.assertTrue('当前' in text or 'Current' in text)
        finally:
            win.close()


def load_tests(loader, tests, pattern):
    suite = loader.suiteClass()
    suite.addTests(loader.loadTestsFromTestCase(TestV28EditUiRuntime))
    # Static fallback checks still run everywhere.
    class TestStatic(unittest.TestCase):
        def test_source_contains_checkbox_ui_instead_of_combo(self):
            text = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
            self.assertIn('_eyedropper_mode_point = QCheckBox', text)
            self.assertIn('_eyedropper_mode_region = QCheckBox', text)
            self.assertNotIn('_eyedropper_mode_combo = QComboBox', text)
            self.assertIn('_current_color_info = QLabel()', text)
    suite.addTests(loader.loadTestsFromTestCase(TestStatic))
    return suite


if __name__ == '__main__':
    unittest.main()
