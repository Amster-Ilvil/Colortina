import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from core.hint_manager import HintManager
from core.project_store import load_project, save_project

ROOT = Path(__file__).resolve().parents[1]


class TestV2VersionAndFilterPersistence(unittest.TestCase):
    def test_left_bottom_version_is_v2(self):
        text = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
        self.assertIn('self._version_label = QLabel("V2")', text)
        self.assertIn('VersionLabel', text)

    def test_project_persists_filter_base(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / 'source.png'
            cv2.imwrite(str(source), np.full((8, 8, 3), 220, np.uint8))
            page = SimpleNamespace(
                path=str(source), hint_manager=HintManager(),
                ai_result_bgr=np.full((8, 8, 3), 100, np.uint8),
                result_bgr=np.full((8, 8, 3), 130, np.uint8),
                filter_base_bgr=np.full((8, 8, 3), 115, np.uint8),
                pipeline_diagnostics={}, forced_character_matches={},
            )
            project = save_project(str(Path(td) / 'test.ccproject'), pages=[page])
            loaded = load_project(project)
            record = loaded['pages'][0]
            self.assertTrue(record.get('filter_base'))
            restored = cv2.imread(record['filter_base'])
            self.assertIsNotNone(restored)
            self.assertEqual(int(restored.mean()), 115)


if __name__ == '__main__':
    unittest.main()
