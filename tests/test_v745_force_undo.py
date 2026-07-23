import unittest
from types import SimpleNamespace
import numpy as np

from core.edit_snapshot import capture_edit_state, restore_edit_state, snapshots_equal
from core.hint_manager import HintManager


def page():
    return SimpleNamespace(
        result_bgr=None,
        ai_result_bgr=None,
        hint_manager=HintManager(),
        pipeline_diagnostics={},
        quality_report=None,
    )


class TestV745ForceUndo(unittest.TestCase):
    def test_empty_state_is_a_real_snapshot(self):
        state = page()
        snap = capture_edit_state(state)
        state.result_bgr = np.full((8, 8, 3), 120, np.uint8)
        state.ai_result_bgr = state.result_bgr.copy()
        restore_edit_state(state, snap)
        self.assertIsNone(state.result_bgr)
        self.assertIsNone(state.ai_result_bgr)

    def test_restore_includes_ai_result_and_manual_hints(self):
        state = page()
        state.result_bgr = np.full((8, 8, 3), 10, np.uint8)
        state.ai_result_bgr = np.full((8, 8, 3), 20, np.uint8)
        state.hint_manager.add_manual_hint(0.5, 0.5, (200, 10, 10), 0.1)
        snap = capture_edit_state(state)
        state.result_bgr[:] = 99
        state.ai_result_bgr[:] = 88
        state.hint_manager.clear_manual_hints()
        restore_edit_state(state, snap)
        self.assertTrue(np.all(state.result_bgr == 10))
        self.assertTrue(np.all(state.ai_result_bgr == 20))
        self.assertEqual(len(state.hint_manager.manual_hints), 1)

    def test_snapshot_detects_noop_and_real_changes(self):
        state = page()
        state.result_bgr = np.full((8, 8, 3), 10, np.uint8)
        before = capture_edit_state(state)
        same = capture_edit_state(state)
        self.assertTrue(snapshots_equal(before, same))
        state.result_bgr[0, 0] = 11
        changed = capture_edit_state(state)
        self.assertFalse(snapshots_equal(before, changed))


if __name__ == '__main__':
    unittest.main()
