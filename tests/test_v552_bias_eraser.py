from pathlib import Path
import numpy as np

from core.bias_brush import add_soft_round_dab, composite_bias_candidate


def test_bias_eraser_composites_only_along_stroke_without_block_expansion():
    base = np.zeros((40, 40, 3), np.uint8)
    current = np.full((40, 40, 3), 120, np.uint8)
    alpha = np.zeros((40, 40), np.float32)
    alpha = add_soft_round_dab(alpha, 20, 20, 4)
    restored, changed = composite_bias_candidate(current, base, alpha)
    assert changed > 0
    # Center of stroke is restored toward base.
    assert restored[20, 20, 0] < current[20, 20, 0]
    # Far-away pixel is untouched, proving there is no whole-block expansion.
    assert np.array_equal(restored[2, 2], current[2, 2])


def test_bias_eraser_tool_and_reference_snapshot_hooks_exist():
    text = (Path(__file__).resolve().parents[1] / 'ui' / 'main_window.py').read_text(encoding='utf-8')
    assert 'tool_hint_bias_eraser' in text
    assert 'def _on_bias_eraser_stroke_started' in text
    assert 'def _on_bias_eraser_stroke_finished' in text
    assert 'self._btn_bias_brush_erase' in text
    snap = (Path(__file__).resolve().parents[1] / 'core' / 'edit_snapshot.py').read_text(encoding='utf-8')
    assert 'bias_brush_reference_bgr' in snap
    assert 'bias_brush_reference_filter_bgr' in snap
