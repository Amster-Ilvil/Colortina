from pathlib import Path

import numpy as np

from core.manual_edit import apply_brush_edit

ROOT = Path(__file__).resolve().parents[1]


def test_live_brush_paints_only_requested_corridor_and_protects_ink():
    source = np.full((72, 72, 3), 255, np.uint8)
    source[:, 36:39] = 0
    result = np.full((72, 72, 3), (170, 180, 190), np.uint8)
    edited, _base, mask, changed = apply_brush_edit(
        source, result, result.copy(), 34, 36, 8, (220, 70, 80),
        opacity=1.0, gap_close=0, mode="shift",
        snap_to_lineart=False, pupil_blend=False)
    assert changed
    assert np.count_nonzero(mask) > 0
    assert np.array_equal(edited[:, 36:39], result[:, 36:39])
    assert np.count_nonzero((edited != result).any(axis=2)[:, :36]) > 0


def test_v5_ui_uses_block_fill_name_and_has_no_brush_release_snap():
    i18n = (ROOT / 'ui' / 'i18n.py').read_text(encoding='utf-8')
    main = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
    config = (ROOT / 'config.py').read_text(encoding='utf-8')

    assert '"tool_bucket": "区块上色"' in i18n
    assert '线条区块上色' not in i18n
    assert 'brush_snap_' not in i18n
    assert 'snap_brush_stroke_mask' not in main
    assert 'gap_close=0, mode=fill_mode' in main
    assert 'APP_VERSION_LABEL = "V5"' in config
