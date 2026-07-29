import numpy as np

from core.bias_brush import (
    add_soft_round_dab,
    add_soft_round_dab_inplace,
    composite_bias_candidate,
    composite_bias_candidate_roi,
)


def _page(h=90, w=120):
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.empty((h, w, 3), np.uint8)
    base[..., 0] = np.clip(50 + xx * 0.5, 0, 255)
    base[..., 1] = np.clip(90 + yy * 0.4, 0, 255)
    base[..., 2] = np.clip(140 + xx * 0.2, 0, 255)
    cand = np.flip(base, axis=2).copy()
    return base, cand


def test_inplace_dab_matches_copy_version_and_returns_roi():
    alpha1 = np.zeros((80, 100), np.float32)
    alpha2 = alpha1.copy()
    out1 = add_soft_round_dab(alpha1, 42, 38, 15)
    out2, roi = add_soft_round_dab_inplace(alpha2, 42, 38, 15)
    assert roi is not None
    assert np.allclose(out1, out2)


def test_roi_composite_matches_full_composite_for_changed_bbox():
    base, cand = _page()
    alpha = np.zeros(base.shape[:2], np.float32)
    alpha, roi = add_soft_round_dab_inplace(alpha, 55, 44, 16)
    full, changed_full = composite_bias_candidate(base, cand, alpha)
    roi_out, changed_roi = composite_bias_candidate_roi(base, cand, alpha, roi)
    assert np.array_equal(full, roi_out)
    assert changed_full == changed_roi


def test_canvas_bias_brush_code_interpolates_between_sparse_points():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / 'ui' / 'canvas.py').read_text(encoding='utf-8')
    start = text.index('def _drop_bias_dab')
    snippet = text[start:start + 1700]
    assert 'np.hypot' in snippet
    assert 'np.ceil(distance / float(min_step))' in snippet
    assert 'for step in range(1, steps + 1)' in snippet
