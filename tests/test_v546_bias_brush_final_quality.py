import numpy as np

from core.bias_brush import (
    add_soft_round_dab,
    add_soft_round_dab_inplace,
    composite_bias_candidate,
    composite_bias_candidate_roi,
)


def _page(h=128, w=176):
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.empty((h, w, 3), np.uint8)
    base[..., 0] = np.clip(35 + xx * 0.65 + yy * 0.10, 0, 255)
    base[..., 1] = np.clip(75 + yy * 0.70, 0, 255)
    base[..., 2] = np.clip(115 + xx * 0.35, 0, 255)
    candidate = np.empty_like(base)
    candidate[..., 0] = np.clip(base[..., 0] * 0.45 + 20, 0, 255)
    candidate[..., 1] = np.clip(base[..., 1] * 0.65 + 80, 0, 255)
    candidate[..., 2] = np.clip(base[..., 2] * 0.75 + 65, 0, 255)
    return base, candidate


def test_final_full_quality_pass_matches_original_multi_dab_compositor_exactly():
    base, candidate = _page()
    points = [(20, 30), (38, 37), (59, 43), (81, 55), (105, 66), (135, 79)]
    radius = 19

    # Earlier high-quality implementation: rebuild full accumulated alpha, then
    # composite the whole page.
    old_alpha = np.zeros(base.shape[:2], np.float32)
    for x, y in points:
        old_alpha = add_soft_round_dab(old_alpha, x, y, radius)
    expected, expected_changed = composite_bias_candidate(base, candidate, old_alpha)

    # New fast preview path updates only each dab ROI in place.
    alpha = np.zeros(base.shape[:2], np.float32)
    preview = base.copy()
    for x, y in points:
        alpha, roi = add_soft_round_dab_inplace(alpha, x, y, radius)
        preview, _ = composite_bias_candidate_roi(
            base, candidate, alpha, roi, dst=preview)

    # Mouse release now performs this exact full-quality calibration pass.
    final, final_changed = composite_bias_candidate(base, candidate, alpha)
    assert np.array_equal(final, expected)
    assert final_changed == expected_changed


def test_main_window_release_contains_final_full_quality_pass():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / 'ui' / 'main_window.py').read_text(encoding='utf-8')
    start = text.index('def _on_bias_brush_stroke_finished')
    snippet = text[start:start + 2200]
    assert 'build_cohesive_stroke_alpha' in snippet
    assert 'composite_bias_candidate' in snippet
    assert 'self._bias_brush_base_result' in snippet
    assert 'self._bias_brush_alpha' in snippet
