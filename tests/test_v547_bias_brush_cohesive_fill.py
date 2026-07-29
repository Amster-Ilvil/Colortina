import numpy as np
import cv2

from core.bias_brush import build_cohesive_stroke_alpha


def test_cohesive_alpha_expands_open_textured_region_beyond_sparse_seed():
    h, w = 120, 140
    original = np.full((h, w, 3), 255, np.uint8)
    # Create an object-like area with many broken vertical texture lines but no
    # fully closed outline, which previously produced fragmented bias patches.
    original[20:100, 30] = 0
    original[20:100, 110] = 0
    original[20, 30:111] = 0
    for x in range(40, 105, 10):
        for y0 in range(28, 92, 14):
            original[y0:y0+6, x:x+1] = 0
    result = np.full((h, w, 3), (170, 175, 180), np.uint8)
    stroke = np.zeros((h, w), np.float32)
    cv2.circle(stroke, (70, 58), 8, 1.0, -1)

    cohesive = build_cohesive_stroke_alpha(stroke, original, result, brush_radius_px=14)

    seed_area = int(np.count_nonzero(stroke > 0.1))
    fill_area = int(np.count_nonzero(cohesive > 0.5))
    assert fill_area > seed_area * 2
    assert fill_area < int(h * w * 0.45)
    # The expansion should reach well beyond the original dab horizontally.
    xs = np.where(cohesive > 0.5)[1]
    assert xs.min() <= 50 and xs.max() >= 90


def test_main_window_final_bias_pass_uses_cohesive_alpha_builder():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / 'ui' / 'main_window.py').read_text(encoding='utf-8')
    start = text.index('def _on_bias_brush_stroke_finished')
    snippet = text[start:start + 1500]
    assert 'build_cohesive_stroke_alpha' in snippet
    assert 'final_alpha = build_cohesive_stroke_alpha' in snippet
