from pathlib import Path

from core.bias_brush import build_cohesive_stroke_alpha


def test_bias_brush_fill_mode_ui_removed():
    text = (Path(__file__).resolve().parents[1] / 'ui' / 'main_window.py').read_text(encoding='utf-8')
    assert '_bias_brush_fill_combo' not in text
    assert 'bias_brush_fill_mode' not in text
    assert 'bias_brush_fill_conservative' not in text
    assert 'bias_brush_fill_standard' not in text
    assert 'bias_brush_fill_aggressive' not in text


def test_bias_brush_core_reverted_to_single_cohesive_mode():
    text = (Path(__file__).resolve().parents[1] / 'core' / 'bias_brush.py').read_text(encoding='utf-8')
    assert 'def _cohesive_fill_profile' not in text
    assert 'fill_mode: str = "standard"' not in text
    assert 'local_reach = max(int(brush_radius_px) * 4, 24)' in text
