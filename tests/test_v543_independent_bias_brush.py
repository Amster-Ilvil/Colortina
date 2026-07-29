from pathlib import Path

import cv2
import numpy as np

from core.bias_brush import (
    BiasBrushConfig,
    add_soft_round_dab,
    composite_bias_candidate,
    empty_stroke_alpha,
    prepare_bias_candidate,
)

ROOT = Path(__file__).resolve().parents[1]


def _page(h=96, w=128):
    yy, xx = np.mgrid[0:h, 0:w]
    result = np.empty((h, w, 3), np.uint8)
    result[..., 0] = np.clip(70 + xx * 0.5, 0, 255)
    result[..., 1] = np.clip(95 + yy * 0.4, 0, 255)
    result[..., 2] = np.clip(135 + xx * 0.25, 0, 255)
    source = np.full((h, w, 3), 190, np.uint8)
    source[:, 42:44] = 0
    return result, source


def test_bias_brush_config_is_independent_and_accepts_ui_percent_strength():
    cfg = BiasBrushConfig.from_dict({
        "rgb": (20, 210, 60),
        "strength": 160,
        "tone_range": "midtones",
        "protect_skin": False,
    })
    assert cfg.rgb == (20, 210, 60)
    assert cfg.strength == 1.6
    assert cfg.tone_range == "midtones"
    assert cfg.protect_skin is False


def test_bias_brush_changes_only_visible_stroke_pixels():
    result, source = _page()
    candidate = prepare_bias_candidate(
        result, source,
        BiasBrushConfig(
            rgb=(25, 70, 235), strength=2.0,
            protect_skin=False, protect_lineart=False,
            protect_saturated=False,
        ),
    )
    alpha = empty_stroke_alpha(result.shape[:2])
    alpha = add_soft_round_dab(alpha, 78, 48, 18)
    edited, changed = composite_bias_candidate(result, candidate, alpha)

    outside = alpha <= 0.0
    inside = alpha > 0.98
    assert changed > 0
    assert np.array_equal(edited[outside], result[outside])
    assert np.any(edited[inside] != result[inside])


def test_overlapping_bias_dabs_use_opacity_union_without_exceeding_one():
    alpha = empty_stroke_alpha((80, 100))
    alpha = add_soft_round_dab(alpha, 42, 40, 16)
    alpha = add_soft_round_dab(alpha, 54, 40, 16)
    assert float(alpha.min()) >= 0.0
    assert float(alpha.max()) <= 1.0
    assert alpha[40, 42] == 1.0
    assert alpha[40, 54] == 1.0
    assert np.count_nonzero(alpha) > 800


def test_ui_routes_bias_brush_through_a_separate_canvas_channel():
    main = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
    canvas = (ROOT / "ui" / "canvas.py").read_text(encoding="utf-8")
    i18n = (ROOT / "ui" / "i18n.py").read_text(encoding="utf-8")

    assert 'TOOL_BIAS_BRUSH = "bias_brush"' in canvas
    assert "bias_brush_dab_added = Signal" in canvas
    assert "bias_brush_stroke_started = Signal" in canvas
    assert "bias_brush_stroke_finished = Signal" in canvas
    assert "_on_bias_brush_dab_added" in main
    assert "prepare_bias_candidate" in main
    assert "composite_bias_candidate" in main
    assert '"bias_brush": {"bias_brush_controls"}' in main
    assert '"tool_bias_brush"' in i18n


def test_bias_brush_color_picker_syncs_with_custom_bias_but_not_normal_brush():
    main = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
    start = main.index("    def _pick_bias_brush_color")
    end = main.index("    def _update_bias_brush_swatch", start)
    method = main[start:end]
    assert "_set_custom_bias_linked_color(color, update_shared_brush=False)" in method
    assert "_set_shared_selected_color" not in method
    assert "self._brush_color" not in method


def test_normal_brush_and_hint_routes_remain_present_and_separate():
    main = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
    assert "apply_brush_edit(" in main
    assert "state.hint_manager.add_manual_hint(" in main
    start = main.index("    def _on_bias_brush_dab_added")
    end = main.index("    def _on_bias_brush_stroke_finished", start)
    bias_method = main[start:end]
    assert "apply_brush_edit" not in bias_method
    assert "hint_manager" not in bias_method
    assert "mc-v2" not in bias_method
