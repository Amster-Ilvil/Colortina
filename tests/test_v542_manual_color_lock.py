import cv2
import numpy as np

from core.hint_manager import HintManager
from core.local_model_recolor import (
    apply_manual_hint_color_lock,
    filtered_hint_manager,
    recolor_selection_with_model,
)


def _closed_page(h=96, w=128):
    original = np.full((h, w, 3), 245, np.uint8)
    cv2.rectangle(original, (28, 20), (96, 78), (0, 0, 0), 2)
    mask = np.zeros((h, w), np.uint8)
    mask[18:81, 26:99] = 255
    current = np.full((h, w, 3), (120, 160, 210), np.uint8)
    return original, current, mask


def _hue_of_bgr(bgr):
    px = np.asarray(bgr, np.uint8).reshape(1, 1, 3)
    return int(cv2.cvtColor(px, cv2.COLOR_BGR2HLS)[0, 0, 0])


def _hue_distance(a, b):
    return min((a - b) % 180, (b - a) % 180)


def test_standard_local_route_keeps_exact_manual_source_and_uses_mixed_renderer():
    original, current, mask = _closed_page()
    hm = HintManager()
    hm.bind_source_image(original)
    hm.add_manual_hint(0.50, 0.50, (20, 40, 230), 0.02, source="manual")
    captured = {}

    def fake_colorize(image, **kwargs):
        captured["source"] = kwargs["hint_manager"].manual_hints[0].source
        captured["rgb"] = kwargs["hint_manager"].manual_hints[0].color
        captured["render_mode"] = kwargs["hint_render_mode"]
        yellow = np.full_like(image, (0, 220, 255))
        return yellow, yellow.copy()

    recolor_selection_with_model(
        original, current, current.copy(), current.copy(), mask, hm,
        feather=0, colorize_fn=fake_colorize)

    assert captured["source"] == "manual"
    assert captured["rgb"] == (20, 40, 230)
    assert captured["render_mode"] == "mixed"


def test_manual_color_lock_turns_always_yellow_model_into_requested_blue():
    original, current, mask = _closed_page()
    hm = HintManager()
    hm.bind_source_image(original)
    target_rgb = (20, 40, 230)
    hm.add_manual_hint(0.50, 0.50, target_rgb, 0.02, source="manual")

    def always_yellow(image, **_kwargs):
        yellow = np.full_like(image, (0, 220, 255))
        return yellow, yellow.copy()

    payload = recolor_selection_with_model(
        original, current, current.copy(), current.copy(), mask, hm,
        feather=0, colorize_fn=always_yellow)

    inside_bgr = payload.result_bgr[48, 64]
    target_bgr = np.array([target_rgb[2], target_rgb[1], target_rgb[0]], np.uint8)
    yellow_bgr = np.array([0, 220, 255], np.uint8)
    assert _hue_distance(_hue_of_bgr(inside_bgr), _hue_of_bgr(target_bgr)) <= 2
    assert _hue_distance(_hue_of_bgr(inside_bgr), _hue_of_bgr(yellow_bgr)) >= 20
    assert payload.diagnostics["local_manual_color_lock_pixels"] > 0
    assert payload.diagnostics["local_manual_color_lock_enabled"] is True


def test_different_manual_colours_produce_different_locked_results():
    original, _current, mask = _closed_page()
    yellow = np.full_like(original, (0, 220, 255))
    outputs = []
    for rgb in ((230, 30, 30), (30, 80, 230), (30, 220, 60)):
        hm = HintManager()
        hm.bind_source_image(original)
        hm.add_manual_hint(0.50, 0.50, rgb, 0.02, source="manual")
        local = filtered_hint_manager(hm, mask)
        local.bind_source_image(original)
        locked, pixels = apply_manual_hint_color_lock(
            yellow, original, mask, local, strength=1.0)
        assert pixels > 0
        outputs.append(tuple(int(v) for v in locked[48, 64]))
    assert len(set(outputs)) == 3


def test_manual_color_lock_does_not_change_pixels_outside_touched_region():
    original, _current, mask = _closed_page()
    yellow = np.full_like(original, (0, 220, 255))
    hm = HintManager()
    hm.bind_source_image(original)
    hm.add_manual_hint(0.50, 0.50, (30, 80, 230), 0.02, source="manual")
    local = filtered_hint_manager(hm, mask)
    local.bind_source_image(original)
    locked, _pixels = apply_manual_hint_color_lock(
        yellow, original, mask, local, strength=1.0)
    assert np.array_equal(locked[5, 5], yellow[5, 5])
    assert not np.array_equal(locked[48, 64], yellow[48, 64])


def test_model_hint_brush_bypasses_manual_style_colour_adapter():
    from pathlib import Path
    main = (Path(__file__).resolve().parents[1] / "ui" / "main_window.py").read_text(encoding="utf-8")
    marker = "# Model Hint means an exact user colour instruction."
    assert marker in main
    block = main[main.index(marker):main.index(marker) + 700]
    assert "paint_rgb = tuple(int(np.clip(v, 0, 255)) for v in rgb)" in block
    assert "paint_rgb = self._manual_target_rgb(rgb)" not in block
