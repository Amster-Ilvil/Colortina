from pathlib import Path

import numpy as np

from core.hint_manager import HintManager
from core.local_model_recolor import (
    build_focus_inference_image,
    filtered_hint_manager,
    merge_inside_selection,
    preview_black_and_white,
    recolor_selection_with_model,
)

ROOT = Path(__file__).resolve().parents[1]


def _page(h=80, w=100):
    original = np.full((h, w, 3), 245, np.uint8)
    original[20:60, 30] = 0
    current = np.full((h, w, 3), (30, 80, 160), np.uint8)
    mask = np.zeros((h, w), np.uint8)
    mask[20:60, 30:75] = 255
    return original, current, mask




def test_focus_inference_image_whitens_outside_and_keeps_context():
    original, _current, mask = _page()
    focused = build_focus_inference_image(
        original, mask, context_expand_px=3, fade_expand_px=8, outside_mode="white")
    assert np.array_equal(focused[40, 40], original[40, 40])
    assert np.array_equal(focused[10, 10], np.array([255, 255, 255], np.uint8))


def test_focus_inference_image_can_fade_outside_selection():
    original, _current, mask = _page()
    focused = build_focus_inference_image(
        original, mask, context_expand_px=1, fade_expand_px=6, outside_mode="fade_white")
    assert np.array_equal(focused[40, 40], original[40, 40])
    assert int(focused[10, 10, 0]) >= int(original[10, 10, 0])
    assert int(focused[10, 10, 0]) <= 255

def test_local_hint_filter_converts_manual_to_model_and_drops_distant_hints():
    original, _current, mask = _page()
    hm = HintManager()
    hm.bind_source_image(original)
    hm.add_manual_hint(0.50, 0.50, (220, 150, 120), source="manual")
    hm.add_manual_hint(0.05, 0.05, (10, 250, 10), source="manual")
    hm.add_eyedropper_hint(0.60, 0.45, (230, 170, 130))

    local = filtered_hint_manager(hm, mask, margin_px=3, only_selection_hints=True)
    assert len(local.manual_hints) == 2
    assert {h.source for h in local.manual_hints} == {"manual", "eyedropper_hint"}
    assert all(h.color != (10, 250, 10) for h in local.manual_hints)


def test_full_original_is_inferred_and_only_selection_is_composited():
    original, current, mask = _page()
    ai = np.full_like(current, (40, 90, 170))
    filter_base = np.full_like(current, (50, 100, 180))
    hm = HintManager()
    hm.bind_source_image(original)
    hm.add_manual_hint(0.50, 0.50, (230, 160, 120), source="manual")
    captured = {}

    def fake_colorize(image, **kwargs):
        captured["image"] = image.copy()
        captured["manager"] = kwargs["hint_manager"]
        captured["preserve"] = kwargs["preserve_empty_auto_hints"]
        captured["learn"] = kwargs["learn_identity"]
        generated = np.full_like(image, (180, 140, 90))
        generated_base = np.full_like(image, (170, 130, 80))
        return generated, generated_base

    payload = recolor_selection_with_model(
        original, current, ai, filter_base, mask, hm,
        feather=4, hint_margin_px=8, focus_outside_mode="none", colorize_fn=fake_colorize)

    assert np.array_equal(captured["image"], original)
    assert captured["preserve"] is True
    assert captured["learn"] is False
    assert captured["manager"].manual_hints[0].source == "manual"
    outside = mask == 0
    assert np.array_equal(payload.result_bgr[outside], current[outside])
    assert np.array_equal(payload.ai_result_bgr[outside], ai[outside])
    assert np.array_equal(payload.filter_base_bgr[outside], filter_base[outside])
    # Deep interior comes from the newly generated page, then receives the
    # deterministic manual-colour lock while preserving model luminance.
    assert not np.array_equal(payload.result_bgr[40, 55], current[40, 55])
    assert payload.diagnostics["local_manual_color_lock_pixels"] > 0
    assert payload.used_hint_count == 1
    assert payload.selection_pixels == int(np.count_nonzero(mask))




def test_local_recolor_can_run_on_selection_focused_page():
    original, current, mask = _page()
    hm = HintManager()
    hm.bind_source_image(original)
    hm.add_manual_hint(0.50, 0.50, (230, 160, 120), source="manual")
    captured = {}

    def fake_colorize(image, **kwargs):
        captured["image"] = image.copy()
        generated = np.full_like(image, (180, 140, 90))
        return generated, generated

    recolor_selection_with_model(
        original, current, current.copy(), current.copy(), mask, hm,
        feather=4, hint_margin_px=8,
        focus_outside_mode="white", focus_context_expand_px=2, focus_fade_expand_px=6,
        colorize_fn=fake_colorize)

    assert np.array_equal(captured["image"][40, 40], original[40, 40])
    assert np.array_equal(captured["image"][5, 5], np.array([255, 255, 255], np.uint8))

def test_inward_feather_never_changes_pixels_outside_selection():
    _original, current, mask = _page()
    generated = np.full_like(current, 255)
    merged = merge_inside_selection(current, generated, mask, feather=12)
    assert np.array_equal(merged[mask == 0], current[mask == 0])
    assert np.any(merged[mask > 0] != current[mask > 0])


def test_black_white_preview_is_non_destructive():
    original, current, mask = _page()
    before = current.copy()
    preview = preview_black_and_white(current, original, mask)
    assert np.array_equal(current, before)
    assert np.array_equal(preview[mask == 0], current[mask == 0])
    assert np.array_equal(preview[mask > 0], original[mask > 0])


def test_ui_and_worker_expose_full_page_local_model_route():
    main = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
    worker = (ROOT / "ui" / "worker.py").read_text(encoding="utf-8")
    core = (ROOT / "core" / "local_model_recolor.py").read_text(encoding="utf-8")
    assert "LocalModelRecolorWorker" in main
    assert "_start_selection_model_recolor" in main
    assert 'QPushButton(tr("selection_ai_recolor"))' in main
    assert "class LocalModelRecolorWorker" in worker
    assert "recolor_selection_with_model" in worker
    assert "merge_inside_selection" in core
    assert "build_focus_inference_image" in core
    assert "selection_focus_outside" in main


def test_real_pipeline_route_passes_full_page_and_model_specs(monkeypatch):
    import pipeline

    original, current, mask = _page(83, 111)
    hm = HintManager()
    hm.bind_source_image(original)
    hm.add_manual_hint(0.55, 0.50, (235, 170, 140), 0.02, source="manual")
    seen = {}

    class FakeColorizer:
        def colorize(self, image, **kwargs):
            seen["image"] = image.copy()
            seen["specs"] = list(kwargs.get("hint_points") or [])
            return np.full_like(image, (160, 130, 100))

    monkeypatch.setattr(pipeline, "get_colorizer", lambda _cfg: FakeColorizer())
    payload = recolor_selection_with_model(
        original, current, current.copy(), current.copy(), mask, hm,
        feather=3, hint_margin_px=8,
        colorize_fn=pipeline.colorize_page,
        colorize_kwargs={
            "style_key": "none",
            "quality_key": "draft",
            "filter_tuning": {},
        },
    )
    assert np.array_equal(seen["image"], original)
    assert len(seen["specs"]) >= 1
    assert any(getattr(spec, "source", "") == "manual" for spec in seen["specs"])
    assert np.array_equal(payload.result_bgr[mask == 0], current[mask == 0])
    assert payload.changed_pixels > 0
