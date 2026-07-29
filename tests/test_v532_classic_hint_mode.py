from pathlib import Path

import numpy as np

from core.hint_manager import HintManager
from core.hint_rasterizer import rasterize_hint_specs_legacy
from core.local_model_recolor import filtered_hint_manager, recolor_selection_with_model
from vendor.manga_colorization_v2.colorizator import MangaColorizator

ROOT = Path(__file__).resolve().parents[1]


def _page(h=80, w=100):
    original = np.full((h, w, 3), 245, np.uint8)
    original[20:60, 30] = 0
    current = np.full((h, w, 3), (30, 80, 160), np.uint8)
    mask = np.zeros((h, w), np.uint8)
    mask[20:60, 30:75] = 255
    return original, current, mask


def test_classic_point_hint_filter_converts_manual_to_point_model_hint():
    original, _current, mask = _page()
    hm = HintManager()
    hm.bind_source_image(original)
    hm.add_manual_hint(0.50, 0.50, (220, 150, 120), source="manual")
    hm.add_manual_hint(0.05, 0.05, (10, 250, 10), source="manual")
    hm.add_eyedropper_hint(0.60, 0.45, (230, 170, 130))

    local = filtered_hint_manager(
        hm, mask, margin_px=3, only_selection_hints=True,
        classic_point_hints=True)
    assert len(local.manual_hints) == 2
    assert {h.source for h in local.manual_hints} == {"eyedropper_hint"}
    assert all(h.color != (10, 250, 10) for h in local.manual_hints)


def test_classic_local_recolor_routes_legacy_hint_rendering():
    original, current, mask = _page()
    hm = HintManager()
    hm.bind_source_image(original)
    hm.add_manual_hint(0.50, 0.50, (230, 160, 120), source="manual")
    captured = {}

    def fake_colorize(image, **kwargs):
        captured["image"] = image.copy()
        captured["manager"] = kwargs["hint_manager"]
        captured["render_mode"] = kwargs["hint_render_mode"]
        generated = np.full_like(image, (180, 140, 90))
        return generated, generated

    payload = recolor_selection_with_model(
        original, current, current.copy(), current.copy(), mask, hm,
        feather=4, hint_margin_px=8, classic_point_hints=True,
        colorize_fn=fake_colorize)

    assert captured["render_mode"] == "legacy"
    assert captured["manager"].manual_hints[0].source == "eyedropper_hint"
    assert payload.changed_pixels > 0


def test_legacy_rasterizer_emits_sparse_binary_mask():
    hint, alpha = rasterize_hint_specs_legacy(
        80, 100, 576,
        [{"x_norm": 0.5, "y_norm": 0.5, "rgb": (200, 100, 50),
          "radius_norm": 0.03, "source": "manual"}]
    )
    used = alpha > 0
    assert used.sum() > 0
    assert used.sum() <= 13  # tiny hard point / disk, not a large soft blob
    assert set(np.unique(alpha[used]).tolist()) == {1.0}
    assert np.all(hint[used] == np.array([200, 100, 50], np.uint8))
    assert np.all(hint[~used] == 0)


def test_vendor_update_hint_multiplies_rgb_by_mask():
    dummy = type("Dummy", (), {"device": "cpu"})()
    hint = np.zeros((3, 3, 3), np.uint8)
    hint[..., 0] = 255
    mask = np.zeros((3, 3), np.float32)
    mask[1, 1] = 1.0
    MangaColorizator.update_hint(dummy, hint, mask)
    tensor = dummy.current_hint.cpu().numpy()
    assert tensor.shape == (1, 4, 3, 3)
    rgb = tensor[0, :3]
    alpha = tensor[0, 3]
    assert alpha[1, 1] == 1.0
    assert np.all(alpha[(slice(None),)] >= 0.0)
    assert np.all(rgb[:, 0, 0] == 0.0)
    assert rgb[0, 1, 1] > 0.9


def test_ui_exposes_independent_classic_hint_toggle():
    main = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
    i18n = (ROOT / "ui" / "i18n.py").read_text(encoding="utf-8")
    worker = (ROOT / "ui" / "worker.py").read_text(encoding="utf-8")
    assert 'selection_classic_hints' in main
    assert 'classic_point_hints' in main
    assert 'selection_classic_hints_hint' in i18n
    assert 'classic_point_hints' in worker
