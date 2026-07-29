import numpy as np

from core.hint_manager import HintManager
from core.local_model_recolor import clear_manual_hints_in_selection, filtered_hint_manager


def _page(h=80, w=100):
    original = np.full((h, w, 3), 245, np.uint8)
    current = np.full((h, w, 3), (30, 80, 160), np.uint8)
    mask = np.zeros((h, w), np.uint8)
    mask[18:62, 28:76] = 255
    return original, current, mask


def test_local_manual_hints_suppress_auto_hints_inside_selection_only():
    original, _current, mask = _page()
    hm = HintManager()
    hm.bind_source_image(original)
    hm.set_auto_hints([
        (0.50, 0.50, (20, 220, 20), 0.02),   # inside selection: should be suppressed
        (0.92, 0.10, (200, 50, 40), 0.02),   # outside selection: should remain
    ])
    hm.add_manual_hint(0.52, 0.52, (240, 180, 130), source="manual")

    local = filtered_hint_manager(hm, mask, margin_px=6, only_selection_hints=False)

    assert len(local.manual_hints) == 1
    assert local.manual_hints[0].source == "manual"
    assert len(local.auto_hints) == 1
    assert local.auto_hints[0].color == (200, 50, 40)
    assert getattr(local, "last_diagnostics", {}).get("manual_override_active") is True
    assert getattr(local, "last_diagnostics", {}).get("suppressed_auto_hints_in_selection") == 1


def test_no_manual_override_keeps_auto_hints_inside_selection():
    original, _current, mask = _page()
    hm = HintManager()
    hm.bind_source_image(original)
    hm.set_auto_hints([
        (0.50, 0.50, (20, 220, 20), 0.02),
    ])

    local = filtered_hint_manager(hm, mask, margin_px=6, only_selection_hints=True)

    assert len(local.manual_hints) == 0
    assert len(local.auto_hints) == 1
    assert local.auto_hints[0].color == (20, 220, 20)
    assert getattr(local, "last_diagnostics", {}).get("manual_override_active") is False


def test_clear_manual_hints_in_selection_preserves_auto_and_outside_manual_hints():
    original, _current, mask = _page()
    hm = HintManager()
    hm.bind_source_image(original)
    hm.set_auto_hints([
        (0.50, 0.50, (20, 220, 20), 0.02),
    ])
    hm.add_manual_hint(0.50, 0.50, (240, 180, 130), source="manual")
    hm.add_eyedropper_hint(0.56, 0.52, (245, 175, 135))
    hm.add_manual_hint(0.08, 0.10, (10, 10, 240), source="manual")

    removed = clear_manual_hints_in_selection(hm, mask)

    assert removed == 2
    assert len(hm.auto_hints) == 1
    assert len(hm.manual_hints) == 1
    assert hm.manual_hints[0].color == (10, 10, 240)
