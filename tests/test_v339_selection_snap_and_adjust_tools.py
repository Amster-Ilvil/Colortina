from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_canvas_selection_adjust_has_explicit_add_and_erase_mode():
    text = (ROOT / "ui" / "canvas.py").read_text(encoding="utf-8")
    assert 'self._selection_adjust_mode = "add"' in text
    assert 'def set_selection_adjust_mode(self, mode: str) -> None:' in text
    assert 'configured_add = self._selection_adjust_mode != "erase"' in text


def test_selection_snap_is_boundary_watershed_not_region_union_growth():
    text = (ROOT / "core" / "selection_snap.py").read_text(encoding="utf-8")
    assert 'marker-controlled watershed' in text
    assert 'cv2.watershed' in text
    assert 'markers[outer == 0] = 1' in text
    assert 'markers[core > 0] = 2' in text
    assert 'region_union' not in text
    assert 'cv2.dilate(raw' not in text
