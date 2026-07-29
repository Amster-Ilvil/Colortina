from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_canvas_hides_eyedropper_hint_overlay_and_emits_selection_adjust():
    text = (ROOT / "ui" / "canvas.py").read_text(encoding="utf-8")
    assert 'if getattr(hint, "source", "") == "eyedropper_hint":' in text
    assert 'selection_adjust_dab = Signal(int, int, int, bool)' in text
    assert 'def set_selection_adjust_enabled' in text
    assert 'def _emit_selection_adjust_dab' in text


def test_main_window_has_selection_adjust_ui_and_handler():
    text = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
    assert 'self._chk_selection_adjust = QCheckBox(tr("selection_adjust_checkbox"))' in text
    assert 'self._selection_adjust_slider = QSlider(Qt.Orientation.Horizontal)' in text
    assert 'self._btn_selection_adjust_add = QPushButton(tr("selection_adjust_add_tool"))' in text
    assert 'self._btn_selection_adjust_erase = QPushButton(tr("selection_adjust_erase_tool"))' in text
    assert 'def _set_selection_adjust_mode(self, mode: str):' in text
    assert 'self._canvas.selection_adjust_dab.connect(self._on_selection_adjust_dab)' in text
    assert 'def _on_selection_adjust_dab(self, ix: int, iy: int, radius_px: int, add_mode: bool):' in text
    assert 'self._clear_closed_preview_cache()' in text
    assert 'cv2.circle(mask, (int(ix), int(iy)), max(1, int(radius_px)), value, thickness=-1)' in text


def test_i18n_contains_selection_adjust_strings():
    text = (ROOT / "ui" / "i18n.py").read_text(encoding="utf-8")
    assert '"selection_adjust_checkbox": "启用蓝色选区手动修选"' in text
    assert '"selection_adjust_add_tool": "＋ 加选画笔"' in text
    assert '"selection_adjust_erase_tool": "⌫ 选区橡皮擦"' in text
    assert '"selection_adjust_checkbox": "Enable manual blue-selection editing"' in text
