from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_custom_bias_and_bias_brush_use_linked_colour_helper():
    text = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
    assert "def _set_custom_bias_linked_color" in text
    assert "self._custom_color_bias_color = QColor(linked)" in text
    assert "self._bias_brush_color = QColor(linked)" in text


def test_custom_bias_picker_is_one_way_and_does_not_call_shared_selected_colour():
    text = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
    snippet_start = text.index("def _pick_custom_bias_color")
    snippet = text[snippet_start: snippet_start + 800]
    assert "_set_custom_bias_linked_color(color, update_shared_brush=False)" in snippet
    assert "_set_shared_selected_color(color)" not in snippet


def test_current_custom_color_bias_reads_custom_bias_colour_not_brush_colour():
    text = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
    snippet_start = text.index("def _current_custom_color_bias")
    snippet = text[snippet_start: snippet_start + 300]
    assert "_custom_color_bias_color" in snippet
    assert "_brush_color" not in snippet.split("return", 1)[0]
