from pathlib import Path


def test_bias_brush_and_eraser_are_merged_and_eyedropper_is_second():
    text = (Path(__file__).resolve().parents[1] / 'ui' / 'main_window.py').read_text(encoding='utf-8')
    assert '_radio_bias_eraser' not in text
    assert 'self._btn_bias_brush_paint' in text
    assert 'self._btn_bias_brush_erase' in text
    assert 'self._current_bias_brush_mode() == "erase"' in text
    tool_buttons_snippet_start = text.index('tool_buttons = (')
    snippet = text[tool_buttons_snippet_start: tool_buttons_snippet_start + 220]
    assert 'self._radio_brush, self._radio_eyedropper, self._radio_bias_brush' in snippet
