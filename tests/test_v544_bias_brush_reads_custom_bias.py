from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_bias_brush_config_reads_custom_bias_color_not_picker_color():
    text=(ROOT/"ui"/"main_window.py").read_text(encoding="utf-8")
    start=text.index("    def _current_bias_brush_config")
    end=text.index("    def _open_color_dialog", start)
    snippet=text[start:end]
    assert "_custom_color_bias_color" in snippet
    assert "_bias_brush_color" in snippet
