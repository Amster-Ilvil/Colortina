from pathlib import Path


def test_main_window_contains_picker_model_hint_controls_and_logic():
    text = Path('ui/main_window.py').read_text(encoding='utf-8')
    assert '_chk_picker_extract_hint = QCheckBox' in text
    assert '"picker_extract_hint": self._picker_extract_hint_enabled()' in text
    assert 'settings.get("picker_extract_hint", False)' in text
    assert 'self._canvas.color_picked.connect(self._on_color_picked)' in text
    assert 'state.hint_manager.add_eyedropper_hint(' in text
    assert 'tr("picker_model_hint_written")' in text
    assert 'tr("picker_model_hint_requires_color")' in text


def test_canvas_emits_normalized_eyedropper_pick_signal():
    text = Path('ui/canvas.py').read_text(encoding='utf-8')
    assert 'color_picked = Signal(tuple, float, float)' in text
    assert 'self.color_picked.emit(color, float(x_norm), float(y_norm))' in text


def test_hint_pipeline_knows_eyedropper_hint_source():
    spec_text = Path('core/hint_spec.py').read_text(encoding='utf-8')
    assert '"eyedropper_hint"' in spec_text
    manager_text = Path('core/hint_manager.py').read_text(encoding='utf-8')
    assert 'def add_eyedropper_hint' in manager_text
    raster_text = Path('core/hint_rasterizer.py').read_text(encoding='utf-8')
    assert '"eyedropper_hint": (2, 40)' in raster_text
    pipeline_text = Path('pipeline.py').read_text(encoding='utf-8')
    assert 'Manual dabs ARE model instructions' in pipeline_text


def test_i18n_contains_model_hint_strings_in_both_languages():
    text = Path('ui/i18n.py').read_text(encoding='utf-8')
    assert '"picker_extract_hint_checkbox": "吸取写入模型提示"' in text
    assert '"picker_model_hint_written": "已写入模型提示；请点击自动上色或重新生成，让 mc-v2 按线稿区域传播该颜色。"' in text
    assert '"picker_model_hint_requires_color": "当前页还没有上色结果；本次仅更新取色，未写入模型提示。"' in text
    assert '"picker_extract_hint_checkbox": "Write picked colour as model hint"' in text
    assert '"picker_model_hint_written": "Model hint written. Click Auto Colorize or Regenerate so mc-v2 can propagate the colour through the line-art region."' in text
