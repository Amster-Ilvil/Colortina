from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_hint_manager_keeps_source_specific_clear_for_picker_hints():
    text = (ROOT / "core" / "hint_manager.py").read_text(encoding="utf-8")
    assert "def clear_manual_hints_by_source" in text


def test_local_model_recolor_uses_full_page_inference_and_selection_composite():
    main = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
    worker = (ROOT / "ui" / "worker.py").read_text(encoding="utf-8")
    core = (ROOT / "core" / "local_model_recolor.py").read_text(encoding="utf-8")
    i18n = (ROOT / "ui" / "i18n.py").read_text(encoding="utf-8")
    assert "LocalModelRecolorWorker" in main
    assert "LocalModelRecolorWorker" in worker
    assert "_start_selection_model_recolor" in main
    assert "recolor_selection_with_model" in core
    assert "original_bgr" in core
    assert "merge_inside_selection" in core
    assert "build_focus_inference_image" in core
    assert "selection_ai_recolor" in i18n
    assert "selection_focus_outside" in i18n
    assert '_btn_picker_clear_model = QPushButton(tr("picker_clear_model_hints"))' in main
    assert 'clear_manual_hints_by_source("eyedropper_hint")' in main
