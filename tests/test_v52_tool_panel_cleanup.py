from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v52_version_and_dynamic_tool_panel_are_declared():
    config = (ROOT / "config.py").read_text(encoding="utf-8")
    main = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
    i18n = (ROOT / "ui" / "i18n.py").read_text(encoding="utf-8")

    assert 'APP_VERSION = "5.4.0"' in config
    assert 'APP_VERSION_LABEL = "V5"' in config
    assert "def _current_manual_tool_key" in main
    assert "def _reset_current_tool_parameters" in main
    assert 'self._tool_param_group.setTitle(tr(title_key))' in main
    assert 'self._compact_hint.setText(tr(hint_key))' in main
    assert '"reset_tool_parameters": "恢复当前工具默认参数"' in i18n


def test_brush_uses_one_live_exact_corridor_path():
    main = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
    worker = (ROOT / "ui" / "worker.py").read_text(encoding="utf-8")

    assert "snap_brush_stroke_mask" not in main
    # Brush still uses one live exact corridor; the separate AI selection
    # worker is allowed because it never runs from brush release.
    assert "LocalModelRecolorWorker" in worker
    brush_start = main.index("    def _on_hint_dab_added")
    brush_end = main.index("    def _undo_last_hint", brush_start)
    assert "_start_selection_model_recolor" not in main[brush_start:brush_end]
    assert "gap_close=0, mode=fill_mode" in main
    assert "snap_to_lineart=False, pupil_blend=False" in main
    assert "def _blend_masked_layer" not in main


def test_removed_region_boundary_ui_and_local_model_modules_stay_absent():
    main = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
    i18n = (ROOT / "ui" / "i18n.py").read_text(encoding="utf-8")

    assert "_chk_show_regions" not in main
    assert "show_regions_checkbox" not in i18n
    assert "show_regions = False" in main
    assert not (ROOT / "core" / "brush_snap.py").exists()
    assert not (ROOT / "core" / "model_region_recolor.py").exists()


def test_selection_adjust_rows_require_a_pending_selection():
    main = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
    assert "pending_selection = self._pending_selection_mask is not None" in main
    assert "pending_selection" in main
    assert 'for row_name in ("selection_adjust_tools", "selection_adjust_radius")' in main
