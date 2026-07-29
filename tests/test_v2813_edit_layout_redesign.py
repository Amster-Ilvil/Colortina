from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_edit_page_restores_single_page_layout_without_nested_detail_tabs():
    source = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
    assert "self._edit_detail_tabs = edit_detail_tabs" not in source
    assert "edit_detail_tabs.addTab" not in source
    assert 'edit_group, edit_layout = make_group(tr("edit_group"))' in source
    assert "edit_tab_layout.addWidget(edit_group, stretch=1)" in source
    assert "edit_tab_layout.addWidget(history_group, stretch=0)" in source
    assert "edit_layout.addStretch(1)" in source


def test_removed_brush_post_snap_ui_and_state_are_absent():
    source = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
    assert "_chk_brush_snap_lineart" not in source
    assert "snap_brush_stroke_mask" not in source


def test_selection_controls_have_distinct_rows_without_collisions():
    source = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
    assert 'self._selection_mode_label = QLabel(tr("selection_mode"))' in source
    assert "edit_grid.addWidget(selection_mode_box, 8, 1, 1, 2)" in source
    assert 'self._selection_feather_label = QLabel(tr("selection_feather"))' in source
    assert "edit_grid.addWidget(self._selection_feather_slider, 9, 1)" in source
    assert 'register_tool_row("selection_mode"' in source


def test_tools_use_two_rows_but_remain_on_same_page():
    source = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
    assert "tool_grid.addWidget(radio, i // 3, i % 3)" in source
    assert "detail_tabs.setCurrentIndex(1 if is_selection else 0)" not in source
