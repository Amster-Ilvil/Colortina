"""Colortina — Phase 2 editor.

Layout: page list (left) | canvas (center) | controls (right), per the
architecture doc. Only one mode — Auto — with an optional edit step:

    import -> auto colorize -> [not happy? draw hints -> regenerate] -> export
"""

from __future__ import annotations

import os

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QAction, QKeySequence, QShortcut, QIcon
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QListWidget, QListWidgetItem, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QSlider, QFileDialog, QSplitter,
    QStatusBar, QColorDialog, QButtonGroup, QRadioButton, QGroupBox,
    QMessageBox, QProgressBar, QCheckBox, QComboBox, QSpinBox, QInputDialog,
    QSizePolicy, QGridLayout, QTabWidget,
)

from config import Config
from core.hint_manager import HintManager
from core.edit_snapshot import capture_edit_state, restore_edit_state, snapshots_equal
from core.lineart_fill import lineart_region_recolor
from core.manual_style import adapt_rgb_to_style
from core.manual_edit import apply_brush_edit, apply_region_edit
from core.pdf_handler import extract_pages
from ui.canvas import HintCanvas
from ui.i18n import tr, set_language, get_language
from ui.worker import BatchColorizeWorker


class PageState:
    """Everything the app tracks for one page.

    Three layers, per the "don't lose the first AI result" request:
      - original_bgr : never modified — the imported black-and-white page.
      - ai_result_bgr: the most recent full mc-v2 run's output. Untouched
                       by region-fill edits, so "恢复AI结果" always has
                       a real fallback even after several manual touch-ups.
      - result_bgr   : the current "Edited" layer — what's displayed and
                       exported. Starts equal to ai_result_bgr after each
                       colorize, then diverges as region-fill edits land.
    """

    def __init__(self, path: str, original_bgr: np.ndarray | None = None):
        self.path = path
        self._original_bgr = original_bgr
        self._load_error: str | None = None
        self.ai_result_bgr: np.ndarray | None = None
        self.result_bgr: np.ndarray | None = None
        # Unfiltered final colorization used for non-cumulative filter changes.
        self.filter_base_bgr: np.ndarray | None = None
        self.hint_manager = HintManager()
        # Do not decode every page or build a region map during import.  Both
        # are loaded lazily when the page is first viewed / edited / generated.
        self.quality_report = None
        self.pipeline_diagnostics: dict = {}
        # Page-local explicit character binding.  Values are character IDs;
        # -1 means the user explicitly disabled identity locking.
        self.forced_character_matches: dict[int, int] = {}
        # Full-state undo/redo.  A snapshot includes the editable result,
        # stored AI base layer and manual hints.  This allows the very first
        # auto-colorize to be undone back to an empty/B&W state and prevents
        # operations that touch more than result_bgr from becoming irreversible.
        self.undo_stack: list[dict] = []
        self.redo_stack: list[dict] = []

    @property
    def original_bgr(self) -> np.ndarray:
        if self._original_bgr is None:
            from core.imageio import imread as _uimread
            image = _uimread(self.path)
            if image is None:
                self._load_error = f"无法读取图片：{self.path}"
                raise RuntimeError(self._load_error)
            self._original_bgr = image
        return self._original_bgr

    @property
    def is_loaded(self) -> bool:
        return self._original_bgr is not None

    def release_original(self) -> None:
        """Release source pixels for untouched, non-current pages."""
        if self.ai_result_bgr is None and self.result_bgr is None:
            self._original_bgr = None

    def edit_snapshot(self) -> dict:
        """Capture every user-visible, editable part of this page."""
        return capture_edit_state(self)

    def snapshot_equals_current(self, snapshot: dict) -> bool:
        return snapshots_equal(snapshot, self.edit_snapshot())

    def restore_snapshot(self, snapshot: dict) -> None:
        """Force the page back to an exact prior edit state."""
        restore_edit_state(self, snapshot)

    def push_undo(self):
        """Call BEFORE any visible mutation; empty states are valid snapshots."""
        snapshot = self.edit_snapshot()
        if not self.undo_stack or not snapshots_equal(self.undo_stack[-1], snapshot):
            self.undo_stack.append(snapshot)
            if len(self.undo_stack) > 20:
                self.undo_stack.pop(0)
        self.redo_stack.clear()

    @staticmethod
    def _snapshot_equal(left: dict, right: dict) -> bool:
        return snapshots_equal(left, right)

    def discard_unchanged_undo(self) -> None:
        """Drop a checkpoint when the attempted operation changed nothing."""
        if self.undo_stack and self.snapshot_equals_current(self.undo_stack[-1]):
            self.undo_stack.pop()



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(tr("window_title"))
        self.resize(1400, 900)
        self.setMinimumSize(1080, 680)  # keep all non-scrolling control tabs usable

        icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "assets", "icon.png")
        if os.path.isfile(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self._pages: dict[str, PageState] = {}
        self._current_path: str | None = None
        self._batch_worker: BatchColorizeWorker | None = None
        self._batch_errors: list[tuple[str, str]] = []
        self._brush_color = QColor(255, 120, 120)
        self._last_picked_rgb_raw: tuple[int, int, int] | None = None
        self._local_brush_stroke_active = False
        self._last_brush_was_local_edit = False
        self._last_local_edit_mode = "paint"
        self._brush_changed_during_stroke = False
        
        # Book-level state (persists across all pages of this session):
        # an optional extracted/loaded StyleProfile (overrides the style
        # preset combo when set) and one CharacterMemory per label that
        # needs multi-character consistency.
        self._character_memories: dict = {}
        # Character-aware Color Assignment: one CharacterLibrary per book,
        # built from color reference pages (角色决定颜色).
        self._character_library = None
        self._scene_palette = None
        self._current_project_path: str | None = None


        self._build_menu_bar()
        self._build_chrome()
        self._rebuild_central_widget()
        self._update_controls_enabled()
        self._refresh_character_diagnostics()

        # Drag & drop import (images / PDFs / folders) — desktop-GUI
        # convenience from Manga-Colorizer-GUI.
        self.setAcceptDrops(True)

    # ── Drag & drop import ─────────────────────────────────────────────

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event):
        image_paths = []
        pdf_paths = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if not path:
                continue
            if os.path.isdir(path):
                for root, _dirs, files in os.walk(path):
                    for f in files:
                        if f.lower().endswith(self._IMAGE_EXTS):
                            image_paths.append(os.path.join(root, f))
            elif path.lower().endswith(".pdf"):
                pdf_paths.append(path)
            elif path.lower().endswith(self._IMAGE_EXTS):
                image_paths.append(path)
        image_paths.sort(key=lambda p: (os.path.dirname(p),
                                        self._natural_key(os.path.basename(p))))
        added = self._add_pages_fast(image_paths)
        for pdf_path in pdf_paths:
            self._import_pdf_path(pdf_path)
        if added:
            self.statusBar().showMessage(tr("imported_n_pages").format(n=added), 5000)
        event.acceptProposedAction()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_responsive_density()

    def _apply_responsive_density(self):
        """Scale the non-scrolling control panel with the window.

        Controls remain on four tabs, while font size, row height, group
        margins and the preferred panel width adapt to the available desktop
        size.  No vertical or horizontal scrollbar is introduced.
        """
        panel = getattr(self, "_right_panel", None)
        tabs = getattr(self, "_right_tabs", None)
        if panel is None or tabs is None:
            return
        height = max(1, self.height())
        if height < 740:
            font_px, button_h, margin, spacing = 10, 24, 4, 2
        elif height < 900:
            font_px, button_h, margin, spacing = 11, 27, 6, 3
        else:
            font_px, button_h, margin, spacing = 12, 30, 8, 5

        preferred = int(np.clip(self.width() * 0.29, 380, 580))
        panel.setMinimumWidth(min(420, preferred))
        panel.setMaximumWidth(max(500, preferred))
        panel.setStyleSheet(f"""
            QWidget {{ font-size: {font_px}px; }}
            QGroupBox {{ margin-top: {font_px + 3}px; padding-top: 2px; }}
            QPushButton {{ min-height: {button_h}px; padding: 2px 5px; }}
            QComboBox, QSpinBox {{ min-height: {max(22, button_h - 2)}px; }}
            QTabBar::tab {{ min-height: {button_h}px; padding: 2px 5px; }}
        """)
        for group in panel.findChildren(QGroupBox):
            layout = group.layout()
            if layout is not None:
                layout.setContentsMargins(margin, margin, margin, margin)
                layout.setSpacing(spacing)
        for button in panel.findChildren(QPushButton):
            button.setMinimumHeight(button_h)
            button.setMaximumHeight(max(button_h + 8, 38))

    def _build_menu_bar(self):
        lang_menu = self.menuBar().addMenu(tr("menu_language"))
        self._lang_actions: dict = {}
        for code, key in (("zh", "lang_zh"), ("en", "lang_en")):
            action = QAction(tr(key), self, checkable=True)
            action.setChecked(get_language() == code)
            action.triggered.connect(lambda checked, c=code: self._set_language(c))
            lang_menu.addAction(action)
            self._lang_actions[code] = action

    def _set_language(self, code: str):
        """Switch UI language immediately (no restart needed) — rebuilds
        the central widget's text while preserving page list / current
        selections / style & quality choices / slider values."""
        if code == get_language():
            return
        set_language(code)
        for c, action in self._lang_actions.items():
            action.setChecked(c == code)
        self.setWindowTitle(tr("window_title"))
        self._lang_button.setText(tr("lang_button"))
        self._rebuild_central_widget()
        self.statusBar().showMessage(tr("ready"), 3000)

    def _toggle_language(self):
        self._set_language("en" if get_language() == "zh" else "zh")

    # ── Slider conversion ──────────────────────────────────────────────
    # Brush remains a 0-100 UI scale. Gap repair is shown directly in pixels
    # because it now means maximum local bridge length, not morphology strength.

    _BRUSH_PX_MIN, _BRUSH_PX_MAX = 2, 60
    _GAP_PX_MIN, _GAP_PX_MAX = 0, 24

    @classmethod
    def _brush_px_from_percent(cls, v: int) -> int:
        lo, hi = cls._BRUSH_PX_MIN, cls._BRUSH_PX_MAX
        return round(lo + (hi - lo) * (v / 100))

    @classmethod
    def _brush_percent_from_px(cls, px: int) -> int:
        lo, hi = cls._BRUSH_PX_MIN, cls._BRUSH_PX_MAX
        return round((px - lo) / (hi - lo) * 100)

    @classmethod
    def _gap_px_from_percent(cls, v: int) -> int:
        # v5.1 projects stored a 0-100 percentage. Values above the new pixel
        # maximum are interpreted as that legacy format during migration.
        v = int(v)
        if v > cls._GAP_PX_MAX:
            return round(cls._GAP_PX_MAX * (min(100, v) / 100.0))
        return max(cls._GAP_PX_MIN, min(cls._GAP_PX_MAX, v))

    @classmethod
    def _gap_percent_from_px(cls, px: int) -> int:
        return max(cls._GAP_PX_MIN, min(cls._GAP_PX_MAX, int(px)))

    # ── UI construction ───────────────────────────────────────────────

    def _build_chrome(self):
        """One-time setup for window furniture that survives language
        switches: status bar, progress indicator, keyboard shortcuts."""
        self.setStatusBar(QStatusBar())
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.setVisible(False)
        self._progress.setFixedWidth(160)
        self.statusBar().addPermanentWidget(self._progress)
        self.statusBar().showMessage(tr("ready"))

        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self._undo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, activated=self._redo)
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=self._redo)

    def _rebuild_central_widget(self):
        """(Re)builds the page list / canvas / controls panels. Called at
        startup and again on every language switch. Captures a handful of
        widget values beforehand (selected style/quality, slider values,
        checkbox state) and re-applies them after rebuilding, so switching
        languages mid-session doesn't reset the user's choices."""
        prev_style_key = getattr(self, "_style_combo", None) and self._style_combo.currentData()
        prev_quality_key = (getattr(self, "_quality_combo", None) and
                            self._quality_combo.currentData()) or Config.DEFAULT_QUALITY_KEY
        style_strength_widget = getattr(self, "_style_strength_slider", None)
        reference_strength_widget = getattr(self, "_reference_strength_slider", None)
        manual_strength_widget = getattr(self, "_manual_strength_slider", None)
        prev_style_strength = style_strength_widget.value() if style_strength_widget is not None else 100
        prev_reference_strength = reference_strength_widget.value() if reference_strength_widget is not None else 100
        prev_manual_strength = manual_strength_widget.value() if manual_strength_widget is not None else 100
        prev_style_color = getattr(self, "_style_color_slider", None)
        prev_style_color = prev_style_color.value() if prev_style_color is not None else 100
        prev_style_brightness = getattr(self, "_style_brightness_slider", None)
        prev_style_brightness = prev_style_brightness.value() if prev_style_brightness is not None else 100
        prev_style_warmth = getattr(self, "_style_warmth_slider", None)
        prev_style_warmth = prev_style_warmth.value() if prev_style_warmth is not None else 100
        prev_style_highlight = getattr(self, "_style_highlight_slider", None)
        prev_style_highlight = prev_style_highlight.value() if prev_style_highlight is not None else 100
        prev_style_softness = getattr(self, "_style_softness_slider", None)
        prev_style_softness = prev_style_softness.value() if prev_style_softness is not None else 100
        prev_style_flatten = getattr(self, "_style_flatten_slider", None)
        prev_style_flatten = prev_style_flatten.value() if prev_style_flatten is not None else 100
        prev_filter_brightness = getattr(self, "_filter_brightness_slider", None)
        prev_filter_brightness = prev_filter_brightness.value() if prev_filter_brightness is not None else 100
        prev_filter_contrast = getattr(self, "_filter_contrast_slider", None)
        prev_filter_contrast = prev_filter_contrast.value() if prev_filter_contrast is not None else 100
        prev_filter_saturation = getattr(self, "_filter_saturation_slider", None)
        prev_filter_saturation = prev_filter_saturation.value() if prev_filter_saturation is not None else 100
        prev_filter_warmth = getattr(self, "_filter_warmth_slider", None)
        prev_filter_warmth = prev_filter_warmth.value() if prev_filter_warmth is not None else 100
        prev_filter_shadow = getattr(self, "_filter_shadow_slider", None)
        prev_filter_shadow = prev_filter_shadow.value() if prev_filter_shadow is not None else 100
        prev_filter_highlight = getattr(self, "_filter_highlight_slider", None)
        prev_filter_highlight = prev_filter_highlight.value() if prev_filter_highlight is not None else 100
        prev_filter_scope = getattr(self, "_filter_scope_combo", None)
        prev_filter_scope = prev_filter_scope.currentData() if prev_filter_scope is not None else "current"
        prev_light3_intensity = getattr(self, "_light3_intensity_slider", None)
        prev_light3_intensity = prev_light3_intensity.value() if prev_light3_intensity is not None else 100
        prev_pastel_person = getattr(self, "_pastel_person_slider", None)
        prev_pastel_person = prev_pastel_person.value() if prev_pastel_person is not None else 100
        prev_pastel_hair = getattr(self, "_pastel_hair_slider", None)
        prev_pastel_hair = prev_pastel_hair.value() if prev_pastel_hair is not None else 100
        prev_pastel_skin = getattr(self, "_pastel_skin_slider", None)
        prev_pastel_skin = prev_pastel_skin.value() if prev_pastel_skin is not None else 100
        prev_pastel_eye = getattr(self, "_pastel_eye_slider", None)
        prev_pastel_eye = prev_pastel_eye.value() if prev_pastel_eye is not None else 100
        prev_pastel_clothing = getattr(self, "_pastel_clothing_slider", None)
        prev_pastel_clothing = prev_pastel_clothing.value() if prev_pastel_clothing is not None else 100
        prev_pastel_environment = getattr(self, "_pastel_environment_slider", None)
        prev_pastel_environment = prev_pastel_environment.value() if prev_pastel_environment is not None else 100
        prev_pastel_skin_warmth = getattr(self, "_pastel_skin_warmth_slider", None)
        prev_pastel_skin_warmth = prev_pastel_skin_warmth.value() if prev_pastel_skin_warmth is not None else 100
        prev_show_regions = bool(getattr(self, "_chk_show_regions", None) and
                                 self._chk_show_regions.isChecked())
        prev_char_mem = getattr(self, "_chk_character_memory", None)
        prev_char_mem = prev_char_mem.isChecked() if prev_char_mem is not None else Config.USE_CHARACTER_MEMORY
        prev_skip_colored = getattr(self, "_chk_skip_colored", None)
        prev_skip_colored = prev_skip_colored.isChecked() if prev_skip_colored is not None else True
        prev_picker_mode = self._current_eyedropper_mode() if hasattr(self, '_current_eyedropper_mode') else "point"
        prev_brush_pct = getattr(self, "_brush_slider", None)
        prev_brush_pct = prev_brush_pct.value() if prev_brush_pct is not None else self._brush_percent_from_px(12)
        prev_picker_lightness = getattr(self, "_picker_lightness_slider", None)
        prev_picker_lightness = prev_picker_lightness.value() if prev_picker_lightness is not None else 0
        prev_gap_pct = getattr(self, "_gap_close_slider", None)
        prev_gap_pct = prev_gap_pct.value() if prev_gap_pct is not None else self._gap_percent_from_px(4)
        prev_manual_match_style = bool(getattr(self, "_chk_manual_match_style", None) and self._chk_manual_match_style.isChecked()) if getattr(self, "_chk_manual_match_style", None) is not None else True
        prev_right_tab = getattr(self, "_right_tabs", None)
        prev_right_tab = prev_right_tab.currentIndex() if prev_right_tab is not None else 0

        old_central = self.centralWidget()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_canvas_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setObjectName("WorkspaceSplitter")
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setChildrenCollapsible(False)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(2, False)
        splitter.setHandleWidth(6)
        splitter.setSizes([300, 840, 460])

        container = QWidget()
        container.setObjectName("CentralWorkspace")
        root_layout = QVBoxLayout(container)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)
        root_layout.addWidget(splitter, stretch=1)
        self.setCentralWidget(container)
        if old_central is not None:
            old_central.deleteLater()

        if prev_style_key in {"monochrome_people", "monochrome_page"}:
            prev_style_key = "monochrome"
        if prev_style_key == "light":
            prev_style_key = "light2"
        if prev_style_key:
            idx = self._style_combo.findData(prev_style_key)
            if idx >= 0:
                self._style_combo.setCurrentIndex(idx)
        idx = self._quality_combo.findData(prev_quality_key)
        if idx >= 0:
            self._quality_combo.setCurrentIndex(idx)
        self._style_strength_slider.setValue(prev_style_strength)
        self._reference_strength_slider.setValue(prev_reference_strength)
        self._manual_strength_slider.setValue(prev_manual_strength)
        self._style_color_slider.setValue(prev_style_color)
        self._style_brightness_slider.setValue(prev_style_brightness)
        self._style_warmth_slider.setValue(prev_style_warmth)
        self._style_highlight_slider.setValue(prev_style_highlight)
        self._style_softness_slider.setValue(prev_style_softness)
        self._style_flatten_slider.setValue(prev_style_flatten)
        self._filter_brightness_slider.setValue(prev_filter_brightness)
        self._filter_contrast_slider.setValue(prev_filter_contrast)
        self._filter_saturation_slider.setValue(prev_filter_saturation)
        self._filter_warmth_slider.setValue(prev_filter_warmth)
        self._filter_shadow_slider.setValue(prev_filter_shadow)
        self._filter_highlight_slider.setValue(prev_filter_highlight)
        idx = self._filter_scope_combo.findData(prev_filter_scope)
        if idx >= 0:
            self._filter_scope_combo.setCurrentIndex(idx)
        self._light3_intensity_slider.setValue(prev_light3_intensity)
        self._pastel_person_slider.setValue(prev_pastel_person)
        self._pastel_hair_slider.setValue(prev_pastel_hair)
        self._pastel_skin_slider.setValue(prev_pastel_skin)
        self._pastel_eye_slider.setValue(prev_pastel_eye)
        self._pastel_clothing_slider.setValue(prev_pastel_clothing)
        self._pastel_environment_slider.setValue(prev_pastel_environment)
        self._pastel_skin_warmth_slider.setValue(prev_pastel_skin_warmth)
        self._chk_show_regions.setChecked(prev_show_regions)
        self._chk_character_memory.setChecked(prev_char_mem)
        self._chk_skip_colored.setChecked(prev_skip_colored)
        self._set_eyedropper_mode(prev_picker_mode)
        self._brush_slider.setValue(prev_brush_pct)
        self._picker_lightness_slider.setValue(prev_picker_lightness)
        self._gap_close_slider.setValue(prev_gap_pct)
        if getattr(self, "_chk_manual_match_style", None) is not None:
            self._chk_manual_match_style.setChecked(prev_manual_match_style)
        self._right_tabs.setCurrentIndex(max(0, min(prev_right_tab, self._right_tabs.count() - 1)))
        self._canvas.set_brush_color(self._brush_color)
        self._update_color_swatch()
        self._sync_workspace_nav()
        self._apply_workspace_theme()
        self._update_tool_specific_visibility()
        self._update_pastel_controls_visibility()

        self._repopulate_page_list()
        self._update_controls_enabled()
        self._update_device_label()
        self._apply_responsive_density()

    def _build_workspace_header(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("WorkspaceHeader")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        title = QLabel("Colortina · Manga Color Studio")
        title.setObjectName("WorkspaceHeaderTitle")
        layout.addWidget(title)
        layout.addSpacing(12)

        self._nav_group = QButtonGroup(self)
        self._workspace_nav_buttons = []

        def add_nav(label: str, handler, *, checked: bool = False):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(checked)
            btn.setObjectName("WorkspaceNavButton")
            btn.clicked.connect(handler)
            self._nav_group.addButton(btn)
            self._workspace_nav_buttons.append(btn)
            layout.addWidget(btn)
            return btn

        self._nav_pages_btn = add_nav("页面", lambda: self._page_list.setFocus(), checked=True)
        self._nav_render_btn = add_nav(tr("right_tab_render"), lambda: self._right_tabs.setCurrentIndex(0))
        self._nav_edit_btn = add_nav(tr("right_tab_edit"), lambda: self._right_tabs.setCurrentIndex(1))
        self._nav_output_btn = add_nav(tr("right_tab_output"), lambda: self._right_tabs.setCurrentIndex(2))
        layout.addStretch(1)
        return bar

    def _sync_workspace_nav(self):
        if not hasattr(self, "_right_tabs"):
            return
        mapping = {
            0: getattr(self, "_nav_render_btn", None),
            1: getattr(self, "_nav_edit_btn", None),
            2: getattr(self, "_nav_output_btn", None),
        }
        current = self._right_tabs.currentIndex()
        btn = mapping.get(current)
        if btn is not None:
            btn.setChecked(True)

    def _apply_workspace_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget#CentralWorkspace { background: #eaf4ff; color: #243244; }
            QMenuBar, QStatusBar { background: #dbe9f7; color: #29405b; }
            QLabel#WorkspaceTitle { font-size: 18px; font-weight: 700; color: #36506f; padding: 6px 2px 2px 2px; }
            QLabel#VersionLabel { color: #6c7f94; font-size: 11px; font-weight: 700; padding: 3px 4px; }
            QPushButton#GhostButton {
                background: #f4f9ff; border: 1px solid #bfd3e8; border-radius: 10px;
                padding: 7px 14px; font-weight: 600; }
            QWidget#LeftPanel, QWidget#CanvasPanel, QWidget#RightPanel {
                background: #f5faff; border: 1px solid #c9dcee; border-radius: 16px; }
            QLabel#PanelSectionTitle { font-size: 14px; font-weight: 700; color: #36506f; }
            QListWidget { background: white; border: 1px solid #cadcf0; border-radius: 12px; padding: 6px; }
            QListWidget::item { border-radius: 9px; padding: 8px 10px; margin: 2px 0px; }
            QListWidget::item:selected { background: #d7e9ff; color: #1f4f88; }
            QPushButton { background: #ffffff; border: 1px solid #bed1e4; border-radius: 10px; padding: 6px 10px; }
            QPushButton:hover { background: #f3f8ff; border-color: #8fb4de; }
            QPushButton:pressed { background: #e5f0ff; }
            QPushButton:checked { background: #dcecff; border-color: #76a4dc; }
            QGroupBox { font-weight: 700; border: 1px solid #cfe0f0; border-radius: 14px; margin-top: 14px; background: #fbfdff; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 2px 6px; color: #36506f; }
            QComboBox, QSpinBox, QLineEdit { background: #ffffff; border: 1px solid #bfd1e4; border-radius: 8px; padding: 4px 8px; }
            QComboBox QAbstractItemView { background: #ffffff; border: 1px solid #9fbce0; selection-background-color: #dcecff; selection-color: #1f4f88; outline: 0; }
            QTabWidget::pane { border: 1px solid #cfdeee; border-radius: 14px; top: -1px; background: #fbfdff; }
            QTabBar::tab { background: #e4effa; border: 1px solid #cfdeee; border-bottom: none;
                           border-top-left-radius: 11px; border-top-right-radius: 11px; padding: 8px 12px; margin-right: 4px; }
            QTabBar::tab:selected { background: #fbfdff; color: #1f4f88; font-weight: 700; }
            QSlider::groove:horizontal { height: 6px; background: #d5e4f1; border-radius: 3px; }
            QSlider::handle:horizontal { width: 16px; margin: -5px 0; background: #5a8ee6; border-radius: 8px; }
            QSplitter::handle { background: transparent; }
            QRadioButton { spacing: 7px; }
        """)

    def _build_left_panel(self) -> QWidget:
        w = QWidget()
        w.setObjectName("LeftPanel")
        self._left_panel = w
        w.setMinimumWidth(300)
        w.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(w)

        header = QLabel("Colortina Studio")
        header.setObjectName("WorkspaceTitle")
        layout.addWidget(header)

        self._lang_button = QPushButton(tr("lang_button"))
        self._lang_button.setObjectName("GhostButton")
        self._lang_button.setToolTip(tr("menu_language"))
        self._lang_button.clicked.connect(self._toggle_language)
        layout.addWidget(self._lang_button)

        btn_import_images = QPushButton(tr("import_images"))
        btn_import_images.clicked.connect(self._import_images)
        btn_import_pdf = QPushButton(tr("import_pdf"))
        btn_import_pdf.clicked.connect(self._import_pdf)
        btn_import_folder = QPushButton(tr("import_folder"))
        btn_import_folder.clicked.connect(self._import_folder)

        self._page_list = QListWidget()
        self._page_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        # Filenames are always fully visible.  Long names wrap to additional
        # lines; a horizontal scrollbar is never introduced on the left.
        self._page_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._page_list.setWordWrap(True)
        self._page_list.setTextElideMode(Qt.TextElideMode.ElideNone)
        self._page_list.setResizeMode(QListView.ResizeMode.Adjust)
        self._page_list.setUniformItemSizes(False)
        self._page_list.setStyleSheet("QListWidget::item { padding: 4px 6px; }")
        self._page_list.currentItemChanged.connect(self._on_page_selected)

        select_row = QHBoxLayout()
        btn_select_all = QPushButton(tr("select_all"))
        btn_select_all.clicked.connect(lambda: self._page_list.selectAll())
        btn_select_none = QPushButton(tr("select_none"))
        btn_select_none.clicked.connect(lambda: self._page_list.clearSelection())
        select_row.addWidget(btn_select_all)
        select_row.addWidget(btn_select_none)

        layout.addWidget(btn_import_images)
        layout.addWidget(btn_import_pdf)
        layout.addWidget(btn_import_folder)
        layout.addWidget(QLabel(tr("pages_label")))
        layout.addLayout(select_row)
        layout.addWidget(self._page_list, stretch=1)

        manage_row = QHBoxLayout()
        btn_move_up = QPushButton(tr("move_up"))
        btn_move_up.clicked.connect(lambda: self._move_selected_page(-1))
        btn_move_down = QPushButton(tr("move_down"))
        btn_move_down.clicked.connect(lambda: self._move_selected_page(1))
        manage_row.addWidget(btn_move_up)
        manage_row.addWidget(btn_move_down)
        layout.addLayout(manage_row)

        btn_delete = QPushButton(tr("delete_pages"))
        btn_delete.clicked.connect(self._delete_selected_pages)
        layout.addWidget(btn_delete)

        project_row = QHBoxLayout()
        btn_load_project = QPushButton(tr("load_project"))
        btn_load_project.clicked.connect(self._load_project)
        btn_save_project = QPushButton(tr("save_project"))
        btn_save_project.clicked.connect(self._save_project)
        project_row.addWidget(btn_load_project)
        project_row.addWidget(btn_save_project)
        layout.addLayout(project_row)

        self._version_label = QLabel("V2")
        self._version_label.setObjectName("VersionLabel")
        self._version_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._version_label.setToolTip("Colortina V2")
        layout.addWidget(self._version_label)
        return w

    def _build_canvas_panel(self) -> QWidget:
        w = QWidget()
        w.setObjectName("CanvasPanel")
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        toolbar = QHBoxLayout()
        title = QLabel("画布 / 预览")
        title.setObjectName("PanelSectionTitle")
        toolbar.addWidget(title)
        toolbar.addSpacing(8)
        btn_fit = QPushButton(tr("fit_view"))
        btn_fit.clicked.connect(lambda: self._canvas.fit_view())
        btn_zoom_in = QPushButton("＋")
        btn_zoom_in.setFixedWidth(32)
        btn_zoom_in.clicked.connect(self._canvas_zoom_in)
        btn_zoom_out = QPushButton("－")
        btn_zoom_out.setFixedWidth(32)
        btn_zoom_out.clicked.connect(self._canvas_zoom_out)
        toolbar.addStretch(1)
        toolbar.addWidget(btn_fit)
        toolbar.addWidget(btn_zoom_out)
        toolbar.addWidget(btn_zoom_in)
        toolbar.addStretch(1)

        self._canvas = HintCanvas()
        self._canvas.hint_dab_added.connect(self._on_hint_dab_added)
        self._canvas.brush_stroke_started.connect(self._on_brush_stroke_started)
        self._canvas.brush_stroke_finished.connect(self._on_brush_stroke_finished)
        self._canvas.color_picked.connect(self._on_color_picked)
        self._canvas.region_fill_requested.connect(self._on_region_fill_requested)

        layout.addLayout(toolbar)
        layout.addWidget(self._canvas, stretch=1)
        return w

    def _build_right_panel(self) -> QWidget:
        """Build a non-scrolling, responsive right control panel.

        The old implementation placed every group in one very tall scrolling panel.
        On ordinary laptop windows the most important buttons were below the
        fold.  The panel is now split into four compact tabs.  Every tab fits in
        the supported minimum window height, every control expands with the
        splitter, and neither the panel nor its tabs use scrollbars.
        """
        panel = QWidget()
        panel.setObjectName("RightPanel")
        self._right_panel = panel
        panel.setMinimumWidth(390)
        panel.setMaximumWidth(560)
        panel.setSizePolicy(QSizePolicy.Policy.Preferred,
                            QSizePolicy.Policy.Expanding)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        self._device_label = QLabel(f"{tr('device_label')}—")
        self._device_label.setSizePolicy(QSizePolicy.Policy.Expanding,
                                         QSizePolicy.Policy.Fixed)
        outer.addWidget(self._device_label)

        self._right_tabs = QTabWidget()
        self._right_tabs.setDocumentMode(True)
        self._right_tabs.setUsesScrollButtons(False)
        self._right_tabs.tabBar().setExpanding(True)
        self._right_tabs.setElideMode(Qt.TextElideMode.ElideNone)
        self._right_tabs.currentChanged.connect(lambda _=None: self._sync_workspace_nav())
        outer.addWidget(self._right_tabs, stretch=1)

        def make_tab() -> tuple[QWidget, QVBoxLayout]:
            tab = QWidget()
            tab.setSizePolicy(QSizePolicy.Policy.Expanding,
                              QSizePolicy.Policy.Expanding)
            lay = QVBoxLayout(tab)
            lay.setContentsMargins(6, 6, 6, 6)
            lay.setSpacing(5)
            return tab, lay

        def make_group(title: str) -> tuple[QGroupBox, QVBoxLayout]:
            group = QGroupBox(title)
            group.setSizePolicy(QSizePolicy.Policy.Expanding,
                                QSizePolicy.Policy.Expanding)
            lay = QVBoxLayout(group)
            lay.setContentsMargins(8, 6, 8, 7)
            lay.setSpacing(4)
            return group, lay

        def tune_button(button: QPushButton, minimum_height: int = 29) -> QPushButton:
            button.setMinimumHeight(minimum_height)
            button.setSizePolicy(QSizePolicy.Policy.Expanding,
                                 QSizePolicy.Policy.Fixed)
            return button

        def add_button_grid(layout: QVBoxLayout, buttons: list[QPushButton],
                            columns: int = 2) -> None:
            grid = QGridLayout()
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(5)
            grid.setVerticalSpacing(4)
            for i, button in enumerate(buttons):
                tune_button(button)
                grid.addWidget(button, i // columns, i % columns)
            for col in range(columns):
                grid.setColumnStretch(col, 1)
            layout.addLayout(grid)

        from core.presets import STYLE_PRESETS, QUALITY_PRESETS

        # ── Tab 1: rendering and automatic colourization ──────────────
        render_tab, render_layout = make_tab()
        style_group, style_layout = make_group(tr("style_quality_group"))

        style_grid = QGridLayout()
        style_grid.setContentsMargins(0, 0, 0, 0)
        style_grid.setHorizontalSpacing(6)
        style_grid.setVerticalSpacing(4)
        style_grid.addWidget(QLabel(tr("style_label")), 0, 0)
        self._style_combo = QComboBox()
        visible_style_keys = ["none", "light2", "light3"]
        for key in visible_style_keys:
            preset = STYLE_PRESETS.get(key)
            if preset is not None:
                self._style_combo.addItem(preset.label, key)
        self._style_combo.currentIndexChanged.connect(self._on_style_combo_changed)
        style_grid.addWidget(self._style_combo, 0, 1, 1, 2)
        style_grid.addWidget(QLabel(tr("quality_label")), 1, 0)
        self._quality_combo = QComboBox()
        self._quality_combo.addItem("Fast", "draft")
        self._quality_combo.setEnabled(False)
        self._quality_combo.setToolTip("已固定为最快生成模式")
        style_grid.addWidget(self._quality_combo, 1, 1, 1, 2)
        style_grid.setColumnStretch(1, 1)
        style_layout.addLayout(style_grid)

        strength_grid = QGridLayout()
        strength_grid.setContentsMargins(0, 0, 0, 0)
        strength_grid.setHorizontalSpacing(5)
        strength_grid.setVerticalSpacing(3)

        def add_strength_row(row: int, label_key: str, attr_name: str,
                             default: int = 100):
            label = QLabel(tr(label_key))
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(default)
            spin = QSpinBox()
            spin.setRange(0, 100)
            spin.setSuffix("%")
            spin.setValue(default)
            spin.setMaximumWidth(72)
            slider.valueChanged.connect(spin.setValue)
            spin.valueChanged.connect(slider.setValue)
            strength_grid.addWidget(label, row, 0)
            strength_grid.addWidget(slider, row, 1)
            strength_grid.addWidget(spin, row, 2)
            strength_grid.setColumnStretch(1, 1)
            setattr(self, attr_name, slider)

        add_strength_row(0, "style_strength_label", "_style_strength_slider")
        add_strength_row(1, "reference_strength_label", "_reference_strength_slider")
        add_strength_row(2, "manual_strength_label", "_manual_strength_slider")
        style_layout.addLayout(strength_grid)
        self._style_strength_hint_label = QLabel(tr("style_strength_hint"))
        self._style_strength_hint_label.setWordWrap(True)
        self._style_strength_hint_label.setStyleSheet("color: #6c7f94; font-size: 10px;")
        style_layout.addWidget(self._style_strength_hint_label)

        # Fine-tuning now lives inside the render tab again, alongside the
        # colourization controls, instead of being a separate main tab.
        self._style_fine_group = QGroupBox(tr("style_fine_controls_group"))
        fine_layout = QVBoxLayout(self._style_fine_group)
        fine_layout.setContentsMargins(10, 10, 10, 8)
        fine_layout.setSpacing(4)
        fine_hint = QLabel(tr("style_fine_controls_hint"))
        fine_hint.setWordWrap(True)
        fine_hint.setStyleSheet("color: #6c7f94; font-size: 10px;")
        fine_layout.addWidget(fine_hint)
        self._style_fine_active_label = QLabel()
        self._style_fine_active_label.setWordWrap(True)
        self._style_fine_active_label.setStyleSheet("font-weight: 600;")
        fine_layout.addWidget(self._style_fine_active_label)
        fine_apply_hint = QLabel(tr("style_fine_apply_hint"))
        fine_apply_hint.setWordWrap(True)
        fine_apply_hint.setStyleSheet("color: #6c7f94; font-size: 10px;")
        fine_layout.addWidget(fine_apply_hint)
        fine_grid = QGridLayout()
        fine_grid.setContentsMargins(0, 0, 0, 0)
        fine_grid.setHorizontalSpacing(5)
        fine_grid.setVerticalSpacing(3)

        def add_tune_row(grid: QGridLayout, row: int, label_key: str, attr_name: str,
                         min_value: int = 0, max_value: int = 180, default: int = 100):
            label = QLabel(tr(label_key))
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(min_value, max_value)
            slider.setValue(default)
            spin = QSpinBox()
            spin.setRange(min_value, max_value)
            spin.setSuffix("%")
            spin.setValue(default)
            spin.setMaximumWidth(72)
            slider.valueChanged.connect(spin.setValue)
            spin.valueChanged.connect(slider.setValue)
            grid.addWidget(label, row, 0)
            grid.addWidget(slider, row, 1)
            grid.addWidget(spin, row, 2)
            grid.setColumnStretch(1, 1)
            setattr(self, attr_name, slider)

        add_tune_row(fine_grid, 0, "style_color_strength", "_style_color_slider")
        add_tune_row(fine_grid, 1, "style_brightness_strength", "_style_brightness_slider")
        add_tune_row(fine_grid, 2, "style_warmth_strength", "_style_warmth_slider")
        add_tune_row(fine_grid, 3, "style_highlight_strength", "_style_highlight_slider")
        add_tune_row(fine_grid, 4, "style_softness_strength", "_style_softness_slider")
        add_tune_row(fine_grid, 5, "style_flatten_strength", "_style_flatten_slider")
        add_tune_row(fine_grid, 6, "light3_intensity", "_light3_intensity_slider", 0, 200, 100)
        self._light3_intensity_label = fine_grid.itemAtPosition(6, 0).widget()
        self._light3_intensity_spin = fine_grid.itemAtPosition(6, 2).widget()
        self._light3_intensity_slider.valueChanged.connect(self._update_color_swatch)
        for slider in (self._style_color_slider, self._style_brightness_slider,
                       self._style_warmth_slider, self._style_highlight_slider,
                       self._style_softness_slider, self._style_flatten_slider):
            slider.valueChanged.connect(self._update_color_swatch)
        fine_layout.addLayout(fine_grid)
        self._btn_reset_style_fine = QPushButton(tr("reset_style_fine_btn"))
        self._btn_reset_style_fine.clicked.connect(self._reset_style_fine_tuning)
        fine_layout.addWidget(self._btn_reset_style_fine)

        self._filter_group = QGroupBox(tr("image_filter_group"))
        filter_layout = QVBoxLayout(self._filter_group)
        filter_layout.setContentsMargins(10, 10, 10, 8)
        filter_layout.setSpacing(4)
        filter_hint = QLabel(tr("image_filter_hint"))
        filter_hint.setWordWrap(True)
        filter_hint.setStyleSheet("color: #6c7f94; font-size: 10px;")
        filter_layout.addWidget(filter_hint)
        filter_grid = QGridLayout()
        filter_grid.setContentsMargins(0, 0, 0, 0)
        filter_grid.setHorizontalSpacing(5)
        filter_grid.setVerticalSpacing(3)
        add_tune_row(filter_grid, 0, "filter_brightness", "_filter_brightness_slider", 0, 200, 100)
        add_tune_row(filter_grid, 1, "filter_contrast", "_filter_contrast_slider", 0, 200, 100)
        add_tune_row(filter_grid, 2, "filter_saturation", "_filter_saturation_slider", 0, 200, 100)
        add_tune_row(filter_grid, 3, "filter_warmth", "_filter_warmth_slider", 0, 200, 100)
        add_tune_row(filter_grid, 4, "filter_shadow_lift", "_filter_shadow_slider", 0, 200, 100)
        add_tune_row(filter_grid, 5, "filter_highlight", "_filter_highlight_slider", 0, 200, 100)
        filter_layout.addLayout(filter_grid)
        filter_scope_row = QHBoxLayout()
        filter_scope_row.addWidget(QLabel(tr("filter_scope_label")))
        self._filter_scope_combo = QComboBox()
        self._filter_scope_combo.addItem(tr("filter_scope_current"), "current")
        self._filter_scope_combo.addItem(tr("filter_scope_all"), "all")
        filter_scope_row.addWidget(self._filter_scope_combo, stretch=1)
        filter_layout.addLayout(filter_scope_row)
        self._btn_filter_reset = QPushButton(tr("filter_reset"))
        self._btn_filter_reset.clicked.connect(self._reset_filter_controls)
        self._btn_filter_apply = QPushButton(tr("filter_apply"))
        self._btn_filter_apply.clicked.connect(self._apply_filter_to_scope)
        filter_button_row = QHBoxLayout()
        filter_button_row.addWidget(self._btn_filter_reset)
        filter_button_row.addWidget(self._btn_filter_apply)
        filter_layout.addLayout(filter_button_row)

        self._pastel_controls_group = QGroupBox(tr("pastel_controls_group"))
        pastel_layout = QVBoxLayout(self._pastel_controls_group)
        pastel_layout.setContentsMargins(10, 10, 10, 8)
        pastel_layout.setSpacing(4)
        pastel_hint = QLabel(tr("pastel_controls_hint"))
        pastel_hint.setWordWrap(True)
        pastel_hint.setStyleSheet("color: #6c7f94; font-size: 10px;")
        pastel_layout.addWidget(pastel_hint)
        pastel_grid = QGridLayout()
        pastel_grid.setContentsMargins(0, 0, 0, 0)
        pastel_grid.setHorizontalSpacing(5)
        pastel_grid.setVerticalSpacing(3)

        def add_pastel_row(row: int, label_key: str, attr_name: str, default: int = 100):
            add_tune_row(pastel_grid, row, label_key, attr_name, 0, 180, default)

        add_pastel_row(0, "pastel_person_strength", "_pastel_person_slider")
        add_pastel_row(1, "pastel_hair_strength", "_pastel_hair_slider")
        add_pastel_row(2, "pastel_skin_strength", "_pastel_skin_slider")
        add_pastel_row(3, "pastel_eye_strength", "_pastel_eye_slider")
        add_pastel_row(4, "pastel_clothing_strength", "_pastel_clothing_slider")
        add_pastel_row(5, "pastel_environment_strength", "_pastel_environment_slider", 100)
        add_pastel_row(6, "pastel_skin_warmth", "_pastel_skin_warmth_slider", 100)
        pastel_layout.addLayout(pastel_grid)
        style_layout.addWidget(self._pastel_controls_group)


        self._chk_character_memory = QCheckBox(tr("character_memory_checkbox"))
        self._chk_character_memory.setChecked(False)
        self._chk_character_memory.hide()

        self._chk_skip_colored = QCheckBox(tr("skip_colored_checkbox"))
        self._chk_skip_colored.setChecked(True)
        self._chk_skip_colored.setToolTip(tr("skip_colored_tooltip"))
        style_layout.addWidget(self._chk_skip_colored)

        auto_group, auto_layout = make_group(tr("auto_group"))
        self._btn_auto = tune_button(QPushButton(tr("auto_btn")), 40)
        self._btn_auto.clicked.connect(self._run_auto_colorize)
        auto_layout.addWidget(self._btn_auto)
        self._auto_status_label = QLabel("准备就绪")
        self._auto_status_label.setWordWrap(True)
        self._auto_status_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._auto_status_label.setStyleSheet("color: #666;")
        self._auto_status_label.setSizePolicy(QSizePolicy.Policy.Expanding,
                                              QSizePolicy.Policy.Expanding)
        auto_layout.addWidget(self._auto_status_label, stretch=1)
        render_detail_tabs = QTabWidget()
        render_detail_tabs.setDocumentMode(True)
        render_detail_tabs.setUsesScrollButtons(False)
        render_detail_tabs.tabBar().setExpanding(True)
        self._render_detail_tabs = render_detail_tabs

        detail_fine_tab, detail_fine_layout = make_tab()
        detail_fine_layout.addWidget(self._style_fine_group, stretch=1)
        render_detail_tabs.addTab(detail_fine_tab, tr("style_detail_tab_fine"))

        detail_filter_tab, detail_filter_layout = make_tab()
        detail_filter_layout.addWidget(self._filter_group, stretch=1)
        render_detail_tabs.addTab(detail_filter_tab, tr("image_filter_group"))

        render_layout.addWidget(style_group, stretch=2)
        render_layout.addWidget(render_detail_tabs, stretch=4)
        # Keep the primary action at the very bottom of the render page.
        render_layout.addWidget(auto_group, stretch=1)
        self._right_tabs.addTab(render_tab, tr("right_tab_render"))

        # ── Tab 2: manual editing ─────────────────────────────────────
        edit_tab, edit_tab_layout = make_tab()
        edit_group, edit_layout = make_group(tr("edit_group"))

        tool_row = QHBoxLayout()
        tool_row.setSpacing(8)
        self._tool_group = QButtonGroup(self)
        self._radio_brush = QRadioButton(tr("tool_brush"))
        self._radio_brush.setChecked(True)
        self._radio_eyedropper = QRadioButton(tr("tool_eyedropper"))
        self._radio_bucket = QRadioButton(tr("tool_bucket"))
        for radio in (self._radio_brush, self._radio_eyedropper,
                      self._radio_bucket):
            self._tool_group.addButton(radio)
            tool_row.addWidget(radio)
        self._radio_brush.toggled.connect(self._on_tool_changed)
        self._radio_eyedropper.toggled.connect(self._on_tool_changed)
        self._radio_bucket.toggled.connect(self._on_tool_changed)
        tool_row.addStretch(1)
        edit_layout.addLayout(tool_row)

        edit_grid = QGridLayout()
        edit_grid.setContentsMargins(0, 0, 0, 0)
        edit_grid.setHorizontalSpacing(6)
        edit_grid.setVerticalSpacing(4)
        self._eyedropper_mode_label = QLabel(tr("eyedropper_mode_label"))
        edit_grid.addWidget(self._eyedropper_mode_label, 0, 0)
        eyedrop_box = QWidget()
        self._eyedropper_mode_box = eyedrop_box
        eyedrop_layout = QHBoxLayout(eyedrop_box)
        eyedrop_layout.setContentsMargins(0, 0, 0, 0)
        eyedrop_layout.setSpacing(8)
        self._eyedropper_mode_group = QButtonGroup(self)
        self._eyedropper_mode_group.setExclusive(True)
        self._eyedropper_mode_point = QCheckBox(tr("eyedropper_mode_point"))
        self._eyedropper_mode_region = QCheckBox(tr("eyedropper_mode_region"))
        self._eyedropper_mode_group.addButton(self._eyedropper_mode_point)
        self._eyedropper_mode_group.addButton(self._eyedropper_mode_region)
        self._eyedropper_mode_point.toggled.connect(lambda checked: checked and self._set_eyedropper_mode("point"))
        self._eyedropper_mode_region.toggled.connect(lambda checked: checked and self._set_eyedropper_mode("region"))
        eyedrop_layout.addWidget(self._eyedropper_mode_point)
        eyedrop_layout.addWidget(self._eyedropper_mode_region)
        eyedrop_layout.addStretch(1)
        edit_grid.addWidget(eyedrop_box, 0, 1, 1, 2)

        edit_grid.addWidget(QLabel(tr("color_label")), 1, 0)
        self._color_swatch = QPushButton()
        self._color_swatch.setFixedSize(54, 30)
        self._color_swatch.clicked.connect(self._pick_color)
        edit_grid.addWidget(self._color_swatch, 1, 1)
        self._current_color_info = QLabel()
        self._current_color_info.setWordWrap(True)
        self._current_color_info.setStyleSheet("color: #5d6f84; font-size: 10px;")
        edit_grid.addWidget(self._current_color_info, 1, 2)
        self._update_color_swatch()
        self._set_eyedropper_mode("point")

        edit_grid.addWidget(QLabel(tr("brush_size_label")), 2, 0)
        self._brush_slider = QSlider(Qt.Orientation.Horizontal)
        self._brush_slider.setRange(0, 100)
        self._brush_slider.setValue(self._brush_percent_from_px(12))
        self._brush_spin = QSpinBox()
        self._brush_spin.setRange(0, 100)
        self._brush_spin.setValue(self._brush_slider.value())
        self._brush_spin.setMaximumWidth(72)
        self._brush_slider.valueChanged.connect(self._brush_spin.setValue)
        self._brush_spin.valueChanged.connect(self._brush_slider.setValue)
        self._brush_slider.valueChanged.connect(
            lambda v: self._canvas.set_brush_radius(self._brush_px_from_percent(v)))
        edit_grid.addWidget(self._brush_slider, 2, 1)
        edit_grid.addWidget(self._brush_spin, 2, 2)

        edit_grid.addWidget(QLabel(tr("picker_lightness_label")), 3, 0)
        self._picker_lightness_slider = QSlider(Qt.Orientation.Horizontal)
        self._picker_lightness_slider.setRange(-100, 100)
        self._picker_lightness_slider.setValue(0)
        self._picker_lightness_slider.setToolTip(tr("picker_lightness_hint"))
        self._picker_lightness_spin = QSpinBox()
        self._picker_lightness_spin.setRange(-100, 100)
        self._picker_lightness_spin.setValue(0)
        self._picker_lightness_spin.setSuffix("%")
        self._picker_lightness_spin.setMaximumWidth(72)
        self._picker_lightness_spin.setToolTip(tr("picker_lightness_hint"))
        self._picker_lightness_slider.valueChanged.connect(self._on_picker_lightness_changed)
        self._picker_lightness_spin.valueChanged.connect(self._picker_lightness_slider.setValue)
        edit_grid.addWidget(self._picker_lightness_slider, 3, 1)
        edit_grid.addWidget(self._picker_lightness_spin, 3, 2)

        edit_grid.addWidget(QLabel(tr("gap_close_label")), 4, 0)
        self._gap_close_slider = QSlider(Qt.Orientation.Horizontal)
        self._gap_close_slider.setRange(self._GAP_PX_MIN, self._GAP_PX_MAX)
        self._gap_close_slider.setValue(6)
        self._gap_close_slider.setToolTip(tr("gap_close_tooltip"))
        self._gap_close_spin = QSpinBox()
        self._gap_close_spin.setRange(self._GAP_PX_MIN, self._GAP_PX_MAX)
        self._gap_close_spin.setSuffix(" px")
        self._gap_close_spin.setValue(self._gap_close_slider.value())
        self._gap_close_spin.setMaximumWidth(72)
        self._gap_close_spin.setToolTip(tr("gap_close_tooltip"))
        self._gap_close_slider.valueChanged.connect(self._gap_close_spin.setValue)
        self._gap_close_spin.valueChanged.connect(self._gap_close_slider.setValue)
        edit_grid.addWidget(self._gap_close_slider, 4, 1)
        edit_grid.addWidget(self._gap_close_spin, 4, 2)

        edit_grid.addWidget(QLabel(tr("fill_mode_label")), 5, 0)
        self._fill_mode_combo = QComboBox()
        self._fill_mode_combo.addItem(tr("fill_mode_shift"), "shift")
        self._fill_mode_combo.addItem(tr("fill_mode_shading"), "shading")
        self._fill_mode_combo.addItem(tr("fill_mode_flat"), "flat")
        self._fill_mode_combo.setToolTip(tr("fill_mode_hint"))
        edit_grid.addWidget(self._fill_mode_combo, 5, 1, 1, 2)
        edit_grid.setColumnStretch(1, 1)
        edit_layout.addLayout(edit_grid)

        self._chk_manual_match_style = QCheckBox(tr("manual_match_style_checkbox"))
        self._chk_manual_match_style.setChecked(True)
        self._chk_manual_match_style.setToolTip(tr("manual_match_style_hint"))
        self._chk_manual_match_style.toggled.connect(self._update_color_swatch)
        edit_layout.addWidget(self._chk_manual_match_style)

        bucket_hint = QLabel(tr("bucket_hint_compact"))
        bucket_hint.setWordWrap(True)
        bucket_hint.setToolTip(tr("bucket_hint"))
        bucket_hint.setStyleSheet("color: #777; font-size: 11px;")
        edit_layout.addWidget(bucket_hint)
        picker_hint = QLabel(tr("picker_keep_tool_hint"))
        picker_hint.setWordWrap(True)
        picker_hint.setStyleSheet("color: #777; font-size: 11px;")
        edit_layout.addWidget(picker_hint)

        self._btn_undo = QPushButton(tr("undo_last_hint"))
        self._btn_undo.clicked.connect(self._undo_last_hint)
        self._btn_clear = QPushButton(tr("clear_manual_hints"))
        self._btn_clear.clicked.connect(self._clear_manual_hints)
        add_button_grid(edit_layout, [self._btn_undo, self._btn_clear])
        self._btn_regenerate = tune_button(QPushButton(tr("regenerate_btn")), 36)
        self._btn_regenerate.clicked.connect(self._run_regenerate)
        edit_layout.addWidget(self._btn_regenerate)
        edit_tab_layout.addWidget(edit_group, stretch=4)

        history_group, history_layout = make_group(tr("edit_history_group"))
        self._btn_undo_edit = QPushButton(tr("undo_edit"))
        self._btn_undo_edit.clicked.connect(self._undo)
        self._btn_redo_edit = QPushButton(tr("redo_edit"))
        self._btn_redo_edit.clicked.connect(self._redo)
        add_button_grid(history_layout,
                        [self._btn_undo_edit, self._btn_redo_edit])
        edit_tab_layout.addWidget(history_group, stretch=1)
        self._right_tabs.addTab(edit_tab, tr("right_tab_edit"))

        # ── Tab 3: view, restore and export ───────────────────────────
        output_tab, output_layout = make_tab()
        view_group, view_layout = make_group(tr("view_group"))
        self._view_tool_group = QButtonGroup(self)
        self._radio_view_original = QRadioButton(tr("view_original"))
        self._radio_view_ai = QRadioButton(tr("view_ai"))
        self._radio_view_edited = QRadioButton(tr("view_edited"))
        self._radio_view_edited.setChecked(True)
        for rb in (self._radio_view_original, self._radio_view_ai,
                   self._radio_view_edited):
            self._view_tool_group.addButton(rb)
            rb.toggled.connect(self._on_view_mode_changed)
            view_layout.addWidget(rb)
        self._btn_restore_ai = QPushButton(tr("restore_ai"))
        self._btn_restore_ai.clicked.connect(self._restore_to_ai_result)
        self._btn_restore_bw = QPushButton(tr("restore_bw"))
        self._btn_restore_bw.clicked.connect(self._restore_to_original)
        add_button_grid(view_layout,
                        [self._btn_restore_ai, self._btn_restore_bw])
        self._chk_show_regions = QCheckBox(tr("show_regions_checkbox"))
        self._chk_show_regions.toggled.connect(
            lambda _checked: self._refresh_hint_overlay())
        view_layout.addWidget(self._chk_show_regions)
        output_layout.addWidget(view_group, stretch=3)

        export_group, export_layout = make_group(tr("export_group"))
        self._btn_export_page = QPushButton(tr("export_page"))
        self._btn_export_page.clicked.connect(self._export_current_page)
        self._btn_export_all = QPushButton(tr("export_all"))
        self._btn_export_all.clicked.connect(self._export_all_pages)
        add_button_grid(export_layout,
                        [self._btn_export_page, self._btn_export_all])
        output_layout.addWidget(export_group, stretch=2)
        self._right_tabs.addTab(output_tab, tr("right_tab_output"))

        return panel

    # ── Import ─────────────────────────────────────────────────────────

    _IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif")

    @staticmethod
    def _natural_key(name: str):
        """Natural sort key so page2 < page10 (folder batch order, like
        Manga-Colorizer-GUI's folder processing)."""
        import re
        return [int(t) if t.isdigit() else t.lower()
                for t in re.split(r"(\d+)", name)]

    def _import_folder(self):
        """Register all image paths immediately; decode only the selected page."""
        folder = QFileDialog.getExistingDirectory(self, tr("import_folder"))
        if not folder:
            return
        paths = []
        for root, _dirs, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(self._IMAGE_EXTS):
                    paths.append(os.path.join(root, f))
        paths.sort(key=lambda p: (os.path.dirname(p),
                                  self._natural_key(os.path.basename(p))))
        if not paths:
            self.statusBar().showMessage(tr("folder_no_images"), 4000)
            return
        added = self._add_pages_fast(paths)
        self.statusBar().showMessage(tr("imported_n_pages").format(n=added), 5000)

    def _import_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, tr("import_images"), "", "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tiff)")
        if paths:
            added = self._add_pages_fast(paths)
            self.statusBar().showMessage(tr("imported_n_pages").format(n=added), 5000)

    def _import_pdf(self):
        pdf_path, _ = QFileDialog.getOpenFileName(self, tr("import_pdf"), "", "PDF (*.pdf)")
        if not pdf_path:
            return
        self._import_pdf_path(pdf_path)

    def _import_pdf_path(self, pdf_path: str):
        out_dir = os.path.join(os.path.dirname(pdf_path),
                               os.path.splitext(os.path.basename(pdf_path))[0] + "_pages")
        self.statusBar().showMessage(tr("splitting_pdf"))
        try:
            page_paths = extract_pages(pdf_path, out_dir, dpi=300)
        except Exception as exc:
            QMessageBox.critical(self, tr("error_title"), tr("pdf_split_failed").format(exc=exc))
            return
        added = self._add_pages_fast(page_paths)
        self.statusBar().showMessage(tr("imported_n_pages").format(n=added), 5000)

    def _register_page_path(self, path: str) -> QListWidgetItem | None:
        path = os.path.abspath(path)
        if path in self._pages or not os.path.isfile(path):
            return None
        self._pages[path] = PageState(path)
        full_name = os.path.basename(path)
        item = QListWidgetItem(full_name)
        item.setToolTip(full_name)
        item.setData(Qt.ItemDataRole.UserRole, path)
        self._page_list.addItem(item)
        return item

    def _add_pages_fast(self, paths: list[str]) -> int:
        """O(n) path registration with one layout pass and one image decode."""
        if not paths:
            return 0
        was_empty = self._page_list.count() == 0
        first_item = None
        added = 0
        self._page_list.blockSignals(True)
        self._page_list.setUpdatesEnabled(False)
        try:
            for path in paths:
                item = self._register_page_path(path)
                if item is not None:
                    first_item = first_item or item
                    added += 1
        finally:
            self._page_list.setUpdatesEnabled(True)
            self._page_list.blockSignals(False)
        if added:
            self._fit_left_panel_to_filenames()
        if was_empty and first_item is not None:
            self._page_list.setCurrentItem(first_item)
        return added

    def _add_page(self, path: str):
        self._add_pages_fast([path])

    def _fit_left_panel_to_filenames(self):
        """Keep the page panel readable without any sideways scrolling."""
        if not hasattr(self, "_page_list") or not hasattr(self, "_left_panel"):
            return
        metrics = self._page_list.fontMetrics()
        longest = 0
        for i in range(self._page_list.count()):
            text = self._page_list.item(i).text().replace("⚠ ", "").replace("✓ ", "")
            longest = max(longest, metrics.horizontalAdvance(text))
        # Ordinary names remain on one line; exceptionally long names wrap.
        preferred = max(300, min(460, longest + 48))
        self._left_panel.setMinimumWidth(preferred)
        self._left_panel.updateGeometry()

    def _delete_selected_pages(self):
        items = self._page_list.selectedItems()
        if not items:
            current = self._page_list.currentItem()
            if current is not None:
                items = [current]
        if not items:
            return
        reply = QMessageBox.question(
            self, tr("confirm_delete_title"), tr("confirm_delete_body").format(n=len(items)),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return

        deleted_paths = set()
        for item in items:
            path = item.data(Qt.ItemDataRole.UserRole)
            deleted_paths.add(path)
            self._pages.pop(path, None)
            row = self._page_list.row(item)
            self._page_list.takeItem(row)

        if self._current_path in deleted_paths:
            self._current_path = None
            if self._page_list.count() > 0:
                self._page_list.setCurrentRow(0)
            else:
                self._canvas.set_image(
                    np.full((10, 10, 3), 255, dtype=np.uint8), fit=True)
        self._update_controls_enabled()
        self.statusBar().showMessage(tr("deleted_n_pages").format(n=len(deleted_paths)), 3000)

    def _move_selected_page(self, direction: int):
        row = self._page_list.currentRow()
        if row < 0:
            return
        new_row = row + direction
        if not (0 <= new_row < self._page_list.count()):
            return
        item = self._page_list.takeItem(row)
        self._page_list.insertItem(new_row, item)
        self._page_list.setCurrentRow(new_row)

    def _repopulate_page_list(self):
        """Rebuild the page-list widget from `self._pages` — needed after
        a language-switch UI rebuild, since `_build_left_panel()` creates
        a brand new (empty) QListWidget each time."""
        selected_row = -1
        for i, (path, state) in enumerate(self._pages.items()):
            label = os.path.basename(path)
            if state.ai_result_bgr is not None:
                marker = "⚠ " if (state.quality_report is not None and
                                    state.quality_report.score < 60) else "✓ "
                label = marker + label
            item = QListWidgetItem(label)
            tooltip = os.path.basename(path)
            if state.quality_report is not None:
                tooltip += "\n" + tr("quality_tooltip").format(
                    score=state.quality_report.score,
                    reasons=", ".join(state.quality_report.reasons) or tr("quality_ok"))
            item.setToolTip(tooltip)
            item.setData(Qt.ItemDataRole.UserRole, path)
            self._page_list.addItem(item)
            if path == self._current_path:
                selected_row = i
        if selected_row < 0 and self._page_list.count() > 0:
            selected_row = 0
        if selected_row >= 0:
            self._page_list.setCurrentRow(selected_row)
        self._fit_left_panel_to_filenames()

    # ── Page selection ────────────────────────────────────────────────

    def _on_page_selected(self, current: QListWidgetItem, _previous):
        if current is None:
            return
        path = current.data(Qt.ItemDataRole.UserRole)
        self._current_path = path
        state = self._pages[path]
        try:
            self._sync_view_after_edit(state)
        except Exception as exc:
            self.statusBar().showMessage(str(exc), 6000)
            return
        # Auto-fit the newly displayed page to the window.
        self._canvas.fit_view()
        self._update_controls_enabled()
        self._refresh_character_diagnostics()

    # ── Colorize actions ─────────────────────────────────────────────

    def showEvent(self, event):
        """First time the window is actually shown the canvas finally has
        its real size — re-fit the current page so it fills the window."""
        super().showEvent(event)
        from PySide6.QtCore import QTimer
        self._apply_responsive_density()
        QTimer.singleShot(0, self._canvas.fit_view)

    def _current_state(self) -> PageState | None:
        if self._current_path is None:
            return None
        return self._pages[self._current_path]

    def _run_auto_colorize(self):
        self._run_colorize(regenerate_auto=True)

    def _run_regenerate(self):
        self._run_colorize(regenerate_auto=False)

    def _selected_paths(self) -> list[str]:
        items = self._page_list.selectedItems()
        if len(items) >= 1:
            return [it.data(Qt.ItemDataRole.UserRole) for it in items]
        if self._current_path is not None:
            return [self._current_path]
        return []

    def _run_colorize(self, regenerate_auto: bool):
        if self._batch_worker is not None and self._batch_worker.isRunning():
            self.statusBar().showMessage("上色任务正在运行，请稍候。", 4000)
            return
        paths = self._selected_paths()
        if not paths:
            QMessageBox.information(self, tr("no_result_title"), "请先导入并选择至少一张页面。")
            return
        pages = [(p, self._pages[p]._original_bgr,
                  self._pages[p].hint_manager,
                  dict(self._pages[p].forced_character_matches))
                 for p in paths]

        character_memories = self._get_or_create_character_memories()

        self._batch_errors = []
        self._set_busy(True, tr("batch_colorizing").format(n=len(pages))
                       if len(pages) > 1 else tr("colorizing"))
        self._batch_worker = BatchColorizeWorker(
            pages, regenerate_auto,
            style_key=self._style_combo.currentData(),
            quality_key="draft",
            character_memories=character_memories,
            character_library=self._character_library,
            scene_palette=self._scene_palette,
            skip_colored=self._chk_skip_colored.isChecked(),
            style_strength=self._style_strength_slider.value() / 100.0,
            reference_strength=self._reference_strength_slider.value() / 100.0,
            manual_strength=self._manual_strength_slider.value() / 100.0,
            pastel_tuning=self._current_pastel_tuning(),
            filter_tuning=self._current_filter_tuning(),
        )
        self._batch_worker.page_done.connect(self._on_batch_page_done)
        self._batch_worker.page_error.connect(self._on_batch_page_error)
        self._batch_worker.status.connect(self._on_worker_status)
        self._batch_worker.finished_all.connect(self._on_batch_finished)
        self._batch_worker.start()

    def _get_or_create_character_memories(self) -> dict:
        """One CharacterMemory per label needing multi-character
        consistency, created lazily and reused (and mutated in place by
        the pipeline) across every colorize run for the rest of this
        session so slots stay consistent across the whole book."""
        if not self._character_memories:
            from core.character_memory import CharacterMemory
            self._character_memories = {"hair": CharacterMemory(label="hair")}
        return self._character_memories

    def _open_style_fine_tab(self):
        self._right_tabs.setCurrentIndex(0)
        if getattr(self, "_render_detail_tabs", None) is not None:
            self._render_detail_tabs.setCurrentIndex(0)
        self._update_pastel_controls_visibility()

    def _reset_style_fine_tuning(self):
        for widget in (
            getattr(self, "_style_color_slider", None),
            getattr(self, "_style_brightness_slider", None),
            getattr(self, "_style_warmth_slider", None),
            getattr(self, "_style_highlight_slider", None),
            getattr(self, "_style_softness_slider", None),
            getattr(self, "_style_flatten_slider", None),
            getattr(self, "_light3_intensity_slider", None),
        ):
            if widget is not None:
                widget.setValue(100)
        self._update_color_swatch()

    def _current_pastel_tuning(self) -> dict:
        return {
            "color_strength": self._style_color_slider.value(),
            "brightness": self._style_brightness_slider.value(),
            "warmth": self._style_warmth_slider.value(),
            "highlight_preserve": self._style_highlight_slider.value(),
            "softness": self._style_softness_slider.value(),
            "flatten": self._style_flatten_slider.value(),
            "light3_intensity": self._light3_intensity_slider.value(),
            "person_strength": self._pastel_person_slider.value(),
            "hair_strength": self._pastel_hair_slider.value(),
            "skin_strength": self._pastel_skin_slider.value(),
            "eye_strength": self._pastel_eye_slider.value(),
            "clothing_strength": self._pastel_clothing_slider.value(),
            "environment_strength": self._pastel_environment_slider.value(),
            "skin_warmth": self._pastel_skin_warmth_slider.value(),
        }

    def _current_filter_tuning(self) -> dict:
        return {
            "brightness": self._filter_brightness_slider.value(),
            "contrast": self._filter_contrast_slider.value(),
            "saturation": self._filter_saturation_slider.value(),
            "warmth": self._filter_warmth_slider.value(),
            "shadow_lift": self._filter_shadow_slider.value(),
            "highlight": self._filter_highlight_slider.value(),
        }

    def _reset_filter_controls(self):
        for slider in (
            self._filter_brightness_slider, self._filter_contrast_slider,
            self._filter_saturation_slider, self._filter_warmth_slider,
            self._filter_shadow_slider, self._filter_highlight_slider,
        ):
            slider.setValue(100)
        self.statusBar().showMessage(tr("filter_reset_done"), 3000)

    def _filter_target_states(self) -> list[PageState]:
        scope = self._filter_scope_combo.currentData() if getattr(self, "_filter_scope_combo", None) else "current"
        if scope == "all":
            return [state for state in self._ordered_states() if state.result_bgr is not None]
        state = self._current_state()
        return [state] if state is not None and state.result_bgr is not None else []

    def _apply_filter_to_scope(self):
        states = self._filter_target_states()
        if not states:
            self.statusBar().showMessage(tr("filter_no_result"), 3500)
            return
        from core.image_filter import apply_image_filter
        tuning = self._current_filter_tuning()
        style_key = self._style_combo.currentData()
        style_strength = self._style_strength_slider.value() / 100.0
        for state in states:
            base = state.filter_base_bgr
            if base is None or base.shape[:2] != state.result_bgr.shape[:2]:
                # Legacy projects have no stored pre-filter layer. Capture the
                # current visible result once, then all later adjustments remain
                # non-cumulative from this stable base.
                state.filter_base_bgr = state.result_bgr.copy()
                base = state.filter_base_bgr
            state.push_undo()
            state.result_bgr = apply_image_filter(
                base.copy(), tuning,
                style_strength=style_strength,
                is_styled=style_key not in {None, "none"},
                source_bw_bgr=state.original_bgr)
            state.discard_unchanged_undo()
            self._mark_page_done(state.path)
        current = self._current_state()
        if current is not None and current in states:
            self._radio_view_edited.setChecked(True)
            self._sync_view_after_edit(current)
        if self._filter_scope_combo.currentData() == "all":
            self.statusBar().showMessage(tr("filter_applied_all").format(n=len(states)), 4500)
        else:
            self.statusBar().showMessage(tr("filter_applied_current"), 4000)

    def _update_pastel_controls_visibility(self):
        style_group = getattr(self, "_style_fine_group", None)
        mono_group = getattr(self, "_pastel_controls_group", None)
        combo = getattr(self, "_style_combo", None)
        if combo is None:
            return
        key = combo.currentData()
        if style_group is not None:
            style_group.setVisible(True)
        active_label = getattr(self, "_style_fine_active_label", None)
        if active_label is not None:
            active_label.setText(tr("style_fine_active").format(name=combo.currentText()))
        is_light3 = key == "light3"
        for widget in (getattr(self, "_light3_intensity_label", None),
                       getattr(self, "_light3_intensity_slider", None),
                       getattr(self, "_light3_intensity_spin", None)):
            if widget is not None:
                widget.setVisible(is_light3)
        if mono_group is not None:
            mono_group.setVisible(False)

    def _on_style_combo_changed(self):
        """React to style changes and expose relevant fine controls."""
        self._update_pastel_controls_visibility()
        self._update_color_swatch()




    def _extract_character_palette(self):
        """Build a fresh character identity library from colour references."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, tr("extract_character_palette_title"), "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        if not paths:
            return
        from core.imageio import imread as _uimread
        images = [image for image in (_uimread(path) for path in paths)
                  if image is not None]
        if not images:
            QMessageBox.warning(self, tr("warning_title"),
                                tr("extract_style_fail_body"))
            return
        try:
            from core.guided_colorist import _get_classifier
            from core.character_library import CharacterLibrary
            classifier = _get_classifier()
            library = CharacterLibrary()
            for image in images:
                library.extract_from_reference(image, classifier=classifier)
        except Exception as exc:
            QMessageBox.warning(self, tr("warning_title"), str(exc))
            return
        if not library.characters:
            QMessageBox.information(
                self, tr("no_result_title"),
                tr("no_characters_extracted") + "\n\n" +
                tr("anime_detector_download_hint"))
            return
        self._character_library = library
        self._refresh_character_diagnostics()
        self.statusBar().showMessage(
            tr("characters_extracted_separate").format(
                n=len(library.characters)), 7000)

    def _manual_add_reference_character(self):
        """Explicitly enrol one identity from a selected colour reference.

        This path is intentionally independent from automatic anime-face/CLIP
        detection.  It is the recommended workflow for dense covers, rotated
        figures and any page where automatic extraction creates a wrong person.
        """
        path, _ = QFileDialog.getOpenFileName(
            self, tr("manual_character_title"), "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tiff)")
        if not path:
            return
        from core.imageio import imread as _uimread
        image = _uimread(path)
        if image is None:
            QMessageBox.warning(self, tr("warning_title"),
                                tr("cannot_read_image").format(path=path))
            return
        try:
            from ui.reference_character_dialog import ReferenceCharacterDialog
            default_name = os.path.splitext(os.path.basename(path))[0]
            dialog = ReferenceCharacterDialog(
                image, self, default_name=default_name)
        except Exception as exc:
            QMessageBox.warning(self, tr("warning_title"), str(exc))
            return
        if not dialog.exec():
            return

        try:
            from core.character_library import CharacterLibrary
            from core.guided_colorist import _get_classifier
            if self._character_library is None:
                self._character_library = CharacterLibrary()
            classifier = _get_classifier()
            profile = self._character_library.add_manual_reference(
                dialog.image_bgr, dialog.head_bbox,
                colors=dialog.colors, name=dialog.character_name,
                rotation=0, classifier=classifier,
                merge_same_name=True)
        except Exception as exc:
            QMessageBox.warning(self, tr("warning_title"), str(exc))
            return
        self._refresh_character_diagnostics()
        self.statusBar().showMessage(
            tr("manual_character_added").format(
                name=profile.name or f"#{profile.char_id}",
                n=len(self._character_library.characters)), 7000)






    def _bind_page_characters(self):
        state = self._current_state()
        if state is None:
            QMessageBox.information(self, tr("no_result_title"),
                                    tr("bind_characters_need_page"))
            return
        if self._character_library is None or not self._character_library.characters:
            QMessageBox.information(self, tr("no_result_title"),
                                    tr("no_characters_msg"))
            return
        context = getattr(state.hint_manager, "last_page_context", None)
        instances = list(getattr(context, "character_instances", []) or [])
        if not instances:
            QMessageBox.information(self, tr("no_result_title"),
                                    tr("bind_characters_need_analysis"))
            return
        from ui.character_match_dialog import CharacterMatchDialog
        dialog = CharacterMatchDialog(
            instances, self._character_library,
            current=state.forced_character_matches, parent=self)
        if not dialog.exec():
            return
        state.forced_character_matches = dialog.bindings()
        # Bindings change identity hints and lock decisions; force fresh page
        # analysis on the next run instead of reusing previous auto hints.
        state.hint_manager.auto_hints = []
        self.statusBar().showMessage(
            tr("bindings_saved").format(n=len(state.forced_character_matches)),
            6000)
        self._refresh_hint_overlay()

    def _manage_characters(self):
        if self._character_library is None or not self._character_library.characters:
            QMessageBox.information(self, tr("no_result_title"), tr("no_characters_msg"))
            return
        from ui.character_dialog import CharacterLibraryDialog
        dialog = CharacterLibraryDialog(self._character_library, self)
        if dialog.exec():
            self._refresh_character_diagnostics()
            self.statusBar().showMessage(tr("characters_updated"), 4000)

    def _save_character_palette(self):
        if self._character_library is None or not self._character_library.characters:
            QMessageBox.information(self, tr("no_result_title"), tr("no_characters_msg"))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, tr("save_palette"), "characters.ccpalette",
            "Colortina Character Palette (*.ccpalette)")
        if not path:
            return
        if not path.endswith(".ccpalette"):
            path += ".ccpalette"
        self._character_library.save(path)
        self.statusBar().showMessage(tr("palette_saved").format(path=path), 5000)

    def _load_character_palette(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("load_palette"), "",
            "Colortina Character Palette (*.ccpalette);;JSON (*.json)")
        if not path:
            return
        try:
            from core.character_library import CharacterLibrary
            self._character_library = CharacterLibrary.load(path)
        except Exception as exc:
            QMessageBox.warning(self, tr("load_style_fail_title"), str(exc))
            return
        self._refresh_character_diagnostics()
        self.statusBar().showMessage(tr("palette_loaded").format(path=path), 5000)

    def _update_scene_palette_label(self):
        if not hasattr(self, "_scene_palette_label"):
            return
        if self._scene_palette is None or not self._scene_palette.colors:
            self._scene_palette_label.setText(tr("scene_palette_unset"))
        else:
            self._scene_palette_label.setText(tr("scene_palette_active").format(
                name=self._scene_palette.name,
                n=len(self._scene_palette.colors)))

    def _refresh_character_diagnostics(self):
        import html

        label = getattr(self, "_character_diag_label", None)
        if label is None:
            return
        if self._character_library is None or not getattr(self._character_library, "characters", None):
            label.setText(html.escape(tr("character_diagnostics_empty")))
            return
        state = self._current_state()
        context = getattr(getattr(state, "hint_manager", None), "last_page_context", None) if state is not None else None
        result_bgr = getattr(state, "result_bgr", None) if state is not None else None
        rows = self._character_library.diagnostic_rows(
            context, result_bgr=result_bgr,
            max_rows=3 if self.height() < 780 else 5)
        if not rows:
            label.setText(html.escape(tr("character_diagnostics_empty")))
            return

        def swatch(value):
            if not value or not isinstance(value, str) or len(value) != 7:
                return "<span style='color:#999'>—</span>"
            safe = html.escape(value)
            border = "#666" if value.lower() not in ("#000000", "#ffffff") else "#aaa"
            return (f"<span style='display:inline-block; background:{safe}; "
                    f"border:1px solid {border}; padding:0 7px; margin-right:2px'>"
                    f"&nbsp;</span><span>{safe}</span>")

        part_names = {
            "upper": tr("clothing_part_upper"),
            "lower": tr("clothing_part_lower"),
            "accessory": tr("clothing_part_accessory"),
        }
        blocks = []
        for row in rows:
            name = html.escape(row.get("name") or tr("character_default_name").format(
                id=row.get("char_id", 0)))
            attrs = ", ".join(row.get("attributes") or [])
            active = int(row.get("active_regions", 0))
            locked = int(row.get("locked_regions", 0))
            header_color = "#b00020" if row.get("drift_alerts") else "#333"
            line = (f"<div style='margin-bottom:3px'><b style='color:{header_color}'>{name}</b> "
                    f"<span style='color:#777'>[{locked}/{active}]</span><br>"
                    f"<span style='color:#777'>发</span> {swatch(row.get('hair'))} &nbsp; "
                    f"<span style='color:#777'>瞳</span> {swatch(row.get('eyes'))} &nbsp; "
                    f"<span style='color:#777'>肤</span> {swatch(row.get('skin'))}<br>"
                    f"<span style='color:#777'>服</span> {swatch(row.get('clothing'))}")
            slots = row.get("clothing_slots") or []
            if slots:
                slot_html = " ".join(swatch(value) for value in slots)
                line += f"<br><span style='color:#777'>{html.escape(tr('character_diagnostics_slots_label'))}</span> {slot_html}"
            part_counts = row.get("part_counts") or {}
            if part_counts:
                parts = ", ".join(
                    f"{html.escape(part_names.get(key, key))} {count}"
                    for key, count in part_counts.items())
                line += f"<br><span style='color:#777'>{html.escape(parts)}</span>"
            alerts = row.get("drift_alerts") or []
            if alerts:
                attrs_text = ", ".join(sorted({
                    part_names.get(a.get("part"), a.get("attribute", ""))
                    if a.get("attribute") == "clothing" else a.get("attribute", "")
                    for a in alerts}))
                line += ("<br><span style='color:#b00020'><b>" +
                         html.escape(tr("character_diagnostics_alert").format(
                             attrs=attrs_text or attrs,
                             delta=row.get("max_delta_e", 0.0))) + "</b></span>")
            elif active:
                line += ("<br><span style='color:#2e7d32'>" +
                         html.escape(tr("character_diagnostics_ok")) + "</span>")
            else:
                line += ("<br><span style='color:#888'>" +
                         html.escape(tr("character_diagnostics_library_only")) + "</span>")
            blocks.append(line + "</div>")
        label.setText("".join(blocks))

    def _build_book_reference_bundle(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, tr("book_reference_bundle_title"), "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tiff)")
        if not paths:
            return
        from core.imageio import imread as _uimread
        images = [img for img in (_uimread(path) for path in paths) if img is not None]
        if not images:
            QMessageBox.warning(self, tr("warning_title"), tr("extract_style_fail_body"))
            return
        try:
            from core.guided_colorist import _get_classifier
            from core.character_library import CharacterLibrary
            from core.region_classifier import RegionClassifier
            from core.scene_palette import ScenePalette

            classifier = _get_classifier()
            library = CharacterLibrary()
            for image in images:
                try:
                    library.extract_from_reference(image, classifier=classifier)
                except Exception:
                    continue

            scene_palette = ScenePalette.extract_from_references(
                images, RegionClassifier(), name=tr("scene_palette_name_default"))
        except Exception as exc:
            QMessageBox.warning(self, tr("warning_title"), str(exc))
            return

        if (not getattr(library, "characters", None)) and (scene_palette is None or not scene_palette.colors):
            QMessageBox.information(self, tr("no_result_title"), tr("book_reference_bundle_empty"))
            return

        if getattr(library, "characters", None):
            self._character_library = library
        if scene_palette is not None and scene_palette.colors:
            self._scene_palette = scene_palette
        self._refresh_character_diagnostics()
        self._update_scene_palette_label()
        chars = len(getattr(self._character_library, "characters", []) or [])
        scenes = len(getattr(self._scene_palette, "colors", {}) or {})
        self.statusBar().showMessage(
            tr("book_reference_bundle_status").format(chars=chars, scenes=scenes, style=self._style_combo.currentText()), 8000)

    def _extract_scene_palette(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, tr("extract_scene_palette"), "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        if not paths:
            return
        from core.imageio import imread as _uimread
        images = [img for img in (_uimread(path) for path in paths) if img is not None]
        if not images:
            QMessageBox.warning(self, tr("warning_title"), tr("extract_style_fail_body"))
            return
        name, ok = QInputDialog.getText(
            self, tr("scene_palette_name_title"), tr("scene_palette_name_label"),
            text=tr("scene_palette_name_default"))
        if not ok:
            return
        try:
            from core.region_classifier import RegionClassifier
            from core.scene_palette import ScenePalette
            self._scene_palette = ScenePalette.extract_from_references(
                images, RegionClassifier(), name=name.strip() or tr("scene_palette_name_default"))
        except Exception as exc:
            QMessageBox.warning(self, tr("warning_title"), str(exc))
            return
        self._update_scene_palette_label()
        self.statusBar().showMessage(tr("scene_palette_extracted").format(
            n=len(self._scene_palette.colors)), 5000)

    def _load_scene_palette(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("load_scene_palette"), "",
            "Colortina Scene Palette (*.ccscene);;JSON (*.json)")
        if not path:
            return
        try:
            from core.scene_palette import ScenePalette
            self._scene_palette = ScenePalette.load(path)
        except Exception as exc:
            QMessageBox.warning(self, tr("warning_title"), str(exc))
            return
        self._update_scene_palette_label()
        self.statusBar().showMessage(tr("scene_palette_loaded").format(path=path), 5000)

    def _save_scene_palette(self):
        if self._scene_palette is None or not self._scene_palette.colors:
            QMessageBox.information(self, tr("no_result_title"),
                                    tr("scene_palette_unset"))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, tr("save_scene_palette"), "scene.ccscene",
            "Colortina Scene Palette (*.ccscene)")
        if not path:
            return
        try:
            saved = self._scene_palette.save(path)
        except Exception as exc:
            QMessageBox.warning(self, tr("warning_title"), str(exc))
            return
        self.statusBar().showMessage(
            tr("scene_palette_saved").format(path=saved), 5000)

    def _clear_scene_palette(self):
        self._scene_palette = None
        self._update_scene_palette_label()
        self.statusBar().showMessage(tr("scene_palette_cleared"), 4000)

    def _ordered_states(self) -> list[PageState]:
        states = []
        for i in range(self._page_list.count()):
            path = self._page_list.item(i).data(Qt.ItemDataRole.UserRole)
            state = self._pages.get(path)
            if state is not None:
                states.append(state)
        return states

    def _project_settings(self) -> dict:
        return {
            "style_key": self._style_combo.currentData(),
            "quality_key": "draft",
            "style_strength": self._style_strength_slider.value(),
            "reference_strength": self._reference_strength_slider.value(),
            "manual_strength": self._manual_strength_slider.value(),
            "style_color_strength": self._style_color_slider.value(),
            "style_brightness_strength": self._style_brightness_slider.value(),
            "style_warmth_strength": self._style_warmth_slider.value(),
            "style_highlight_strength": self._style_highlight_slider.value(),
            "style_softness_strength": self._style_softness_slider.value(),
            "style_flatten_strength": self._style_flatten_slider.value(),
            "light3_intensity": self._light3_intensity_slider.value(),
            "filter_brightness": self._filter_brightness_slider.value(),
            "filter_contrast": self._filter_contrast_slider.value(),
            "filter_saturation": self._filter_saturation_slider.value(),
            "filter_warmth": self._filter_warmth_slider.value(),
            "filter_shadow": self._filter_shadow_slider.value(),
            "filter_highlight": self._filter_highlight_slider.value(),
            "filter_scope": self._filter_scope_combo.currentData(),
            "pastel_person_strength": self._pastel_person_slider.value(),
            "pastel_hair_strength": self._pastel_hair_slider.value(),
            "pastel_skin_strength": self._pastel_skin_slider.value(),
            "pastel_eye_strength": self._pastel_eye_slider.value(),
            "pastel_clothing_strength": self._pastel_clothing_slider.value(),
            "pastel_environment_strength": self._pastel_environment_slider.value(),
            "pastel_skin_warmth": self._pastel_skin_warmth_slider.value(),
            "gap_close": self._gap_close_slider.value(),
            "brush_size": self._brush_slider.value(),
            "picker_lightness": self._picker_lightness_slider.value(),
            "manual_match_style": self._chk_manual_match_style.isChecked(),
            "skip_colored": self._chk_skip_colored.isChecked(),
            "character_memory": self._chk_character_memory.isChecked(),
        }

    def _save_project(self):
        default = self._current_project_path or "colortina_project.ccproject"
        path, _ = QFileDialog.getSaveFileName(
            self, tr("save_project"), default,
            "Colortina Project (*.ccproject)")
        if not path:
            return
        try:
            from core.project_store import save_project
            path = save_project(
                path, pages=self._ordered_states(),
                    character_library=self._character_library,
                character_memories=self._character_memories,
                scene_palette=self._scene_palette,
                settings=self._project_settings())
        except Exception as exc:
            QMessageBox.warning(self, tr("error_title"),
                                tr("project_save_failed").format(exc=exc))
            return
        self._current_project_path = path
        self.statusBar().showMessage(tr("project_saved").format(path=path), 5000)

    def _load_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("load_project"), "", "Colortina Project (*.ccproject)")
        if not path:
            return
        try:
            from core.project_store import load_project
            data = load_project(path)
        except Exception as exc:
            QMessageBox.warning(self, tr("error_title"),
                                tr("project_load_failed").format(exc=exc))
            return

        self._pages.clear()
        self._page_list.clear()
        self._current_path = None
        missing = 0
        from core.imageio import imread as _uimread
        valid_records = []
        for record in data["pages"]:
            source = record.get("path")
            if not source or not os.path.isfile(source):
                missing += 1
                continue
            record = dict(record)
            record["path"] = os.path.abspath(source)
            valid_records.append(record)
        self._add_pages_fast([record["path"] for record in valid_records])
        for record in valid_records:
            source = record["path"]
            state = self._pages[source]
            state.hint_manager = record["hint_manager"]
            state.pipeline_diagnostics = dict(record.get("diagnostics", {}) or {})
            state.forced_character_matches = dict(
                record.get("forced_character_matches", {}) or {})
            ai_path = record.get("ai_result")
            result_path = record.get("result")
            filter_base_path = record.get("filter_base")
            if ai_path and os.path.isfile(ai_path):
                state.ai_result_bgr = _uimread(ai_path)
            if result_path and os.path.isfile(result_path):
                state.result_bgr = _uimread(result_path)
            elif state.ai_result_bgr is not None:
                state.result_bgr = state.ai_result_bgr.copy()
            if filter_base_path and os.path.isfile(filter_base_path):
                state.filter_base_bgr = _uimread(filter_base_path)
            elif state.result_bgr is not None:
                # Compatibility with projects saved before V2 filter-base persistence.
                state.filter_base_bgr = state.result_bgr.copy()

        settings = data.get("settings", {})
        loaded_style_key = settings.get("style_key")
        if loaded_style_key in {"monochrome_people", "monochrome_page"}:
            loaded_style_key = "monochrome"
        if loaded_style_key == "light":
            loaded_style_key = "light2"
            settings = dict(settings)
            if "pastel_environment_strength" not in settings:
                settings["pastel_environment_strength"] = 0 if settings.get("style_key") == "monochrome_people" else 100
        for widget, key in ((self._style_combo, "style_key"),
                            (self._quality_combo, "quality_key")):
            desired = loaded_style_key if key == "style_key" else settings.get(key)
            idx = widget.findData(desired)
            if idx >= 0:
                widget.setCurrentIndex(idx)
        self._character_library = data["character_library"]
        self._character_memories = data["character_memories"]
        self._scene_palette = data.get("scene_palette")
        self._update_scene_palette_label()
        self._style_strength_slider.setValue(settings.get("style_strength", 100))
        self._reference_strength_slider.setValue(settings.get("reference_strength", 100))
        self._manual_strength_slider.setValue(settings.get("manual_strength", 100))
        self._style_color_slider.setValue(settings.get("style_color_strength", 100))
        self._style_brightness_slider.setValue(settings.get("style_brightness_strength", 100))
        self._style_warmth_slider.setValue(settings.get("style_warmth_strength", 100))
        self._style_highlight_slider.setValue(settings.get("style_highlight_strength", 100))
        self._style_softness_slider.setValue(settings.get("style_softness_strength", 100))
        self._style_flatten_slider.setValue(settings.get("style_flatten_strength", 100))
        self._light3_intensity_slider.setValue(settings.get("light3_intensity", 100))
        self._filter_brightness_slider.setValue(settings.get("filter_brightness", 100))
        self._filter_contrast_slider.setValue(settings.get("filter_contrast", 100))
        self._filter_saturation_slider.setValue(settings.get("filter_saturation", 100))
        self._filter_warmth_slider.setValue(settings.get("filter_warmth", 100))
        self._filter_shadow_slider.setValue(settings.get("filter_shadow", 100))
        self._filter_highlight_slider.setValue(settings.get("filter_highlight", 100))
        idx = self._filter_scope_combo.findData(settings.get("filter_scope", "current"))
        if idx >= 0:
            self._filter_scope_combo.setCurrentIndex(idx)
        self._pastel_person_slider.setValue(settings.get("pastel_person_strength", 100))
        self._pastel_hair_slider.setValue(settings.get("pastel_hair_strength", 100))
        self._pastel_skin_slider.setValue(settings.get("pastel_skin_strength", 100))
        self._pastel_eye_slider.setValue(settings.get("pastel_eye_strength", 100))
        self._pastel_clothing_slider.setValue(settings.get("pastel_clothing_strength", 100))
        legacy_env = 0 if settings.get("style_key") == "monochrome_people" else 100
        self._pastel_environment_slider.setValue(settings.get("pastel_environment_strength", legacy_env))
        self._pastel_skin_warmth_slider.setValue(settings.get("pastel_skin_warmth", 100))
        self._gap_close_slider.setValue(self._gap_px_from_percent(
            settings.get("gap_close", self._gap_close_slider.value())))
        self._brush_slider.setValue(settings.get("brush_size", self._brush_slider.value()))
        self._picker_lightness_slider.setValue(settings.get("picker_lightness", 0))
        if getattr(self, '_chk_manual_match_style', None) is not None:
            self._chk_manual_match_style.setChecked(settings.get("manual_match_style", True))
        self._chk_skip_colored.setChecked(settings.get("skip_colored", True))
        self._chk_character_memory.setChecked(settings.get("character_memory", True))
        self._current_project_path = path
        if self._page_list.count():
            self._page_list.setCurrentRow(0)
        self._update_controls_enabled()
        self._update_pastel_controls_visibility()
        message = tr("project_loaded").format(path=path)
        if missing:
            message += tr("project_missing_pages").format(n=missing)
        self.statusBar().showMessage(message, 8000)

    def _on_worker_status(self, message: str):
        self.statusBar().showMessage(message, 0)
        if hasattr(self, "_auto_status_label"):
            self._auto_status_label.setText(message)

    def _on_batch_page_done(self, path: str, result_payload):
        state = self._pages.get(path)
        if state is None:
            return
        if isinstance(result_payload, tuple) and len(result_payload) == 2:
            result_bgr, filter_base_bgr = result_payload
        else:
            result_bgr, filter_base_bgr = result_payload, None
        state.push_undo()  # snapshot whatever was there before this run
        state.ai_result_bgr = result_bgr.copy()
        state.result_bgr = result_bgr.copy()
        state.filter_base_bgr = (filter_base_bgr.copy() if filter_base_bgr is not None
                                 else result_bgr.copy())
        state.pipeline_diagnostics = dict(
            getattr(state.hint_manager, "last_diagnostics", {}) or {})
        self._refresh_character_diagnostics()
        try:
            from core.quality_score import assess_colorization
            state.quality_report = assess_colorization(state.original_bgr, result_bgr)
        except Exception:
            state.quality_report = None
        self._mark_page_done(path)
        self._canvas.clear_dabs()
        if path == self._current_path:
            self._radio_view_edited.setChecked(True)
            self._sync_view_after_edit(state)

    def _on_batch_page_error(self, path: str, message: str):
        name = os.path.basename(path)
        self._batch_errors.append((name, message))
        visible = tr("page_colorize_failed").format(name=name, message=message)
        self.statusBar().showMessage(visible, 8000)
        if hasattr(self, "_auto_status_label"):
            self._auto_status_label.setText(visible)

    @staticmethod
    def _release_compute_memory():
        """Free accelerator caches after a batch so the app doesn't sit
        on GPU/unified memory between jobs."""
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    def _on_batch_finished(self):
        self._release_compute_memory()
        worker = self._batch_worker
        self._batch_worker = None
        if self._batch_errors:
            name, message = self._batch_errors[0]
            self._set_busy(False, "上色未完成")
            if hasattr(self, "_auto_status_label"):
                self._auto_status_label.setText(f"失败：{name} — {message}")
            QMessageBox.critical(
                self, "自动上色失败",
                f"{name} 上色失败：\n{message}\n\n请检查模型文件、网络或终端日志。")
        else:
            self._set_busy(False, tr("colorize_done"))
            if hasattr(self, "_auto_status_label"):
                self._auto_status_label.setText(tr("colorize_done"))
        if worker is not None:
            worker.deleteLater()
        self._update_device_label()

    def _mark_page_done(self, path: str):
        state = self._pages.get(path)
        for i in range(self._page_list.count()):
            item = self._page_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == path:
                diagnostics = dict(getattr(state, "pipeline_diagnostics", {}) or {}) if state else {}
                warning = bool(state and state.quality_report is not None and
                               state.quality_report.score < 60)
                warning = warning or float(diagnostics.get("hint_blob_score", 0.0)) >= 14.0
                warning = warning or int(diagnostics.get("identity_drift_alerts", 0) or 0) > 0
                item.setText(("⚠ " if warning else "✓ ") + os.path.basename(path))
                tooltip_parts = []
                if state and state.quality_report is not None:
                    tooltip_parts.append(tr("quality_tooltip").format(
                        score=state.quality_report.score,
                        reasons=", ".join(state.quality_report.reasons) or tr("quality_ok")))
                if diagnostics:
                    tooltip_parts.append(tr("pipeline_diagnostics").format(
                        matched=diagnostics.get("matched", 0),
                        ambiguous=diagnostics.get("ambiguous", 0),
                        locks=diagnostics.get("lock_regions", 0),
                        hints=diagnostics.get("composed_hint_count", diagnostics.get("hint_count", 0)),
                        drift=diagnostics.get("identity_drift_alerts", 0),
                        retry=tr("yes") if diagnostics.get("hint_retry") else tr("no")))
                if tooltip_parts:
                    item.setToolTip("\n".join(tooltip_parts))
                if warning and state and state.quality_report is not None:
                    self.statusBar().showMessage(
                        tr("quality_warning").format(
                            name=os.path.basename(path),
                            score=state.quality_report.score), 7000)
                break

    def _unmark_page_done(self, path: str):
        for i in range(self._page_list.count()):
            item = self._page_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == path:
                item.setText(os.path.basename(path))
                break

    def _set_busy(self, busy: bool, message: str = ""):
        self._progress.setVisible(busy)
        self._btn_auto.setText("正在上色…" if busy else tr("auto_btn"))
        for btn in (self._btn_auto, self._btn_regenerate, self._btn_undo,
                   self._btn_clear, self._btn_undo_edit, self._btn_redo_edit,
                   self._btn_restore_ai, self._btn_restore_bw,
                   self._btn_export_page, self._btn_export_all):
            btn.setEnabled(not busy)
        if message:
            self.statusBar().showMessage(message, 0 if busy else 5000)

    def _update_device_label(self):
        """Show device/model readiness without loading models on the UI thread."""
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif (getattr(torch.backends, "mps", None) is not None
                  and torch.backends.mps.is_available()):
                device = "mps"
            else:
                device = "cpu"
            from core.model_downloader import models_ready
            suffix = "模型已就绪" if models_ready(Config.WEIGHTS_DIR) else "首次上色将下载模型"
            self._device_label.setText(f"{tr('device_label')}{device} · {suffix}")
        except Exception as exc:
            self._device_label.setText(f"{tr('device_label')}检测失败：{exc}")

    # ── Editing (manual hints) ───────────────────────────────────────

    def _manual_color_mode(self) -> str:
        return "custom"

    def _uses_ai_original_color(self) -> bool:
        return False

    def _update_tool_specific_visibility(self):
        is_eyedropper = getattr(self, "_radio_eyedropper", None) is not None and self._radio_eyedropper.isChecked()
        if getattr(self, "_eyedropper_mode_label", None) is not None:
            self._eyedropper_mode_label.setVisible(is_eyedropper)
        mode_box = getattr(self, "_eyedropper_mode_box", None)
        if mode_box is not None:
            mode_box.setVisible(is_eyedropper)
        else:
            for widget_name in ("_eyedropper_mode_point", "_eyedropper_mode_region"):
                widget = getattr(self, widget_name, None)
                if widget is not None:
                    widget.setVisible(is_eyedropper)

    def _on_tool_changed(self):
        if self._radio_brush.isChecked():
            tool = HintCanvas.TOOL_BRUSH
        elif self._radio_eyedropper.isChecked():
            tool = HintCanvas.TOOL_EYEDROPPER
        else:
            tool = HintCanvas.TOOL_BUCKET
        self._canvas.set_tool(tool)
        self._update_tool_specific_visibility()

    def _current_eyedropper_mode(self) -> str:
        if getattr(self, '_eyedropper_mode_region', None) is not None and self._eyedropper_mode_region.isChecked():
            return 'region'
        return 'point'

    def _set_eyedropper_mode(self, mode: str):
        mode = 'region' if mode == 'region' else 'point'
        point = getattr(self, '_eyedropper_mode_point', None)
        region = getattr(self, '_eyedropper_mode_region', None)
        if point is not None:
            point.blockSignals(True)
        if region is not None:
            region.blockSignals(True)
        if point is not None:
            point.setChecked(mode == 'point')
        if region is not None:
            region.setChecked(mode == 'region')
        if point is not None:
            point.blockSignals(False)
        if region is not None:
            region.blockSignals(False)
        self._canvas.set_eyedropper_mode(mode)

    @staticmethod
    def _adjust_picker_lightness(rgb: tuple[int, int, int], amount: int) -> tuple[int, int, int]:
        amount = int(np.clip(amount, -100, 100))
        if amount == 0:
            return tuple(int(np.clip(v, 0, 255)) for v in rgb)
        r, g, b = [int(np.clip(v, 0, 255)) for v in rgb]
        px = np.array([[[b, g, r]]], dtype=np.uint8)
        lab = cv2.cvtColor(px, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = float(lab[0, 0, 0])
        if amount > 0:
            L = L + (255.0 - L) * (amount / 100.0)
        else:
            L = L * (1.0 + amount / 100.0)
        lab[0, 0, 0] = np.clip(L, 0.0, 255.0)
        out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)[0, 0]
        return int(out[2]), int(out[1]), int(out[0])

    def _apply_picked_color(self, rgb: tuple[int, int, int], *, remember_raw: bool = False):
        if remember_raw:
            self._last_picked_rgb_raw = tuple(int(np.clip(v, 0, 255)) for v in rgb)
        elif rgb is not None:
            self._last_picked_rgb_raw = None
        adjusted = self._adjust_picker_lightness(
            tuple(int(np.clip(v, 0, 255)) for v in rgb),
            self._picker_lightness_slider.value() if hasattr(self, '_picker_lightness_slider') else 0)
        self._brush_color = QColor(*adjusted)
        self._canvas.set_brush_color(self._brush_color)
        self._update_color_swatch()

    def _on_picker_lightness_changed(self, value: int):
        if getattr(self, '_picker_lightness_spin', None) is not None and self._picker_lightness_spin.value() != value:
            self._picker_lightness_spin.setValue(value)
        if self._last_picked_rgb_raw is not None:
            self._apply_picked_color(self._last_picked_rgb_raw, remember_raw=True)

    def _pick_color(self):
        color = QColorDialog.getColor(self._brush_color, self, tr("pick_color_title"))
        if color.isValid():
            self._last_picked_rgb_raw = None
            self._brush_color = color
            self._canvas.set_brush_color(color)
            self._update_color_swatch()

    def _update_color_swatch(self):
        if getattr(self, "_color_swatch", None) is None:
            return
        info = getattr(self, '_current_color_info', None)
        if self._uses_ai_original_color():
            self._color_swatch.setToolTip(tr("brush_color_mode_ai"))
            self._color_swatch.setText('')
            self._color_swatch.setStyleSheet(
                "background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #ffffff, stop:1 #cfd8ea); "
                "border: 1px dashed #7e8aa6;")
            if info is not None:
                info.setText(tr("brush_color_mode_ai"))
                info.setToolTip(tr("brush_color_mode_ai"))
        else:
            raw_rgb = (self._brush_color.red(), self._brush_color.green(), self._brush_color.blue())
            shown_rgb = self._manual_target_rgb(raw_rgb) if self._manual_style_match_enabled() else raw_rgb
            shown_hex = '#%02x%02x%02x' % shown_rgb
            raw_hex = self._brush_color.name()
            self._color_swatch.setText('')
            if shown_hex.lower() != raw_hex.lower():
                tip = tr("manual_color_preview_tooltip").format(raw=raw_hex, adapted=shown_hex)
                self._color_swatch.setToolTip(tip)
                self._color_swatch.setStyleSheet(
                    f"background-color: {shown_hex}; border: 2px solid {raw_hex};")
                if info is not None:
                    info.setText(tr("current_color_value").format(current=raw_hex) + "\n" +
                                 tr("adapted_color_value").format(adapted=shown_hex))
                    info.setToolTip(tr("current_color_tooltip").format(current=raw_hex, adapted=shown_hex))
            else:
                self._color_swatch.setToolTip(raw_hex)
                self._color_swatch.setStyleSheet(
                    f"background-color: {raw_hex}; border: 1px solid #888;")
                if info is not None:
                    info.setText(tr("current_color_value").format(current=raw_hex))
                    info.setToolTip(tr("current_color_tooltip").format(current=raw_hex, adapted=shown_hex))

    def _manual_style_match_enabled(self) -> bool:
        return bool(getattr(self, '_chk_manual_match_style', None) and self._chk_manual_match_style.isChecked())

    def _active_manual_style_preset(self):
        if not self._manual_style_match_enabled():
            return None
        key = None
        combo = getattr(self, '_style_combo', None)
        if combo is not None:
            key = combo.currentData()
        style = None
        try:
            from core.presets import get_style
            style = get_style(key)
            from pipeline import _apply_pastel_tuning
            style = _apply_pastel_tuning(style, self._current_pastel_tuning())
        except Exception:
            return None
        if style is None or getattr(style, 'key', 'none') == 'none':
            return None
        return style

    def _manual_target_rgb(self, rgb: tuple[int, int, int]) -> tuple[int, int, int]:
        rgb = tuple(int(np.clip(v, 0, 255)) for v in rgb)
        style = self._active_manual_style_preset()
        if style is None:
            return rgb
        return adapt_rgb_to_style(rgb, style)

    def _on_color_picked(self, rgb: tuple):
        self._apply_picked_color(rgb, remember_raw=True)
        self.statusBar().showMessage(tr("picker_keep_tool_hint"), 2500)

    def _on_brush_stroke_started(self):
        """Snapshot once per local brush stroke, not once per mouse-move dab."""
        state = self._current_state()
        self._local_brush_stroke_active = bool(
            state is not None and state.result_bgr is not None)
        self._brush_changed_during_stroke = False
        if self._local_brush_stroke_active:
            state.push_undo()
            self._last_brush_was_local_edit = True

    def _on_hint_dab_added(self, x_norm: float, y_norm: float, rgb: tuple,
                           radius_norm: float):
        state = self._current_state()
        if state is None:
            return
        gap_close = self._gap_px_from_percent(self._gap_close_slider.value())
        region_map = state.hint_manager.bind_source_image(
            state.original_bgr, gap_close=gap_close)

        if state.result_bgr is not None:
            # Be robust even if a platform drops the stroke-start signal: a dab
            # must still create an undo checkpoint and become visible.
            if not self._local_brush_stroke_active:
                self._local_brush_stroke_active = True
                self._brush_changed_during_stroke = False
                state.push_undo()
                self._last_brush_was_local_edit = True
            # The visible brush is a true local post-edit.  It never becomes a
            # whole-face/whole-region model instruction.
            h, w = state.result_bgr.shape[:2]
            ix = min(w - 1, max(0, int(round(x_norm * (w - 1)))))
            iy = min(h - 1, max(0, int(round(y_norm * (h - 1)))))
            radius_px = max(1, int(round(radius_norm * w)))
            strength = self._manual_strength_slider.value() / 100.0
            if self._uses_ai_original_color():
                if state.ai_result_bgr is None:
                    self.statusBar().showMessage(tr("manual_restore_unavailable"), 4000)
                    return
                from core.local_brush import restore_local_brush_from_reference
                before_restore = state.result_bgr.copy()
                state.result_bgr, _mask = restore_local_brush_from_reference(
                    state.original_bgr, state.result_bgr, state.ai_result_bgr,
                    ix, iy, radius_px, opacity=0.88 * strength, region_map=region_map,
                    gap_close=gap_close)
                if state.filter_base_bgr is not None:
                    state.filter_base_bgr, _ = restore_local_brush_from_reference(
                        state.original_bgr, state.filter_base_bgr, state.ai_result_bgr,
                        ix, iy, radius_px, opacity=0.88 * strength, region_map=region_map,
                        gap_close=gap_close)
                self._brush_changed_during_stroke = self._brush_changed_during_stroke or not np.array_equal(before_restore, state.result_bgr)
                self._last_local_edit_mode = "restore"
            else:
                paint_rgb = self._manual_target_rgb(rgb)
                state.result_bgr, state.filter_base_bgr, _mask, changed = apply_brush_edit(
                    state.original_bgr, state.result_bgr, state.filter_base_bgr,
                    ix, iy, radius_px, paint_rgb,
                    opacity=min(1.0, 1.00 * strength), region_map=region_map,
                    gap_close=gap_close)
                self._brush_changed_during_stroke = self._brush_changed_during_stroke or changed
                self._last_local_edit_mode = "paint"
            if self._brush_changed_during_stroke:
                self._radio_view_edited.setChecked(True)
                # Immediate pixel refresh: the edit is visible while dragging,
                # not only after a mouse-release event.
                if hasattr(self._canvas, "update_image_pixels"):
                    self._canvas.update_image_pixels(state.result_bgr)
                else:
                    self._sync_view_after_edit(state)
            return

        # Before the first AI result, retain a *local* model hint.  Ordinary
        # manual hints no longer expand into highlight/mid/shadow points across
        # the entire connected region.
        state.push_undo()
        changed = state.hint_manager.add_manual_hint(x_norm, y_norm, rgb, radius_norm)
        if not changed:
            state.discard_unchanged_undo()
        self._refresh_hint_overlay()

    def _on_brush_stroke_finished(self):
        if not self._local_brush_stroke_active:
            return
        state = self._current_state()
        self._local_brush_stroke_active = False
        if state is None or state.result_bgr is None:
            return
        state.discard_unchanged_undo()
        self._canvas.clear_dabs()
        self._radio_view_edited.setChecked(True)
        self._sync_view_after_edit(state)
        if self._brush_changed_during_stroke:
            message = tr("local_brush_done")
        else:
            message = tr("manual_edit_no_change")
        self.statusBar().showMessage(message, 3500)
        self._brush_changed_during_stroke = False

    def _undo_last_hint(self):
        """Use the same force-undo history as Ctrl+Z.

        Keeping a second, hint-only undo path caused the canvas overlay and the
        actual page state to diverge.  One history now owns every edit.
        """
        state = self._current_state()
        if state is None:
            return
        if state.undo_stack:
            self._undo()
            self._last_brush_was_local_edit = False
            return
        self.statusBar().showMessage(tr("nothing_to_undo"), 1800)

    def _clear_manual_hints(self):
        state = self._current_state()
        if state is None:
            return
        state.push_undo()
        state.hint_manager.clear_manual_hints()
        state.discard_unchanged_undo()
        self._canvas.clear_dabs()
        self._refresh_hint_overlay()

    def _on_region_fill_requested(self, ix: int, iy: int):
        state = self._current_state()
        if state is None:
            return
        if state.result_bgr is None:
            QMessageBox.information(self, tr("no_result_title"), tr("no_result_body"))
            return

        paint_rgb = self._manual_target_rgb((self._brush_color.red(), self._brush_color.green(), self._brush_color.blue()))
        hex_color = '#%02x%02x%02x' % paint_rgb
        gap_close = self._gap_px_from_percent(self._gap_close_slider.value())
        fill_mode = self._fill_mode_combo.currentData()

        region_map = state.hint_manager.bind_source_image(
            state.original_bgr, gap_close=gap_close)
        new_img, new_base, mask, changed = apply_region_edit(
            state.original_bgr, state.result_bgr, state.filter_base_bgr,
            ix, iy, hex_color, gap_close=gap_close, mode=fill_mode,
            feather=2, region_map=region_map)
        if not mask.any():
            self.statusBar().showMessage(tr("no_fill_area"), 5000)
            return

        state.push_undo()  # snapshot `before` state for Ctrl+Z
        if self._uses_ai_original_color() and state.ai_result_bgr is not None:
            state.result_bgr = state.result_bgr.copy()
            restore = state.ai_result_bgr
            if restore.shape[:2] != state.result_bgr.shape[:2]:
                restore = cv2.resize(restore, (state.result_bgr.shape[1], state.result_bgr.shape[0]), interpolation=cv2.INTER_AREA)
            state.result_bgr[mask > 0] = restore[mask > 0]
            if state.filter_base_bgr is not None:
                state.filter_base_bgr[mask > 0] = restore[mask > 0]
            message = tr("region_restore_done")
        else:
            state.result_bgr = new_img
            state.filter_base_bgr = new_base
            message = tr("region_fill_done") if changed else tr("manual_edit_no_change")
        state.discard_unchanged_undo()
        self._radio_view_edited.setChecked(True)
        self._sync_view_after_edit(state)
        self.statusBar().showMessage(message, 3000)

    def _undo(self):
        """Force-revert to the nearest genuinely different prior state."""
        state = self._current_state()
        if state is None or not state.undo_stack:
            self.statusBar().showMessage(tr("nothing_to_undo"), 1800)
            return
        current = state.edit_snapshot()
        target = None
        # Skip stale/no-op checkpoints so one Ctrl+Z always produces a visible
        # state change whenever an earlier state exists.
        while state.undo_stack:
            candidate = state.undo_stack.pop()
            if not state._snapshot_equal(candidate, current):
                target = candidate
                break
        if target is None:
            self.statusBar().showMessage(tr("nothing_to_undo"), 1800)
            return
        state.redo_stack.append(current)
        if len(state.redo_stack) > 20:
            state.redo_stack.pop(0)
        state.restore_snapshot(target)
        self._radio_view_edited.setChecked(state.result_bgr is not None)
        if state.result_bgr is None:
            self._radio_view_original.setChecked(True)
            self._unmark_page_done(state.path)
        else:
            self._mark_page_done(state.path)
        self._canvas.clear_dabs()
        self._sync_view_after_edit(state)
        self._refresh_character_diagnostics()
        self.statusBar().showMessage(tr("force_undone"), 2200)

    def _redo(self):
        """Restore the nearest genuinely different forward state."""
        state = self._current_state()
        if state is None or not state.redo_stack:
            self.statusBar().showMessage(tr("nothing_to_redo"), 1800)
            return
        current = state.edit_snapshot()
        target = None
        while state.redo_stack:
            candidate = state.redo_stack.pop()
            if not state._snapshot_equal(candidate, current):
                target = candidate
                break
        if target is None:
            self.statusBar().showMessage(tr("nothing_to_redo"), 1800)
            return
        state.undo_stack.append(current)
        if len(state.undo_stack) > 20:
            state.undo_stack.pop(0)
        state.restore_snapshot(target)
        self._radio_view_edited.setChecked(state.result_bgr is not None)
        if state.result_bgr is None:
            self._radio_view_original.setChecked(True)
            self._unmark_page_done(state.path)
        else:
            self._mark_page_done(state.path)
        self._canvas.clear_dabs()
        self._sync_view_after_edit(state)
        self._refresh_character_diagnostics()
        self.statusBar().showMessage(tr("redone"), 2000)

    def _restore_to_original(self):
        """Destructive: wipes this page's colorize/edit history entirely."""
        state = self._current_state()
        if state is None:
            return
        reply = QMessageBox.question(
            self, tr("confirm_reset_title"), tr("confirm_reset_body"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        state.push_undo()
        state.ai_result_bgr = None
        state.result_bgr = None
        state.filter_base_bgr = None
        state.quality_report = None
        state.hint_manager = HintManager()
        state.hint_manager.bind_source_image(
            state.original_bgr,
            gap_close=self._gap_px_from_percent(self._gap_close_slider.value()))
        self._canvas.clear_dabs()
        self._radio_view_edited.setChecked(False)
        self._radio_view_original.setChecked(True)
        self._sync_view_after_edit(state)
        self._unmark_page_done(state.path)

    def _restore_to_ai_result(self):
        """Discard region-fill touch-ups, keep the last full mc-v2 run."""
        state = self._current_state()
        if state is None or state.ai_result_bgr is None:
            return
        state.push_undo()
        state.result_bgr = state.ai_result_bgr.copy()
        self._radio_view_edited.setChecked(True)
        self._sync_view_after_edit(state)
        self.statusBar().showMessage(tr("restored_ai"), 3000)

    def _on_view_mode_changed(self):
        state = self._current_state()
        if state is not None:
            self._sync_view_after_edit(state)

    def _sync_view_after_edit(self, state: "PageState"):
        """Refresh the canvas according to the current view-mode radio,
        falling back sensibly if the selected layer doesn't exist yet."""
        if self._radio_view_original.isChecked():
            self._canvas.set_image(state.original_bgr, fit=False)
        elif self._radio_view_ai.isChecked() and state.ai_result_bgr is not None:
            self._canvas.set_image(state.ai_result_bgr, fit=False)
        elif state.result_bgr is not None:
            self._canvas.set_image(state.result_bgr, fit=False)
        else:
            self._canvas.set_image(state.original_bgr, fit=False)
        self._refresh_hint_overlay()

    def _refresh_hint_overlay(self):
        state = self._current_state()
        if state is None or not hasattr(self, "_canvas"):
            return
        show_regions = bool(getattr(self, "_chk_show_regions", None) and
                            self._chk_show_regions.isChecked())
        region_map = state.hint_manager.region_map
        labels = region_map.labels if region_map is not None else None
        self._canvas.set_hint_overlay(
            state.hint_manager.preview_hints(), labels,
            show_regions=show_regions,
            context=getattr(state.hint_manager, "last_page_context", None))

    # ── Export ────────────────────────────────────────────────────────

    def _export_current_page(self):
        state = self._current_state()
        if state is None or state.result_bgr is None:
            QMessageBox.information(self, tr("no_result_title"), tr("no_colorized_page"))
            return
        default_name = os.path.splitext(os.path.basename(state.path))[0] + "_colorized.png"
        out_path, _ = QFileDialog.getSaveFileName(self, tr("export_page"), default_name,
                                                   "PNG (*.png);;JPEG (*.jpg)")
        if not out_path:
            return
        from core.imageio import imwrite as _uimwrite
        _uimwrite(out_path, state.result_bgr)
        self.statusBar().showMessage(tr("exported_to").format(path=out_path), 5000)

    def _export_all_pages(self):
        done = [s for s in self._pages.values() if s.result_bgr is not None]
        if not done:
            QMessageBox.information(self, tr("no_result_title"), tr("no_colorized_pages"))
            return
        out_dir = QFileDialog.getExistingDirectory(self, tr("select_export_folder"))
        if not out_dir:
            return
        for state in done:
            name = os.path.splitext(os.path.basename(state.path))[0] + "_colorized.png"
            from core.imageio import imwrite as _uimwrite
            _uimwrite(os.path.join(out_dir, name), state.result_bgr)
        self.statusBar().showMessage(
            tr("exported_n_pages").format(n=len(done), dir=out_dir), 5000)

    # ── Misc ──────────────────────────────────────────────────────────

    def _canvas_zoom_in(self):
        self._canvas.zoom_in()

    def _canvas_zoom_out(self):
        self._canvas.zoom_out()

    def _update_controls_enabled(self):
        has_page = self._current_path is not None
        for btn in (self._btn_auto, self._btn_regenerate, self._btn_undo,
                   self._btn_clear, self._btn_undo_edit, self._btn_redo_edit,
                   self._btn_restore_ai, self._btn_restore_bw,
                   self._btn_export_page):
            btn.setEnabled(has_page)
