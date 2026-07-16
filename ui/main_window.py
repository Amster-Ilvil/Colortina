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
)

from config import Config
from core.hint_manager import HintManager
from core.lineart_fill import lineart_region_recolor
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

    def __init__(self, path: str, original_bgr: np.ndarray):
        self.path = path
        self.original_bgr = original_bgr
        self.ai_result_bgr: np.ndarray | None = None
        self.result_bgr: np.ndarray | None = None
        self.hint_manager = HintManager()
        # Unified undo/redo over the "Edited" layer — covers both a full
        # colorize run and a region-fill touch-up, so Ctrl+Z steps back
        # through whatever you actually did, in order.
        self.undo_stack: list[np.ndarray] = []
        self.redo_stack: list[np.ndarray] = []

    def push_undo(self):
        """Call BEFORE mutating result_bgr — snapshots the pre-edit state."""
        if self.result_bgr is not None:
            self.undo_stack.append(self.result_bgr.copy())
            if len(self.undo_stack) > 30:
                self.undo_stack.pop(0)
        self.redo_stack.clear()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(tr("window_title"))
        self.resize(1400, 900)

        icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "assets", "icon.png")
        if os.path.isfile(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self._pages: dict[str, PageState] = {}
        self._current_path: str | None = None
        self._batch_worker: BatchColorizeWorker | None = None
        self._brush_color = QColor(255, 120, 120)

        # Book-level state (persists across all pages of this session):
        # an optional extracted/loaded StyleProfile (overrides the style
        # preset combo when set) and one CharacterMemory per label that
        # needs multi-character consistency.
        self._style_profile = None
        self._character_memories: dict = {}

        from core.style_engine import StyleEngine
        self._style_engine = StyleEngine(styles_dir=Config.STYLES_DIR)

        self._build_menu_bar()
        self._build_chrome()
        self._rebuild_central_widget()
        self._update_controls_enabled()

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
        added = 0
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if not path:
                continue
            if os.path.isdir(path):
                sub = []
                for root, _dirs, files in os.walk(path):
                    for f in files:
                        if f.lower().endswith(self._IMAGE_EXTS):
                            sub.append(os.path.join(root, f))
                sub.sort(key=lambda p: (os.path.dirname(p),
                                        self._natural_key(os.path.basename(p))))
                for p in sub:
                    self._add_page(p)
                added += len(sub)
            elif path.lower().endswith(".pdf"):
                self._import_pdf_path(path)
            elif path.lower().endswith(self._IMAGE_EXTS):
                self._add_page(path)
                added += 1
        if added:
            self.statusBar().showMessage(tr("imported_n_pages").format(n=added), 5000)
        event.acceptProposedAction()

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

    # ── 0-100 slider <-> underlying pixel-unit conversion ─────────────
    # Both sliders are shown to the user as a plain 0-100 scale; these
    # map that back to the actual units the canvas / paint_bucket code
    # expects (brush radius in image px, gap-close dilation in px).

    _BRUSH_PX_MIN, _BRUSH_PX_MAX = 2, 60
    _GAP_PX_MIN, _GAP_PX_MAX = 0, 12

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
        lo, hi = cls._GAP_PX_MIN, cls._GAP_PX_MAX
        return round(lo + (hi - lo) * (v / 100))

    @classmethod
    def _gap_percent_from_px(cls, px: int) -> int:
        lo, hi = cls._GAP_PX_MIN, cls._GAP_PX_MAX
        return round((px - lo) / (hi - lo) * 100)

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
        prev_char_mem = getattr(self, "_chk_character_memory", None)
        prev_char_mem = prev_char_mem.isChecked() if prev_char_mem is not None else Config.USE_CHARACTER_MEMORY
        prev_skip_colored = getattr(self, "_chk_skip_colored", None)
        prev_skip_colored = prev_skip_colored.isChecked() if prev_skip_colored is not None else True
        prev_brush_pct = getattr(self, "_brush_slider", None)
        prev_brush_pct = prev_brush_pct.value() if prev_brush_pct is not None else self._brush_percent_from_px(12)
        prev_gap_pct = getattr(self, "_gap_close_slider", None)
        prev_gap_pct = prev_gap_pct.value() if prev_gap_pct is not None else self._gap_percent_from_px(4)

        old_central = self.centralWidget()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_canvas_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([220, 900, 280])
        self.setCentralWidget(splitter)
        if old_central is not None:
            old_central.deleteLater()

        if prev_style_key:
            idx = self._style_combo.findData(prev_style_key)
            if idx >= 0:
                self._style_combo.setCurrentIndex(idx)
        self._chk_character_memory.setChecked(prev_char_mem)
        self._chk_skip_colored.setChecked(prev_skip_colored)
        self._brush_slider.setValue(prev_brush_pct)
        self._gap_close_slider.setValue(prev_gap_pct)
        self._canvas.set_brush_color(self._brush_color)
        self._update_color_swatch()
        self._refresh_custom_style_combo()
        self._update_style_profile_label()

        self._repopulate_page_list()
        self._update_controls_enabled()
        self._update_device_label()

    def _build_left_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        self._lang_button = QPushButton(tr("lang_button"))
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
        return w

    def _build_canvas_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        toolbar = QHBoxLayout()
        btn_fit = QPushButton(tr("fit_view"))
        btn_fit.clicked.connect(lambda: self._canvas.fit_view())
        btn_zoom_in = QPushButton("＋")
        btn_zoom_in.setFixedWidth(32)
        btn_zoom_in.clicked.connect(self._canvas_zoom_in)
        btn_zoom_out = QPushButton("－")
        btn_zoom_out.setFixedWidth(32)
        btn_zoom_out.clicked.connect(self._canvas_zoom_out)
        toolbar.addWidget(btn_fit)
        toolbar.addWidget(btn_zoom_out)
        toolbar.addWidget(btn_zoom_in)
        toolbar.addStretch(1)

        self._canvas = HintCanvas()
        self._canvas.hint_dab_added.connect(self._on_hint_dab_added)
        self._canvas.color_picked.connect(self._on_color_picked)
        self._canvas.region_fill_requested.connect(self._on_region_fill_requested)

        layout.addLayout(toolbar)
        layout.addWidget(self._canvas, stretch=1)
        return w

    def _build_right_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        self._device_label = QLabel(f"{tr('device_label')}—")
        layout.addWidget(self._device_label)

        # Style / Character Memory
        style_group = QGroupBox(tr("style_quality_group"))
        style_layout = QVBoxLayout(style_group)

        from core.presets import STYLE_PRESETS

        style_row = QHBoxLayout()
        style_row.addWidget(QLabel(tr("style_label")))
        self._style_combo = QComboBox()
        for key, preset in STYLE_PRESETS.items():
            self._style_combo.addItem(preset.label, key)
        self._style_combo.currentIndexChanged.connect(self._on_style_combo_changed)
        style_row.addWidget(self._style_combo, stretch=1)
        style_layout.addLayout(style_row)

        self._style_profile_label = QLabel(tr("style_profile_unset"))
        self._style_profile_label.setWordWrap(True)
        self._style_profile_label.setStyleSheet("color: #888; font-size: 11px;")
        style_layout.addWidget(self._style_profile_label)

        self._chk_character_memory = QCheckBox(tr("character_memory_checkbox"))
        self._chk_character_memory.setChecked(Config.USE_CHARACTER_MEMORY)
        self._chk_character_memory.setToolTip(tr("character_memory_tooltip"))
        style_layout.addWidget(self._chk_character_memory)

        self._chk_skip_colored = QCheckBox(tr("skip_colored_checkbox"))
        self._chk_skip_colored.setChecked(True)
        self._chk_skip_colored.setToolTip(tr("skip_colored_tooltip"))
        style_layout.addWidget(self._chk_skip_colored)

        layout.addWidget(style_group)

        # Custom style library — new (extract+name+save) / load from disk /
        # apply a saved one / delete / clear back to the preset combo above.
        custom_style_group = QGroupBox(tr("custom_style_group"))
        custom_style_layout = QVBoxLayout(custom_style_group)

        self._custom_style_combo = QComboBox()
        custom_style_layout.addWidget(self._custom_style_combo)

        self._btn_new_style = QPushButton(tr("btn_new_style"))
        self._btn_new_style.clicked.connect(self._new_style_from_reference)
        custom_style_layout.addWidget(self._btn_new_style)

        self._btn_load_style_file = QPushButton(tr("btn_load_style_file"))
        self._btn_load_style_file.clicked.connect(self._load_style_file)
        custom_style_layout.addWidget(self._btn_load_style_file)

        apply_delete_row = QHBoxLayout()
        self._btn_apply_saved_style = QPushButton(tr("btn_apply_saved_style"))
        self._btn_apply_saved_style.clicked.connect(self._apply_saved_style)
        self._btn_delete_style = QPushButton(tr("btn_delete_style"))
        self._btn_delete_style.clicked.connect(self._delete_saved_style)
        apply_delete_row.addWidget(self._btn_apply_saved_style)
        apply_delete_row.addWidget(self._btn_delete_style)
        custom_style_layout.addLayout(apply_delete_row)

        self._btn_clear_style_profile = QPushButton(tr("btn_clear_style_profile"))
        self._btn_clear_style_profile.clicked.connect(self._clear_style_profile)
        custom_style_layout.addWidget(self._btn_clear_style_profile)

        layout.addWidget(custom_style_group)
        self._refresh_custom_style_combo()

        # Auto colorize
        auto_group = QGroupBox(tr("auto_group"))
        auto_layout = QVBoxLayout(auto_group)
        self._btn_auto = QPushButton(tr("auto_btn"))
        self._btn_auto.clicked.connect(self._run_auto_colorize)
        auto_layout.addWidget(self._btn_auto)
        layout.addWidget(auto_group)

        # Edit tools
        edit_group = QGroupBox(tr("edit_group"))
        edit_layout = QVBoxLayout(edit_group)

        tool_row = QHBoxLayout()
        self._tool_group = QButtonGroup(self)
        self._radio_brush = QRadioButton(tr("tool_brush"))
        self._radio_brush.setChecked(True)
        self._radio_eyedropper = QRadioButton(tr("tool_eyedropper"))
        self._radio_bucket = QRadioButton(tr("tool_bucket"))
        self._tool_group.addButton(self._radio_brush)
        self._tool_group.addButton(self._radio_eyedropper)
        self._tool_group.addButton(self._radio_bucket)
        self._radio_brush.toggled.connect(self._on_tool_changed)
        self._radio_eyedropper.toggled.connect(self._on_tool_changed)
        self._radio_bucket.toggled.connect(self._on_tool_changed)
        tool_row.addWidget(self._radio_brush)
        tool_row.addWidget(self._radio_eyedropper)
        tool_row.addWidget(self._radio_bucket)
        edit_layout.addLayout(tool_row)
        bucket_hint = QLabel(tr("bucket_hint"))
        bucket_hint.setWordWrap(True)
        bucket_hint.setStyleSheet("color: #888; font-size: 11px;")
        edit_layout.addWidget(bucket_hint)

        color_row = QHBoxLayout()
        color_row.addWidget(QLabel(tr("color_label")))
        self._color_swatch = QPushButton()
        self._color_swatch.setFixedSize(32, 24)
        self._color_swatch.clicked.connect(self._pick_color)
        self._update_color_swatch()
        color_row.addWidget(self._color_swatch)
        color_row.addStretch(1)
        edit_layout.addLayout(color_row)

        # Brush size: slider AND a 0-100 spinbox (typeable), kept in sync;
        # mapped internally to an actual pixel radius of
        # BRUSH_RADIUS_PX_RANGE so brush behavior is unchanged, just the
        # displayed unit.
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel(tr("brush_size_label")))
        self._brush_slider = QSlider(Qt.Orientation.Horizontal)
        self._brush_slider.setRange(0, 100)
        self._brush_slider.setValue(self._brush_percent_from_px(12))
        self._brush_spin = QSpinBox()
        self._brush_spin.setRange(0, 100)
        self._brush_spin.setValue(self._brush_slider.value())
        self._brush_slider.valueChanged.connect(self._brush_spin.setValue)
        self._brush_spin.valueChanged.connect(self._brush_slider.setValue)
        self._brush_slider.valueChanged.connect(
            lambda v: self._canvas.set_brush_radius(self._brush_px_from_percent(v)))
        size_row.addWidget(self._brush_slider, stretch=1)
        size_row.addWidget(self._brush_spin)
        edit_layout.addLayout(size_row)

        # Gap closing: same 0-100 scale (slider + typeable spinbox),
        # mapped to the actual 0-12px dilation kernel
        # lineart_region_recolor() expects.
        tol_row = QHBoxLayout()
        tol_row.addWidget(QLabel(tr("gap_close_label")))
        self._gap_close_slider = QSlider(Qt.Orientation.Horizontal)
        self._gap_close_slider.setRange(0, 100)
        self._gap_close_slider.setValue(self._gap_percent_from_px(4))
        self._gap_close_slider.setToolTip(tr("gap_close_tooltip"))
        self._gap_close_spin = QSpinBox()
        self._gap_close_spin.setRange(0, 100)
        self._gap_close_spin.setValue(self._gap_close_slider.value())
        self._gap_close_spin.setToolTip(tr("gap_close_tooltip"))
        self._gap_close_slider.valueChanged.connect(self._gap_close_spin.setValue)
        self._gap_close_spin.valueChanged.connect(self._gap_close_slider.setValue)
        tol_row.addWidget(self._gap_close_slider, stretch=1)
        tol_row.addWidget(self._gap_close_spin)
        edit_layout.addLayout(tol_row)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel(tr("fill_mode_label")))
        self._fill_mode_combo = QComboBox()
        self._fill_mode_combo.addItem(tr("fill_mode_shift"), "shift")
        self._fill_mode_combo.addItem(tr("fill_mode_shading"), "shading")
        self._fill_mode_combo.addItem(tr("fill_mode_flat"), "flat")
        mode_row.addWidget(self._fill_mode_combo, stretch=1)
        edit_layout.addLayout(mode_row)
        mode_hint = QLabel(tr("fill_mode_hint"))
        mode_hint.setWordWrap(True)
        mode_hint.setStyleSheet("color: #888; font-size: 11px;")
        edit_layout.addWidget(mode_hint)

        undo_row = QHBoxLayout()
        self._btn_undo = QPushButton(tr("undo_last_hint"))
        self._btn_undo.clicked.connect(self._undo_last_hint)
        self._btn_clear = QPushButton(tr("clear_manual_hints"))
        self._btn_clear.clicked.connect(self._clear_manual_hints)
        undo_row.addWidget(self._btn_undo)
        undo_row.addWidget(self._btn_clear)
        edit_layout.addLayout(undo_row)


        self._btn_regenerate = QPushButton(tr("regenerate_btn"))
        self._btn_regenerate.clicked.connect(self._run_regenerate)
        edit_layout.addWidget(self._btn_regenerate)

        layout.addWidget(edit_group)

        # Undo / redo — unified over the "Edited" layer (colorize runs +
        # region-fill touch-ups), independent from the hint-dab undo above
        undo_redo_row = QHBoxLayout()
        self._btn_undo_edit = QPushButton(tr("undo_edit"))
        self._btn_undo_edit.clicked.connect(self._undo)
        self._btn_redo_edit = QPushButton(tr("redo_edit"))
        self._btn_redo_edit.clicked.connect(self._redo)
        undo_redo_row.addWidget(self._btn_undo_edit)
        undo_redo_row.addWidget(self._btn_redo_edit)
        layout.addLayout(undo_redo_row)

        # View / restore
        view_group = QGroupBox(tr("view_group"))
        view_layout = QVBoxLayout(view_group)

        self._view_tool_group = QButtonGroup(self)
        self._radio_view_original = QRadioButton(tr("view_original"))
        self._radio_view_ai = QRadioButton(tr("view_ai"))
        self._radio_view_edited = QRadioButton(tr("view_edited"))
        self._radio_view_edited.setChecked(True)
        for rb in (self._radio_view_original, self._radio_view_ai, self._radio_view_edited):
            self._view_tool_group.addButton(rb)
            rb.toggled.connect(self._on_view_mode_changed)
            view_layout.addWidget(rb)

        self._btn_restore_ai = QPushButton(tr("restore_ai"))
        self._btn_restore_ai.clicked.connect(self._restore_to_ai_result)
        view_layout.addWidget(self._btn_restore_ai)

        self._btn_restore_bw = QPushButton(tr("restore_bw"))
        self._btn_restore_bw.clicked.connect(self._restore_to_original)
        view_layout.addWidget(self._btn_restore_bw)

        layout.addWidget(view_group)

        # Export
        export_group = QGroupBox(tr("export_group"))
        export_layout = QVBoxLayout(export_group)
        self._btn_export_page = QPushButton(tr("export_page"))
        self._btn_export_page.clicked.connect(self._export_current_page)
        self._btn_export_all = QPushButton(tr("export_all"))
        self._btn_export_all.clicked.connect(self._export_all_pages)
        export_layout.addWidget(self._btn_export_page)
        export_layout.addWidget(self._btn_export_all)
        layout.addWidget(export_group)

        layout.addStretch(1)
        return w

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
        """Batch-import every image inside a folder (recursive), in
        natural page order — the folder workflow from Manga-Colorizer-GUI."""
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
        for p in paths:
            self._add_page(p)
        self.statusBar().showMessage(tr("imported_n_pages").format(n=len(paths)), 5000)

    def _import_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, tr("import_images"), "", "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tiff)")
        for p in paths:
            self._add_page(p)

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
        for p in page_paths:
            self._add_page(p)
        self.statusBar().showMessage(tr("imported_n_pages").format(n=len(page_paths)), 5000)

    def _add_page(self, path: str):
        image_bgr = cv2.imread(path)
        if image_bgr is None:
            QMessageBox.warning(self, tr("warning_title"), tr("cannot_read_image").format(path=path))
            return
        if path in self._pages:
            return
        self._pages[path] = PageState(path, image_bgr)
        item = QListWidgetItem(os.path.basename(path))
        item.setData(Qt.ItemDataRole.UserRole, path)
        self._page_list.addItem(item)
        if self._page_list.count() == 1:
            self._page_list.setCurrentItem(item)

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
                label = "✓ " + label
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, path)
            self._page_list.addItem(item)
            if path == self._current_path:
                selected_row = i
        if selected_row < 0 and self._page_list.count() > 0:
            selected_row = 0
        if selected_row >= 0:
            self._page_list.setCurrentRow(selected_row)

    # ── Page selection ────────────────────────────────────────────────

    def _on_page_selected(self, current: QListWidgetItem, _previous):
        if current is None:
            return
        path = current.data(Qt.ItemDataRole.UserRole)
        self._current_path = path
        state = self._pages[path]
        self._sync_view_after_edit(state)
        # Auto-fit the newly displayed page to the window.
        self._canvas.fit_view()
        self._update_controls_enabled()

    # ── Colorize actions ─────────────────────────────────────────────

    def showEvent(self, event):
        """First time the window is actually shown the canvas finally has
        its real size — re-fit the current page so it fills the window."""
        super().showEvent(event)
        from PySide6.QtCore import QTimer
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
        paths = self._selected_paths()
        if not paths:
            return
        pages = [(p, self._pages[p].original_bgr, self._pages[p].hint_manager)
                 for p in paths]

        character_memories = None
        if self._chk_character_memory.isChecked():
            character_memories = self._get_or_create_character_memories()

        self._set_busy(True, tr("batch_colorizing").format(n=len(pages))
                       if len(pages) > 1 else tr("colorizing"))
        self._batch_worker = BatchColorizeWorker(
            pages, regenerate_auto,
            style_key=self._style_combo.currentData(),
            quality_key=Config.DEFAULT_QUALITY_KEY,
            style_profile=self._style_profile,
            character_memories=character_memories,
            skip_colored=self._chk_skip_colored.isChecked(),
        )
        self._batch_worker.page_done.connect(self._on_batch_page_done)
        self._batch_worker.page_error.connect(self._on_batch_page_error)
        self._batch_worker.status.connect(lambda msg: self.statusBar().showMessage(msg, 0))
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

    def _on_style_combo_changed(self):
        """Selecting the built-in 'None' preset means the user wants the
        literal, unmodified mc-v2 output — that only actually happens if
        no custom StyleProfile is overriding it (StyleProfile always wins
        over the preset combo, see pipeline.colorize_page), so clear it
        automatically and let them know."""
        if self._style_combo.currentData() == "none" and self._style_profile is not None:
            self._style_profile = None
            self._update_style_profile_label()
            self.statusBar().showMessage(tr("style_cleared_msg").format(name=self._style_combo.currentText()), 4000)

    def _refresh_custom_style_combo(self):
        self._custom_style_combo.clear()
        entries = self._style_engine.list_styles_with_names()
        if not entries:
            self._custom_style_combo.addItem(tr("custom_style_combo_placeholder"), None)
            return
        for filename, name in entries:
            self._custom_style_combo.addItem(name, filename)

    def _update_style_profile_label(self):
        if self._style_profile is not None:
            p = self._style_profile
            self._style_profile_label.setText(
                tr("style_profile_active").format(
                    name=p.name, saturation=p.saturation,
                    contrast=p.contrast, temperature=p.temperature))
        else:
            self._style_profile_label.setText(tr("style_profile_unset"))

    def _new_style_from_reference(self):
        """New style: pick one or more color reference images, name it,
        extract a StyleProfile, save it to the library, and make it the
        active style."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, tr("extract_style_dialog_title"), "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        if not paths:
            return

        images = []
        for p in paths:
            img = cv2.imread(p)
            if img is not None:
                images.append(img)
        if not images:
            QMessageBox.warning(self, tr("extract_style_fail_title"),
                               tr("extract_style_fail_body"))
            return

        default_name = os.path.splitext(os.path.basename(paths[0]))[0]
        name, ok = QInputDialog.getText(
            self, tr("new_style_name_title"), tr("new_style_name_label"),
            text=default_name or tr("new_style_name_default"))
        if not ok:
            return
        name = (name or "").strip() or default_name or tr("new_style_name_default")

        try:
            from core.guided_colorist import _get_classifier
            classifier = _get_classifier()
        except Exception:
            classifier = None

        if len(images) == 1:
            profile = self._style_engine.extract_from_reference(images[0], name=name, classifier=classifier)
        else:
            profile = self._style_engine.extract_from_references(images, name=name, classifier=classifier)
        self._style_profile = profile
        self._style_engine.save_style(profile)

        # References also seed the hair CharacterMemory with REAL colors
        # (if any hair-like regions were found), instead of the rotating
        # fallback palette — resolves same-tone/different-hue ambiguity
        # for however many characters appear across the references.
        memories = self._get_or_create_character_memories()
        for img in images:
            if memories["hair"].seed_from_reference(img, classifier=classifier):
                break  # first reference with usable hair regions wins the seeding

        self._update_style_profile_label()
        self._refresh_custom_style_combo()
        idx = self._custom_style_combo.findData(f"{name.lower().replace(' ', '_')}.ccstyle")
        if idx >= 0:
            self._custom_style_combo.setCurrentIndex(idx)
        self.statusBar().showMessage(tr("extract_style_status").format(name=name), 5000)

    def _load_style_file(self):
        """Load an existing .ccstyle file from anywhere on disk and make
        it the active style. A copy is also kept in the style library
        folder so it can be re-selected/deleted from the combo above."""
        path, _ = QFileDialog.getOpenFileName(
            self, tr("load_style_dialog_title"), "", "Colortina Style (*.ccstyle)")
        if not path:
            return
        from core.style_engine import StyleProfile
        try:
            profile = StyleProfile.load(path)
        except Exception as exc:
            QMessageBox.warning(self, tr("load_style_fail_title"),
                               tr("load_style_fail_body").format(exc=exc))
            return

        self._style_profile = profile
        styles_dir = os.path.abspath(self._style_engine.styles_dir)
        if os.path.dirname(os.path.abspath(path)) != styles_dir:
            self._style_engine.save_style(profile, filename=os.path.basename(path))

        self._update_style_profile_label()
        self._refresh_custom_style_combo()
        idx = self._custom_style_combo.findData(os.path.basename(path))
        if idx >= 0:
            self._custom_style_combo.setCurrentIndex(idx)
        self.statusBar().showMessage(tr("style_loaded_msg").format(name=profile.name), 4000)

    def _apply_saved_style(self):
        filename = self._custom_style_combo.currentData()
        if not filename:
            QMessageBox.information(self, tr("no_result_title"), tr("no_style_selected_msg"))
            return
        try:
            profile = self._style_engine.load_style(filename)
        except Exception as exc:
            QMessageBox.warning(self, tr("load_style_fail_title"),
                               tr("load_style_fail_body").format(exc=exc))
            return
        self._style_profile = profile
        self._update_style_profile_label()
        self.statusBar().showMessage(tr("style_loaded_msg").format(name=profile.name), 4000)

    def _delete_saved_style(self):
        filename = self._custom_style_combo.currentData()
        if not filename:
            QMessageBox.information(self, tr("no_result_title"), tr("no_style_selected_msg"))
            return
        name = self._custom_style_combo.currentText()
        reply = QMessageBox.question(
            self, tr("confirm_delete_style_title"), tr("confirm_delete_style_body").format(name=name),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._style_engine.delete_style(filename)
        if self._style_profile is not None and self._style_profile.name == name:
            self._style_profile = None
            self._update_style_profile_label()
        self._refresh_custom_style_combo()
        self.statusBar().showMessage(tr("style_deleted_msg").format(name=name), 4000)

    def _clear_style_profile(self):
        self._style_profile = None
        self._update_style_profile_label()
        self.statusBar().showMessage(
            tr("style_cleared_msg").format(name=self._style_combo.currentText()), 4000)

    def _on_batch_page_done(self, path: str, result_bgr: np.ndarray):
        state = self._pages.get(path)
        if state is None:
            return
        state.push_undo()  # snapshot whatever was there before this run
        state.ai_result_bgr = result_bgr
        state.result_bgr = result_bgr
        self._mark_page_done(path)
        self._canvas.clear_dabs()
        if path == self._current_path:
            self._radio_view_edited.setChecked(True)
            self._sync_view_after_edit(state)

    def _on_batch_page_error(self, path: str, message: str):
        name = path.split("/")[-1]
        self.statusBar().showMessage(tr("page_colorize_failed").format(name=name, message=message), 8000)

    def _on_batch_finished(self):
        self._set_busy(False, tr("colorize_done"))
        self._update_device_label()

    def _mark_page_done(self, path: str):
        for i in range(self._page_list.count()):
            item = self._page_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == path:
                item.setText("✓ " + os.path.basename(path))
                break

    def _unmark_page_done(self, path: str):
        for i in range(self._page_list.count()):
            item = self._page_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == path:
                item.setText(os.path.basename(path))
                break

    def _set_busy(self, busy: bool, message: str = ""):
        self._progress.setVisible(busy)
        for btn in (self._btn_auto, self._btn_regenerate, self._btn_undo,
                   self._btn_clear, self._btn_undo_edit, self._btn_redo_edit,
                   self._btn_restore_ai, self._btn_restore_bw,
                   self._btn_export_page, self._btn_export_all):
            btn.setEnabled(not busy)
        if message:
            self.statusBar().showMessage(message, 0 if busy else 5000)

    def _update_device_label(self):
        try:
            from pipeline import get_colorizer
            colorizer = get_colorizer(Config)
            text = f"{tr('device_label')}{colorizer.device_name}"
            if colorizer.device_warning:
                text += f"  ⚠ {colorizer.device_warning}"
            self._device_label.setText(text)
        except Exception:
            pass

    # ── Editing (manual hints) ───────────────────────────────────────

    def _on_tool_changed(self):
        if self._radio_brush.isChecked():
            tool = HintCanvas.TOOL_BRUSH
        elif self._radio_eyedropper.isChecked():
            tool = HintCanvas.TOOL_EYEDROPPER
        else:
            tool = HintCanvas.TOOL_BUCKET
        self._canvas.set_tool(tool)

    def _pick_color(self):
        color = QColorDialog.getColor(self._brush_color, self, tr("pick_color_title"))
        if color.isValid():
            self._brush_color = color
            self._canvas.set_brush_color(color)
            self._update_color_swatch()

    def _update_color_swatch(self):
        self._color_swatch.setStyleSheet(
            f"background-color: {self._brush_color.name()}; border: 1px solid #888;")

    def _on_color_picked(self, rgb: tuple):
        self._brush_color = QColor(*rgb)
        self._canvas.set_brush_color(self._brush_color)
        self._update_color_swatch()
        # Picking a color is a natural "now go paint with it" cue
        self._radio_brush.setChecked(True)

    def _on_hint_dab_added(self, x_norm: float, y_norm: float, rgb: tuple,
                           radius_norm: float):
        state = self._current_state()
        if state is None:
            return
        state.hint_manager.add_manual_hint(x_norm, y_norm, rgb, radius_norm)

    def _undo_last_hint(self):
        state = self._current_state()
        if state is None:
            return
        state.hint_manager.undo_last_manual()
        self._canvas.undo_last_dab()

    def _clear_manual_hints(self):
        state = self._current_state()
        if state is None:
            return
        state.hint_manager.clear_manual_hints()
        self._canvas.clear_dabs()

    def _on_region_fill_requested(self, ix: int, iy: int):
        state = self._current_state()
        if state is None:
            return
        if state.result_bgr is None:
            QMessageBox.information(self, tr("no_result_title"), tr("no_result_body"))
            return

        hex_color = self._brush_color.name()  # '#rrggbb'
        gap_close = self._gap_px_from_percent(self._gap_close_slider.value())
        fill_mode = self._fill_mode_combo.currentData()

        before = state.result_bgr
        new_img, mask = lineart_region_recolor(
            state.original_bgr, state.result_bgr.copy(), ix, iy, hex_color,
            gap_close=gap_close, mode=fill_mode, feather=3)
        if not mask.any():
            self.statusBar().showMessage(tr("no_fill_area"), 5000)
            return

        state.push_undo()  # snapshot `before` state for Ctrl+Z
        state.result_bgr = new_img
        self._sync_view_after_edit(state)
        self.statusBar().showMessage(tr("region_fill_done"), 3000)

    def _undo(self):
        state = self._current_state()
        if state is None or not state.undo_stack:
            return
        if state.result_bgr is not None:
            state.redo_stack.append(state.result_bgr.copy())
        state.result_bgr = state.undo_stack.pop()
        self._sync_view_after_edit(state)
        self.statusBar().showMessage(tr("undone"), 2000)

    def _redo(self):
        state = self._current_state()
        if state is None or not state.redo_stack:
            return
        if state.result_bgr is not None:
            state.undo_stack.append(state.result_bgr.copy())
        state.result_bgr = state.redo_stack.pop()
        self._sync_view_after_edit(state)
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
        state.ai_result_bgr = None
        state.result_bgr = None
        state.hint_manager = HintManager()
        state.undo_stack.clear()
        state.redo_stack.clear()
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
        cv2.imwrite(out_path, state.result_bgr)
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
            cv2.imwrite(os.path.join(out_dir, name), state.result_bgr)
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
