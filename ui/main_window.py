"""Colortina — Phase 2 editor.

Layout: page list (left) | canvas (center) | controls (right), per the
architecture doc. Only one mode — Auto — with an optional edit step:

    import -> auto colorize -> [not happy? draw hints -> regenerate] -> export
"""

from __future__ import annotations

import os

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QAction, QKeySequence, QShortcut, QIcon
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QListWidget, QListWidgetItem, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QSlider, QFileDialog, QSplitter,
    QStatusBar, QColorDialog, QButtonGroup, QRadioButton, QGroupBox, QListView,
    QMessageBox, QProgressBar, QCheckBox, QComboBox, QSpinBox, QInputDialog,
    QSizePolicy, QGridLayout, QTabWidget, QDialog,
)

from config import Config
from core.hint_manager import HintManager
from core.edit_snapshot import capture_edit_state, restore_edit_state, snapshots_equal
from core.manual_style import adapt_rgb_to_style
from core.manual_edit import (
    apply_brush_edit, apply_region_edit, apply_selection_edit,
    build_region_edit_mask, build_selection_edit_mask,
    build_polygon_selection_mask, build_rect_selection_mask,
    combine_selection_masks,
)
from core.pdf_handler import extract_pages
from core.selection_snap import snap_selection_mask_to_lineart
from ui.canvas import HintCanvas
from ui.i18n import tr, set_language, get_language
from ui.worker import (BatchColorizeWorker, LocalModelRecolorWorker,
                       MangaLineExtractionWorker)


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
        # Reference layer for the dedicated colour-bias eraser. It tracks the
        # latest non-bias page state so the eraser can restore only the bias
        # brush effect along the painted stroke.
        self.bias_brush_reference_bgr: np.ndarray | None = None
        self.bias_brush_reference_filter_bgr: np.ndarray | None = None
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
        self._local_recolor_worker: LocalModelRecolorWorker | None = None
        self._local_recolor_page_path: str | None = None
        self._local_recolor_selection_mask = None
        self._line_extract_worker: MangaLineExtractionWorker | None = None
        self._line_extract_page_path: str | None = None
        self._line_extract_raw_mask: np.ndarray | None = None
        # Cached AI inference for instant closed-region expansion preview.
        # Moving the slider must never re-download/re-run MangaLineExtraction.
        self._closed_preview_page_path: str | None = None
        self._closed_preview_raw_mask: np.ndarray | None = None
        self._closed_preview_probability: np.ndarray | None = None
        self._closed_preview_device: str = ""
        self._closed_preview_base_mask: np.ndarray | None = None
        self._closed_preview_combine_mode: str = "replace"
        self._closed_expand_preview_timer = QTimer(self)
        self._closed_expand_preview_timer.setSingleShot(True)
        self._closed_expand_preview_timer.setInterval(70)
        self._closed_expand_preview_timer.timeout.connect(
            self._refresh_closed_selection_preview)
        self._active_color_dialog: QColorDialog | None = None
        self._batch_errors: list[tuple[str, str]] = []
        self._brush_color = QColor(255, 120, 120)
        # Independent from the ordinary brush and global page colour bias.
        self._bias_brush_color = QColor(120, 160, 255)
        self._last_picked_rgb_raw: tuple[int, int, int] | None = None
        self._local_brush_stroke_active = False
        self._brush_changed_during_stroke = False
        self._bias_brush_stroke_active = False
        self._bias_brush_changed_during_stroke = False
        self._bias_brush_alpha = None
        self._bias_brush_base_result = None
        self._bias_brush_base_filter = None
        self._bias_brush_candidate_result = None
        self._bias_brush_candidate_filter = None
        self._bias_brush_last_preview_ts = 0.0
        self._bias_brush_last_radius_px = 18
        self._bias_brush_preview_interval = 1.0 / 30.0
        self._bias_brush_last_radius_px = 18
        self._bias_eraser_stroke_active = False
        self._bias_eraser_changed_during_stroke = False
        self._bias_eraser_alpha = None
        self._bias_eraser_base_result = None
        self._bias_eraser_base_filter = None
        self._bias_eraser_candidate_result = None
        self._bias_eraser_candidate_filter = None
        self._bias_eraser_last_preview_ts = 0.0
        self._bias_eraser_last_radius_px = 18
        self._bias_eraser_preview_interval = 1.0 / 30.0
        self._pending_selection_mask = None
        self._pending_selection_kind = None
        
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

        self._preload_colorizer_engine()

    def _preload_colorizer_engine(self):
        """Warm up mc-v2 in the background right after the window opens.

        Loading the model used to happen inside the first 自动上色 click, so
        that click absorbed the whole cold-start cost. get_colorizer() is
        lock-guarded and cached, so a click issued while this is still
        running simply waits for the same load instead of doubling it.
        """
        import sys as _sys
        if "pytest" in _sys.modules:
            # Unit tests construct MainWindow; a live preload thread would
            # race with tests that patch pipeline.MangaColorizer.
            return

        import threading

        def _load():
            try:
                from pipeline import get_colorizer
                get_colorizer()
            except Exception:
                pass

        threading.Thread(target=_load, daemon=True,
                         name="mcv2-engine-preload").start()

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
        # 自定义颜色倾向的选项必须始终完整可见：给它硬性最小高度，
        # 空间不足时让其它组收缩，而不是把这些控件压成 0 高度。
        bias_group = getattr(self, "_custom_color_bias_group", None)
        if bias_group is not None:
            bias_group.updateGeometry()
            bias_group.setMinimumHeight(bias_group.sizeHint().height())

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
        self._selection_apply_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Return), self, activated=self._apply_pending_selection_fill)
        self._selection_apply_shortcut2 = QShortcut(QKeySequence(Qt.Key.Key_Enter), self, activated=self._apply_pending_selection_fill)
        self._selection_cancel_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self, activated=self._cancel_pending_selection_fill)

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
        prev_filter_color_enabled = bool(getattr(self, "_chk_filter_color", None) and self._chk_filter_color.isChecked())
        prev_filter_color_strength = getattr(self, "_filter_color_strength_slider", None)
        prev_filter_color_strength = prev_filter_color_strength.value() if prev_filter_color_strength is not None else 35
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
        prev_char_mem = getattr(self, "_chk_character_memory", None)
        prev_char_mem = prev_char_mem.isChecked() if prev_char_mem is not None else Config.USE_CHARACTER_MEMORY
        prev_skip_colored = getattr(self, "_chk_skip_colored", None)
        prev_skip_colored = prev_skip_colored.isChecked() if prev_skip_colored is not None else True
        prev_picker_mode = self._current_eyedropper_mode() if hasattr(self, '_current_eyedropper_mode') else "point"
        prev_brush_pct = getattr(self, "_brush_slider", None)
        prev_brush_pct = prev_brush_pct.value() if prev_brush_pct is not None else self._brush_percent_from_px(12)
        prev_bias_brush_color = QColor(getattr(self, "_bias_brush_color", QColor(120, 160, 255)))
        prev_bias_brush_pct = getattr(self, "_bias_brush_size_slider", None)
        prev_bias_brush_pct = prev_bias_brush_pct.value() if prev_bias_brush_pct is not None else self._brush_percent_from_px(18)
        prev_bias_brush_strength = getattr(self, "_bias_brush_strength_slider", None)
        prev_bias_brush_strength = prev_bias_brush_strength.value() if prev_bias_brush_strength is not None else 80
        prev_bias_brush_tone = getattr(self, "_bias_brush_tone_combo", None)
        prev_bias_brush_tone = prev_bias_brush_tone.currentData() if prev_bias_brush_tone is not None else "all"
        prev_bias_eraser_pct = getattr(self, "_bias_eraser_size_slider", None)
        prev_bias_eraser_pct = prev_bias_eraser_pct.value() if prev_bias_eraser_pct is not None else self._brush_percent_from_px(18)
        prev_bias_brush_protect_skin = bool(getattr(self, "_chk_bias_brush_protect_skin", None) and self._chk_bias_brush_protect_skin.isChecked()) if getattr(self, "_chk_bias_brush_protect_skin", None) is not None else True
        prev_bias_brush_protect_lineart = bool(getattr(self, "_chk_bias_brush_protect_lineart", None) and self._chk_bias_brush_protect_lineart.isChecked()) if getattr(self, "_chk_bias_brush_protect_lineart", None) is not None else True
        prev_bias_brush_protect_saturated = bool(getattr(self, "_chk_bias_brush_protect_saturated", None) and self._chk_bias_brush_protect_saturated.isChecked()) if getattr(self, "_chk_bias_brush_protect_saturated", None) is not None else True
        prev_picker_lightness = getattr(self, "_picker_lightness_slider", None)
        prev_picker_lightness = prev_picker_lightness.value() if prev_picker_lightness is not None else 0
        prev_gap_pct = getattr(self, "_gap_close_slider", None)
        prev_gap_pct = prev_gap_pct.value() if prev_gap_pct is not None else self._gap_percent_from_px(4)
        prev_manual_match_style = bool(getattr(self, "_chk_manual_match_style", None) and self._chk_manual_match_style.isChecked()) if getattr(self, "_chk_manual_match_style", None) is not None else False
        prev_custom_bias_enabled = bool(getattr(self, "_chk_custom_color_bias", None) and self._chk_custom_color_bias.isChecked())
        prev_custom_bias_color = QColor(getattr(self, "_custom_color_bias_color", QColor(255, 180, 180)))
        prev_custom_bias_strength = getattr(self, "_custom_color_bias_slider", None)
        prev_custom_bias_strength = prev_custom_bias_strength.value() if prev_custom_bias_strength is not None else 35
        prev_custom_bias_scope = getattr(self, "_custom_color_bias_scope", None)
        prev_custom_bias_scope = prev_custom_bias_scope.currentData() if prev_custom_bias_scope is not None else "page"
        prev_custom_bias_tone = getattr(self, "_custom_color_bias_tone_range", None)
        prev_custom_bias_tone = prev_custom_bias_tone.currentData() if prev_custom_bias_tone is not None else "all"
        prev_custom_bias_protect_skin = bool(getattr(self, "_chk_custom_bias_protect_skin", None) and self._chk_custom_bias_protect_skin.isChecked()) if getattr(self, "_chk_custom_bias_protect_skin", None) is not None else True
        prev_custom_bias_protect_lineart = bool(getattr(self, "_chk_custom_bias_protect_lineart", None) and self._chk_custom_bias_protect_lineart.isChecked()) if getattr(self, "_chk_custom_bias_protect_lineart", None) is not None else True
        prev_custom_bias_protect_saturated = bool(getattr(self, "_chk_custom_bias_protect_saturated", None) and self._chk_custom_bias_protect_saturated.isChecked()) if getattr(self, "_chk_custom_bias_protect_saturated", None) is not None else True
        prev_fill_mode = self._current_fill_mode() if hasattr(self, '_current_fill_mode') else "shift"
        prev_selection_mode = self._current_selection_mode() if hasattr(self, '_current_selection_mode') else "replace"
        prev_selection_feather = getattr(self, "_selection_feather_slider", None)
        prev_selection_feather = prev_selection_feather.value() if prev_selection_feather is not None else 2
        prev_selection_closed_expand = getattr(self, "_selection_closed_expand_slider", None)
        prev_selection_closed_expand = prev_selection_closed_expand.value() if prev_selection_closed_expand is not None else 0
        prev_selection_closed_min_area = getattr(self, "_selection_closed_min_area_slider", None)
        prev_selection_closed_min_area = prev_selection_closed_min_area.value() if prev_selection_closed_min_area is not None else 6
        prev_selection_closed_min_thickness = getattr(self, "_selection_closed_min_thickness_slider", None)
        prev_selection_closed_min_thickness = prev_selection_closed_min_thickness.value() if prev_selection_closed_min_thickness is not None else 3
        prev_selection_closed_only = bool(getattr(self, "_chk_selection_closed_only", None) and self._chk_selection_closed_only.isChecked())
        prev_selection_adjust = bool(getattr(self, "_chk_selection_adjust", None) and self._chk_selection_adjust.isChecked())
        prev_selection_adjust_radius = getattr(self, "_selection_adjust_slider", None)
        prev_selection_adjust_radius = prev_selection_adjust_radius.value() if prev_selection_adjust_radius is not None else 18
        prev_selection_adjust_mode = self._current_selection_adjust_mode() if hasattr(self, "_current_selection_adjust_mode") else "add"
        prev_selection_snap = bool(getattr(self, "_chk_selection_snap_lineart", None) and self._chk_selection_snap_lineart.isChecked()) if hasattr(self, "_chk_selection_snap_lineart") else False
        prev_selection_snap_distance = getattr(self, "_selection_snap_distance_slider", None)
        prev_selection_snap_distance = prev_selection_snap_distance.value() if prev_selection_snap_distance is not None else 8
        prev_selection_classic_hints = bool(getattr(self, "_chk_selection_classic_hints", None) and self._chk_selection_classic_hints.isChecked())
        prev_selection_focus_outside = bool(getattr(self, "_chk_selection_focus_outside", None) and self._chk_selection_focus_outside.isChecked())
        prev_selection_focus_context = getattr(self, "_selection_focus_context_spin", None)
        prev_selection_focus_context = prev_selection_focus_context.value() if prev_selection_focus_context is not None else 32
        prev_selection_focus_fade = getattr(self, "_selection_focus_fade_spin", None)
        prev_selection_focus_fade = prev_selection_focus_fade.value() if prev_selection_focus_fade is not None else 96
        prev_right_tab = getattr(self, "_right_tabs", None)
        prev_right_tab = prev_right_tab.currentIndex() if prev_right_tab is not None else 0

        # A language-triggered rebuild replaces the canvas. Do not keep an
        # invisible pending selection that could later be applied accidentally.
        self._pending_selection_mask = None
        self._pending_selection_kind = None
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
        self._on_style_combo_changed()
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
        if getattr(self, "_chk_filter_color", None) is not None:
            self._chk_filter_color.setChecked(prev_filter_color_enabled)
        if getattr(self, "_filter_color_strength_slider", None) is not None:
            self._filter_color_strength_slider.setValue(prev_filter_color_strength)
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
        self._chk_character_memory.setChecked(prev_char_mem)
        self._chk_skip_colored.setChecked(prev_skip_colored)
        self._set_eyedropper_mode(prev_picker_mode)
        self._brush_slider.setValue(prev_brush_pct)
        self._bias_brush_color = QColor(prev_bias_brush_color)
        self._bias_brush_size_slider.setValue(prev_bias_brush_pct)
        self._bias_brush_strength_slider.setValue(prev_bias_brush_strength)
        idx = self._bias_brush_tone_combo.findData(prev_bias_brush_tone)
        if idx >= 0:
            self._bias_brush_tone_combo.setCurrentIndex(idx)
        self._bias_eraser_size_slider.setValue(prev_bias_eraser_pct)
        self._chk_bias_brush_protect_skin.setChecked(prev_bias_brush_protect_skin)
        self._chk_bias_brush_protect_lineart.setChecked(prev_bias_brush_protect_lineart)
        self._chk_bias_brush_protect_saturated.setChecked(prev_bias_brush_protect_saturated)
        self._update_bias_brush_swatch()
        self._set_bias_brush_mode("paint")
        self._picker_lightness_slider.setValue(prev_picker_lightness)
        self._gap_close_slider.setValue(prev_gap_pct)
        if getattr(self, "_chk_manual_match_style", None) is not None:
            self._chk_manual_match_style.setChecked(prev_manual_match_style)
        self._chk_custom_color_bias.setChecked(prev_custom_bias_enabled)
        self._custom_color_bias_color = prev_custom_bias_color
        self._custom_color_bias_slider.setValue(prev_custom_bias_strength)
        idx = self._custom_color_bias_scope.findData(prev_custom_bias_scope)
        if idx >= 0:
            self._custom_color_bias_scope.setCurrentIndex(idx)
        idx = self._custom_color_bias_tone_range.findData(prev_custom_bias_tone)
        if idx >= 0:
            self._custom_color_bias_tone_range.setCurrentIndex(idx)
        self._chk_custom_bias_protect_skin.setChecked(prev_custom_bias_protect_skin)
        self._chk_custom_bias_protect_lineart.setChecked(prev_custom_bias_protect_lineart)
        self._chk_custom_bias_protect_saturated.setChecked(prev_custom_bias_protect_saturated)
        self._update_custom_color_bias_swatch()
        self._set_fill_mode(prev_fill_mode)
        self._set_selection_mode(prev_selection_mode)
        self._selection_feather_slider.setValue(prev_selection_feather)
        self._selection_closed_expand_slider.setValue(prev_selection_closed_expand)
        self._selection_closed_min_area_slider.setValue(prev_selection_closed_min_area)
        self._selection_closed_min_thickness_slider.setValue(prev_selection_closed_min_thickness)
        self._chk_selection_closed_only.setChecked(prev_selection_closed_only)
        self._selection_adjust_slider.setValue(prev_selection_adjust_radius)
        self._set_selection_adjust_mode(prev_selection_adjust_mode)
        self._chk_selection_adjust.setChecked(prev_selection_adjust)
        self._selection_snap_distance_slider.setValue(prev_selection_snap_distance)
        self._chk_selection_snap_lineart.setChecked(prev_selection_snap)
        if getattr(self, '_chk_selection_classic_hints', None) is not None:
            self._chk_selection_classic_hints.setChecked(prev_selection_classic_hints)
        if getattr(self, '_chk_selection_focus_outside', None) is not None:
            self._chk_selection_focus_outside.setChecked(prev_selection_focus_outside)
        if getattr(self, '_selection_focus_context_spin', None) is not None:
            self._selection_focus_context_spin.setValue(prev_selection_focus_context)
        if getattr(self, '_selection_focus_fade_spin', None) is not None:
            self._selection_focus_fade_spin.setValue(max(prev_selection_focus_context, prev_selection_focus_fade))
        self._update_selection_focus_controls_enabled()
        self._update_custom_bias_controls_enabled()
        self._update_filter_color_controls_enabled()
        self._right_tabs.setCurrentIndex(max(0, min(prev_right_tab, self._right_tabs.count() - 1)))
        self._canvas.set_brush_color(self._brush_color)
        self._canvas.set_bias_brush_color(self._bias_brush_color)
        self._canvas.set_bias_brush_radius(
            self._brush_px_from_percent(self._bias_brush_size_slider.value()))
        self._update_color_swatch()
        self._update_bias_brush_swatch()
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

        self._version_label = QLabel(Config.APP_VERSION_LABEL)
        self._version_label.setObjectName("VersionLabel")
        self._version_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._version_label.setToolTip(f"Colortina {Config.APP_VERSION_LABEL} ({Config.APP_VERSION})")
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
        self._canvas.bias_brush_dab_added.connect(self._on_bias_brush_dab_added)
        self._canvas.bias_brush_stroke_started.connect(self._on_bias_brush_stroke_started)
        self._canvas.bias_brush_stroke_finished.connect(self._on_bias_brush_stroke_finished)
        self._canvas.bias_eraser_dab_added.connect(self._on_bias_eraser_dab_added)
        self._canvas.bias_eraser_stroke_started.connect(self._on_bias_eraser_stroke_started)
        self._canvas.bias_eraser_stroke_finished.connect(self._on_bias_eraser_stroke_finished)
        self._canvas.color_picked.connect(self._on_color_picked)
        self._canvas.region_fill_requested.connect(self._on_region_fill_requested)
        self._canvas.polygon_fill_requested.connect(self._on_polygon_fill_requested)
        self._canvas.rect_fill_requested.connect(self._on_rect_fill_requested)
        self._canvas.selection_preview_active.connect(self._on_selection_preview_active)
        self._canvas.selection_adjust_dab.connect(self._on_selection_adjust_dab)

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
        # 质量已固定为最快模式：控件保留供设置读写，但不再显示。
        self._quality_combo = QComboBox()
        self._quality_combo.addItem("Fast", "draft")
        self._quality_combo.hide()
        style_grid.setColumnStretch(1, 1)
        style_layout.addLayout(style_grid)

        strength_grid = QGridLayout()
        strength_grid.setContentsMargins(0, 0, 0, 0)
        strength_grid.setHorizontalSpacing(5)
        strength_grid.setVerticalSpacing(0)

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
        self._chk_filter_color = QCheckBox(tr("filter_color_enable"))
        self._chk_filter_color.setToolTip(tr("filter_color_hint"))
        self._chk_filter_color.toggled.connect(self._update_filter_color_controls_enabled)
        filter_layout.addWidget(self._chk_filter_color)
        filter_color_grid = QGridLayout()
        filter_color_grid.setContentsMargins(0, 0, 0, 0)
        filter_color_grid.setHorizontalSpacing(5)
        filter_color_grid.setVerticalSpacing(3)
        filter_color_grid.addWidget(QLabel(tr("filter_color_strength")), 0, 0)
        self._filter_color_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self._filter_color_strength_slider.setRange(0, 150)
        self._filter_color_strength_slider.setValue(35)
        self._filter_color_strength_spin = QSpinBox()
        self._filter_color_strength_spin.setRange(0, 150)
        self._filter_color_strength_spin.setSuffix("%")
        self._filter_color_strength_spin.setValue(35)
        self._filter_color_strength_spin.setMaximumWidth(72)
        self._filter_color_strength_slider.valueChanged.connect(self._filter_color_strength_spin.setValue)
        self._filter_color_strength_spin.valueChanged.connect(self._filter_color_strength_slider.setValue)
        filter_color_grid.addWidget(self._filter_color_strength_slider, 0, 1)
        filter_color_grid.addWidget(self._filter_color_strength_spin, 0, 2)
        filter_color_grid.setColumnStretch(1, 1)
        filter_layout.addLayout(filter_color_grid)
        self._filter_color_hint = QLabel(tr("filter_color_hint"))
        self._filter_color_hint.setWordWrap(True)
        self._filter_color_hint.setStyleSheet("color: #6c7f94; font-size: 10px;")
        filter_layout.addWidget(self._filter_color_hint)
        self._update_filter_color_controls_enabled()
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

        self._custom_color_bias_group = QGroupBox(tr("custom_color_bias_group"))
        custom_bias_layout = QVBoxLayout(self._custom_color_bias_group)
        custom_bias_layout.setContentsMargins(10, 10, 10, 8)
        custom_bias_layout.setSpacing(4)
        self._chk_custom_color_bias = QCheckBox(tr("custom_color_bias_enable"))
        self._chk_custom_color_bias.toggled.connect(self._update_custom_bias_controls_enabled)
        custom_bias_layout.addWidget(self._chk_custom_color_bias)
        # 详细选项集中放进一个容器：未启用时整体隐藏，右侧面板不再为
        # 灰色控件付出高度；启用时显示且不允许被压缩到显示不全。
        self._custom_color_bias_details = QWidget()
        custom_bias_details_layout = QVBoxLayout(self._custom_color_bias_details)
        custom_bias_details_layout.setContentsMargins(0, 0, 0, 0)
        custom_bias_details_layout.setSpacing(4)
        custom_bias_grid = QGridLayout()
        custom_bias_grid.setContentsMargins(0, 0, 0, 0)
        custom_bias_grid.setHorizontalSpacing(5)
        custom_bias_grid.setVerticalSpacing(3)
        custom_bias_grid.addWidget(QLabel(tr("custom_color_bias_color")), 0, 0)
        self._custom_color_bias_btn = QPushButton()
        self._custom_color_bias_btn.setFixedSize(54, 30)
        self._custom_color_bias_btn.clicked.connect(self._pick_custom_bias_color)
        custom_bias_grid.addWidget(self._custom_color_bias_btn, 0, 1)
        self._custom_color_bias_color = QColor(255, 180, 180)
        self._custom_color_bias_info = QLabel()
        self._custom_color_bias_info.setStyleSheet("color: #5d6f84; font-size: 10px;")
        custom_bias_grid.addWidget(self._custom_color_bias_info, 0, 2)
        custom_bias_grid.addWidget(QLabel(tr("custom_color_bias_scope")), 1, 0)
        self._custom_color_bias_scope = QComboBox()
        self._custom_color_bias_scope.addItem(tr("custom_color_bias_scope_page"), "page")
        self._custom_color_bias_scope.addItem(tr("custom_color_bias_scope_characters"), "characters")
        self._custom_color_bias_scope.addItem(tr("custom_color_bias_scope_background"), "background")
        custom_bias_grid.addWidget(self._custom_color_bias_scope, 1, 1, 1, 2)
        custom_bias_grid.addWidget(QLabel(tr("custom_color_bias_tone_range")), 2, 0)
        self._custom_color_bias_tone_range = QComboBox()
        self._custom_color_bias_tone_range.addItem(tr("custom_color_bias_tone_all"), "all")
        self._custom_color_bias_tone_range.addItem(tr("custom_color_bias_tone_highlights"), "highlights")
        self._custom_color_bias_tone_range.addItem(tr("custom_color_bias_tone_midtones"), "midtones")
        self._custom_color_bias_tone_range.addItem(tr("custom_color_bias_tone_shadows"), "shadows")
        custom_bias_grid.addWidget(self._custom_color_bias_tone_range, 2, 1, 1, 2)
        custom_bias_grid.addWidget(QLabel(tr("custom_color_bias_strength")), 3, 0)
        self._custom_color_bias_slider = QSlider(Qt.Orientation.Horizontal)
        self._custom_color_bias_slider.setRange(0, 200)
        self._custom_color_bias_slider.setValue(35)
        self._custom_color_bias_spin = QSpinBox()
        self._custom_color_bias_spin.setRange(0, 200)
        self._custom_color_bias_spin.setSuffix("%")
        self._custom_color_bias_spin.setValue(35)
        self._custom_color_bias_spin.setMaximumWidth(72)
        self._custom_color_bias_slider.valueChanged.connect(self._custom_color_bias_spin.setValue)
        self._custom_color_bias_spin.valueChanged.connect(self._custom_color_bias_slider.setValue)
        custom_bias_grid.addWidget(self._custom_color_bias_slider, 3, 1)
        custom_bias_grid.addWidget(self._custom_color_bias_spin, 3, 2)
        custom_bias_grid.setColumnStretch(1, 1)
        custom_bias_details_layout.addLayout(custom_bias_grid)
        protection_label = QLabel(tr("custom_color_bias_protection"))
        protection_label.setStyleSheet("font-weight: 600;")
        custom_bias_details_layout.addWidget(protection_label)
        protection_grid = QGridLayout()
        protection_grid.setContentsMargins(0, 0, 0, 0)
        protection_grid.setHorizontalSpacing(8)
        protection_grid.setVerticalSpacing(2)
        self._chk_custom_bias_protect_skin = QCheckBox(tr("custom_color_bias_protect_skin"))
        self._chk_custom_bias_protect_lineart = QCheckBox(tr("custom_color_bias_protect_lineart"))
        self._chk_custom_bias_protect_saturated = QCheckBox(tr("custom_color_bias_protect_saturated"))
        protections = (self._chk_custom_bias_protect_skin,
                       self._chk_custom_bias_protect_lineart,
                       self._chk_custom_bias_protect_saturated)
        for index, checkbox in enumerate(protections):
            checkbox.setChecked(True)
            checkbox.setToolTip(tr("custom_color_bias_protection_hint"))
            protection_grid.addWidget(checkbox, index // 2, index % 2)
        protection_grid.setColumnStretch(1, 1)
        custom_bias_details_layout.addLayout(protection_grid)
        custom_bias_layout.addWidget(self._custom_color_bias_details)
        style_layout.addWidget(self._custom_color_bias_group)
        self._update_custom_color_bias_swatch()
        self._update_custom_bias_controls_enabled()

        auto_group, auto_layout = make_group(tr("auto_group"))
        self._chk_protect_text = QCheckBox("保护文字气泡（AI 检测文字区域，保持原稿黑白）")
        self._chk_protect_text.setChecked(True)
        self._chk_protect_text.setToolTip(
            "用本地文字检测模型找到对话框文字像素，上色后把这些区域还原为原稿的"
            "纸白与墨黑，避免文字被染色。模型文件缺失时自动跳过，不影响上色。")
        auto_layout.addWidget(self._chk_protect_text)
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
        # Restore the single-page editor used before the nested detail tabs.
        # The main editing group owns the available vertical space and extends
        # toward the middle/bottom; history remains a compact fixed-height row.
        edit_tab, edit_tab_layout = make_tab()
        edit_tab_layout.setSpacing(5)
        edit_group, edit_layout = make_group(tr("edit_group"))
        edit_group.setSizePolicy(QSizePolicy.Policy.Expanding,
                                 QSizePolicy.Policy.Expanding)
        edit_layout.setSpacing(5)

        tool_grid = QGridLayout()
        tool_grid.setContentsMargins(0, 0, 0, 0)
        tool_grid.setHorizontalSpacing(9)
        tool_grid.setVerticalSpacing(4)
        self._tool_group = QButtonGroup(self)
        self._radio_brush = QRadioButton(tr("tool_brush"))
        self._radio_brush.setChecked(True)
        self._radio_eyedropper = QRadioButton(tr("tool_eyedropper"))
        self._radio_bias_brush = QRadioButton(tr("tool_bias_brush"))
        self._radio_bucket = QRadioButton(tr("tool_bucket"))
        self._radio_lasso_bucket = QRadioButton(tr("tool_lasso_bucket"))
        self._radio_rect_bucket = QRadioButton(tr("tool_rect_bucket"))
        tool_buttons = (
            self._radio_brush, self._radio_eyedropper, self._radio_bias_brush,
            self._radio_bucket, self._radio_lasso_bucket, self._radio_rect_bucket,
        )
        for i, radio in enumerate(tool_buttons):
            self._tool_group.addButton(radio)
            tool_grid.addWidget(radio, i // 3, i % 3)
        for col in range(3):
            tool_grid.setColumnStretch(col, 1)
        self._radio_brush.toggled.connect(self._on_tool_changed)
        self._radio_bias_brush.toggled.connect(self._on_tool_changed)
        self._radio_eyedropper.toggled.connect(self._on_tool_changed)
        self._radio_bucket.toggled.connect(self._on_tool_changed)
        self._radio_lasso_bucket.toggled.connect(self._on_tool_changed)
        self._radio_rect_bucket.toggled.connect(self._on_tool_changed)
        edit_layout.addLayout(tool_grid)

        self._tool_visibility_rows = {}

        def register_tool_row(name: str, *widgets: QWidget) -> None:
            self._tool_visibility_rows[name] = tuple(
                widget for widget in widgets if widget is not None)

        self._tool_param_group, tool_param_layout = make_group(
            tr("tool_parameters_group"))
        self._tool_param_group.setSizePolicy(QSizePolicy.Policy.Expanding,
                                             QSizePolicy.Policy.Fixed)
        edit_layout.addWidget(self._tool_param_group)

        edit_grid = QGridLayout()
        edit_grid.setContentsMargins(0, 0, 0, 0)
        edit_grid.setHorizontalSpacing(7)
        edit_grid.setVerticalSpacing(5)

        self._eyedropper_mode_label = QLabel(tr("eyedropper_mode_label"))
        edit_grid.addWidget(self._eyedropper_mode_label, 0, 0)
        eyedrop_box = QWidget()
        self._eyedropper_mode_box = eyedrop_box
        eyedrop_layout = QHBoxLayout(eyedrop_box)
        eyedrop_layout.setContentsMargins(0, 0, 0, 0)
        eyedrop_layout.setSpacing(9)
        self._eyedropper_mode_group = QButtonGroup(self)
        self._eyedropper_mode_group.setExclusive(True)
        self._eyedropper_mode_point = QCheckBox(tr("eyedropper_mode_point"))
        self._eyedropper_mode_region = QCheckBox(tr("eyedropper_mode_region"))
        self._eyedropper_mode_group.addButton(self._eyedropper_mode_point)
        self._eyedropper_mode_group.addButton(self._eyedropper_mode_region)
        self._eyedropper_mode_point.toggled.connect(
            lambda checked: checked and self._set_eyedropper_mode("point"))
        self._eyedropper_mode_region.toggled.connect(
            lambda checked: checked and self._set_eyedropper_mode("region"))
        eyedrop_layout.addWidget(self._eyedropper_mode_point)
        eyedrop_layout.addWidget(self._eyedropper_mode_region)
        eyedrop_layout.addStretch(1)
        edit_grid.addWidget(eyedrop_box, 0, 1, 1, 2)
        register_tool_row("eyedropper_mode", self._eyedropper_mode_label, eyedrop_box)

        self._color_label = QLabel(tr("color_label"))
        edit_grid.addWidget(self._color_label, 1, 0)
        self._color_swatch = QPushButton()
        self._color_swatch.setFixedSize(54, 30)
        self._color_swatch.clicked.connect(self._pick_color)
        edit_grid.addWidget(self._color_swatch, 1, 1)
        self._current_color_info = QLabel()
        self._current_color_info.setWordWrap(True)
        self._current_color_info.setStyleSheet(
            "color: #5d6f84; font-size: 10px;")
        edit_grid.addWidget(self._current_color_info, 1, 2)
        register_tool_row("color", self._color_label, self._color_swatch, self._current_color_info)

        self._brush_size_label = QLabel(tr("brush_size_label"))
        edit_grid.addWidget(self._brush_size_label, 2, 0)
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
            lambda v: self._canvas.set_brush_radius(
                self._brush_px_from_percent(v)))
        edit_grid.addWidget(self._brush_slider, 2, 1)
        edit_grid.addWidget(self._brush_spin, 2, 2)
        register_tool_row("brush_size", self._brush_size_label, self._brush_slider, self._brush_spin)

        self._picker_lightness_label = QLabel(tr("picker_lightness_label"))
        edit_grid.addWidget(self._picker_lightness_label, 5, 0)
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
        self._picker_lightness_slider.valueChanged.connect(
            self._on_picker_lightness_changed)
        self._picker_lightness_spin.valueChanged.connect(
            self._picker_lightness_slider.setValue)
        edit_grid.addWidget(self._picker_lightness_slider, 5, 1)
        edit_grid.addWidget(self._picker_lightness_spin, 5, 2)
        register_tool_row("picker_lightness", self._picker_lightness_label, self._picker_lightness_slider, self._picker_lightness_spin)

        self._gap_close_label = QLabel(tr("gap_close_label"))
        edit_grid.addWidget(self._gap_close_label, 6, 0)
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
        self._gap_close_slider.valueChanged.connect(
            self._gap_close_spin.setValue)
        self._gap_close_spin.valueChanged.connect(
            self._gap_close_slider.setValue)
        edit_grid.addWidget(self._gap_close_slider, 6, 1)
        edit_grid.addWidget(self._gap_close_spin, 6, 2)
        register_tool_row("gap_close", self._gap_close_label, self._gap_close_slider, self._gap_close_spin)

        self._fill_mode_label = QLabel(tr("fill_mode_label"))
        edit_grid.addWidget(self._fill_mode_label, 7, 0)
        self._fill_mode_combo = QComboBox()
        self._fill_mode_combo.addItem(tr("fill_mode_shift"), "shift")
        self._fill_mode_combo.addItem(tr("fill_mode_shading"), "shading")
        self._fill_mode_combo.addItem(tr("fill_mode_flat"), "flat")
        self._fill_mode_combo.setToolTip(tr("fill_mode_hint"))
        edit_grid.addWidget(self._fill_mode_combo, 7, 1, 1, 2)
        register_tool_row("fill_mode", self._fill_mode_label, self._fill_mode_combo)

        self._selection_mode_label = QLabel(tr("selection_mode"))
        edit_grid.addWidget(self._selection_mode_label, 8, 0)
        selection_mode_box = QWidget()
        selection_mode_layout = QHBoxLayout(selection_mode_box)
        selection_mode_layout.setContentsMargins(0, 0, 0, 0)
        selection_mode_layout.setSpacing(8)
        self._selection_mode_group = QButtonGroup(self)
        self._selection_mode_group.setExclusive(True)
        self._selection_mode_radios = {}
        for mode, key in (("replace", "selection_mode_replace"),
                          ("add", "selection_mode_add"),
                          ("subtract", "selection_mode_subtract")):
            radio = QRadioButton(tr(key))
            radio.setToolTip(tr("selection_mode_hint"))
            radio.toggled.connect(
                lambda checked, m=mode: checked and self._set_selection_mode(m))
            self._selection_mode_group.addButton(radio)
            self._selection_mode_radios[mode] = radio
            selection_mode_layout.addWidget(radio)
        selection_mode_layout.addStretch(1)
        edit_grid.addWidget(selection_mode_box, 8, 1, 1, 2)
        self._selection_mode_box = selection_mode_box
        register_tool_row("selection_mode", self._selection_mode_label, selection_mode_box)

        self._set_selection_mode("replace")

        self._selection_feather_label = QLabel(tr("selection_feather"))
        edit_grid.addWidget(self._selection_feather_label, 9, 0)
        self._selection_feather_slider = QSlider(Qt.Orientation.Horizontal)
        self._selection_feather_slider.setRange(0, 30)
        self._selection_feather_slider.setValue(2)
        self._selection_feather_slider.setToolTip(
            tr("selection_feather_hint"))
        self._selection_feather_spin = QSpinBox()
        self._selection_feather_spin.setRange(0, 30)
        self._selection_feather_spin.setSuffix(" px")
        self._selection_feather_spin.setValue(2)
        self._selection_feather_spin.setMaximumWidth(72)
        self._selection_feather_spin.setToolTip(tr("selection_feather_hint"))
        self._selection_feather_slider.valueChanged.connect(
            self._selection_feather_spin.setValue)
        self._selection_feather_spin.valueChanged.connect(
            self._selection_feather_slider.setValue)
        edit_grid.addWidget(self._selection_feather_slider, 9, 1)
        edit_grid.addWidget(self._selection_feather_spin, 9, 2)
        register_tool_row("selection_feather", self._selection_feather_label, self._selection_feather_slider, self._selection_feather_spin)

        self._selection_closed_expand_label = QLabel(tr("selection_closed_expand"))
        edit_grid.addWidget(self._selection_closed_expand_label, 10, 0)
        self._selection_closed_expand_slider = QSlider(Qt.Orientation.Horizontal)
        self._selection_closed_expand_slider.setRange(0, 20)
        self._selection_closed_expand_slider.setValue(0)
        self._selection_closed_expand_slider.setToolTip(
            tr("selection_closed_expand_hint"))
        self._selection_closed_expand_spin = QSpinBox()
        self._selection_closed_expand_spin.setRange(0, 20)
        self._selection_closed_expand_spin.setSuffix(" px")
        self._selection_closed_expand_spin.setValue(0)
        self._selection_closed_expand_spin.setMaximumWidth(72)
        self._selection_closed_expand_spin.setToolTip(
            tr("selection_closed_expand_hint"))
        self._selection_closed_expand_slider.valueChanged.connect(
            self._selection_closed_expand_spin.setValue)
        self._selection_closed_expand_spin.valueChanged.connect(
            self._selection_closed_expand_slider.setValue)
        self._selection_closed_expand_slider.valueChanged.connect(
            self._schedule_closed_preview_refresh)
        edit_grid.addWidget(self._selection_closed_expand_slider, 10, 1)
        edit_grid.addWidget(self._selection_closed_expand_spin, 10, 2)
        register_tool_row("selection_closed_expand", self._selection_closed_expand_label, self._selection_closed_expand_slider, self._selection_closed_expand_spin)

        self._selection_closed_min_area_label = QLabel(tr("selection_closed_min_area"))
        edit_grid.addWidget(self._selection_closed_min_area_label, 11, 0)
        self._selection_closed_min_area_slider = QSlider(Qt.Orientation.Horizontal)
        self._selection_closed_min_area_slider.setRange(0, 10000)
        self._selection_closed_min_area_slider.setValue(6)
        self._selection_closed_min_area_slider.setToolTip(
            tr("selection_closed_min_area_hint"))
        self._selection_closed_min_area_spin = QSpinBox()
        self._selection_closed_min_area_spin.setRange(0, 10000)
        self._selection_closed_min_area_spin.setSuffix(" px²")
        self._selection_closed_min_area_spin.setValue(6)
        self._selection_closed_min_area_spin.setMaximumWidth(96)
        self._selection_closed_min_area_spin.setToolTip(
            tr("selection_closed_min_area_hint"))
        self._selection_closed_min_area_slider.valueChanged.connect(
            self._selection_closed_min_area_spin.setValue)
        self._selection_closed_min_area_spin.valueChanged.connect(
            self._selection_closed_min_area_slider.setValue)
        self._selection_closed_min_area_slider.valueChanged.connect(
            self._schedule_closed_preview_refresh)
        edit_grid.addWidget(self._selection_closed_min_area_slider, 11, 1)
        edit_grid.addWidget(self._selection_closed_min_area_spin, 11, 2)
        register_tool_row("selection_closed_min_area", self._selection_closed_min_area_label, self._selection_closed_min_area_slider, self._selection_closed_min_area_spin)

        self._selection_closed_min_thickness_label = QLabel(tr("selection_closed_min_thickness"))
        edit_grid.addWidget(self._selection_closed_min_thickness_label, 12, 0)
        self._selection_closed_min_thickness_slider = QSlider(Qt.Orientation.Horizontal)
        self._selection_closed_min_thickness_slider.setRange(0, 100)
        self._selection_closed_min_thickness_slider.setValue(3)
        self._selection_closed_min_thickness_slider.setToolTip(
            tr("selection_closed_min_thickness_hint"))
        self._selection_closed_min_thickness_spin = QSpinBox()
        self._selection_closed_min_thickness_spin.setRange(0, 100)
        self._selection_closed_min_thickness_spin.setSuffix(" px")
        self._selection_closed_min_thickness_spin.setValue(3)
        self._selection_closed_min_thickness_spin.setMaximumWidth(80)
        self._selection_closed_min_thickness_spin.setToolTip(
            tr("selection_closed_min_thickness_hint"))
        self._selection_closed_min_thickness_slider.valueChanged.connect(
            self._selection_closed_min_thickness_spin.setValue)
        self._selection_closed_min_thickness_spin.valueChanged.connect(
            self._selection_closed_min_thickness_slider.setValue)
        self._selection_closed_min_thickness_slider.valueChanged.connect(
            self._schedule_closed_preview_refresh)
        edit_grid.addWidget(self._selection_closed_min_thickness_slider, 12, 1)
        edit_grid.addWidget(self._selection_closed_min_thickness_spin, 12, 2)
        register_tool_row("selection_closed_min_thickness", self._selection_closed_min_thickness_label, self._selection_closed_min_thickness_slider, self._selection_closed_min_thickness_spin)
        edit_grid.setColumnStretch(1, 1)
        tool_param_layout.addLayout(edit_grid)

        # Independent custom-colour-bias brush controls. These widgets own
        # their state and never mutate the normal brush or global page-bias UI.
        self._bias_brush_box = QWidget()
        bias_brush_layout = QGridLayout(self._bias_brush_box)
        bias_brush_layout.setContentsMargins(0, 0, 0, 0)
        bias_brush_layout.setHorizontalSpacing(7)
        bias_brush_layout.setVerticalSpacing(5)

        self._bias_brush_mode_label = QLabel(tr("bias_brush_mode"))
        self._bias_brush_mode_box = QWidget()
        bias_brush_mode_layout = QHBoxLayout(self._bias_brush_mode_box)
        bias_brush_mode_layout.setContentsMargins(0, 0, 0, 0)
        bias_brush_mode_layout.setSpacing(6)
        self._bias_brush_mode_group = QButtonGroup(self)
        self._bias_brush_mode_group.setExclusive(True)
        self._btn_bias_brush_paint = QPushButton(tr("bias_brush_mode_paint"))
        self._btn_bias_brush_paint.setCheckable(True)
        self._btn_bias_brush_paint.setChecked(True)
        self._btn_bias_brush_paint.setToolTip(tr("tool_hint_bias_brush"))
        self._btn_bias_brush_erase = QPushButton(tr("bias_brush_mode_erase"))
        self._btn_bias_brush_erase.setCheckable(True)
        self._btn_bias_brush_erase.setToolTip(tr("tool_hint_bias_eraser"))
        self._bias_brush_mode_group.addButton(self._btn_bias_brush_paint)
        self._bias_brush_mode_group.addButton(self._btn_bias_brush_erase)
        self._btn_bias_brush_paint.toggled.connect(
            lambda checked: self._set_bias_brush_mode("paint") if checked else None)
        self._btn_bias_brush_erase.toggled.connect(
            lambda checked: self._set_bias_brush_mode("erase") if checked else None)
        bias_brush_mode_layout.addWidget(self._btn_bias_brush_paint)
        bias_brush_mode_layout.addWidget(self._btn_bias_brush_erase)
        bias_brush_mode_layout.addStretch(1)
        bias_brush_layout.addWidget(self._bias_brush_mode_label, 0, 0)
        bias_brush_layout.addWidget(self._bias_brush_mode_box, 0, 1, 1, 2)

        self._bias_brush_color_label = QLabel(tr("bias_brush_color"))
        self._bias_brush_color_btn = QPushButton()
        self._bias_brush_color_btn.setFixedSize(54, 30)
        self._bias_brush_color_btn.clicked.connect(self._pick_bias_brush_color)
        self._bias_brush_color_info = QLabel()
        self._bias_brush_color_info.setStyleSheet("color: #5d6f84; font-size: 10px;")
        bias_brush_layout.addWidget(self._bias_brush_color_label, 1, 0)
        bias_brush_layout.addWidget(self._bias_brush_color_btn, 1, 1)
        bias_brush_layout.addWidget(self._bias_brush_color_info, 1, 2)

        self._bias_brush_size_label = QLabel(tr("bias_brush_size"))
        self._bias_brush_size_slider = QSlider(Qt.Orientation.Horizontal)
        self._bias_brush_size_slider.setRange(0, 100)
        self._bias_brush_size_slider.setValue(self._brush_percent_from_px(18))
        self._bias_brush_size_spin = QSpinBox()
        self._bias_brush_size_spin.setRange(0, 100)
        self._bias_brush_size_spin.setValue(self._bias_brush_size_slider.value())
        self._bias_brush_size_spin.setMaximumWidth(72)
        self._bias_brush_size_slider.valueChanged.connect(self._bias_brush_size_spin.setValue)
        self._bias_brush_size_spin.valueChanged.connect(self._bias_brush_size_slider.setValue)
        self._bias_brush_size_slider.valueChanged.connect(
            lambda v: self._canvas.set_bias_brush_radius(self._brush_px_from_percent(v)))
        bias_brush_layout.addWidget(self._bias_brush_size_label, 2, 0)
        bias_brush_layout.addWidget(self._bias_brush_size_slider, 2, 1)
        bias_brush_layout.addWidget(self._bias_brush_size_spin, 2, 2)

        self._bias_brush_strength_label = QLabel(tr("bias_brush_strength"))
        self._bias_brush_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self._bias_brush_strength_slider.setRange(0, 200)
        self._bias_brush_strength_slider.setValue(80)
        self._bias_brush_strength_spin = QSpinBox()
        self._bias_brush_strength_spin.setRange(0, 200)
        self._bias_brush_strength_spin.setSuffix("%")
        self._bias_brush_strength_spin.setValue(80)
        self._bias_brush_strength_spin.setMaximumWidth(76)
        self._bias_brush_strength_slider.valueChanged.connect(self._bias_brush_strength_spin.setValue)
        self._bias_brush_strength_spin.valueChanged.connect(self._bias_brush_strength_slider.setValue)
        bias_brush_layout.addWidget(self._bias_brush_strength_label, 3, 0)
        bias_brush_layout.addWidget(self._bias_brush_strength_slider, 3, 1)
        bias_brush_layout.addWidget(self._bias_brush_strength_spin, 3, 2)

        self._bias_brush_tone_label = QLabel(tr("bias_brush_tone"))
        self._bias_brush_tone_combo = QComboBox()
        self._bias_brush_tone_combo.addItem(tr("custom_color_bias_tone_all"), "all")
        self._bias_brush_tone_combo.addItem(tr("custom_color_bias_tone_highlights"), "highlights")
        self._bias_brush_tone_combo.addItem(tr("custom_color_bias_tone_midtones"), "midtones")
        self._bias_brush_tone_combo.addItem(tr("custom_color_bias_tone_shadows"), "shadows")
        bias_brush_layout.addWidget(self._bias_brush_tone_label, 4, 0)
        bias_brush_layout.addWidget(self._bias_brush_tone_combo, 4, 1, 1, 2)

        self._bias_brush_protection_box = QWidget()
        bias_protect_layout = QHBoxLayout(self._bias_brush_protection_box)
        bias_protect_layout.setContentsMargins(0, 0, 0, 0)
        bias_protect_layout.setSpacing(8)
        self._chk_bias_brush_protect_skin = QCheckBox(tr("custom_color_bias_protect_skin"))
        self._chk_bias_brush_protect_lineart = QCheckBox(tr("custom_color_bias_protect_lineart"))
        self._chk_bias_brush_protect_saturated = QCheckBox(tr("custom_color_bias_protect_saturated"))
        self._chk_bias_brush_protect_skin.setChecked(True)
        self._chk_bias_brush_protect_lineart.setChecked(True)
        self._chk_bias_brush_protect_saturated.setChecked(True)
        bias_protect_layout.addWidget(self._chk_bias_brush_protect_skin)
        bias_protect_layout.addWidget(self._chk_bias_brush_protect_lineart)
        bias_protect_layout.addWidget(self._chk_bias_brush_protect_saturated)
        bias_protect_layout.addStretch(1)
        bias_brush_layout.addWidget(QLabel(tr("custom_color_bias_protection")), 5, 0)
        bias_brush_layout.addWidget(self._bias_brush_protection_box, 5, 1, 1, 2)
        bias_brush_layout.setColumnStretch(1, 1)
        tool_param_layout.addWidget(self._bias_brush_box)
        register_tool_row("bias_brush_controls", self._bias_brush_box)
        self._update_bias_brush_swatch()

        self._bias_eraser_size_label = QLabel(tr("bias_eraser_size"))
        self._bias_eraser_size_slider = QSlider(Qt.Orientation.Horizontal)
        self._bias_eraser_size_slider.setRange(0, 100)
        self._bias_eraser_size_slider.setValue(self._brush_percent_from_px(18))
        self._bias_eraser_size_spin = QSpinBox()
        self._bias_eraser_size_spin.setRange(0, 100)
        self._bias_eraser_size_spin.setValue(self._bias_eraser_size_slider.value())
        self._bias_eraser_size_spin.setMaximumWidth(72)
        self._bias_eraser_size_slider.valueChanged.connect(self._bias_eraser_size_spin.setValue)
        self._bias_eraser_size_spin.valueChanged.connect(self._bias_eraser_size_slider.setValue)
        self._bias_eraser_size_slider.valueChanged.connect(
            lambda v: self._canvas.set_bias_eraser_radius(self._brush_px_from_percent(v)))
        self._bias_eraser_note = QLabel(tr("bias_eraser_note"))
        self._bias_eraser_note.setWordWrap(True)
        self._bias_eraser_note.setStyleSheet("color: #5d6f84; font-size: 10px;")
        self._bias_eraser_controls_box = QWidget()
        bias_eraser_layout = QGridLayout(self._bias_eraser_controls_box)
        bias_eraser_layout.setContentsMargins(0, 0, 0, 0)
        bias_eraser_layout.setHorizontalSpacing(7)
        bias_eraser_layout.setVerticalSpacing(5)
        bias_eraser_layout.addWidget(self._bias_eraser_size_label, 0, 0)
        bias_eraser_layout.addWidget(self._bias_eraser_size_slider, 0, 1)
        bias_eraser_layout.addWidget(self._bias_eraser_size_spin, 0, 2)
        bias_eraser_layout.addWidget(self._bias_eraser_note, 1, 0, 1, 3)
        bias_eraser_layout.setColumnStretch(1, 1)
        bias_brush_layout.addWidget(self._bias_eraser_controls_box, 6, 0, 1, 3)

        self._chk_picker_extract_hint = QCheckBox(tr("picker_extract_hint_checkbox"))
        self._chk_picker_extract_hint.setChecked(False)
        self._chk_picker_extract_hint.setToolTip(tr("picker_extract_hint_hint"))
        self._chk_picker_extract_hint.setMinimumHeight(25)
        tool_param_layout.addWidget(self._chk_picker_extract_hint)
        register_tool_row("picker_extract_hint", self._chk_picker_extract_hint)

        self._chk_brush_model_hint = QCheckBox(tr("brush_model_hint_checkbox"))
        self._chk_brush_model_hint.setChecked(False)
        self._chk_brush_model_hint.setToolTip(tr("brush_model_hint_hint"))
        tool_param_layout.addWidget(self._chk_brush_model_hint)
        register_tool_row("brush_model_hint", self._chk_brush_model_hint)

        self._chk_manual_match_style = QCheckBox(
            tr("manual_match_style_checkbox"))
        self._chk_manual_match_style.setChecked(False)
        self._chk_manual_match_style.setToolTip(
            tr("manual_match_style_hint"))
        self._chk_manual_match_style.toggled.connect(
            self._update_color_swatch)
        tool_param_layout.addWidget(self._chk_manual_match_style)
        register_tool_row("manual_match_style", self._chk_manual_match_style)

        self._chk_selection_closed_only = QCheckBox(
            tr("selection_closed_only"))
        self._chk_selection_closed_only.setToolTip(
            tr("selection_closed_only_hint"))
        self._chk_selection_closed_only.toggled.connect(
            self._update_closed_expand_controls_enabled)
        tool_param_layout.addWidget(self._chk_selection_closed_only)
        register_tool_row("selection_closed_only", self._chk_selection_closed_only)
        self._update_closed_expand_controls_enabled(False)

        self._selection_action_box = QWidget()
        selection_action_row = QHBoxLayout(self._selection_action_box)
        selection_action_row.setContentsMargins(0, 0, 0, 0)
        selection_action_row.setSpacing(6)
        self._btn_apply_selection = QPushButton(tr("selection_apply"))
        self._btn_apply_selection.setToolTip(tr("selection_apply_hint"))
        self._btn_apply_selection.clicked.connect(
            self._apply_pending_selection_fill)
        self._btn_ai_recolor_selection = QPushButton(tr("selection_ai_recolor"))
        self._btn_ai_recolor_selection.setToolTip(tr("selection_ai_recolor_hint"))
        self._btn_ai_recolor_selection.clicked.connect(
            self._start_selection_model_recolor)
        self._btn_cancel_selection = QPushButton(tr("selection_cancel"))
        self._btn_cancel_selection.clicked.connect(
            self._cancel_pending_selection_fill)
        selection_action_row.addWidget(self._btn_apply_selection)
        selection_action_row.addWidget(self._btn_ai_recolor_selection)
        selection_action_row.addWidget(self._btn_cancel_selection)
        tool_param_layout.addWidget(self._selection_action_box)
        register_tool_row("selection_actions", self._selection_action_box)

        self._selection_ai_options_box = QWidget()
        selection_ai_options = QGridLayout(self._selection_ai_options_box)
        selection_ai_options.setContentsMargins(0, 0, 0, 0)
        selection_ai_options.setHorizontalSpacing(7)
        selection_ai_options.setVerticalSpacing(4)
        self._chk_selection_bw_preview = QCheckBox(tr("selection_bw_preview"))
        self._chk_selection_bw_preview.setChecked(False)
        self._chk_selection_bw_preview.setToolTip(tr("selection_bw_preview_hint"))
        self._chk_selection_bw_preview.toggled.connect(
            self._refresh_selection_display_preview)
        self._chk_selection_local_hints_only = QCheckBox(
            tr("selection_local_hints_only"))
        self._chk_selection_local_hints_only.setChecked(False)
        self._chk_selection_local_hints_only.setToolTip(
            tr("selection_local_hints_only_hint"))
        self._selection_hint_margin_label = QLabel(tr("selection_hint_margin"))
        self._selection_hint_margin_spin = QSpinBox()
        self._selection_hint_margin_spin.setRange(0, 96)
        self._selection_hint_margin_spin.setValue(16)
        self._selection_hint_margin_spin.setSuffix(" px")
        self._selection_hint_margin_spin.setToolTip(tr("selection_hint_margin_hint"))
        self._chk_selection_classic_hints = QCheckBox(tr("selection_classic_hints"))
        self._chk_selection_classic_hints.setChecked(True)
        self._chk_selection_classic_hints.setToolTip(tr("selection_classic_hints_hint"))
        self._chk_selection_focus_outside = QCheckBox(tr("selection_focus_outside"))
        self._chk_selection_focus_outside.setChecked(True)
        self._chk_selection_focus_outside.setToolTip(tr("selection_focus_outside_hint"))
        self._chk_selection_focus_outside.toggled.connect(self._update_selection_focus_controls_enabled)
        self._selection_focus_context_label = QLabel(tr("selection_focus_context"))
        self._selection_focus_context_spin = QSpinBox()
        self._selection_focus_context_spin.setRange(0, 256)
        self._selection_focus_context_spin.setValue(32)
        self._selection_focus_context_spin.setSuffix(" px")
        self._selection_focus_context_spin.setToolTip(tr("selection_focus_context_hint"))
        self._selection_focus_fade_label = QLabel(tr("selection_focus_fade"))
        self._selection_focus_fade_spin = QSpinBox()
        self._selection_focus_fade_spin.setRange(0, 512)
        self._selection_focus_fade_spin.setValue(96)
        self._selection_focus_fade_spin.setSuffix(" px")
        self._selection_focus_fade_spin.setToolTip(tr("selection_focus_fade_hint"))
        self._selection_focus_context_spin.valueChanged.connect(self._ensure_selection_focus_fade_not_smaller)
        self._btn_clear_selection_hints = QPushButton(
            tr("selection_clear_local_hints"))
        self._btn_clear_selection_hints.setToolTip(
            tr("selection_clear_local_hints_hint"))
        self._btn_clear_selection_hints.clicked.connect(
            self._clear_selection_model_hints)
        selection_ai_options.addWidget(self._chk_selection_bw_preview, 0, 0, 1, 2)
        selection_ai_options.addWidget(self._chk_selection_local_hints_only, 1, 0, 1, 2)
        selection_ai_options.addWidget(self._selection_hint_margin_label, 2, 0)
        selection_ai_options.addWidget(self._selection_hint_margin_spin, 2, 1)
        selection_ai_options.addWidget(self._chk_selection_classic_hints, 3, 0, 1, 2)
        selection_ai_options.addWidget(self._chk_selection_focus_outside, 4, 0, 1, 2)
        selection_ai_options.addWidget(self._selection_focus_context_label, 5, 0)
        selection_ai_options.addWidget(self._selection_focus_context_spin, 5, 1)
        selection_ai_options.addWidget(self._selection_focus_fade_label, 6, 0)
        selection_ai_options.addWidget(self._selection_focus_fade_spin, 6, 1)
        selection_ai_options.addWidget(self._btn_clear_selection_hints, 7, 0, 1, 2)
        tool_param_layout.addWidget(self._selection_ai_options_box)
        register_tool_row("selection_ai_options", self._selection_ai_options_box)
        self._update_selection_focus_controls_enabled()

        self._chk_selection_adjust = QCheckBox(tr("selection_adjust_checkbox"))
        self._chk_selection_adjust.setToolTip(tr("selection_adjust_hint"))
        self._chk_selection_adjust.toggled.connect(self._on_selection_adjust_toggled)
        tool_param_layout.addWidget(self._chk_selection_adjust)
        register_tool_row("selection_adjust_toggle", self._chk_selection_adjust)

        self._selection_adjust_tools_box = QWidget()
        selection_adjust_tools = QHBoxLayout(self._selection_adjust_tools_box)
        selection_adjust_tools.setContentsMargins(0, 0, 0, 0)
        selection_adjust_tools.setSpacing(6)
        self._selection_adjust_group = QButtonGroup(self)
        self._selection_adjust_group.setExclusive(True)
        self._btn_selection_adjust_add = QPushButton(tr("selection_adjust_add_tool"))
        self._btn_selection_adjust_add.setCheckable(True)
        self._btn_selection_adjust_add.setChecked(True)
        self._btn_selection_adjust_add.setToolTip(tr("selection_adjust_add_tool_hint"))
        self._btn_selection_adjust_erase = QPushButton(tr("selection_adjust_erase_tool"))
        self._btn_selection_adjust_erase.setCheckable(True)
        self._btn_selection_adjust_erase.setToolTip(tr("selection_adjust_erase_tool_hint"))
        self._selection_adjust_group.addButton(self._btn_selection_adjust_add)
        self._selection_adjust_group.addButton(self._btn_selection_adjust_erase)
        self._btn_selection_adjust_add.toggled.connect(
            lambda checked: self._set_selection_adjust_mode("add") if checked else None)
        self._btn_selection_adjust_erase.toggled.connect(
            lambda checked: self._set_selection_adjust_mode("erase") if checked else None)
        selection_adjust_tools.addWidget(self._btn_selection_adjust_add)
        selection_adjust_tools.addWidget(self._btn_selection_adjust_erase)
        tool_param_layout.addWidget(self._selection_adjust_tools_box)
        register_tool_row("selection_adjust_tools", self._selection_adjust_tools_box)

        self._selection_adjust_row_box = QWidget()
        selection_adjust_row = QHBoxLayout(self._selection_adjust_row_box)
        selection_adjust_row.setContentsMargins(0, 0, 0, 0)
        selection_adjust_row.setSpacing(6)
        self._selection_adjust_brush_label = QLabel(tr("selection_adjust_brush_label"))
        selection_adjust_row.addWidget(self._selection_adjust_brush_label)
        self._selection_adjust_slider = QSlider(Qt.Orientation.Horizontal)
        self._selection_adjust_slider.setRange(2, 80)
        self._selection_adjust_slider.setValue(18)
        self._selection_adjust_slider.valueChanged.connect(self._on_selection_adjust_radius_changed)
        self._selection_adjust_spin = QSpinBox()
        self._selection_adjust_spin.setRange(2, 80)
        self._selection_adjust_spin.setValue(18)
        self._selection_adjust_slider.valueChanged.connect(self._selection_adjust_spin.setValue)
        self._selection_adjust_spin.valueChanged.connect(self._selection_adjust_slider.setValue)
        selection_adjust_row.addWidget(self._selection_adjust_slider, 1)
        selection_adjust_row.addWidget(self._selection_adjust_spin)
        tool_param_layout.addWidget(self._selection_adjust_row_box)
        register_tool_row("selection_adjust_radius", self._selection_adjust_row_box)
        self._on_selection_adjust_radius_changed(self._selection_adjust_slider.value())
        self._set_selection_adjust_mode("add")

        self._chk_selection_snap_lineart = QCheckBox(tr("selection_snap_lineart_checkbox"))
        self._chk_selection_snap_lineart.setChecked(False)
        self._chk_selection_snap_lineart.setToolTip(tr("selection_snap_lineart_hint"))
        tool_param_layout.addWidget(self._chk_selection_snap_lineart)
        register_tool_row("selection_snap_toggle", self._chk_selection_snap_lineart)

        self._selection_snap_row_box = QWidget()
        selection_snap_row = QHBoxLayout(self._selection_snap_row_box)
        selection_snap_row.setContentsMargins(0, 0, 0, 0)
        selection_snap_row.setSpacing(6)
        self._selection_snap_distance_label = QLabel(tr("selection_snap_distance_label"))
        selection_snap_row.addWidget(self._selection_snap_distance_label)
        self._selection_snap_distance_slider = QSlider(Qt.Orientation.Horizontal)
        self._selection_snap_distance_slider.setRange(2, 24)
        self._selection_snap_distance_slider.setValue(8)
        self._selection_snap_distance_spin = QSpinBox()
        self._selection_snap_distance_spin.setRange(2, 24)
        self._selection_snap_distance_spin.setValue(8)
        self._selection_snap_distance_spin.setSuffix(" px")
        self._selection_snap_distance_slider.valueChanged.connect(
            self._selection_snap_distance_spin.setValue)
        self._selection_snap_distance_spin.valueChanged.connect(
            self._selection_snap_distance_slider.setValue)
        selection_snap_row.addWidget(self._selection_snap_distance_slider, 1)
        selection_snap_row.addWidget(self._selection_snap_distance_spin)
        tool_param_layout.addWidget(self._selection_snap_row_box)
        register_tool_row("selection_snap_distance", self._selection_snap_row_box)

        self._btn_reset_tool_parameters = QPushButton(tr("reset_tool_parameters"))
        self._btn_reset_tool_parameters.clicked.connect(
            self._reset_current_tool_parameters)
        tool_param_layout.addWidget(self._btn_reset_tool_parameters)

        self._compact_hint = QLabel(tr("manual_tool_context_hint"))
        self._compact_hint.setWordWrap(True)
        self._compact_hint.setStyleSheet("color: #6c7f94; font-size: 10px;")
        edit_layout.addWidget(self._compact_hint)

        self._hint_group, hint_layout = make_group(tr("hint_actions_group"))
        self._hint_group.setSizePolicy(QSizePolicy.Policy.Expanding,
                                       QSizePolicy.Policy.Fixed)
        edit_layout.addWidget(self._hint_group)

        self._btn_undo = QPushButton(tr("undo_last_hint"))
        self._btn_undo.clicked.connect(self._undo_last_hint)
        self._btn_clear = QPushButton(tr("clear_manual_hints"))
        self._btn_clear.clicked.connect(self._clear_manual_hints)
        add_button_grid(hint_layout, [self._btn_undo, self._btn_clear])

        self._btn_picker_clear_model = QPushButton(tr("picker_clear_model_hints"))
        self._btn_picker_clear_model.setToolTip(tr("picker_clear_model_hints_hint"))
        self._btn_picker_clear_model.clicked.connect(self._clear_eyedropper_model_hints)
        add_button_grid(hint_layout, [self._btn_picker_clear_model], columns=1)

        self._btn_regenerate = tune_button(
            QPushButton(tr("regenerate_btn")), 35)
        self._btn_regenerate.clicked.connect(self._run_regenerate)
        hint_layout.addWidget(self._btn_regenerate)
        edit_layout.addStretch(1)

        self._update_color_swatch()
        self._set_eyedropper_mode("point")
        self._update_selection_buttons(False)
        edit_tab_layout.addWidget(edit_group, stretch=1)

        history_group, history_layout = make_group(tr("edit_history_group"))
        history_group.setSizePolicy(QSizePolicy.Policy.Expanding,
                                    QSizePolicy.Policy.Fixed)
        self._btn_undo_edit = QPushButton(tr("undo_edit"))
        self._btn_undo_edit.clicked.connect(self._undo)
        self._btn_redo_edit = QPushButton(tr("redo_edit"))
        self._btn_redo_edit.clicked.connect(self._redo)
        add_button_grid(history_layout,
                        [self._btn_undo_edit, self._btn_redo_edit])
        edit_tab_layout.addWidget(history_group, stretch=0)
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
        self._cancel_pending_selection_fill()
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
            protect_text=self._chk_protect_text.isChecked(),
            style_strength=self._style_strength_slider.value() / 100.0,
            reference_strength=self._reference_strength_slider.value() / 100.0,
            manual_strength=self._manual_strength_slider.value() / 100.0,
            pastel_tuning=self._current_pastel_tuning(),
            filter_tuning=self._current_filter_tuning(),
            custom_color_bias=self._current_custom_color_bias(),
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
            "color_filter_enabled": bool(getattr(self, '_chk_filter_color', None) and self._chk_filter_color.isChecked()),
            "color_filter_strength": int(getattr(self, '_filter_color_strength_slider', None).value() if getattr(self, '_filter_color_strength_slider', None) is not None else 35),
            "color_filter_rgb": (self._brush_color.red(), self._brush_color.green(), self._brush_color.blue()),
            "color_filter_color": self._brush_color.name(),
        }

    def _reset_filter_controls(self):
        for slider in (
            self._filter_brightness_slider, self._filter_contrast_slider,
            self._filter_saturation_slider, self._filter_warmth_slider,
            self._filter_shadow_slider, self._filter_highlight_slider,
        ):
            slider.setValue(100)
        if getattr(self, '_chk_filter_color', None) is not None:
            self._chk_filter_color.setChecked(False)
        if getattr(self, '_filter_color_strength_slider', None) is not None:
            self._filter_color_strength_slider.setValue(35)
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
        combo = getattr(self, "_style_combo", None)
        key = combo.currentData() if combo is not None else None




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
            "filter_color_enabled": self._chk_filter_color.isChecked() if getattr(self, '_chk_filter_color', None) is not None else False,
            "filter_color_strength": self._filter_color_strength_slider.value() if getattr(self, '_filter_color_strength_slider', None) is not None else 35,
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
            "bias_brush_color": self._bias_brush_color.name(),
            "bias_brush_size": self._bias_brush_size_slider.value(),
            "bias_brush_strength": self._bias_brush_strength_slider.value(),
            "bias_brush_tone_range": self._bias_brush_tone_combo.currentData(),
            "bias_brush_mode": self._current_bias_brush_mode(),
            "bias_eraser_size": self._bias_eraser_size_slider.value(),
            "bias_brush_protect_skin": self._chk_bias_brush_protect_skin.isChecked(),
            "bias_brush_protect_lineart": self._chk_bias_brush_protect_lineart.isChecked(),
            "bias_brush_protect_saturated": self._chk_bias_brush_protect_saturated.isChecked(),
            "picker_lightness": self._picker_lightness_slider.value(),
            "picker_extract_hint": self._picker_extract_hint_enabled(),
            "selection_adjust_enabled": bool(getattr(self, "_chk_selection_adjust", None) and self._chk_selection_adjust.isChecked()),
            "selection_adjust_radius": int(getattr(self, "_selection_adjust_slider", None).value() if getattr(self, "_selection_adjust_slider", None) is not None else 18),
            "selection_adjust_mode": self._current_selection_adjust_mode(),
            "selection_snap_lineart": bool(getattr(self, "_chk_selection_snap_lineart", None) and self._chk_selection_snap_lineart.isChecked()),
            "selection_snap_distance": int(getattr(self, "_selection_snap_distance_slider", None).value() if getattr(self, "_selection_snap_distance_slider", None) is not None else 8),
            "shared_selected_color": self._brush_color.name(),
            "manual_match_style": self._chk_manual_match_style.isChecked(),
            "skip_colored": self._chk_skip_colored.isChecked(),
            "character_memory": self._chk_character_memory.isChecked(),
            "custom_color_bias_enabled": self._chk_custom_color_bias.isChecked(),
            "custom_color_bias_color": self._custom_color_bias_color.name(),
            "custom_color_bias_strength": self._custom_color_bias_slider.value(),
            "custom_color_bias_scope": self._custom_color_bias_scope.currentData(),
            "custom_color_bias_tone_range": self._custom_color_bias_tone_range.currentData(),
            "custom_color_bias_protect_skin": self._chk_custom_bias_protect_skin.isChecked(),
            "custom_color_bias_protect_lineart": self._chk_custom_bias_protect_lineart.isChecked(),
            "custom_color_bias_protect_saturated": self._chk_custom_bias_protect_saturated.isChecked(),
            "fill_mode": self._current_fill_mode(),
            "selection_closed_only": self._chk_selection_closed_only.isChecked(),
            "selection_mode": self._current_selection_mode(),
            "selection_feather": self._selection_feather_slider.value(),
            "selection_classic_hints": self._chk_selection_classic_hints.isChecked() if getattr(self, '_chk_selection_classic_hints', None) is not None else True,
            "selection_focus_outside": self._chk_selection_focus_outside.isChecked() if getattr(self, '_chk_selection_focus_outside', None) is not None else True,
            "selection_focus_context": self._selection_focus_context_spin.value() if getattr(self, '_selection_focus_context_spin', None) is not None else 32,
            "selection_focus_fade": self._selection_focus_fade_spin.value() if getattr(self, '_selection_focus_fade_spin', None) is not None else 96,
            "selection_closed_expand": self._selection_closed_expand_slider.value(),
            "selection_closed_min_area": self._selection_closed_min_area_slider.value(),
            "selection_closed_min_thickness": self._selection_closed_min_thickness_slider.value(),
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
            if state.result_bgr is not None:
                state.bias_brush_reference_bgr = state.result_bgr.copy()
                state.bias_brush_reference_filter_bgr = (
                    state.filter_base_bgr.copy() if state.filter_base_bgr is not None
                    else state.result_bgr.copy())

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
        if getattr(self, '_chk_filter_color', None) is not None:
            self._chk_filter_color.setChecked(settings.get("filter_color_enabled", False))
        if getattr(self, '_filter_color_strength_slider', None) is not None:
            self._filter_color_strength_slider.setValue(settings.get("filter_color_strength", 35))
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
        shared_color_name = settings.get("shared_selected_color")
        if shared_color_name and QColor(shared_color_name).isValid():
            self._brush_color = QColor(shared_color_name)
        self._brush_slider.setValue(settings.get("brush_size", self._brush_slider.value()))
        bias_color_name = settings.get("bias_brush_color", "#78a0ff")
        if QColor(bias_color_name).isValid():
            self._bias_brush_color = QColor(bias_color_name)
        self._bias_brush_size_slider.setValue(
            settings.get("bias_brush_size", self._bias_brush_size_slider.value()))
        self._bias_brush_strength_slider.setValue(
            settings.get("bias_brush_strength", 80))
        idx = self._bias_brush_tone_combo.findData(
            settings.get("bias_brush_tone_range", "all"))
        if idx >= 0:
            self._bias_brush_tone_combo.setCurrentIndex(idx)
        self._bias_eraser_size_slider.setValue(
            settings.get("bias_eraser_size", self._bias_eraser_size_slider.value()))
        self._set_bias_brush_mode(settings.get("bias_brush_mode", "paint"))
        self._chk_bias_brush_protect_skin.setChecked(
            settings.get("bias_brush_protect_skin", True))
        self._chk_bias_brush_protect_lineart.setChecked(
            settings.get("bias_brush_protect_lineart", True))
        self._chk_bias_brush_protect_saturated.setChecked(
            settings.get("bias_brush_protect_saturated", True))
        self._update_bias_brush_swatch()
        self._picker_lightness_slider.setValue(settings.get("picker_lightness", 0))
        if getattr(self, "_chk_picker_extract_hint", None) is not None:
            self._chk_picker_extract_hint.setChecked(settings.get("picker_extract_hint", False))
        if getattr(self, "_selection_adjust_slider", None) is not None:
            self._selection_adjust_slider.setValue(int(settings.get("selection_adjust_radius", 18)))
        self._set_selection_adjust_mode(str(settings.get("selection_adjust_mode", "add")))
        if getattr(self, "_chk_selection_adjust", None) is not None:
            self._chk_selection_adjust.setChecked(bool(settings.get("selection_adjust_enabled", False)))
        if getattr(self, "_selection_snap_distance_slider", None) is not None:
            self._selection_snap_distance_slider.setValue(int(settings.get("selection_snap_distance", 8)))
        if getattr(self, "_chk_selection_snap_lineart", None) is not None:
            self._chk_selection_snap_lineart.setChecked(bool(settings.get("selection_snap_lineart", False)))
        if getattr(self, '_chk_manual_match_style', None) is not None:
            self._chk_manual_match_style.setChecked(settings.get("manual_match_style", False))
        self._chk_skip_colored.setChecked(settings.get("skip_colored", True))
        self._chk_character_memory.setChecked(settings.get("character_memory", True))
        if getattr(self, '_chk_custom_color_bias', None) is not None:
            self._chk_custom_color_bias.setChecked(settings.get("custom_color_bias_enabled", False))
            self._custom_color_bias_slider.setValue(settings.get("custom_color_bias_strength", 35))
            color_name = settings.get("custom_color_bias_color", settings.get("shared_selected_color") or "#ffb4b4")
            valid_color = QColor(color_name) if QColor(color_name).isValid() else QColor(255, 180, 180)
            self._set_custom_bias_linked_color(valid_color, update_shared_brush=False)
            scope = settings.get("custom_color_bias_scope", "page")
            idx = self._custom_color_bias_scope.findData(scope)
            if idx >= 0:
                self._custom_color_bias_scope.setCurrentIndex(idx)
            tone_range = settings.get("custom_color_bias_tone_range", "all")
            idx = self._custom_color_bias_tone_range.findData(tone_range)
            if idx >= 0:
                self._custom_color_bias_tone_range.setCurrentIndex(idx)
            self._chk_custom_bias_protect_skin.setChecked(settings.get("custom_color_bias_protect_skin", True))
            self._chk_custom_bias_protect_lineart.setChecked(settings.get("custom_color_bias_protect_lineart", True))
            self._chk_custom_bias_protect_saturated.setChecked(settings.get("custom_color_bias_protect_saturated", True))
            self._update_custom_color_bias_swatch()
            self._update_custom_bias_controls_enabled()
        self._update_filter_color_controls_enabled()
        self._canvas.set_brush_color(self._brush_color)
        self._canvas.set_bias_brush_color(self._bias_brush_color)
        self._canvas.set_bias_brush_radius(
            self._brush_px_from_percent(self._bias_brush_size_slider.value()))
        self._update_color_swatch()
        self._update_bias_brush_swatch()
        self._set_fill_mode(settings.get("fill_mode", "shift"))
        if getattr(self, '_chk_selection_closed_only', None) is not None:
            self._chk_selection_closed_only.setChecked(settings.get("selection_closed_only", False))
        if getattr(self, '_selection_feather_slider', None) is not None:
            self._selection_feather_slider.setValue(settings.get("selection_feather", 2))
        if getattr(self, '_chk_selection_classic_hints', None) is not None:
            self._chk_selection_classic_hints.setChecked(bool(settings.get("selection_classic_hints", True)))
        if getattr(self, '_chk_selection_focus_outside', None) is not None:
            self._chk_selection_focus_outside.setChecked(bool(settings.get("selection_focus_outside", True)))
        if getattr(self, '_selection_focus_context_spin', None) is not None:
            self._selection_focus_context_spin.setValue(int(settings.get("selection_focus_context", 32)))
        if getattr(self, '_selection_focus_fade_spin', None) is not None:
            self._selection_focus_fade_spin.setValue(max(int(settings.get("selection_focus_context", 32)), int(settings.get("selection_focus_fade", 96))))
        self._update_selection_focus_controls_enabled()
        if getattr(self, '_selection_closed_expand_slider', None) is not None:
            self._selection_closed_expand_slider.setValue(settings.get("selection_closed_expand", 0))
        if getattr(self, '_selection_closed_min_area_slider', None) is not None:
            self._selection_closed_min_area_slider.setValue(settings.get("selection_closed_min_area", 6))
        if getattr(self, '_selection_closed_min_thickness_slider', None) is not None:
            self._selection_closed_min_thickness_slider.setValue(
                settings.get("selection_closed_min_thickness", 3))
        self._set_selection_mode(settings.get("selection_mode", "replace"))
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
        self._sync_bias_reference_to_current(state)
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
                   self._btn_export_page, self._btn_export_all,
                   self._btn_apply_selection, self._btn_ai_recolor_selection,
                   self._btn_clear_selection_hints, self._btn_cancel_selection):
            btn.setEnabled(not busy)
        if not busy:
            self._update_selection_buttons(
                self._pending_selection_mask is not None)
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

    def _on_selection_preview_active(self, active: bool):
        self._update_selection_buttons(active and self._pending_selection_mask is not None)

    def _update_selection_buttons(self, active: bool | None = None):
        if active is None:
            active = self._pending_selection_mask is not None
        if getattr(self, '_btn_apply_selection', None) is not None:
            self._btn_apply_selection.setEnabled(bool(active))
        if getattr(self, '_btn_cancel_selection', None) is not None:
            self._btn_cancel_selection.setEnabled(bool(active))
        if getattr(self, '_btn_ai_recolor_selection', None) is not None:
            self._btn_ai_recolor_selection.setEnabled(bool(active))
        if getattr(self, '_btn_clear_selection_hints', None) is not None:
            self._btn_clear_selection_hints.setEnabled(bool(active))
        adjust_ok = bool(active)
        for widget in (getattr(self, '_chk_selection_adjust', None),
                       getattr(self, '_btn_selection_adjust_add', None),
                       getattr(self, '_btn_selection_adjust_erase', None),
                       getattr(self, '_selection_adjust_slider', None),
                       getattr(self, '_selection_adjust_spin', None)):
            if widget is not None:
                widget.setEnabled(adjust_ok)
        self._update_tool_specific_visibility()

    def _selection_snap_enabled(self) -> bool:
        checkbox = getattr(self, '_chk_selection_snap_lineart', None)
        return False if checkbox is None else bool(checkbox.isChecked())

    def _snap_selection_to_lineart(self, mask: np.ndarray | None,
                                    *, status: bool = True) -> np.ndarray | None:
        if mask is None or not np.any(mask) or not self._selection_snap_enabled():
            return mask
        state = self._current_state()
        if state is None:
            return mask
        gap_close = self._gap_px_from_percent(self._gap_close_slider.value())
        region_map = state.hint_manager.bind_source_image(
            state.original_bgr, gap_close=gap_close)
        max_distance = int(
            getattr(self, '_selection_snap_distance_slider', None).value()
            if getattr(self, '_selection_snap_distance_slider', None) is not None else 8)
        snapped, diagnostics = snap_selection_mask_to_lineart(
            state.original_bgr, mask, region_map=region_map,
            gap_close=gap_close, max_distance=max_distance)
        if status:
            if diagnostics.used_fallback:
                self.statusBar().showMessage(tr('selection_snap_no_match'), 2200)
            else:
                self.statusBar().showMessage(
                    tr('selection_snap_status').format(
                        regions=len(diagnostics.selected_regions),
                        distance=max_distance), 2600)
        return snapped if np.any(snapped) else mask

    def _current_bias_brush_mode(self) -> str:
        button = getattr(self, '_btn_bias_brush_erase', None)
        return "erase" if button is not None and button.isChecked() else "paint"

    def _set_bias_brush_mode(self, mode: str):
        mode = "erase" if str(mode) == "erase" else "paint"
        paint_button = getattr(self, '_btn_bias_brush_paint', None)
        erase_button = getattr(self, '_btn_bias_brush_erase', None)
        if paint_button is not None and paint_button.isChecked() != (mode == "paint"):
            paint_button.blockSignals(True)
            paint_button.setChecked(mode == "paint")
            paint_button.blockSignals(False)
        if erase_button is not None and erase_button.isChecked() != (mode == "erase"):
            erase_button.blockSignals(True)
            erase_button.setChecked(mode == "erase")
            erase_button.blockSignals(False)
        self._update_bias_brush_mode_ui()
        if getattr(self, '_radio_bias_brush', None) is not None and self._radio_bias_brush.isChecked():
            self._on_tool_changed()
        self.statusBar().showMessage(
            tr("bias_brush_mode_paint_selected") if mode == "paint"
            else tr("bias_brush_mode_erase_selected"), 1800)

    def _update_bias_brush_mode_ui(self):
        paint_mode = self._current_bias_brush_mode() == "paint"
        for widget in (
            getattr(self, '_bias_brush_color_label', None),
            getattr(self, '_bias_brush_color_btn', None),
            getattr(self, '_bias_brush_color_info', None),
            getattr(self, '_bias_brush_size_label', None),
            getattr(self, '_bias_brush_size_slider', None),
            getattr(self, '_bias_brush_size_spin', None),
            getattr(self, '_bias_brush_strength_label', None),
            getattr(self, '_bias_brush_strength_slider', None),
            getattr(self, '_bias_brush_strength_spin', None),
            getattr(self, '_bias_brush_tone_label', None),
            getattr(self, '_bias_brush_tone_combo', None),
            getattr(self, '_bias_brush_protection_box', None),
        ):
            if widget is not None:
                widget.setVisible(paint_mode)
        # the static protection label was inserted inline, so toggle by row index through layout items
        if getattr(self, '_bias_brush_box', None) is not None:
            layout = self._bias_brush_box.layout()
            if layout is not None:
                item = layout.itemAtPosition(5, 0) if hasattr(layout, 'itemAtPosition') else None
                if item is not None and item.widget() is not None:
                    item.widget().setVisible(paint_mode)
        eraser_box = getattr(self, '_bias_eraser_controls_box', None)
        if eraser_box is not None:
            eraser_box.setVisible(not paint_mode)

    def _current_selection_adjust_mode(self) -> str:
        button = getattr(self, '_btn_selection_adjust_erase', None)
        return "erase" if button is not None and button.isChecked() else "add"

    def _set_selection_adjust_mode(self, mode: str):
        mode = "erase" if str(mode) == "erase" else "add"
        add_button = getattr(self, '_btn_selection_adjust_add', None)
        erase_button = getattr(self, '_btn_selection_adjust_erase', None)
        if add_button is not None and add_button.isChecked() != (mode == "add"):
            add_button.blockSignals(True)
            add_button.setChecked(mode == "add")
            add_button.blockSignals(False)
        if erase_button is not None and erase_button.isChecked() != (mode == "erase"):
            erase_button.blockSignals(True)
            erase_button.setChecked(mode == "erase")
            erase_button.blockSignals(False)
        if getattr(self, '_canvas', None) is not None:
            self._canvas.set_selection_adjust_mode(mode)
        self.statusBar().showMessage(
            tr("selection_adjust_add_selected") if mode == "add"
            else tr("selection_adjust_erase_selected"), 1800)

    def _on_selection_adjust_toggled(self, enabled: bool):
        if getattr(self, '_canvas', None) is not None:
            self._canvas.set_selection_adjust_enabled(bool(enabled))
        self._update_tool_specific_visibility()
        if enabled:
            self.statusBar().showMessage(tr("selection_adjust_status"), 3500)

    def _on_selection_adjust_radius_changed(self, value: int):
        if getattr(self, '_canvas', None) is not None:
            self._canvas.set_selection_adjust_radius(int(value))

    def _on_selection_adjust_dab(self, ix: int, iy: int, radius_px: int, add_mode: bool):
        if self._pending_selection_mask is None:
            return
        # Manual corrections become authoritative.  Stop the cached AI preview
        # from regenerating on Apply and overwriting the user's add/erase work.
        self._clear_closed_preview_cache()
        mask = self._pending_selection_mask.copy()
        value = 255 if add_mode else 0
        cv2.circle(mask, (int(ix), int(iy)), max(1, int(radius_px)), value, thickness=-1)
        self._set_pending_selection(mask, self._pending_selection_kind, combine=False)
        self.statusBar().showMessage(
            tr("selection_adjust_add_status") if add_mode else tr("selection_adjust_erase_status"),
            1200)

    def _current_selection_mode(self) -> str:
        radios = getattr(self, '_selection_mode_radios', {}) or {}
        for mode in ('replace', 'add', 'subtract'):
            radio = radios.get(mode)
            if radio is not None and radio.isChecked():
                return mode
        return 'replace'

    def _set_selection_mode(self, mode: str):
        mode = mode if mode in {'replace', 'add', 'subtract'} else 'replace'
        for key, radio in (getattr(self, '_selection_mode_radios', {}) or {}).items():
            desired = key == mode
            if radio.isChecked() != desired:
                radio.blockSignals(True)
                radio.setChecked(desired)
                radio.blockSignals(False)
        canvas = getattr(self, '_canvas', None)
        if canvas is not None and hasattr(canvas, 'set_selection_combine_mode'):
            canvas.set_selection_combine_mode(mode)

    def _combine_selection_mask(self, incoming_mask: np.ndarray | None) -> np.ndarray | None:
        return combine_selection_masks(
            self._pending_selection_mask, incoming_mask,
            self._current_selection_mode())

    def _set_pending_selection(self, mask: np.ndarray | None,
                               kind: str | None = None, *,
                               combine: bool = True):
        combined = (self._combine_selection_mask(mask) if combine else
                    (None if mask is None or not np.any(mask) else
                     np.where(mask > 0, 255, 0).astype(np.uint8)))
        self._pending_selection_mask = None if combined is None else combined.copy()
        self._pending_selection_kind = kind
        active = combined is not None and bool(np.any(combined))
        self._update_selection_buttons(active)
        if active:
            mode = self._current_selection_mode()
            if (getattr(self, '_chk_selection_bw_preview', None) is not None
                    and self._chk_selection_bw_preview.isChecked()):
                self._refresh_selection_display_preview()
            else:
                self._canvas.set_selection_mask_overlay(
                    self._pending_selection_mask, mode=mode)
            suffix = {'replace': '（替换）', 'add': '（加选）', 'subtract': '（减选）'}.get(mode, '')
            self.statusBar().showMessage(tr('selection_pending_hint') + suffix, 5000)
        else:
            self._canvas.clear_selection_preview()

    def _update_closed_expand_controls_enabled(self, enabled: bool | None = None):
        if enabled is None:
            enabled = bool(getattr(self, '_chk_selection_closed_only', None)
                           and self._chk_selection_closed_only.isChecked())
        for widget in (getattr(self, '_selection_closed_expand_slider', None),
                       getattr(self, '_selection_closed_expand_spin', None),
                       getattr(self, '_selection_closed_min_area_slider', None),
                       getattr(self, '_selection_closed_min_area_spin', None),
                       getattr(self, '_selection_closed_min_thickness_slider', None),
                       getattr(self, '_selection_closed_min_thickness_spin', None)):
            if widget is not None:
                widget.setEnabled(bool(enabled))

    def _clear_closed_preview_cache(self):
        timer = getattr(self, '_closed_expand_preview_timer', None)
        if timer is not None:
            timer.stop()
        self._closed_preview_page_path = None
        self._closed_preview_raw_mask = None
        self._closed_preview_probability = None
        self._closed_preview_device = ""
        self._closed_preview_base_mask = None
        self._closed_preview_combine_mode = "replace"

    def _schedule_closed_preview_refresh(self, _value: int | None = None):
        if (self._closed_preview_probability is None
                or self._closed_preview_raw_mask is None
                or self._closed_preview_page_path != self._current_path):
            return
        self._closed_expand_preview_timer.start()

    def _refresh_closed_selection_preview(self) -> bool:
        page_path = self._closed_preview_page_path
        raw_mask = self._closed_preview_raw_mask
        probability = self._closed_preview_probability
        if (page_path is None or raw_mask is None or probability is None
                or page_path != self._current_path):
            return False
        state = self._pages.get(page_path)
        if state is None:
            return False
        expand_px = int(self._selection_closed_expand_slider.value())
        min_area = int(self._selection_closed_min_area_slider.value())
        min_thickness = int(self._selection_closed_min_thickness_slider.value())
        mask = build_selection_edit_mask(
            state.original_bgr,
            raw_mask,
            closed_only=True,
            reject_dominant=True,
            extra_probability=probability,
            expand_px=expand_px,
            min_area=min_area,
            min_thickness=min_thickness,
        )
        if not np.any(mask):
            self._canvas.clear_selection_preview()
            self._pending_selection_mask = None
            self._pending_selection_kind = None
            self._update_selection_buttons(False)
            self.statusBar().showMessage(tr("selection_fill_no_closed"), 3500)
            return False
        combined = combine_selection_masks(
            self._closed_preview_base_mask,
            mask,
            self._closed_preview_combine_mode)
        self._set_pending_selection(combined, 'rect_closed', combine=False)
        device = self._closed_preview_device
        self.statusBar().showMessage(
            tr("selection_closed_preview_status").format(
                expand=expand_px,
                area=min_area,
                thickness=min_thickness,
                device=(f" · {device}" if device else "")),
            2500)
        return True

    def _cancel_pending_selection_fill(self):
        self._clear_closed_preview_cache()
        # A running inference cannot always be interrupted safely, but clearing
        # these tokens makes its eventual result stale and therefore ignored.
        self._line_extract_page_path = None
        self._line_extract_raw_mask = None
        if self._pending_selection_mask is None:
            self._canvas.clear_selection_preview()
            self._update_selection_buttons(False)
            return
        self._pending_selection_mask = None
        self._pending_selection_kind = None
        self._canvas.clear_selection_preview()
        self._update_selection_buttons(False)
        state = self._current_state()
        if state is not None:
            self._sync_view_after_edit(state)
        self.statusBar().showMessage(tr('selection_cancel'), 2500)

    def _apply_pending_selection_fill(self):
        if self._closed_preview_probability is not None:
            self._closed_expand_preview_timer.stop()
            self._refresh_closed_selection_preview()
        if self._pending_selection_mask is None:
            return
        mask = self._pending_selection_mask.copy()
        kind = self._pending_selection_kind
        self._clear_closed_preview_cache()
        self._pending_selection_mask = None
        self._pending_selection_kind = None
        self._canvas.clear_selection_preview()
        self._update_selection_buttons(False)
        self._apply_selection_fill_mask(mask, selection_kind=kind)


    def _refresh_selection_display_preview(self, _checked: bool | None = None):
        """Refresh non-destructive B&W-in-selection preview."""
        state = self._current_state()
        if state is None:
            return
        self._sync_view_after_edit(state)

    def _ensure_selection_focus_fade_not_smaller(self, value: int):
        spin = getattr(self, '_selection_focus_fade_spin', None)
        if spin is None:
            return
        minimum = int(value)
        if spin.value() < minimum:
            spin.setValue(minimum)

    def _update_selection_focus_controls_enabled(self, _checked: bool | None = None):
        enabled = bool(getattr(self, '_chk_selection_focus_outside', None)
                       and self._chk_selection_focus_outside.isChecked())
        for widget in (getattr(self, '_selection_focus_context_label', None),
                       getattr(self, '_selection_focus_context_spin', None),
                       getattr(self, '_selection_focus_fade_label', None),
                       getattr(self, '_selection_focus_fade_spin', None)):
            if widget is not None:
                widget.setEnabled(enabled)

    def _clear_selection_model_hints(self):
        state = self._current_state()
        mask = self._pending_selection_mask
        if state is None or mask is None or not np.any(mask):
            self.statusBar().showMessage(tr("selection_fill_empty"), 2500)
            return
        from core.local_model_recolor import clear_manual_hints_in_selection
        state.push_undo()
        removed = clear_manual_hints_in_selection(state.hint_manager, mask)
        if removed <= 0:
            state.discard_unchanged_undo()
            self.statusBar().showMessage(tr("selection_no_local_hints"), 3000)
            return
        self._refresh_hint_overlay()
        self.statusBar().showMessage(
            tr("selection_local_hints_cleared").format(count=removed), 3500)

    def _start_selection_model_recolor(self):
        if (self._local_recolor_worker is not None
                and self._local_recolor_worker.isRunning()):
            self.statusBar().showMessage(tr("selection_ai_busy"), 3500)
            return
        if self._batch_worker is not None and self._batch_worker.isRunning():
            self.statusBar().showMessage("上色任务正在运行，请稍候。", 4000)
            return
        state = self._current_state()
        mask = self._pending_selection_mask
        if state is None or state.result_bgr is None:
            QMessageBox.information(self, tr("no_result_title"), tr("no_result_body"))
            return
        if mask is None or not np.any(mask):
            self.statusBar().showMessage(tr("selection_fill_empty"), 3000)
            return

        from core.local_model_recolor import count_hints_in_selection
        margin = int(self._selection_hint_margin_spin.value())
        local_hint_count = count_hints_in_selection(
            state.hint_manager, mask, margin_px=margin)
        state.push_undo()
        self._local_recolor_page_path = state.path
        self._local_recolor_selection_mask = mask.copy()
        worker = LocalModelRecolorWorker(
            state.original_bgr,
            state.result_bgr,
            state.ai_result_bgr,
            state.filter_base_bgr,
            mask,
            state.hint_manager,
            feather=int(self._selection_feather_slider.value()),
            hint_margin_px=margin,
            gap_close=self._gap_px_from_percent(self._gap_close_slider.value()),
            only_selection_hints=bool(
                self._chk_selection_local_hints_only.isChecked()),
            classic_point_hints=bool(getattr(self, '_chk_selection_classic_hints', None)
                                     and self._chk_selection_classic_hints.isChecked()),
            focus_outside_mode=("fade_white" if bool(getattr(self, '_chk_selection_focus_outside', None)
                                                     and self._chk_selection_focus_outside.isChecked()) else "none"),
            focus_context_expand_px=int(getattr(self, '_selection_focus_context_spin', None).value()
                                        if getattr(self, '_selection_focus_context_spin', None) is not None else 32),
            focus_fade_expand_px=int(getattr(self, '_selection_focus_fade_spin', None).value()
                                     if getattr(self, '_selection_focus_fade_spin', None) is not None else 96),
            style_key=self._style_combo.currentData(),
            character_memories=self._get_or_create_character_memories(),
            character_library=self._character_library,
            scene_palette=self._scene_palette,
            style_strength=self._style_strength_slider.value() / 100.0,
            reference_strength=self._reference_strength_slider.value() / 100.0,
            manual_strength=self._manual_strength_slider.value() / 100.0,
            pastel_tuning=self._current_pastel_tuning(),
            filter_tuning=self._current_filter_tuning(),
            custom_color_bias=self._current_custom_color_bias(),
            forced_matches=dict(state.forced_character_matches),
            parent=self,
        )
        self._local_recolor_worker = worker
        worker.status.connect(self._on_worker_status)
        worker.finished_ok.connect(self._on_local_recolor_done)
        worker.finished_err.connect(self._on_local_recolor_error)
        self._set_busy(True, tr("selection_ai_running").format(hints=local_hint_count))
        worker.start()

    def _finish_local_recolor_worker(self):
        worker = self._local_recolor_worker
        self._local_recolor_worker = None
        self._local_recolor_page_path = None
        self._local_recolor_selection_mask = None
        if worker is not None:
            worker.deleteLater()
        self._release_compute_memory()
        self._set_busy(False)
        self._update_selection_buttons(self._pending_selection_mask is not None)

    def _on_local_recolor_done(self, payload):
        page_path = self._local_recolor_page_path
        selection_mask = (self._local_recolor_selection_mask.copy()
                          if isinstance(self._local_recolor_selection_mask, np.ndarray)
                          else None)
        state = self._pages.get(page_path) if page_path else None
        if state is None:
            self._finish_local_recolor_worker()
            return
        state.result_bgr = payload.result_bgr.copy()
        state.ai_result_bgr = payload.ai_result_bgr.copy()
        state.filter_base_bgr = payload.filter_base_bgr.copy()
        state.pipeline_diagnostics = dict(payload.diagnostics or {})
        self._sync_bias_reference_to_current(state)
        consumed_hint_count = 0
        if selection_mask is not None and np.any(selection_mask):
            from core.local_model_recolor import clear_manual_hints_in_selection
            consumed_hint_count = int(clear_manual_hints_in_selection(
                state.hint_manager, selection_mask))
        try:
            from core.quality_score import assess_colorization
            state.quality_report = assess_colorization(
                state.original_bgr, state.result_bgr)
        except Exception:
            state.quality_report = None
        state.discard_unchanged_undo()
        self._mark_page_done(state.path)

        if state.path == self._current_path:
            self._clear_closed_preview_cache()
            self._pending_selection_mask = None
            self._pending_selection_kind = None
            self._canvas.clear_selection_preview()
            self._radio_view_edited.setChecked(True)
            self._sync_view_after_edit(state)
            self._refresh_hint_overlay()
        used = int(getattr(payload, "used_hint_count", 0))
        pixels = int(getattr(payload, "selection_pixels", 0))
        changed = int(getattr(payload, "changed_pixels", 0))
        self._finish_local_recolor_worker()
        if changed <= 0:
            self.statusBar().showMessage(
                tr("selection_ai_no_change").format(hints=used), 7000)
        else:
            message = tr("selection_ai_done").format(
                hints=used, pixels=pixels, changed=changed)
            if consumed_hint_count > 0:
                message += " " + tr("selection_local_hints_cleared").format(
                    count=consumed_hint_count)
            self.statusBar().showMessage(message, 6000)

    def _on_local_recolor_error(self, message: str):
        page_path = self._local_recolor_page_path
        state = self._pages.get(page_path) if page_path else None
        if state is not None:
            state.discard_unchanged_undo()
        self._finish_local_recolor_worker()
        QMessageBox.critical(
            self, tr("selection_ai_failed_title"),
            tr("selection_ai_failed").format(message=message))

    # Compatibility marker for the legacy route test: mode=self._current_fill_mode()
    def _current_fill_mode(self) -> str:
        combo = getattr(self, "_fill_mode_combo", None)
        value = str(combo.currentData() if combo is not None else "shift").strip().lower()
        return value if value in {"shift", "shading", "flat"} else "shift"

    def _set_fill_mode(self, mode: str):
        combo = getattr(self, "_fill_mode_combo", None)
        if combo is None:
            return
        value = str(mode or "shift").strip().lower()
        if value not in {"shift", "shading", "flat"}:
            value = "shift"
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _current_manual_tool_key(self) -> str:
        if getattr(self, "_radio_bias_brush", None) is not None and self._radio_bias_brush.isChecked():
            return "bias_brush"
        if getattr(self, "_radio_eyedropper", None) is not None and self._radio_eyedropper.isChecked():
            return "eyedropper"
        if getattr(self, "_radio_bucket", None) is not None and self._radio_bucket.isChecked():
            return "bucket"
        if getattr(self, "_radio_lasso_bucket", None) is not None and self._radio_lasso_bucket.isChecked():
            return "lasso"
        if getattr(self, "_radio_rect_bucket", None) is not None and self._radio_rect_bucket.isChecked():
            return "rect"
        return "brush"

    def _update_tool_specific_visibility(self):
        tool = self._current_manual_tool_key()
        selection_tool = tool in {"lasso", "rect"}
        visible_rows = {
            "brush": {"color", "brush_size", "fill_mode", "manual_match_style", "brush_model_hint"},
            "bias_brush": {"bias_brush_controls"},
            "eyedropper": {"eyedropper_mode", "color", "picker_lightness", "picker_extract_hint"},
            "bucket": {"color", "gap_close", "fill_mode", "manual_match_style"},
            "lasso": {
                "color", "gap_close", "fill_mode", "manual_match_style",
                "selection_mode", "selection_feather", "selection_closed_only",
                "selection_closed_expand", "selection_closed_min_area",
                "selection_closed_min_thickness", "selection_actions",
                "selection_ai_options", "selection_adjust_toggle", "selection_adjust_tools",
                "selection_adjust_radius", "selection_snap_toggle",
                "selection_snap_distance"
            },
            "rect": {
                "color", "gap_close", "fill_mode", "manual_match_style",
                "selection_mode", "selection_feather", "selection_closed_only",
                "selection_closed_expand", "selection_closed_min_area",
                "selection_closed_min_thickness", "selection_actions",
                "selection_ai_options", "selection_adjust_toggle", "selection_adjust_tools",
                "selection_adjust_radius", "selection_snap_toggle",
                "selection_snap_distance"
            },
        }.get(tool, set())

        for row_name, widgets in (getattr(self, "_tool_visibility_rows", {}) or {}).items():
            visible = row_name in visible_rows
            for widget in widgets:
                widget.setVisible(visible)

        # Manual expansion/erase only makes sense after a selection exists.
        pending_selection = self._pending_selection_mask is not None
        adjust_enabled = bool(
            pending_selection
            and getattr(self, "_chk_selection_adjust", None)
            and self._chk_selection_adjust.isChecked())
        for row_name in ("selection_adjust_tools", "selection_adjust_radius"):
            for widget in (getattr(self, "_tool_visibility_rows", {}) or {}).get(row_name, ()):
                widget.setVisible(selection_tool and adjust_enabled)

        picker_clear = getattr(self, "_btn_picker_clear_model", None)
        if picker_clear is not None:
            picker_clear.setVisible(tool == "eyedropper")

        if tool == "bias_brush" and self._current_bias_brush_mode() == "erase":
            title_key = "tool_parameters_bias_eraser"
        else:
            title_key = {
                "brush": "tool_parameters_brush",
                "bias_brush": "tool_parameters_bias_brush",
            "eyedropper": "tool_parameters_eyedropper",
            "bucket": "tool_parameters_bucket",
                "lasso": "tool_parameters_lasso",
                "rect": "tool_parameters_rect",
            }.get(tool, "tool_parameters_group")
        if tool == "bias_brush" and self._current_bias_brush_mode() == "erase":
            hint_key = "tool_hint_bias_eraser"
        else:
            hint_key = {
                "brush": "tool_hint_brush",
                "bias_brush": "tool_hint_bias_brush",
            "eyedropper": "tool_hint_eyedropper",
            "bucket": "tool_hint_bucket",
            "lasso": "tool_hint_selection",
            "rect": "tool_hint_selection",
        }.get(tool, "manual_tool_context_hint")
        if getattr(self, "_tool_param_group", None) is not None:
            self._tool_param_group.setTitle(tr(title_key))
        if getattr(self, "_compact_hint", None) is not None:
            self._compact_hint.setText(tr(hint_key))

    def _reset_current_tool_parameters(self):
        tool = self._current_manual_tool_key()
        if tool == "brush":
            self._brush_slider.setValue(self._brush_percent_from_px(12))
            self._set_fill_mode("shift")
            self._chk_manual_match_style.setChecked(False)
            self._chk_brush_model_hint.setChecked(False)
        elif tool == "bias_brush":
            self._set_bias_brush_mode("paint")
            self._bias_brush_size_slider.setValue(self._brush_percent_from_px(18))
            self._bias_eraser_size_slider.setValue(self._brush_percent_from_px(18))
            self._bias_brush_strength_slider.setValue(80)
            idx = self._bias_brush_tone_combo.findData("all")
            if idx >= 0:
                self._bias_brush_tone_combo.setCurrentIndex(idx)
            self._chk_bias_brush_protect_skin.setChecked(True)
            self._chk_bias_brush_protect_lineart.setChecked(True)
            self._chk_bias_brush_protect_saturated.setChecked(True)
        elif tool == "eyedropper":
            self._set_eyedropper_mode("point")
            self._picker_lightness_slider.setValue(0)
            self._chk_picker_extract_hint.setChecked(False)
        elif tool == "bucket":
            self._gap_close_slider.setValue(6)
            self._set_fill_mode("shift")
            self._chk_manual_match_style.setChecked(False)
        else:
            self._gap_close_slider.setValue(6)
            self._set_fill_mode("shift")
            self._chk_manual_match_style.setChecked(False)
            self._set_selection_mode("replace")
            self._selection_feather_slider.setValue(2)
            self._chk_selection_closed_only.setChecked(False)
            self._selection_closed_expand_slider.setValue(0)
            self._selection_closed_min_area_slider.setValue(6)
            self._selection_closed_min_thickness_slider.setValue(3)
            self._chk_selection_adjust.setChecked(False)
            self._selection_adjust_slider.setValue(18)
            self._set_selection_adjust_mode("add")
            self._chk_selection_snap_lineart.setChecked(False)
            self._selection_snap_distance_slider.setValue(8)
        self._update_tool_specific_visibility()
        self.statusBar().showMessage(tr("tool_parameters_reset_done"), 2200)

    def _on_tool_changed(self):
        if self._radio_brush.isChecked():
            tool = HintCanvas.TOOL_BRUSH
        elif getattr(self, '_radio_bias_brush', None) is not None and self._radio_bias_brush.isChecked():
            tool = (HintCanvas.TOOL_BIAS_ERASER
                    if self._current_bias_brush_mode() == "erase"
                    else HintCanvas.TOOL_BIAS_BRUSH)
        elif self._radio_eyedropper.isChecked():
            tool = HintCanvas.TOOL_EYEDROPPER
        elif self._radio_bucket.isChecked():
            tool = HintCanvas.TOOL_BUCKET
        elif getattr(self, '_radio_lasso_bucket', None) is not None and self._radio_lasso_bucket.isChecked():
            tool = HintCanvas.TOOL_LASSO_BUCKET
        else:
            tool = HintCanvas.TOOL_RECT_BUCKET
        self._canvas.set_tool(tool)
        # Keep a prepared blue selection while switching to the hint brush or
        # eyedropper. This is required by the local mc-v2 workflow: select the
        # bad region first, add/remove hints second, then return and execute.
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

    def _update_custom_bias_controls_enabled(self):
        enabled = bool(getattr(self, '_chk_custom_color_bias', None) and self._chk_custom_color_bias.isChecked())
        for widget_name in (
            '_custom_color_bias_btn', '_custom_color_bias_info',
            '_custom_color_bias_scope', '_custom_color_bias_tone_range',
            '_custom_color_bias_slider', '_custom_color_bias_spin',
            '_chk_custom_bias_protect_skin', '_chk_custom_bias_protect_lineart',
            '_chk_custom_bias_protect_saturated',
        ):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.setEnabled(enabled)

    def _update_filter_color_controls_enabled(self):
        enabled = bool(getattr(self, '_chk_filter_color', None) and self._chk_filter_color.isChecked())
        for widget_name in ('_filter_color_strength_slider', '_filter_color_strength_spin', '_filter_color_hint'):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.setEnabled(enabled)

    def _set_custom_bias_linked_color(self, color: QColor, *, update_shared_brush: bool = False):
        """Update the linked custom-bias colour pair.

        The global custom-colour-bias control and the independent colour-bias
        brush intentionally share one colour.  The normal brush / eyedropper
        selection is *not* driven by this helper unless explicitly requested,
        which keeps the sync one-way: eyedropper -> custom bias is allowed,
        but editing the custom-bias colour does not rewrite the normal brush.
        """
        if color is None or not color.isValid():
            return
        linked = QColor(color)
        self._custom_color_bias_color = QColor(linked)
        self._bias_brush_color = QColor(linked)
        if update_shared_brush:
            self._brush_color = QColor(linked)
            if getattr(self, '_canvas', None) is not None:
                self._canvas.set_brush_color(self._brush_color)
            self._update_color_swatch()
        if getattr(self, '_canvas', None) is not None:
            self._canvas.set_bias_brush_color(self._bias_brush_color)
        self._update_custom_color_bias_swatch()
        self._update_bias_brush_swatch()

    def _set_shared_selected_color(self, color: QColor):
        if color is None or not color.isValid():
            return
        self._brush_color = QColor(color)
        if getattr(self, '_canvas', None) is not None:
            self._canvas.set_brush_color(self._brush_color)
        self._update_color_swatch()
        # One-way sync: the active picked colour may update custom colour bias
        # and the bias brush, but picking a custom-bias colour must not push
        # back into the normal brush / eyedropper state.
        self._set_custom_bias_linked_color(QColor(color), update_shared_brush=False)

    def _pick_bias_brush_color(self):
        color = self._open_color_dialog(
            QColor(getattr(self, '_bias_brush_color', QColor(120, 160, 255))),
            tr("bias_brush_color"))
        if not color.isValid():
            return
        self._set_custom_bias_linked_color(color, update_shared_brush=False)

    def _update_bias_brush_swatch(self):
        btn = getattr(self, '_bias_brush_color_btn', None)
        color = QColor(getattr(self, '_bias_brush_color', QColor(120, 160, 255)))
        if btn is not None:
            btn.setText('')
            btn.setStyleSheet(
                f"background-color: {color.name()}; border: 1px solid #888;")
            btn.setToolTip(color.name())
        info = getattr(self, '_bias_brush_color_info', None)
        if info is not None:
            info.setText(color.name())
            info.setToolTip(tr("bias_brush_independent_hint"))

    def _current_bias_brush_config(self) -> dict:
        # The bias brush follows the custom colour bias target, not the last
        # picked/normal brush colour.  The previous implementation introduced
        # an accidental dependency on the independent eyedropper colour.
        color = QColor(getattr(self, '_custom_color_bias_color',
                              getattr(self, '_bias_brush_color', QColor(120, 160, 255))))
        return {
            'rgb': (color.red(), color.green(), color.blue()),
            'strength': int(getattr(self, '_bias_brush_strength_slider', None).value()
                            if getattr(self, '_bias_brush_strength_slider', None) is not None else 80),
            'tone_range': str(getattr(self, '_bias_brush_tone_combo', None).currentData()
                              if getattr(self, '_bias_brush_tone_combo', None) is not None else 'all'),
            'protect_skin': bool(getattr(self, '_chk_bias_brush_protect_skin', None)
                                 and self._chk_bias_brush_protect_skin.isChecked()),
            'protect_lineart': bool(getattr(self, '_chk_bias_brush_protect_lineart', None)
                                    and self._chk_bias_brush_protect_lineart.isChecked()),
            'protect_saturated': bool(getattr(self, '_chk_bias_brush_protect_saturated', None)
                                      and self._chk_bias_brush_protect_saturated.isChecked()),
        }

    def _open_color_dialog(self, initial: QColor, title: str) -> QColor:
        """Use a Qt-owned dialog so Cancel and the title-bar close button always work.

        The native macOS colour panel can outlive the static ``getColor`` call or
        lose its parent modality after the editor rebuilds its central widget.
        A single non-native dialog instance has deterministic Accept/Reject
        semantics on macOS, Windows and Linux.
        """
        if self._active_color_dialog is not None:
            self._active_color_dialog.raise_()
            self._active_color_dialog.activateWindow()
            return QColor()
        dialog = QColorDialog(QColor(initial), self)
        self._active_color_dialog = dialog
        dialog.setWindowTitle(title)
        dialog.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog, True)
        dialog.setOption(QColorDialog.ColorDialogOption.ShowAlphaChannel, False)
        flags = dialog.windowFlags() | Qt.WindowType.WindowCloseButtonHint
        flags &= ~Qt.WindowType.WindowContextHelpButtonHint
        dialog.setWindowFlags(flags)
        dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        try:
            accepted = dialog.exec() == QDialog.DialogCode.Accepted
            return QColor(dialog.selectedColor()) if accepted else QColor()
        finally:
            dialog.close()
            dialog.deleteLater()
            self._active_color_dialog = None

    def _pick_custom_bias_color(self):
        color = self._open_color_dialog(
            self._custom_color_bias_color, tr("custom_color_bias_color"))
        if not color.isValid():
            return
        self._last_picked_rgb_raw = None
        # Deliberately one-way: editing the custom colour bias updates the
        # linked bias-brush colour, but must not rewrite the normal brush /
        # eyedropper selected colour.
        self._set_custom_bias_linked_color(color, update_shared_brush=False)

    def _update_custom_color_bias_swatch(self):
        btn = getattr(self, '_custom_color_bias_btn', None)
        if btn is None:
            return
        color = getattr(self, '_custom_color_bias_color', QColor(255, 180, 180))
        hex_color = color.name()
        btn.setText('')
        btn.setStyleSheet(f"background-color: {hex_color}; border: 1px solid #888;")
        info = getattr(self, '_custom_color_bias_info', None)
        if info is not None:
            info.setText(hex_color)
            info.setToolTip(hex_color)

    def _current_custom_color_bias(self) -> dict:
        color = QColor(getattr(self, '_custom_color_bias_color', QColor(255, 180, 180)))
        return {
            'enabled': bool(getattr(self, '_chk_custom_color_bias', None) and self._chk_custom_color_bias.isChecked()),
            'rgb': (color.red(), color.green(), color.blue()),
            'strength': int(getattr(self, '_custom_color_bias_slider', None).value() if getattr(self, '_custom_color_bias_slider', None) is not None else 35),
            'scope': str(getattr(self, '_custom_color_bias_scope', None).currentData() if getattr(self, '_custom_color_bias_scope', None) is not None else 'page'),
            'tone_range': str(getattr(self, '_custom_color_bias_tone_range', None).currentData() if getattr(self, '_custom_color_bias_tone_range', None) is not None else 'all'),
            'protect_skin': bool(getattr(self, '_chk_custom_bias_protect_skin', None) and self._chk_custom_bias_protect_skin.isChecked()),
            'protect_lineart': bool(getattr(self, '_chk_custom_bias_protect_lineart', None) and self._chk_custom_bias_protect_lineart.isChecked()),
            'protect_saturated': bool(getattr(self, '_chk_custom_bias_protect_saturated', None) and self._chk_custom_bias_protect_saturated.isChecked()),
        }

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

    def _picker_extract_hint_enabled(self) -> bool:
        checkbox = getattr(self, "_chk_picker_extract_hint", None)
        return False if checkbox is None else bool(checkbox.isChecked())

    @staticmethod
    def _representative_rgb_from_mask(image_bgr: np.ndarray, mask: np.ndarray,
                                      *, center_bgr: np.ndarray | None = None) -> tuple[int, int, int] | None:
        if image_bgr is None or mask is None:
            return None
        region = mask > 0
        if not np.any(region):
            return None
        pixels = image_bgr[region].reshape(-1, 3).astype(np.float32)
        if pixels.size == 0:
            return None
        brightness = pixels.mean(axis=1)
        usable = pixels[brightness > 20.0]
        if usable.shape[0] >= max(8, pixels.shape[0] // 5):
            pixels = usable
        if pixels.shape[0] > 32000:
            step = int(np.ceil(pixels.shape[0] / 32000.0))
            pixels = pixels[::step]
        if center_bgr is not None and pixels.shape[0] >= 12:
            center = center_bgr.astype(np.float32).reshape(1, 3)
            d = np.linalg.norm(pixels - center, axis=1)
            near = pixels[d <= max(16.0, float(np.percentile(d, 55)))]
            if near.shape[0] >= max(8, pixels.shape[0] // 6):
                pixels = near
        b, g, r = np.median(pixels, axis=0)
        return int(np.clip(r, 0, 255)), int(np.clip(g, 0, 255)), int(np.clip(b, 0, 255))

    @classmethod
    def _local_hint_color_from_image(cls, image_bgr: np.ndarray, ix: int, iy: int) -> tuple[int, int, int] | None:
        if image_bgr is None or image_bgr.ndim != 3:
            return None
        h, w = image_bgr.shape[:2]
        if not (0 <= ix < w and 0 <= iy < h):
            return None
        smooth = cv2.medianBlur(image_bgr, 5)
        mask = np.zeros((h + 2, w + 2), np.uint8)
        flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        try:
            cv2.floodFill(smooth, mask, (int(ix), int(iy)), 0,
                          loDiff=(14, 14, 14), upDiff=(14, 14, 14), flags=flags)
        except cv2.error:
            return None
        region = mask[1:-1, 1:-1] > 0
        if int(np.count_nonzero(region)) < 8:
            return None
        return cls._representative_rgb_from_mask(
            image_bgr, region, center_bgr=image_bgr[iy, ix])

    def _extract_hint_color_from_state(self, state, ix: int, iy: int,
                                       fallback_rgb: tuple[int, int, int] | None = None) -> tuple[int, int, int] | None:
        color_layer = getattr(state, "filter_base_bgr", None)
        if color_layer is None:
            color_layer = getattr(state, "result_bgr", None)
        if color_layer is None or getattr(state, "original_bgr", None) is None:
            return fallback_rgb
        h, w = color_layer.shape[:2]
        if not (0 <= ix < w and 0 <= iy < h):
            return fallback_rgb
        gap_close = self._gap_px_from_percent(self._gap_close_slider.value())
        region_map = state.hint_manager.bind_source_image(state.original_bgr, gap_close=gap_close)
        region_id = region_map.region_at(ix, iy, search_radius=max(4, gap_close + 2))
        if region_id > 0 and not region_map.is_background_region(region_id, max_area_ratio=0.45):
            binary = (region_map.labels == int(region_id)).astype(np.uint8)
            distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
            safe = (binary > 0) & (distance >= 1.25)
            if int(np.count_nonzero(safe)) < 12:
                safe = binary > 0
            picked = self._representative_rgb_from_mask(
                color_layer, safe, center_bgr=color_layer[iy, ix])
            if picked is not None:
                return picked
        picked = self._local_hint_color_from_image(color_layer, ix, iy)
        return picked if picked is not None else fallback_rgb

    def _apply_picked_color(self, rgb: tuple[int, int, int], *, remember_raw: bool = False):
        if remember_raw:
            self._last_picked_rgb_raw = tuple(int(np.clip(v, 0, 255)) for v in rgb)
        elif rgb is not None:
            self._last_picked_rgb_raw = None
        adjusted = self._adjust_picker_lightness(
            tuple(int(np.clip(v, 0, 255)) for v in rgb),
            self._picker_lightness_slider.value() if hasattr(self, '_picker_lightness_slider') else 0)
        self._set_shared_selected_color(QColor(*adjusted))

    def _on_picker_lightness_changed(self, value: int):
        if getattr(self, '_picker_lightness_spin', None) is not None and self._picker_lightness_spin.value() != value:
            self._picker_lightness_spin.setValue(value)
        if self._last_picked_rgb_raw is not None:
            self._apply_picked_color(self._last_picked_rgb_raw, remember_raw=True)

    def _pick_color(self):
        color = self._open_color_dialog(self._brush_color, tr("pick_color_title"))
        if color.isValid():
            self._last_picked_rgb_raw = None
            self._set_shared_selected_color(color)

    def _update_color_swatch(self):
        if getattr(self, "_color_swatch", None) is None:
            return
        info = getattr(self, '_current_color_info', None)
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

    def _on_color_picked(self, rgb: tuple, x_norm: float, y_norm: float):
        picked = tuple(int(np.clip(v, 0, 255)) for v in rgb)
        self._apply_picked_color(picked, remember_raw=True)

        wrote_hint = False
        if self._picker_extract_hint_enabled():
            state = self._current_state()
            if state is not None and getattr(state, "result_bgr", None) is not None:
                gap_close = self._gap_px_from_percent(self._gap_close_slider.value())
                state.hint_manager.bind_source_image(state.original_bgr, gap_close=gap_close)
                width = max(1, state.original_bgr.shape[1])
                brush_px = self._brush_px_from_percent(self._brush_slider.value())
                hint_radius_px = max(2, int(round(brush_px * 0.55)))
                radius_norm = float(np.clip(hint_radius_px / max(1, width), 0.0015, 0.05))
                hint_strength = (self._manual_strength_slider.value() / 100.0
                                 if getattr(self, '_manual_strength_slider', None) is not None else 1.0)
                wrote_hint = state.hint_manager.add_eyedropper_hint(
                    float(x_norm), float(y_norm), picked, radius_norm,
                    strength=hint_strength)
                if wrote_hint:
                    self._refresh_hint_overlay()
        if wrote_hint:
            self.statusBar().showMessage(tr("picker_model_hint_written"), 3500)
        elif self._picker_extract_hint_enabled():
            self.statusBar().showMessage(tr("picker_model_hint_requires_color"), 3500)
        else:
            self.statusBar().showMessage(tr("picker_keep_tool_hint"), 2500)

    def _on_brush_stroke_started(self):
        """Create one undo checkpoint for the whole visible brush stroke."""
        state = self._current_state()
        self._local_brush_stroke_active = bool(
            state is not None and state.result_bgr is not None)
        self._brush_changed_during_stroke = False
        if self._local_brush_stroke_active:
            state.push_undo()

    def _on_hint_dab_added(self, x_norm: float, y_norm: float, rgb: tuple,
                           radius_norm: float):
        state = self._current_state()
        if state is None:
            return

        # Before the first auto-colorize, brush dabs remain model hints.
        if state.result_bgr is None:
            state.push_undo()
            changed = state.hint_manager.add_manual_hint(
                x_norm, y_norm, rgb, radius_norm)
            if not changed:
                state.discard_unchanged_undo()
            self._refresh_hint_overlay()
            return

        if not self._local_brush_stroke_active:
            self._local_brush_stroke_active = True
            self._brush_changed_during_stroke = False
            state.push_undo()

        if (getattr(self, '_chk_brush_model_hint', None) is not None
                and self._chk_brush_model_hint.isChecked()):
            gap_close = self._gap_px_from_percent(self._gap_close_slider.value())
            state.hint_manager.bind_source_image(
                state.original_bgr, gap_close=gap_close)
            strength = self._manual_strength_slider.value() / 100.0
            # Model Hint means an exact user colour instruction. Do not run it
            # through the optional manual-style adapter, which is appropriate
            # for direct paint/bucket edits but can shift a requested blue/red
            # before mc-v2 ever sees it.
            paint_rgb = tuple(int(np.clip(v, 0, 255)) for v in rgb)
            changed = state.hint_manager.add_manual_hint(
                x_norm, y_norm, paint_rgb, radius_norm,
                source="manual", strength=strength)
            self._brush_changed_during_stroke |= changed
            if changed:
                self._refresh_hint_overlay()
            return

        h, w = state.result_bgr.shape[:2]
        ix = min(w - 1, max(0, int(round(x_norm * (w - 1)))))
        iy = min(h - 1, max(0, int(round(y_norm * (h - 1)))))
        radius_px = max(1, int(round(radius_norm * w)))
        strength = self._manual_strength_slider.value() / 100.0
        paint_rgb = self._manual_target_rgb(rgb)
        fill_mode = self._current_fill_mode()

        # The V5 brush has one predictable path: paint the exact live corridor.
        # It never waits for mouse release, expands into a region, or snaps to
        # line art. Solid ink pixels are still protected by apply_brush_edit.
        state.result_bgr, state.filter_base_bgr, _mask, changed = apply_brush_edit(
            state.original_bgr, state.result_bgr, state.filter_base_bgr,
            ix, iy, radius_px, paint_rgb, opacity=min(1.0, strength),
            region_map=None, gap_close=0, mode=fill_mode,
            snap_to_lineart=False, pupil_blend=False)
        self._brush_changed_during_stroke |= changed

        if changed:
            self._radio_view_edited.setChecked(True)
            if hasattr(self._canvas, "update_image_pixels"):
                self._canvas.update_image_pixels(state.result_bgr)
            else:
                self._sync_view_after_edit(state)

    def _on_brush_stroke_finished(self):
        if not self._local_brush_stroke_active:
            return
        state = self._current_state()
        self._local_brush_stroke_active = False
        self._canvas.clear_dabs()
        if state is None or state.result_bgr is None:
            self._brush_changed_during_stroke = False
            return

        state.discard_unchanged_undo()
        if self._brush_changed_during_stroke:
            self._sync_bias_reference_to_current(state)
        self._radio_view_edited.setChecked(True)
        self._sync_view_after_edit(state)
        if (getattr(self, '_chk_brush_model_hint', None) is not None
                and self._chk_brush_model_hint.isChecked()
                and self._brush_changed_during_stroke):
            message = tr("brush_model_hint_done")
        else:
            message = (tr("local_brush_done") if self._brush_changed_during_stroke
                       else tr("manual_edit_no_change"))
        self.statusBar().showMessage(message, 3000)
        self._brush_changed_during_stroke = False

    def _sync_bias_reference_to_current(self, state: "PageState" | None):
        if state is None or state.result_bgr is None:
            if state is not None:
                state.bias_brush_reference_bgr = None
                state.bias_brush_reference_filter_bgr = None
            return
        state.bias_brush_reference_bgr = state.result_bgr.copy()
        state.bias_brush_reference_filter_bgr = (
            state.filter_base_bgr.copy() if state.filter_base_bgr is not None
            else state.result_bgr.copy())

    def _ensure_bias_reference_exists(self, state: "PageState" | None):
        if state is None or state.result_bgr is None:
            return
        ref = getattr(state, 'bias_brush_reference_bgr', None)
        ref_filter = getattr(state, 'bias_brush_reference_filter_bgr', None)
        if ref is None or ref.shape[:2] != state.result_bgr.shape[:2]:
            state.bias_brush_reference_bgr = state.result_bgr.copy()
        if ref_filter is None or ref_filter.shape[:2] != state.result_bgr.shape[:2]:
            state.bias_brush_reference_filter_bgr = (
                state.filter_base_bgr.copy() if state.filter_base_bgr is not None
                else state.result_bgr.copy())

    def _reset_bias_brush_runtime(self):
        self._bias_brush_stroke_active = False
        self._bias_brush_changed_during_stroke = False
        self._bias_brush_alpha = None
        self._bias_brush_base_result = None
        self._bias_brush_base_filter = None
        self._bias_brush_candidate_result = None
        self._bias_brush_candidate_filter = None
        self._bias_brush_last_preview_ts = 0.0
        self._bias_brush_last_radius_px = 18

    def _on_bias_brush_stroke_started(self):
        """Prepare one independent colour-bias candidate for this stroke."""
        self._reset_bias_brush_runtime()
        state = self._current_state()
        if state is None or state.result_bgr is None:
            self.statusBar().showMessage(tr("bias_brush_requires_result"), 3500)
            return
        self._ensure_bias_reference_exists(state)
        state.push_undo()
        try:
            from core.bias_brush import empty_stroke_alpha, prepare_bias_candidate
            import time
            self._bias_brush_base_result = state.result_bgr.copy()
            self._bias_brush_base_filter = (
                state.filter_base_bgr.copy() if state.filter_base_bgr is not None
                else state.result_bgr.copy())
            config = self._current_bias_brush_config()
            self._bias_brush_candidate_result = prepare_bias_candidate(
                self._bias_brush_base_result, state.original_bgr, config)
            if np.array_equal(self._bias_brush_base_filter,
                              self._bias_brush_base_result):
                self._bias_brush_candidate_filter = (
                    self._bias_brush_candidate_result.copy())
            else:
                self._bias_brush_candidate_filter = prepare_bias_candidate(
                    self._bias_brush_base_filter, state.original_bgr, config)
            self._bias_brush_alpha = empty_stroke_alpha(state.result_bgr.shape[:2])
            self._bias_brush_stroke_active = True
            self._bias_brush_last_preview_ts = time.monotonic()
            self._bias_brush_last_radius_px = int(self._brush_px_from_percent(self._bias_brush_size_slider.value())) if getattr(self, '_bias_brush_size_slider', None) is not None else 18
        except Exception as exc:  # noqa: BLE001
            state.discard_unchanged_undo()
            self._reset_bias_brush_runtime()
            self.statusBar().showMessage(
                tr("bias_brush_failed").format(message=f"{type(exc).__name__}: {exc}"),
                6000)

    def _on_bias_brush_dab_added(self, x_norm: float, y_norm: float,
                                 radius_norm: float):
        if not self._bias_brush_stroke_active:
            return
        state = self._current_state()
        if state is None or state.result_bgr is None:
            return
        try:
            import time
            from core.bias_brush import (
                add_soft_round_dab_inplace,
                composite_bias_candidate_roi,
            )
            h, w = state.result_bgr.shape[:2]
            ix = min(w - 1, max(0, int(round(float(x_norm) * max(0, w - 1)))))
            iy = min(h - 1, max(0, int(round(float(y_norm) * max(0, h - 1)))))
            radius_px = max(1, int(round(float(radius_norm) * max(1, w))))
            self._bias_brush_last_radius_px = radius_px
            self._bias_brush_alpha, roi = add_soft_round_dab_inplace(
                self._bias_brush_alpha, ix, iy, radius_px)
            if roi is None:
                return
            state.result_bgr, changed_result = composite_bias_candidate_roi(
                self._bias_brush_base_result,
                self._bias_brush_candidate_result,
                self._bias_brush_alpha,
                roi,
                dst=state.result_bgr)
            state.filter_base_bgr, changed_filter = composite_bias_candidate_roi(
                self._bias_brush_base_filter,
                self._bias_brush_candidate_filter,
                self._bias_brush_alpha,
                roi,
                dst=state.filter_base_bgr)
            changed = bool(changed_result or changed_filter)
            self._bias_brush_changed_during_stroke |= changed
            if changed:
                self._radio_view_edited.setChecked(True)
                now = time.monotonic()
                if (now - float(getattr(self, '_bias_brush_last_preview_ts', 0.0)) >=
                        float(getattr(self, '_bias_brush_preview_interval', 1.0 / 30.0))):
                    self._bias_brush_last_preview_ts = now
                    if hasattr(self._canvas, "update_image_pixels"):
                        self._canvas.update_image_pixels(state.result_bgr)
                    else:
                        self._sync_view_after_edit(state)
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(
                tr("bias_brush_failed").format(message=f"{type(exc).__name__}: {exc}"),
                6000)

    def _on_bias_brush_stroke_finished(self):
        self._canvas.clear_dabs()
        state = self._current_state()
        active = self._bias_brush_stroke_active
        changed = self._bias_brush_changed_during_stroke
        if active and state is not None and state.result_bgr is not None:
            try:
                # Final-quality pass: recompute the complete accumulated stroke
                # with the exact pre-optimization full-page compositor.  The
                # ROI path above is preview-only; this guarantees that soft
                # edges, overlapping dabs and colour strength are byte-identical
                # to the earlier higher-quality bias-brush implementation.
                from core.bias_brush import (
                    build_cohesive_stroke_alpha,
                    composite_bias_candidate,
                )
                if (self._bias_brush_base_result is not None and
                        self._bias_brush_candidate_result is not None and
                        self._bias_brush_alpha is not None):
                    final_alpha = build_cohesive_stroke_alpha(
                        self._bias_brush_alpha,
                        state.original_bgr,
                        self._bias_brush_base_result,
                        brush_radius_px=int(getattr(self, '_bias_brush_last_radius_px', 18)))
                    state.result_bgr, final_result_changed = composite_bias_candidate(
                        self._bias_brush_base_result,
                        self._bias_brush_candidate_result,
                        final_alpha)
                    state.filter_base_bgr, final_filter_changed = composite_bias_candidate(
                        self._bias_brush_base_filter,
                        self._bias_brush_candidate_filter,
                        final_alpha)
                    changed = bool(final_result_changed or final_filter_changed)
            except Exception as exc:  # noqa: BLE001
                self.statusBar().showMessage(
                    tr("bias_brush_failed").format(
                        message=f"{type(exc).__name__}: {exc}"), 6000)
            state.discard_unchanged_undo()
            self._radio_view_edited.setChecked(True)
            self._sync_view_after_edit(state)
        self._reset_bias_brush_runtime()
        if not active:
            return
        self.statusBar().showMessage(
            tr("bias_brush_done") if changed else tr("manual_edit_no_change"),
            3000)

    def _reset_bias_eraser_runtime(self):
        self._bias_eraser_stroke_active = False
        self._bias_eraser_changed_during_stroke = False
        self._bias_eraser_alpha = None
        self._bias_eraser_base_result = None
        self._bias_eraser_base_filter = None
        self._bias_eraser_candidate_result = None
        self._bias_eraser_candidate_filter = None
        self._bias_eraser_last_preview_ts = 0.0
        self._bias_eraser_last_radius_px = 18

    def _on_bias_eraser_stroke_started(self):
        self._reset_bias_eraser_runtime()
        state = self._current_state()
        if state is None or state.result_bgr is None:
            self.statusBar().showMessage(tr("bias_brush_requires_result"), 3500)
            return
        if getattr(state, 'bias_brush_reference_bgr', None) is None:
            self.statusBar().showMessage(tr("bias_eraser_requires_reference"), 3500)
            return
        state.push_undo()
        try:
            from core.bias_brush import empty_stroke_alpha
            import time
            self._bias_eraser_base_result = state.result_bgr.copy()
            self._bias_eraser_base_filter = (
                state.filter_base_bgr.copy() if state.filter_base_bgr is not None
                else state.result_bgr.copy())
            self._bias_eraser_candidate_result = state.bias_brush_reference_bgr.copy()
            ref_filter = getattr(state, 'bias_brush_reference_filter_bgr', None)
            if ref_filter is not None and ref_filter.shape[:2] == state.result_bgr.shape[:2]:
                self._bias_eraser_candidate_filter = ref_filter.copy()
            else:
                self._bias_eraser_candidate_filter = state.bias_brush_reference_bgr.copy()
            self._bias_eraser_alpha = empty_stroke_alpha(state.result_bgr.shape[:2])
            self._bias_eraser_stroke_active = True
            self._bias_eraser_last_preview_ts = time.monotonic()
            self._bias_eraser_last_radius_px = int(self._brush_px_from_percent(self._bias_eraser_size_slider.value())) if getattr(self, '_bias_eraser_size_slider', None) is not None else 18
        except Exception as exc:  # noqa: BLE001
            state.discard_unchanged_undo()
            self._reset_bias_eraser_runtime()
            self.statusBar().showMessage(
                tr("bias_brush_failed").format(message=f"{type(exc).__name__}: {exc}"),
                6000)

    def _on_bias_eraser_dab_added(self, x_norm: float, y_norm: float,
                                  radius_norm: float):
        if not self._bias_eraser_stroke_active:
            return
        state = self._current_state()
        if state is None or state.result_bgr is None:
            return
        try:
            import time
            from core.bias_brush import add_soft_round_dab_inplace, composite_bias_candidate_roi
            h, w = state.result_bgr.shape[:2]
            ix = min(w - 1, max(0, int(round(float(x_norm) * max(0, w - 1)))))
            iy = min(h - 1, max(0, int(round(float(y_norm) * max(0, h - 1)))))
            radius_px = max(1, int(round(float(radius_norm) * max(1, w))))
            self._bias_eraser_last_radius_px = radius_px
            self._bias_eraser_alpha, roi = add_soft_round_dab_inplace(
                self._bias_eraser_alpha, ix, iy, radius_px)
            if roi is None:
                return
            state.result_bgr, changed_result = composite_bias_candidate_roi(
                self._bias_eraser_base_result,
                self._bias_eraser_candidate_result,
                self._bias_eraser_alpha,
                roi,
                dst=state.result_bgr)
            state.filter_base_bgr, changed_filter = composite_bias_candidate_roi(
                self._bias_eraser_base_filter,
                self._bias_eraser_candidate_filter,
                self._bias_eraser_alpha,
                roi,
                dst=state.filter_base_bgr)
            changed = bool(changed_result or changed_filter)
            self._bias_eraser_changed_during_stroke |= changed
            if changed:
                self._radio_view_edited.setChecked(True)
                now = time.monotonic()
                if (now - float(getattr(self, '_bias_eraser_last_preview_ts', 0.0)) >=
                        float(getattr(self, '_bias_eraser_preview_interval', 1.0 / 30.0))):
                    self._bias_eraser_last_preview_ts = now
                    if hasattr(self._canvas, 'update_image_pixels'):
                        self._canvas.update_image_pixels(state.result_bgr)
                    else:
                        self._sync_view_after_edit(state)
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(
                tr("bias_brush_failed").format(message=f"{type(exc).__name__}: {exc}"),
                6000)

    def _on_bias_eraser_stroke_finished(self):
        self._canvas.clear_dabs()
        state = self._current_state()
        active = self._bias_eraser_stroke_active
        changed = self._bias_eraser_changed_during_stroke
        if active and state is not None and state.result_bgr is not None:
            try:
                from core.bias_brush import composite_bias_candidate
                if (self._bias_eraser_base_result is not None and
                        self._bias_eraser_candidate_result is not None and
                        self._bias_eraser_alpha is not None):
                    state.result_bgr, final_result_changed = composite_bias_candidate(
                        self._bias_eraser_base_result,
                        self._bias_eraser_candidate_result,
                        self._bias_eraser_alpha)
                    state.filter_base_bgr, final_filter_changed = composite_bias_candidate(
                        self._bias_eraser_base_filter,
                        self._bias_eraser_candidate_filter,
                        self._bias_eraser_alpha)
                    changed = bool(final_result_changed or final_filter_changed)
            except Exception as exc:  # noqa: BLE001
                self.statusBar().showMessage(
                    tr("bias_brush_failed").format(
                        message=f"{type(exc).__name__}: {exc}"), 6000)
            state.discard_unchanged_undo()
            self._radio_view_edited.setChecked(True)
            self._sync_view_after_edit(state)
        self._reset_bias_eraser_runtime()
        if not active:
            return
        self.statusBar().showMessage(
            tr("bias_eraser_done") if changed else tr("manual_edit_no_change"),
            3000)

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

    def _clear_eyedropper_model_hints(self):
        state = self._current_state()
        if state is None:
            return
        state.push_undo()
        removed = state.hint_manager.clear_manual_hints_by_source("eyedropper_hint")
        if removed <= 0:
            state.discard_unchanged_undo()
            self.statusBar().showMessage(tr("picker_no_model_hint"), 3000)
            return
        self._refresh_hint_overlay()
        self.statusBar().showMessage(tr("picker_model_hint_cleared").format(count=removed), 3000)

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
        fill_mode = self._current_fill_mode()
        region_map = state.hint_manager.bind_source_image(
            state.original_bgr, gap_close=gap_close)
        mask = build_region_edit_mask(
            state.original_bgr, state.result_bgr, ix, iy,
            gap_close=gap_close, region_map=region_map)
        if not mask.any():
            self.statusBar().showMessage(tr("no_fill_area"), 5000)
            return

        state.push_undo()
        new_img, new_base, _mask, changed = apply_region_edit(
            state.original_bgr, state.result_bgr, state.filter_base_bgr,
            ix, iy, hex_color, gap_close=gap_close, mode=fill_mode,
            feather=2 if fill_mode != "flat" else 0, region_map=region_map)
        state.result_bgr = new_img
        state.filter_base_bgr = new_base
        if changed:
            self._sync_bias_reference_to_current(state)
        message = tr("region_fill_done") if changed else tr("manual_edit_no_change")
        state.discard_unchanged_undo()
        self._radio_view_edited.setChecked(True)
        self._sync_view_after_edit(state)
        self.statusBar().showMessage(message, 3000)

    def _apply_selection_fill_mask(self, selection_mask: np.ndarray,
                                   selection_kind: str | None = None):
        state = self._current_state()
        if state is None:
            return
        if state.result_bgr is None:
            QMessageBox.information(self, tr("no_result_title"), tr("no_result_body"))
            return
        paint_rgb = self._manual_target_rgb((self._brush_color.red(), self._brush_color.green(), self._brush_color.blue()))
        hex_color = '#%02x%02x%02x' % paint_rgb
        requested_closed_only = bool(
            getattr(self, '_chk_selection_closed_only', None)
            and self._chk_selection_closed_only.isChecked())
        already_closed = str(selection_kind or '').endswith('_closed')
        feather = int(getattr(self, '_selection_feather_slider', None).value()
                      if getattr(self, '_selection_feather_slider', None) is not None else 2)
        fill_mode = self._current_fill_mode()
        # Rectangle closed-fill is filtered at mouse-release time and the exact
        # resulting mask is previewed. Other selection routes are filtered here.
        closed_expand_px = int(
            getattr(self, '_selection_closed_expand_slider', None).value()
            if getattr(self, '_selection_closed_expand_slider', None) is not None else 0)
        closed_min_area = int(
            getattr(self, '_selection_closed_min_area_slider', None).value()
            if getattr(self, '_selection_closed_min_area_slider', None) is not None else 6)
        closed_min_thickness = int(
            getattr(self, '_selection_closed_min_thickness_slider', None).value()
            if getattr(self, '_selection_closed_min_thickness_slider', None) is not None else 3)
        mask = (np.where(selection_mask > 0, 255, 0).astype(np.uint8)
                if already_closed else build_selection_edit_mask(
                    state.original_bgr, selection_mask,
                    closed_only=requested_closed_only,
                    expand_px=closed_expand_px,
                    min_area=closed_min_area,
                    min_thickness=closed_min_thickness))
        if not mask.any():
            self.statusBar().showMessage(
                tr("selection_fill_no_closed")
                if requested_closed_only or already_closed
                else tr("selection_fill_empty"), 5000)
            return
        state.push_undo()
        # Pass the authoritative mask directly. Re-running with the original
        # rectangle here was the path that could silently fall back to full-box
        # colouring when UI state changed between drawing and confirmation.
        new_img, new_base, _mask, changed = apply_selection_edit(
            state.original_bgr, state.result_bgr, state.filter_base_bgr,
            mask, hex_color, feather=feather, closed_only=False,
            mode=fill_mode)
        state.result_bgr = new_img
        state.filter_base_bgr = new_base
        if changed:
            self._sync_bias_reference_to_current(state)
        state.discard_unchanged_undo()
        self._radio_view_edited.setChecked(True)
        self._sync_view_after_edit(state)
        self.statusBar().showMessage(tr('selection_confirmed') if changed else tr('manual_edit_no_change'), 3000)

    def _on_polygon_fill_requested(self, points):
        state = self._current_state()
        if state is None or state.result_bgr is None:
            if state is None:
                return
            QMessageBox.information(self, tr("no_result_title"), tr("no_result_body"))
            self._cancel_pending_selection_fill()
            return
        mask = build_polygon_selection_mask(state.result_bgr.shape[:2], list(points))
        mask = self._snap_selection_to_lineart(mask)
        self._clear_closed_preview_cache()
        self._set_pending_selection(mask, 'polygon')

    def _on_rect_fill_requested(self, x1: int, y1: int, x2: int, y2: int):
        state = self._current_state()
        if state is None or state.result_bgr is None:
            if state is None:
                return
            QMessageBox.information(self, tr("no_result_title"), tr("no_result_body"))
            self._cancel_pending_selection_fill()
            return
        raw_mask = build_rect_selection_mask(
            state.result_bgr.shape[:2], x1, y1, x2, y2)
        closed_only = bool(
            getattr(self, '_chk_selection_closed_only', None)
            and self._chk_selection_closed_only.isChecked())
        if closed_only:
            # Closed rectangles use the integrated MangaLineExtraction model on
            # the original B/W crop.  Running download/load/inference in a
            # worker keeps the Qt UI responsive and prevents a missing model
            # from silently falling back to a full-box mask.
            if (self._line_extract_worker is not None
                    and self._line_extract_worker.isRunning()):
                self.statusBar().showMessage("漫画结构线正在识别，请稍候…", 3500)
                return
            self._clear_closed_preview_cache()
            self._closed_preview_base_mask = (
                None if self._pending_selection_mask is None else
                self._pending_selection_mask.copy())
            self._closed_preview_combine_mode = self._current_selection_mode()
            self._line_extract_page_path = self._current_path
            self._line_extract_raw_mask = raw_mask.copy()
            worker = MangaLineExtractionWorker(
                state.original_bgr, raw_mask, parent=self)
            self._line_extract_worker = worker
            worker.status.connect(self._on_worker_status)
            worker.finished_ok.connect(self._on_manga_line_finished)
            worker.finished_err.connect(self._on_manga_line_failed)
            worker.finished.connect(self._on_manga_line_thread_finished)
            self._update_selection_buttons(False)
            self.statusBar().showMessage("正在识别矩形内的漫画结构线…", 0)
            worker.start()
        else:
            self._clear_closed_preview_cache()
            self._set_pending_selection(raw_mask, 'rect')

    def _on_manga_line_thread_finished(self):
        worker = self._line_extract_worker
        self._line_extract_worker = None
        self._line_extract_page_path = None
        self._line_extract_raw_mask = None
        if worker is not None:
            worker.deleteLater()

    def _on_manga_line_finished(self, inference):
        page_path = self._line_extract_page_path
        raw_mask = self._line_extract_raw_mask
        try:
            if page_path is None or raw_mask is None:
                return
            state = self._pages.get(page_path)
            if state is None or page_path != self._current_path:
                self.statusBar().showMessage("页面已切换，已丢弃线条识别结果", 3500)
                return
            probability = getattr(inference, "probability", None)
            if probability is None:
                raise RuntimeError("漫画结构线模型没有返回概率图")
            self._closed_preview_page_path = page_path
            self._closed_preview_raw_mask = raw_mask.copy()
            self._closed_preview_probability = np.asarray(
                probability, dtype=np.float32).copy()
            self._closed_preview_device = str(getattr(inference, "device", "") or "")
            if not self._refresh_closed_selection_preview():
                self._clear_closed_preview_cache()
                return
        except Exception as exc:  # noqa: BLE001
            self._cancel_pending_selection_fill()
            QMessageBox.warning(
                self, "漫画线条识别失败",
                f"无法生成闭合区域：\n{type(exc).__name__}: {exc}")

    def _on_manga_line_failed(self, message: str):
        self._cancel_pending_selection_fill()
        self.statusBar().showMessage("漫画结构线识别失败", 5000)
        QMessageBox.warning(
            self, "漫画线条识别失败",
            "MangaLineExtraction 模型未能完成识别，因此没有使用普通矩形"
            "作为替代，避免再次整框上色。\n\n" + str(message))

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
        state.bias_brush_reference_bgr = None
        state.bias_brush_reference_filter_bgr = None
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
        self._sync_bias_reference_to_current(state)
        self._radio_view_edited.setChecked(True)
        self._sync_view_after_edit(state)
        self.statusBar().showMessage(tr("restored_ai"), 3000)

    def _on_view_mode_changed(self):
        state = self._current_state()
        if state is not None:
            self._sync_view_after_edit(state)

    def _sync_view_after_edit(self, state: "PageState"):
        """Refresh the current layer and optional non-destructive B&W preview."""
        if self._radio_view_original.isChecked():
            image = state.original_bgr
        elif self._radio_view_ai.isChecked() and state.ai_result_bgr is not None:
            image = state.ai_result_bgr
        elif state.result_bgr is not None:
            image = state.result_bgr
        else:
            image = state.original_bgr

        mask = self._pending_selection_mask
        preview_on = bool(
            mask is not None and np.any(mask)
            and getattr(self, '_chk_selection_bw_preview', None) is not None
            and self._chk_selection_bw_preview.isChecked()
            and not self._radio_view_original.isChecked())
        if preview_on:
            from core.local_model_recolor import preview_black_and_white
            image = preview_black_and_white(image, state.original_bgr, mask)
        self._canvas.set_image(image, fit=False)
        self._refresh_hint_overlay()
        if mask is not None and np.any(mask):
            self._canvas.set_selection_mask_overlay(
                mask, mode=self._current_selection_mode())

    def _refresh_hint_overlay(self):
        state = self._current_state()
        if state is None or not hasattr(self, "_canvas"):
            return
        show_regions = False
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
