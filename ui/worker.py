"""Background worker — runs the (slow) Auto pipeline off the UI thread."""

from __future__ import annotations

import traceback

import numpy as np
from PySide6.QtCore import QThread, Signal

from core.hint_manager import HintManager
from pipeline import colorize_page


class ColorizeWorker(QThread):
    """Runs colorize_page() in a background thread.

    Emits `finished_ok(result_bgr)` on success, `finished_err(message)` on
    failure, and `status(message)` for progress text (e.g. first-run model
    downloads) — never touches Qt widgets directly, the caller updates UI
    from the connected slots (Qt marshals queued-connection signals back
    onto the main thread automatically).
    """

    finished_ok = Signal(object)
    finished_err = Signal(str)
    status = Signal(str)

    def __init__(self, image_bgr: np.ndarray, hint_manager: HintManager,
                 regenerate_auto: bool, parent=None,
                 style_key: str | None = None, quality_key: str | None = None,
                 character_memories: dict | None = None,
                 character_library=None, scene_palette=None,
                 style_strength: float = 1.0,
                 reference_strength: float = 1.0,
                 manual_strength: float = 1.0,
                 pastel_tuning: dict | None = None,
                 filter_tuning: dict | None = None,
                 custom_color_bias: dict | None = None,
                 forced_matches: dict[int, int] | None = None):
        super().__init__(parent)
        self._character_library = character_library
        self._scene_palette = scene_palette
        self._image_bgr = image_bgr
        self._hint_manager = hint_manager
        self._regenerate_auto = regenerate_auto
        self._style_key = style_key
        self._quality_key = "draft"
        self._character_memories = character_memories
        self._style_strength = style_strength
        self._reference_strength = reference_strength
        self._manual_strength = manual_strength
        self._pastel_tuning = dict(pastel_tuning or {})
        self._filter_tuning = dict(filter_tuning or {})
        self._custom_color_bias = dict(custom_color_bias or {})
        self._forced_matches = dict(forced_matches or {})

    def run(self):
        try:
            from config import Config
            from core.model_downloader import ensure_models_downloaded
            ensure_models_downloaded(Config.WEIGHTS_DIR, callback=self.status.emit)

            self.status.emit("正在上色...")
            result = colorize_page(
                self._image_bgr,
                hint_manager=self._hint_manager,
                regenerate_auto=self._regenerate_auto,
                style_key=self._style_key,
                quality_key=self._quality_key,
                character_memories=self._character_memories,
                character_library=self._character_library,
                scene_palette=self._scene_palette,
                style_strength=self._style_strength,
                reference_strength=self._reference_strength,
                manual_strength=self._manual_strength,
                pastel_tuning=self._pastel_tuning,
                filter_tuning=self._filter_tuning,
                custom_color_bias=self._custom_color_bias,
                forced_matches=self._forced_matches,
                return_filter_base=True,
            )
            self.finished_ok.emit(result)
        except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
            traceback.print_exc()
            self.finished_err.emit(f"{type(exc).__name__}: {exc}")


class BatchColorizeWorker(QThread):
    """Runs colorize_page() over several selected pages, one at a time.

    `pages` is a list of (path, image_bgr, hint_manager[, forced_matches]) tuples. Emits
    `page_done(path, result_bgr)` as each page finishes (so the UI can
    update that page's thumbnail/state immediately), `page_error(path,
    message)` if one page fails (batch continues with the rest), and
    `finished_all()` once every page has been attempted.
    """

    page_done = Signal(str, object)
    page_error = Signal(str, str)
    status = Signal(str)
    finished_all = Signal()

    def __init__(self, pages: list,
                 regenerate_auto: bool, parent=None,
                 style_key: str | None = None, quality_key: str | None = None,
                 character_memories: dict | None = None,
                 character_library=None, scene_palette=None,
                 skip_colored: bool = True,
                 protect_text: bool = True,
                 style_strength: float = 1.0,
                 reference_strength: float = 1.0,
                 manual_strength: float = 1.0,
                 pastel_tuning: dict | None = None,
                 filter_tuning: dict | None = None,
                 custom_color_bias: dict | None = None):
        super().__init__(parent)
        self._pages = pages
        self._regenerate_auto = regenerate_auto
        # Manga-Colorization-FJ style: skip pages that are already in color.
        self._skip_colored = skip_colored
        self._protect_text = bool(protect_text)
        self._character_library = character_library
        self._scene_palette = scene_palette
        self._style_key = style_key
        self._quality_key = "draft"
        # Same dict (same CharacterMemory instances) reused across every
        # page in the batch, so character slots learned on page 1 carry
        # over to page 50 — this is what makes CharacterMemory book-level
        # rather than per-page.
        self._character_memories = character_memories
        self._style_strength = style_strength
        self._reference_strength = reference_strength
        self._manual_strength = manual_strength
        self._pastel_tuning = dict(pastel_tuning or {})
        self._filter_tuning = dict(filter_tuning or {})
        self._custom_color_bias = dict(custom_color_bias or {})

    def run(self):
        """Run the batch and *always* release the UI busy state.

        Earlier versions imported/downloader-initialized outside the guarded
        block.  Any import or first-run model error could terminate the thread
        before ``finished_all`` was emitted, leaving the Auto button disabled
        and making the application look unresponsive.
        """
        try:
            from config import Config
            from core.model_downloader import ensure_models_downloaded
            self.status.emit("正在检查 mc-v2 模型文件...")
            ensure_models_downloaded(Config.WEIGHTS_DIR, callback=self.status.emit)

            # Load the large model once while a visible status is shown.  The
            # following colorize_page() calls reuse the cached instance.
            self.status.emit("正在加载上色模型，首次运行可能需要一些时间...")
            from pipeline import get_colorizer
            get_colorizer(Config, ensure_weights=False)

            total = len(self._pages)
            for i, item in enumerate(self._pages, start=1):
                path, image_bgr, hint_manager = item[:3]
                forced_matches = dict(item[3] or {}) if len(item) > 3 else {}
                self.status.emit(f"正在读取并上色 ({i}/{total})：{path.split('/')[-1]}")
                try:
                    if image_bgr is None:
                        from core.imageio import imread as _uimread
                        image_bgr = _uimread(path)
                        if image_bgr is None:
                            raise RuntimeError(f"无法读取图片：{path}")
                    if self._skip_colored:
                        from core.color_detect import is_already_colored
                        if is_already_colored(image_bgr):
                            self.status.emit(
                                f"检测为彩色页，已跳过 ({i}/{total})：{path.split('/')[-1]}")
                            self.page_done.emit(path, image_bgr.copy())
                            continue
                    result = colorize_page(
                        image_bgr,
                        hint_manager=hint_manager,
                        regenerate_auto=self._regenerate_auto,
                        style_key=self._style_key,
                        quality_key=self._quality_key,
                                character_memories=self._character_memories,
                        character_library=self._character_library,
                        scene_palette=self._scene_palette,
                        style_strength=self._style_strength,
                        reference_strength=self._reference_strength,
                        manual_strength=self._manual_strength,
                        pastel_tuning=self._pastel_tuning,
                        filter_tuning=self._filter_tuning,
                        custom_color_bias=self._custom_color_bias,
                        forced_matches=forced_matches,
                        return_filter_base=True,
                        protect_text=self._protect_text,
                    )
                    self.page_done.emit(path, result)
                except Exception as exc:  # noqa: BLE001 — keep going per page
                    traceback.print_exc()
                    self.page_error.emit(path, f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # covers imports, downloads and model load
            traceback.print_exc()
            message = f"初始化上色模型失败：{type(exc).__name__}: {exc}"
            for item in self._pages:
                self.page_error.emit(item[0], message)
        finally:
            self.finished_all.emit()


class LocalModelRecolorWorker(QThread):
    """Run full-page mc-v2 and merge only the selected region.

    Inputs are copied on construction so the UI can keep displaying and the
    user can switch pages without the worker reading mutable page state.
    """

    finished_ok = Signal(object)
    finished_err = Signal(str)
    status = Signal(str)

    def __init__(self, original_bgr: np.ndarray, current_result_bgr: np.ndarray,
                 current_ai_result_bgr: np.ndarray | None,
                 current_filter_base_bgr: np.ndarray | None,
                 selection_mask: np.ndarray, hint_manager: HintManager,
                 *, feather: int = 4, hint_margin_px: int = 16,
                 gap_close: int = 4, only_selection_hints: bool = True,
                 classic_point_hints: bool = False,
                 focus_outside_mode: str = "fade_white",
                 focus_context_expand_px: int = 32,
                 focus_fade_expand_px: int = 96,
                 style_key: str | None = None,
                 character_memories: dict | None = None,
                 character_library=None, scene_palette=None,
                 style_strength: float = 1.0,
                 reference_strength: float = 1.0,
                 manual_strength: float = 1.0,
                 pastel_tuning: dict | None = None,
                 filter_tuning: dict | None = None,
                 custom_color_bias: dict | None = None,
                 forced_matches: dict[int, int] | None = None,
                 parent=None):
        super().__init__(parent)
        self._original_bgr = original_bgr.copy()
        self._current_result_bgr = current_result_bgr.copy()
        self._current_ai_result_bgr = (None if current_ai_result_bgr is None
                                       else current_ai_result_bgr.copy())
        self._current_filter_base_bgr = (None if current_filter_base_bgr is None
                                         else current_filter_base_bgr.copy())
        self._selection_mask = selection_mask.copy()
        self._hint_manager = HintManager.from_dict(hint_manager.to_dict())
        self._feather = int(feather)
        self._hint_margin_px = int(hint_margin_px)
        self._gap_close = int(gap_close)
        self._only_selection_hints = bool(only_selection_hints)
        self._classic_point_hints = bool(classic_point_hints)
        self._focus_outside_mode = str(focus_outside_mode or "none")
        self._focus_context_expand_px = int(focus_context_expand_px)
        self._focus_fade_expand_px = int(focus_fade_expand_px)
        self._style_key = style_key
        self._character_memories = character_memories
        self._character_library = character_library
        self._scene_palette = scene_palette
        self._style_strength = float(style_strength)
        self._reference_strength = float(reference_strength)
        self._manual_strength = float(manual_strength)
        self._pastel_tuning = dict(pastel_tuning or {})
        self._filter_tuning = dict(filter_tuning or {})
        self._custom_color_bias = dict(custom_color_bias or {})
        self._forced_matches = dict(forced_matches or {})

    def run(self):
        try:
            from config import Config
            from core.model_downloader import ensure_models_downloaded
            ensure_models_downloaded(Config.WEIGHTS_DIR, callback=self.status.emit)
            if self._focus_outside_mode and self._focus_outside_mode not in {"none", "off", "disabled"}:
                self.status.emit("正在构建局部聚焦黑白页并运行 mc-v2…")
            else:
                self.status.emit("正在用原始黑白整页运行 mc-v2…")
            from core.local_model_recolor import recolor_selection_with_model
            payload = recolor_selection_with_model(
                self._original_bgr,
                self._current_result_bgr,
                self._current_ai_result_bgr,
                self._current_filter_base_bgr,
                self._selection_mask,
                self._hint_manager,
                feather=self._feather,
                hint_margin_px=self._hint_margin_px,
                gap_close=self._gap_close,
                only_selection_hints=self._only_selection_hints,
                classic_point_hints=self._classic_point_hints,
                focus_outside_mode=self._focus_outside_mode,
                focus_context_expand_px=self._focus_context_expand_px,
                focus_fade_expand_px=self._focus_fade_expand_px,
                colorize_kwargs={
                    "style_key": self._style_key,
                    "quality_key": "draft",
                    "character_memories": self._character_memories,
                    "character_library": self._character_library,
                    "scene_palette": self._scene_palette,
                    "style_strength": self._style_strength,
                    "reference_strength": self._reference_strength,
                    "manual_strength": self._manual_strength,
                    "pastel_tuning": self._pastel_tuning,
                    "filter_tuning": self._filter_tuning,
                    "custom_color_bias": self._custom_color_bias,
                    "forced_matches": self._forced_matches,
                },
            )
            self.finished_ok.emit(payload)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self.finished_err.emit(f"{type(exc).__name__}: {exc}")


class MangaLineExtractionWorker(QThread):
    """Download/load MangaLineExtraction and infer one selection crop."""

    finished_ok = Signal(object)
    finished_err = Signal(str)
    status = Signal(str)

    def __init__(self, source_bgr: np.ndarray, selection_mask: np.ndarray,
                 parent=None):
        super().__init__(parent)
        self._source_bgr = source_bgr.copy()
        self._selection_mask = selection_mask.copy()

    def run(self):
        try:
            from config import Config
            from core.manga_line_extractor import extract_line_probability
            from core.model_downloader import ensure_manga_line_model_downloaded

            self.status.emit("正在准备 MangaLineExtraction 漫画结构线模型...")
            path = ensure_manga_line_model_downloaded(
                Config, callback=self.status.emit)
            self.status.emit("正在识别矩形内的漫画结构线...")
            result = extract_line_probability(
                self._source_bgr,
                self._selection_mask,
                weights_path=path,
                requested_device=Config.ML_DEVICE,
                max_side=Config.MANGA_LINE_MAX_SIDE,
            )
            self.finished_ok.emit(result)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self.finished_err.emit(f"{type(exc).__name__}: {exc}")
