"""Background worker — runs the (slow) Auto pipeline off the UI thread."""

from __future__ import annotations

import traceback
import gc
import threading
import copy

import numpy as np
from PySide6.QtCore import QThread, Signal

from core.hint_manager import HintManager
from core.cancellation import OperationCancelled, check_cancelled
from pipeline import colorize_page


class _CancellableThread(QThread):
    """QThread with a safe cooperative stop flag.

    ``requestInterruption`` alone is only advisory.  The Event gives worker
    code and non-Qt pipeline helpers a cheap thread-safe cancellation probe.
    """

    cancelled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # Heavy first-use imports can overflow Qt's small default worker stack.
        self.setStackSize(8 * 1024 * 1024)
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()
        self.requestInterruption()

    def cancel_requested(self) -> bool:
        return self._cancel_event.is_set() or self.isInterruptionRequested()

    def _check_cancelled(self) -> None:
        check_cancelled(self.cancel_requested)

    def _emit_status(self, message: str) -> None:
        self._check_cancelled()
        self.status.emit(message)


class ColorizeWorker(_CancellableThread):
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
                 forced_matches: dict[int, int] | None = None,
                 auto_black_hair_guard: bool | str = False,
                 auto_black_hair_guard_scope: str = "page",
                 auto_black_hair_guard_limit_mask: np.ndarray | None = None,
                 auto_black_hair_guard_target_rgb: tuple[int, int, int] | None = None,
                 ai_hair_parser_enabled: bool = True,
                 ai_hair_parser_backend: str = "auto",
                 ai_hair_parser_allow_download: bool = False,
                 semantic_mask_controls: dict | None = None,
                 ink_fidelity: bool = False,
                 hint_render_mode: str = "mixed"):
        super().__init__(parent)
        self._character_library = character_library
        self._scene_palette = scene_palette
        self._image_bgr = image_bgr
        self._hint_manager = hint_manager
        self._regenerate_auto = regenerate_auto
        self._style_key = str(style_key or "none")
        self._quality_key = str(quality_key or "draft")
        self._character_memories = character_memories
        self._style_strength = float(style_strength)
        self._reference_strength = float(reference_strength)
        self._manual_strength = float(manual_strength)
        self._pastel_tuning = dict(pastel_tuning or {})
        self._filter_tuning = dict(filter_tuning or {})
        self._custom_color_bias = dict(custom_color_bias or {})
        self._forced_matches = dict(forced_matches or {})
        self._auto_black_hair_guard = auto_black_hair_guard
        self._auto_black_hair_guard_scope = str(auto_black_hair_guard_scope or "page")
        self._auto_black_hair_guard_limit_mask = (None if auto_black_hair_guard_limit_mask is None else np.where(auto_black_hair_guard_limit_mask > 0, 255, 0).astype(np.uint8))
        self._auto_black_hair_guard_target_rgb = (None if auto_black_hair_guard_target_rgb is None else tuple(int(v) for v in auto_black_hair_guard_target_rgb))
        self._ai_hair_parser_enabled = bool(ai_hair_parser_enabled)
        self._ai_hair_parser_backend = str(ai_hair_parser_backend or "auto")
        self._ai_hair_parser_allow_download = bool(ai_hair_parser_allow_download)
        self._semantic_mask_controls = copy.deepcopy(semantic_mask_controls or {})
        self._ink_fidelity = bool(ink_fidelity)
        self._hint_render_mode = ("native_soft" if str(hint_render_mode) == "native_soft"
                                  else "mixed")

    def run(self):
        try:
            from config import Config
            from core.model_downloader import ensure_models_downloaded
            self._check_cancelled()
            ensure_models_downloaded(Config.WEIGHTS_DIR, callback=self._emit_status)
            self._check_cancelled()

            self._emit_status("正在上色...")
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
                auto_black_hair_guard=self._auto_black_hair_guard,
                auto_black_hair_guard_scope=self._auto_black_hair_guard_scope,
                auto_black_hair_guard_limit_mask=self._auto_black_hair_guard_limit_mask,
                auto_black_hair_guard_target_rgb=self._auto_black_hair_guard_target_rgb,
                ai_hair_parser_enabled=self._ai_hair_parser_enabled,
                ai_hair_parser_backend=self._ai_hair_parser_backend,
                ai_hair_parser_allow_download=self._ai_hair_parser_allow_download,
                semantic_mask_controls=self._semantic_mask_controls,
                ink_fidelity=self._ink_fidelity,
                hint_render_mode=self._hint_render_mode,
                cancel_check=self.cancel_requested,
            )
            self._check_cancelled()
            if self._hint_render_mode == "native_soft":
                from pipeline import get_colorizer
                spot_report = getattr(
                    get_colorizer(Config, ensure_weights=False),
                    "last_native_hint_spot_report", None)
                if spot_report and spot_report.get("reject"):
                    self._emit_status(
                        "检测到原生 Hint 圆形色斑，已保留无提示 Generator 结果；"
                        "本次指定发色未生效，请使用人工选区重着色。")
            self.finished_ok.emit(result)
        except OperationCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
            traceback.print_exc()
            self.finished_err.emit(f"{type(exc).__name__}: {exc}")


class BatchColorizeWorker(_CancellableThread):
    """Runs colorize_page() over several selected pages, one at a time.

    `pages` is a list of (path, image_bgr, hint_manager[, forced_matches]) tuples. Emits
    `page_done(path, result_bgr)` as each page finishes (so the UI can
    update that page's thumbnail/state immediately), `page_skipped(path,
    reason)` when an opt-in skip rule leaves a page untouched,
    `page_error(path, message)` if one page fails (batch continues with the
    rest), and `finished_all()` once every page has been attempted.
    """

    page_done = Signal(str, object)
    page_skipped = Signal(str, str)
    page_error = Signal(str, str)
    status = Signal(str)
    finished_all = Signal()

    def __init__(self, pages: list,
                 regenerate_auto: bool, parent=None,
                 style_key: str | None = None, quality_key: str | None = None,
                 character_memories: dict | None = None,
                 character_library=None, scene_palette=None,
                 skip_colored: bool = False,
                 protect_text: bool = True,
                 allow_model_download: bool = False,
                 style_strength: float = 1.0,
                 reference_strength: float = 1.0,
                 manual_strength: float = 1.0,
                 pastel_tuning: dict | None = None,
                 filter_tuning: dict | None = None,
                 custom_color_bias: dict | None = None,
                 auto_black_hair_guard: bool | str = False,
                 auto_black_hair_guard_scope: str = "page",
                 auto_black_hair_guard_limit_mask: np.ndarray | None = None,
                 auto_black_hair_guard_target_rgb: tuple[int, int, int] | None = None,
                 ai_hair_parser_enabled: bool = True,
                 ai_hair_parser_backend: str = "auto",
                 ai_hair_parser_allow_download: bool = False,
                 semantic_mask_controls: dict | None = None,
                 ink_fidelity: bool = False,
                 hint_render_mode: str = "mixed"):
        super().__init__(parent)
        self._pages = pages
        self._regenerate_auto = regenerate_auto
        # Manga-Colorization-FJ style: skip pages that are already in color.
        self._skip_colored = skip_colored
        self._protect_text = bool(protect_text)
        self._allow_model_download = bool(allow_model_download)
        self._character_library = character_library
        self._scene_palette = scene_palette
        self._style_key = str(style_key or "none")
        self._quality_key = str(quality_key or "draft")
        # Same dict (same CharacterMemory instances) reused across every
        # page in the batch, so character slots learned on page 1 carry
        # over to page 50 — this is what makes CharacterMemory book-level
        # rather than per-page.
        self._character_memories = character_memories
        self._style_strength = float(style_strength)
        self._reference_strength = float(reference_strength)
        self._manual_strength = float(manual_strength)
        self._pastel_tuning = dict(pastel_tuning or {})
        self._filter_tuning = dict(filter_tuning or {})
        self._custom_color_bias = dict(custom_color_bias or {})
        self._auto_black_hair_guard = auto_black_hair_guard
        self._auto_black_hair_guard_scope = str(auto_black_hair_guard_scope or "page")
        self._auto_black_hair_guard_limit_mask = (
            None if auto_black_hair_guard_limit_mask is None
            else np.where(auto_black_hair_guard_limit_mask > 0, 255, 0).astype(np.uint8))
        self._auto_black_hair_guard_target_rgb = (
            None if auto_black_hair_guard_target_rgb is None
            else tuple(int(v) for v in auto_black_hair_guard_target_rgb))
        self._ai_hair_parser_enabled = bool(ai_hair_parser_enabled)
        self._ai_hair_parser_backend = str(ai_hair_parser_backend or "auto")
        self._ai_hair_parser_allow_download = bool(ai_hair_parser_allow_download)
        self._semantic_mask_controls = copy.deepcopy(semantic_mask_controls or {})
        self._ink_fidelity = bool(ink_fidelity)
        self._hint_render_mode = ("native_soft" if str(hint_render_mode) == "native_soft"
                                  else "mixed")

    def run(self):
        """Run the batch with cooperative cancellation and guaranteed cleanup."""
        try:
            from config import Config
            from core.model_downloader import ensure_models_downloaded
            self._emit_status("正在检查 mc-v2 模型文件...")
            ensure_models_downloaded(
                Config.WEIGHTS_DIR, callback=self._emit_status,
                allow_download=self._allow_model_download)
            self._check_cancelled()

            self._emit_status("正在加载上色模型，首次运行可能需要一些时间...")
            from pipeline import get_colorizer
            get_colorizer(Config, ensure_weights=False)
            self._check_cancelled()

            total = len(self._pages)
            for i, item in enumerate(self._pages, start=1):
                self._check_cancelled()
                path, image_bgr, hint_manager = item[:3]
                forced_matches = dict(item[3] or {}) if len(item) > 3 else {}
                self._emit_status(f"正在读取并上色 ({i}/{total})：{path.split('/')[-1]}")
                result = None
                result_bgr = None
                filter_base_bgr = None
                quality_report = None
                auto_hints_before = copy.deepcopy(getattr(hint_manager, "auto_hints", []))
                diagnostics_before = copy.deepcopy(getattr(hint_manager, "last_diagnostics", None))
                page_context_before = getattr(hint_manager, "last_page_context", None)
                try:
                    self._check_cancelled()
                    if image_bgr is None:
                        from core.imageio import imread as _uimread
                        image_bgr = _uimread(path)
                        if image_bgr is None:
                            raise RuntimeError(f"无法读取图片：{path}")
                    self._check_cancelled()
                    if self._skip_colored:
                        from core.color_detect import is_already_colored
                        if is_already_colored(image_bgr):
                            self._check_cancelled()
                            self._emit_status(
                                f"检测为彩色页，已跳过 ({i}/{total})：{path.split('/')[-1]}")
                            self.page_skipped.emit(path, "检测为彩色页")
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
                        auto_black_hair_guard=self._auto_black_hair_guard,
                        auto_black_hair_guard_scope=self._auto_black_hair_guard_scope,
                        auto_black_hair_guard_limit_mask=self._auto_black_hair_guard_limit_mask,
                        auto_black_hair_guard_target_rgb=self._auto_black_hair_guard_target_rgb,
                        ai_hair_parser_enabled=self._ai_hair_parser_enabled,
                        ai_hair_parser_backend=self._ai_hair_parser_backend,
                        ai_hair_parser_allow_download=self._ai_hair_parser_allow_download,
                        semantic_mask_controls=self._semantic_mask_controls,
                        ink_fidelity=self._ink_fidelity,
                        hint_render_mode=self._hint_render_mode,
                        cancel_check=self.cancel_requested,
                    )
                    self._check_cancelled()
                    if self._hint_render_mode == "native_soft":
                        from pipeline import get_colorizer
                        spot_report = getattr(
                            get_colorizer(Config, ensure_weights=False),
                            "last_native_hint_spot_report", None)
                        if spot_report and spot_report.get("reject"):
                            self._emit_status(
                                f"({i}/{total}) 原生 Hint 出现色斑，"
                                "已自动使用无提示模型结果。")
                    if isinstance(result, tuple) and len(result) == 2:
                        result_bgr, filter_base_bgr = result
                    else:
                        result_bgr, filter_base_bgr = result, None
                    try:
                        self._check_cancelled()
                        from core.quality_score import assess_colorization
                        quality_report = assess_colorization(image_bgr, result_bgr)
                    except OperationCancelled:
                        raise
                    except Exception:
                        pass
                    self._check_cancelled()
                    self.page_done.emit(path, {
                        "result_bgr": result_bgr,
                        "filter_base_bgr": filter_base_bgr,
                        "quality_report": quality_report,
                    })
                except OperationCancelled:
                    # A stopped page must not leave half-regenerated automatic
                    # hints/diagnostics behind when no result is accepted.
                    hint_manager.auto_hints = auto_hints_before
                    if diagnostics_before is None:
                        if hasattr(hint_manager, "last_diagnostics"):
                            delattr(hint_manager, "last_diagnostics")
                    else:
                        hint_manager.last_diagnostics = diagnostics_before
                    if page_context_before is None:
                        if hasattr(hint_manager, "last_page_context"):
                            hint_manager.last_page_context = None
                    else:
                        hint_manager.last_page_context = page_context_before
                    raise
                except Exception as exc:  # noqa: BLE001 — keep going per page
                    traceback.print_exc()
                    if not self.cancel_requested():
                        self.page_error.emit(path, f"{type(exc).__name__}: {exc}")
                finally:
                    image_bgr = None
                    result = None
                    result_bgr = None
                    filter_base_bgr = None
                    quality_report = None
                    try:
                        self._pages[i - 1] = (path, None, hint_manager, forced_matches)
                    except Exception:
                        pass
                    try:
                        from pipeline import clear_transient_page_caches
                        clear_transient_page_caches()
                    except Exception:
                        pass
                    gc.collect()
                    if i % 2 == 0 or i == total or self.cancel_requested():
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            if (getattr(torch.backends, "mps", None) and
                                    torch.backends.mps.is_available()):
                                torch.mps.empty_cache()
                        except Exception:
                            pass
        except OperationCancelled:
            self.cancelled.emit()
        except Exception as exc:  # covers imports, downloads and model load
            if not self.cancel_requested():
                traceback.print_exc()
                message = f"初始化上色模型失败：{type(exc).__name__}: {exc}"
                for item in self._pages:
                    self.page_error.emit(item[0], message)
        finally:
            self.finished_all.emit()


class LocalModelRecolorWorker(_CancellableThread):
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
                 region_control: dict | None = None,
                 protect_text_overlay: bool = True,
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
        self._region_control = dict(region_control or {})
        self._protect_text_overlay = bool(protect_text_overlay)
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
            self._check_cancelled()
            ensure_models_downloaded(Config.WEIGHTS_DIR, callback=self._emit_status)
            self._check_cancelled()
            if self._focus_outside_mode and self._focus_outside_mode not in {"none", "off", "disabled"}:
                self._emit_status("正在构建局部聚焦黑白页并运行 mc-v2…")
            else:
                self._emit_status("正在用原始黑白整页运行 mc-v2…")
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
                region_control=self._region_control,
                protect_text_overlay=self._protect_text_overlay,
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
                    "protect_text": self._protect_text_overlay,
                    "cancel_check": self.cancel_requested,
                },
            )
            self._check_cancelled()
            self.finished_ok.emit(payload)
        except OperationCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self.finished_err.emit(f"{type(exc).__name__}: {exc}")


class SemanticSelectionWorker(_CancellableThread):
    """Build the editable mc-v2 semantic mask off the Qt main thread."""

    finished_ok = Signal(object)
    finished_err = Signal(str)
    status = Signal(str)

    def __init__(self, original_bgr: np.ndarray, colorized_bgr: np.ndarray,
                 seed_xy: tuple[int, int], limit_mask: np.ndarray | None = None,
                 complete_hair_object: bool = False, parent=None):
        super().__init__(parent)
        # Page arrays can be changed/spooled while the worker is running; own
        # compact copies so the algorithm never observes mutable UI state.
        self._original_bgr = np.asarray(original_bgr).copy()
        self._colorized_bgr = np.asarray(colorized_bgr).copy()
        self._seed_xy = (int(seed_xy[0]), int(seed_xy[1]))
        self._limit_mask = (None if limit_mask is None else
                            np.asarray(limit_mask).copy())
        self._complete_hair_object = bool(complete_hair_object)

    def run(self):
        try:
            self._check_cancelled()
            self._emit_status(
                "正在重建完整头发对象、补全发梢与文字遮挡…"
                if self._complete_hair_object else
                "正在补全 mc-v2 语义区域、发梢与文字遮挡…")
            from core.mcv2_semantic_selection import select_semantic_region
            mask, diagnostics = select_semantic_region(
                self._original_bgr, self._colorized_bgr, self._seed_xy,
                limit_mask=self._limit_mask,
                complete_hair_object=self._complete_hair_object)
            self._check_cancelled()
            self.finished_ok.emit({
                "mask": mask,
                "diagnostics": diagnostics,
            })
        except OperationCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self.finished_err.emit(f"{type(exc).__name__}: {exc}")


class MangaLineExtractionWorker(_CancellableThread):
    """Download/load MangaLineExtraction and infer one selection crop."""

    finished_ok = Signal(object)
    finished_err = Signal(str)
    status = Signal(str)

    def __init__(self, source_bgr: np.ndarray, selection_mask: np.ndarray,
                 parent=None, *, allow_model_download: bool = False):
        super().__init__(parent)
        self._source_bgr = source_bgr.copy()
        self._selection_mask = selection_mask.copy()
        self._allow_model_download = bool(allow_model_download)

    def run(self):
        try:
            from config import Config
            from core.manga_line_extractor import extract_line_probability
            from core.model_downloader import ensure_manga_line_model_downloaded

            self._emit_status("正在准备 MangaLineExtraction 漫画结构线模型...")
            self._check_cancelled()
            path = ensure_manga_line_model_downloaded(
                Config, callback=self._emit_status,
                cancel_check=self.cancel_requested,
                allow_download=self._allow_model_download)
            self._check_cancelled()
            self._emit_status("正在识别矩形内的漫画结构线...")
            result = extract_line_probability(
                self._source_bgr,
                self._selection_mask,
                weights_path=path,
                requested_device=Config.ML_DEVICE,
                max_side=Config.MANGA_LINE_MAX_SIDE,
            )
            self._check_cancelled()
            self.finished_ok.emit(result)
        except OperationCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self.finished_err.emit(f"{type(exc).__name__}: {exc}")



class HairModelDownloadWorker(_CancellableThread):
    """Download/validate optional AI hair models outside the UI thread."""

    finished_ok = Signal(object)
    finished_err = Signal(str)
    status = Signal(str)

    def __init__(self, keys: list[str] | tuple[str, ...], parent=None,
                 *, force: bool = False):
        super().__init__(parent)
        self._keys = list(keys)
        self._force = bool(force)

    def run(self):
        try:
            from config import Config
            from core.model_downloader import ensure_hair_models_downloaded
            self._check_cancelled()
            result = ensure_hair_models_downloaded(
                Config, keys=self._keys, callback=self._emit_status,
                force=self._force, cancel_check=self.cancel_requested)
            self._check_cancelled()
            self.finished_ok.emit(result)
        except OperationCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self.finished_err.emit(f"{type(exc).__name__}: {exc}")


class HairModelImportWorker(_CancellableThread):
    """Validate and install one user-selected local AI hair model."""

    finished_ok = Signal(object)
    finished_err = Signal(str)
    status = Signal(str)

    def __init__(self, key: str, source_path: str, parent=None):
        super().__init__(parent)
        self._key = str(key)
        self._source_path = str(source_path)

    def run(self):
        try:
            self._check_cancelled()
            from config import Config
            from core.hair_model_manager import install_local_model
            result = install_local_model(
                self._key, self._source_path, config=Config,
                callback=self._emit_status)
            self._check_cancelled()
            self.finished_ok.emit(result)
        except OperationCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self.finished_err.emit(f"{type(exc).__name__}: {exc}")

class HairSelectionRefineWorker(_CancellableThread):
    """Refine a rough user hair selection without blocking Qt's GUI thread."""

    finished_ok = Signal(object)
    finished_err = Signal(str)
    status = Signal(str)

    def __init__(self, original_bgr, colorized_bgr, rough_mask,
                 positive=None, negative=None, *, grow_px=22, parent=None):
        super().__init__(parent)
        self._original = np.ascontiguousarray(original_bgr).copy()
        self._color = np.ascontiguousarray(colorized_bgr).copy()
        self._rough = np.ascontiguousarray(rough_mask).copy()
        self._positive = None if positive is None else np.ascontiguousarray(positive).copy()
        self._negative = None if negative is None else np.ascontiguousarray(negative).copy()
        self._grow_px = int(grow_px)

    def run(self):
        try:
            self._check_cancelled()
            self.status.emit("正在根据墨线、mc-v2 配色与正负画笔贴合发丝…")
            from core.hair_selection_refiner import refine_hair_selection
            from dataclasses import asdict
            mask, diagnostic = refine_hair_selection(
                self._original, self._color, self._rough,
                positive=self._positive, negative=self._negative,
                grow_px=self._grow_px)
            self._check_cancelled()
            self.finished_ok.emit({'mask': mask, 'diagnostics': asdict(diagnostic)})
        except OperationCancelled:
            self.cancelled.emit()
        except Exception as exc:
            traceback.print_exc()
            self.finished_err.emit(f'{type(exc).__name__}: {exc}')
