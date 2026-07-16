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

    finished_ok = Signal(np.ndarray)
    finished_err = Signal(str)
    status = Signal(str)

    def __init__(self, image_bgr: np.ndarray, hint_manager: HintManager,
                 regenerate_auto: bool, parent=None,
                 style_key: str | None = None, quality_key: str | None = None,
                 style_profile=None, character_memories: dict | None = None):
        super().__init__(parent)
        self._image_bgr = image_bgr
        self._hint_manager = hint_manager
        self._regenerate_auto = regenerate_auto
        self._style_key = style_key
        self._quality_key = quality_key
        self._style_profile = style_profile
        self._character_memories = character_memories

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
                style_profile=self._style_profile,
                character_memories=self._character_memories,
            )
            self.finished_ok.emit(result)
        except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
            traceback.print_exc()
            self.finished_err.emit(f"{type(exc).__name__}: {exc}")


class BatchColorizeWorker(QThread):
    """Runs colorize_page() over several selected pages, one at a time.

    `pages` is a list of (path, image_bgr, hint_manager) tuples. Emits
    `page_done(path, result_bgr)` as each page finishes (so the UI can
    update that page's thumbnail/state immediately), `page_error(path,
    message)` if one page fails (batch continues with the rest), and
    `finished_all()` once every page has been attempted.
    """

    page_done = Signal(str, np.ndarray)
    page_error = Signal(str, str)
    status = Signal(str)
    finished_all = Signal()

    def __init__(self, pages: list[tuple[str, np.ndarray, HintManager]],
                 regenerate_auto: bool, parent=None,
                 style_key: str | None = None, quality_key: str | None = None,
                 style_profile=None, character_memories: dict | None = None,
                 skip_colored: bool = True):
        super().__init__(parent)
        self._pages = pages
        self._regenerate_auto = regenerate_auto
        # Manga-Colorization-FJ style: skip pages that are already in color.
        self._skip_colored = skip_colored
        self._style_key = style_key
        self._quality_key = quality_key
        self._style_profile = style_profile
        # Same dict (same CharacterMemory instances) reused across every
        # page in the batch, so character slots learned on page 1 carry
        # over to page 50 — this is what makes CharacterMemory book-level
        # rather than per-page.
        self._character_memories = character_memories

    def run(self):
        from config import Config
        from core.model_downloader import ensure_models_downloaded
        try:
            ensure_models_downloaded(Config.WEIGHTS_DIR, callback=self.status.emit)
        except Exception as exc:
            traceback.print_exc()
            for path, _img, _hm in self._pages:
                self.page_error.emit(path, f"权重下载失败：{exc}")
            self.finished_all.emit()
            return

        total = len(self._pages)
        for i, (path, image_bgr, hint_manager) in enumerate(self._pages, start=1):
            self.status.emit(f"正在上色 ({i}/{total})：{path.split('/')[-1]}")
            try:
                if self._skip_colored:
                    from core.color_detect import is_already_colored
                    if is_already_colored(image_bgr):
                        self.status.emit(
                            f"已是彩色，跳过 ({i}/{total})：{path.split('/')[-1]}")
                        self.page_done.emit(path, image_bgr.copy())
                        continue
                result = colorize_page(
                    image_bgr,
                    hint_manager=hint_manager,
                    regenerate_auto=self._regenerate_auto,
                    style_key=self._style_key,
                    quality_key=self._quality_key,
                    style_profile=self._style_profile,
                    character_memories=self._character_memories,
                )
                self.page_done.emit(path, result)
            except Exception as exc:  # noqa: BLE001 — keep going on per-page failure
                traceback.print_exc()
                self.page_error.emit(path, f"{type(exc).__name__}: {exc}")
        self.finished_all.emit()
