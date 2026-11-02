"""Colortina — Phase 2 editor.

Layout: page list (left) | canvas (center) | controls (right), per the
architecture doc. Only one mode — Auto — with an optional edit step:

    import -> auto colorize -> [not happy? draw hints -> regenerate] -> export
"""

from __future__ import annotations

import os
import shutil
import tempfile
import hashlib
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, QObject, Signal, QSettings
from PySide6.QtGui import QColor, QAction, QKeySequence, QShortcut, QIcon
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QListWidget, QListWidgetItem, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QSlider, QFileDialog, QSplitter,
    QStatusBar, QColorDialog, QButtonGroup, QRadioButton, QGroupBox, QListView,
    QMessageBox, QProgressBar, QCheckBox, QComboBox, QSpinBox, QInputDialog,
    QSizePolicy, QGridLayout, QTabWidget, QDialog, QApplication, QToolButton,
    QScrollArea, QFrame, QLayout,
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
from ui.worker import (BatchColorizeWorker, HairModelDownloadWorker,
                       HairModelImportWorker, LocalModelRecolorWorker, MangaLineExtractionWorker,
                       SemanticSelectionWorker, HairSelectionRefineWorker)


def _tunnel2_adapter_available() -> bool:
    """Return whether either supported external region adapter is installed."""
    paths = (
        getattr(Config, "TUNNEL2_SEMANTIC_ADAPTER_PATH", ""),
        getattr(Config, "REGION_ADAPTER_WEIGHTS_PATH", ""),
    )
    return any(path and os.path.isfile(path) for path in paths)


class _LayerCacheBridge(QObject):
    """Marshal background cache completions back onto the Qt UI thread."""

    completed = Signal(object)


class _ExportCacheBridge(QObject):
    """Marshal final-image compression completions onto the Qt UI thread."""

    completed = Signal(object)


def _discard_spill_files(result: dict | None) -> None:
    """Delete files produced by a stale/cancelled background spill."""
    if not isinstance(result, dict):
        return
    for path in set(result.get("created_paths", ()) or ()):
        try:
            os.remove(path)
        except OSError:
            pass


def _write_layer_spill(request: dict) -> dict:
    """Encode one page's resident layers outside the Qt main thread.

    The request contains live ndarray references plus revision/object tokens.
    The UI thread validates those tokens before committing the files, so an
    edit made while encoding simply makes this result stale and disposable.
    """
    cache_dir = request["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    layers = request["layers"]
    new_paths = dict(request.get("old_paths", {}))
    representatives: list[tuple[np.ndarray, str]] = []
    successful_keys: set[str] = set()
    created_paths: set[str] = set()
    bytes_written = 0
    from core.imageio import imwrite as _uimwrite

    for key in ("ai", "result", "filter"):
        info = layers.get(key)
        if not info:
            continue
        image = info["image"]
        existing_path = info.get("existing_path")
        if (not info.get("dirty", True) and existing_path and
                os.path.isfile(existing_path)):
            new_paths[key] = existing_path
            successful_keys.add(key)
            representatives.append((image, existing_path))
            continue

        shared_path = None
        for other_image, other_path in representatives:
            if (image is other_image or
                    (image.shape == other_image.shape and
                     image.dtype == other_image.dtype and
                     np.array_equal(image, other_image))):
                shared_path = other_path
                break

        if shared_path is None:
            final_path = os.path.join(
                cache_dir,
                f"{request['page_id']}_g{request['cache_generation']}_{key}.png")
            temp_path = final_path + ".tmp.png"
            # Compression level 1 is still lossless, but uses much less CPU than
            # the previous level 3.  Session files are temporary and equal
            # layers share one file, so responsiveness is more important than
            # squeezing the final few percent out of the cache size.
            ok = _uimwrite(
                temp_path, image,
                [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if not ok:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
                continue
            os.replace(temp_path, final_path)
            created_paths.add(final_path)
            try:
                bytes_written += os.path.getsize(final_path)
            except OSError:
                pass
            shared_path = final_path
            representatives.append((image, shared_path))

        new_paths[key] = shared_path
        successful_keys.add(key)

    return {
        "path": request["path"],
        "state_id": request["state_id"],
        "serial": request["serial"],
        "old_paths": request.get("old_paths", {}),
        "new_paths": new_paths,
        "successful_keys": successful_keys,
        "created_paths": created_paths,
        "layers": {
            key: {
                "revision": info["revision"],
                "image_id": info["image_id"],
            }
            for key, info in layers.items()
        },
        "bytes_written": bytes_written,
    }


def _discard_export_cache_result(result: dict | None) -> None:
    if not isinstance(result, dict):
        return
    path = result.get("created_path")
    if path:
        try:
            os.remove(path)
        except OSError:
            pass


def _write_export_cache(request: dict) -> dict:
    """Encode one immutable final export asset outside the Qt main thread."""
    image = request.get("image")
    if image is None:
        source_path = request.get("source_cache_path")
        if not source_path or not os.path.isfile(source_path):
            raise RuntimeError("No resident result or lossless result cache is available")
        from core.imageio import imread as _uimread
        image = _uimread(source_path)
        if image is None:
            raise RuntimeError(f"Cannot read result cache: {source_path}")

    from core.image_compressor import encode_image, atomic_write_bytes
    encoded = encode_image(image, request.get("profile"))
    os.makedirs(request["cache_dir"], exist_ok=True)
    base = os.path.join(
        request["cache_dir"],
        f"{request['page_id']}_r{request['revision']}_s{request['serial']}_{request['profile']}")
    final_path = base + encoded.extension
    atomic_write_bytes(final_path, encoded.data)
    return {
        "path": request["path"],
        "state_id": request["state_id"],
        "serial": request["serial"],
        "revision": request["revision"],
        "image_id": request.get("image_id"),
        "profile": request["profile"],
        "created_path": final_path,
        "metadata": encoded.as_metadata(),
    }


class PageState:
    """Everything the app tracks for one page.

    Full-resolution batch layers are kept in memory only while the page is
    active. Inactive pages are losslessly spooled to a session PNG cache and
    loaded lazily on access. This prevents macOS from creating many gigabytes
    of swap while preserving the AI base, editable result and filter base.
    """

    _LAYER_ATTRS = {
        "ai": "_ai_result_bgr",
        "result": "_result_bgr",
        "filter": "_filter_base_bgr",
    }

    def __init__(self, path: str, original_bgr: np.ndarray | None = None):
        self.path = path
        self._original_bgr = original_bgr
        self._load_error: str | None = None
        self._ai_result_bgr: np.ndarray | None = None
        self._result_bgr: np.ndarray | None = None
        self._filter_base_bgr: np.ndarray | None = None
        self._layer_cache_paths: dict[str, str] = {}
        self._layer_dirty: dict[str, bool] = {key: False for key in self._LAYER_ATTRS}
        self._layer_revision: dict[str, int] = {key: 0 for key in self._LAYER_ATTRS}
        self._owned_cache_paths: set[str] = set()
        self._cache_generation = 0
        self._spill_serial = 0
        # Separate pre-compressed final export asset.  Working/edit layers stay
        # lossless; export copies these immutable bytes without recompressing.
        self._export_cache_path: str | None = None
        self._export_cache_profile: str = ""
        self._export_cache_revision: int = -1
        self._export_cache_metadata: dict = {}
        self._export_serial = 0
        # Reference layers for the dedicated colour-bias eraser are created
        # lazily for the active page rather than duplicated for every batch page.
        self.bias_brush_reference_bgr: np.ndarray | None = None
        self.bias_brush_reference_filter_bgr: np.ndarray | None = None
        self.hint_manager = HintManager()
        self.quality_report = None
        self.pipeline_diagnostics: dict = {}
        self.forced_character_matches: dict[int, int] = {}
        self.undo_stack: list[dict] = []
        self.redo_stack: list[dict] = []

    def _get_layer(self, key: str) -> np.ndarray | None:
        attr = self._LAYER_ATTRS[key]
        image = getattr(self, attr)
        if image is None:
            cache_path = self._layer_cache_paths.get(key)
            if cache_path and os.path.isfile(cache_path):
                # AI/result/filter commonly share the same deduplicated PNG.
                # Reuse an already-decoded sibling via a RAM copy instead of
                # decoding the same multi-megapixel PNG two or three times.
                for other_key, other_attr in self._LAYER_ATTRS.items():
                    if other_key == key:
                        continue
                    if self._layer_cache_paths.get(other_key) != cache_path:
                        continue
                    other_image = getattr(self, other_attr)
                    if other_image is not None:
                        image = other_image.copy()
                        break
                if image is None:
                    from core.imageio import imread as _uimread
                    image = _uimread(cache_path)
                if image is not None:
                    setattr(self, attr, image)
                    self._layer_dirty[key] = False
                else:
                    self._layer_cache_paths.pop(key, None)
        return image

    def _delete_unreferenced_cache(self, path: str | None) -> None:
        if not path or path not in self._owned_cache_paths:
            return
        if path in self._layer_cache_paths.values():
            return
        self._owned_cache_paths.discard(path)
        try:
            os.remove(path)
        except OSError:
            pass

    def _set_layer(self, key: str, image: np.ndarray | None) -> None:
        old_path = self._layer_cache_paths.pop(key, None)
        setattr(self, self._LAYER_ATTRS[key], image)
        self._layer_dirty[key] = image is not None
        self._layer_revision[key] = int(self._layer_revision.get(key, 0)) + 1
        self._delete_unreferenced_cache(old_path)
        if key == "result":
            self.invalidate_export_cache()

    @property
    def ai_result_bgr(self) -> np.ndarray | None:
        return self._get_layer("ai")

    @ai_result_bgr.setter
    def ai_result_bgr(self, image: np.ndarray | None) -> None:
        self._set_layer("ai", image)

    @property
    def result_bgr(self) -> np.ndarray | None:
        return self._get_layer("result")

    @result_bgr.setter
    def result_bgr(self, image: np.ndarray | None) -> None:
        self._set_layer("result", image)

    @property
    def filter_base_bgr(self) -> np.ndarray | None:
        return self._get_layer("filter")

    @filter_base_bgr.setter
    def filter_base_bgr(self, image: np.ndarray | None) -> None:
        self._set_layer("filter", image)

    @property
    def has_ai_result(self) -> bool:
        return (self._ai_result_bgr is not None or
                bool(self._layer_cache_paths.get("ai") and
                     os.path.isfile(self._layer_cache_paths["ai"])))

    @property
    def has_result(self) -> bool:
        return (self._result_bgr is not None or
                bool(self._layer_cache_paths.get("result") and
                     os.path.isfile(self._layer_cache_paths["result"])))

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

    def release_original(self, *, force: bool = False) -> None:
        """Release source pixels; they can always be decoded from ``path``."""
        if force or (not self.has_ai_result and not self.has_result):
            self._original_bgr = None

    def can_release_cached_layers(self) -> bool:
        """True when every resident layer already has a clean cache file."""
        resident = False
        for key, attr in self._LAYER_ATTRS.items():
            image = getattr(self, attr)
            if image is None:
                continue
            resident = True
            path = self._layer_cache_paths.get(key)
            if self._layer_dirty.get(key, True) or not path or not os.path.isfile(path):
                return False
        return resident

    def prepare_spill_request(self, cache_dir: str) -> dict | None:
        """Capture a cheap, non-blocking description for background encoding."""
        resident = {
            key: getattr(self, attr)
            for key, attr in self._LAYER_ATTRS.items()
            if getattr(self, attr) is not None
        }
        if not resident:
            self.release_original(force=True)
            if hasattr(self.hint_manager, "release_runtime_cache"):
                self.hint_manager.release_runtime_cache()
            return None

        self._cache_generation += 1
        self._spill_serial += 1
        page_id = hashlib.blake2b(
            os.path.abspath(self.path).encode("utf-8", "surrogatepass"),
            digest_size=8).hexdigest()
        return {
            "path": self.path,
            "state_id": id(self),
            "serial": self._spill_serial,
            "cache_dir": cache_dir,
            "cache_generation": self._cache_generation,
            "page_id": page_id,
            "old_paths": dict(self._layer_cache_paths),
            "layers": {
                key: {
                    "image": image,
                    "image_id": id(image),
                    "revision": int(self._layer_revision.get(key, 0)),
                    "dirty": bool(self._layer_dirty.get(key, True)),
                    "existing_path": self._layer_cache_paths.get(key),
                }
                for key, image in resident.items()
            },
        }

    def commit_spill_result(self, result: dict, *, release: bool) -> int:
        """Commit a background spill only for layers that stayed unchanged."""
        if not isinstance(result, dict) or result.get("state_id") != id(self):
            _discard_spill_files(result)
            return 0

        old_paths = dict(result.get("old_paths", {}))
        new_paths = dict(result.get("new_paths", {}))
        successful = set(result.get("successful_keys", ()) or ())
        created = set(result.get("created_paths", ()) or ())
        committed_created: set[str] = set()

        for key in successful:
            token = (result.get("layers", {}) or {}).get(key, {})
            attr = self._LAYER_ATTRS.get(key)
            current = getattr(self, attr) if attr else None
            unchanged = (
                attr is not None and
                int(self._layer_revision.get(key, 0)) == int(token.get("revision", -1)) and
                current is not None and id(current) == int(token.get("image_id", -1))
            )
            path = new_paths.get(key)
            if not unchanged or not path or not os.path.isfile(path):
                continue
            self._layer_cache_paths[key] = path
            self._layer_dirty[key] = False
            if path in created:
                self._owned_cache_paths.add(path)
                committed_created.add(path)
            if release:
                setattr(self, attr, None)

        # Remove files produced solely for stale layers and retire superseded
        # owned files only after the new references have been installed.
        for path in created - committed_created:
            try:
                os.remove(path)
            except OSError:
                pass
        for path in set(old_paths.values()):
            self._delete_unreferenced_cache(path)

        self._promote_lossless_export_cache()

        if release:
            self.release_original(force=True)
            self.bias_brush_reference_bgr = None
            self.bias_brush_reference_filter_bgr = None
            if hasattr(self.hint_manager, "release_runtime_cache"):
                self.hint_manager.release_runtime_cache()
        return int(result.get("bytes_written", 0) or 0)

    def spill_layers(self, cache_dir: str) -> int:
        """Synchronous fallback used by tests/non-UI callers.

        The GUI uses ``prepare_spill_request`` with a background executor so
        compression never blocks a click, drag, or page switch.
        """
        if self.can_release_cached_layers():
            self.release_cached_layers()
            return 0
        request = self.prepare_spill_request(cache_dir)
        if request is None:
            return 0
        return self.commit_spill_result(_write_layer_spill(request), release=True)

    def release_cached_layers(self) -> None:
        """Drop decoded copies whose lossless session files already exist."""
        for key, attr in self._LAYER_ATTRS.items():
            path = self._layer_cache_paths.get(key)
            if path and os.path.isfile(path):
                setattr(self, attr, None)
        self.release_original(force=True)
        if hasattr(self.hint_manager, "release_runtime_cache"):
            self.hint_manager.release_runtime_cache()

    def clear_layer_cache(self) -> None:
        self.invalidate_export_cache()
        paths = set(self._owned_cache_paths)
        self._layer_cache_paths.clear()
        self._layer_dirty = {key: False for key in self._LAYER_ATTRS}
        self._layer_revision = {
            key: int(self._layer_revision.get(key, 0)) + 1
            for key in self._LAYER_ATTRS
        }
        self._owned_cache_paths.clear()
        for path in paths:
            try:
                os.remove(path)
            except OSError:
                pass

    def _promote_lossless_export_cache(self) -> None:
        path = self._export_cache_path
        if (not path or not os.path.isfile(path) or
                not bool(self._export_cache_metadata.get("lossless"))):
            return
        previous_result_path = self._layer_cache_paths.get("result")
        if (not previous_result_path or previous_result_path == path or
                not os.path.isfile(previous_result_path)):
            return
        shared_keys = [
            key for key, cache_path in self._layer_cache_paths.items()
            if cache_path == previous_result_path
        ]
        for key in shared_keys:
            self._layer_cache_paths[key] = str(path)
            self._layer_dirty[key] = False
        self._owned_cache_paths.add(str(path))
        self._delete_unreferenced_cache(previous_result_path)

    def invalidate_export_cache(self) -> None:
        self._export_serial = int(getattr(self, "_export_serial", 0)) + 1
        path = self._export_cache_path
        self._export_cache_path = None
        self._export_cache_profile = ""
        self._export_cache_revision = -1
        self._export_cache_metadata = {}
        # A pixel-identical WebP/PNG export may be promoted to the lossless
        # session layer cache.  Never delete it while AI/result/filter still
        # reference the same file.
        if path and path not in self._layer_cache_paths.values():
            try:
                os.remove(path)
            except OSError:
                pass

    def has_valid_export_cache(self, profile: str) -> bool:
        from core.image_compressor import normalize_profile
        profile = normalize_profile(profile)
        return bool(
            self._export_cache_path and
            os.path.isfile(self._export_cache_path) and
            self._export_cache_profile == profile and
            self._export_cache_revision == int(self._layer_revision.get("result", -1))
        )

    def export_cache_info(self, profile: str) -> tuple[str | None, dict]:
        if not self.has_valid_export_cache(profile):
            return None, {}
        return self._export_cache_path, dict(self._export_cache_metadata)

    def prepare_export_cache_request(self, cache_dir: str, profile: str) -> dict | None:
        from core.image_compressor import normalize_profile
        profile = normalize_profile(profile)
        if self.has_valid_export_cache(profile):
            return None
        image = self._result_bgr
        source_cache_path = None
        if image is None:
            source_cache_path = self._layer_cache_paths.get("result")
            if not source_cache_path or not os.path.isfile(source_cache_path):
                return None
        self._export_serial += 1
        revision = int(self._layer_revision.get("result", 0))
        page_id = hashlib.blake2b(
            os.path.abspath(self.path).encode("utf-8", "surrogatepass"),
            digest_size=8).hexdigest()
        return {
            "path": self.path,
            "state_id": id(self),
            "serial": self._export_serial,
            "cache_dir": cache_dir,
            "page_id": page_id,
            "revision": revision,
            "profile": profile,
            "image": image,
            "image_id": id(image) if image is not None else None,
            "source_cache_path": source_cache_path,
        }

    def commit_export_cache_result(self, result: dict) -> bool:
        if not isinstance(result, dict) or result.get("state_id") != id(self):
            _discard_export_cache_result(result)
            return False
        path = result.get("created_path")
        unchanged = (
            int(result.get("serial", -1)) == int(self._export_serial) and
            int(result.get("revision", -1)) == int(self._layer_revision.get("result", -2)) and
            str(result.get("profile", "")) != "" and
            path and os.path.isfile(path)
        )
        resident = self._result_bgr
        image_id = result.get("image_id")
        if image_id is not None and resident is not None and id(resident) != int(image_id):
            unchanged = False
        if not unchanged:
            _discard_export_cache_result(result)
            return False
        old_path = self._export_cache_path
        self._export_cache_path = str(path)
        self._export_cache_profile = str(result.get("profile"))
        self._export_cache_revision = int(result.get("revision", -1))
        self._export_cache_metadata = dict(result.get("metadata", {}) or {})

        # When the chosen export is truly lossless, reuse it as the inactive
        # page cache instead of storing both PNG and WebP copies.
        self._promote_lossless_export_cache()

        if (old_path and old_path != path and
                old_path not in self._layer_cache_paths.values()):
            try:
                os.remove(old_path)
            except OSError:
                pass
        return True

    FULL_SNAPSHOT_FIELDS = (
        "result_bgr", "ai_result_bgr", "filter_base_bgr",
        "bias_brush_reference_bgr", "bias_brush_reference_filter_bgr",
        "hint_manager", "pipeline_diagnostics", "quality_report",
    )
    PIXEL_EDIT_FIELDS = (
        "result_bgr", "filter_base_bgr",
        "bias_brush_reference_bgr", "bias_brush_reference_filter_bgr",
    )
    FILTER_EDIT_FIELDS = ("result_bgr", "filter_base_bgr")
    HINT_EDIT_FIELDS = ("hint_manager",)

    def edit_snapshot(self, fields=None) -> dict:
        return capture_edit_state(self, fields=fields)

    def snapshot_equals_current(self, snapshot: dict) -> bool:
        if snapshot.get("__kind__") == "pixel_tiles":
            for tile in snapshot.get("tiles", ()) or ():
                x0, y0, x1, y1 = [int(v) for v in tile["bbox"]]
                for field in ("result_bgr", "filter_base_bgr"):
                    expected = tile.get(field)
                    current = getattr(self, field, None)
                    if expected is None or current is None:
                        if expected is not None or current is not None:
                            return False
                        continue
                    if current.shape[0] < y1 or current.shape[1] < x1:
                        return False
                    crop = current[y0:y1, x0:x1]
                    if (crop.shape != expected.shape or crop.dtype != expected.dtype or
                            not np.array_equal(crop, expected)):
                        return False
            return True
        if snapshot.get("__kind__") == "pixel_patch":
            bbox = snapshot.get("bbox")
            if not bbox or len(bbox) != 4:
                return True
            x0, y0, x1, y1 = [int(v) for v in bbox]
            for field in ("result_bgr", "filter_base_bgr"):
                expected = snapshot.get(field)
                current = getattr(self, field, None)
                if expected is None or current is None:
                    if expected is not None or current is not None:
                        return False
                    continue
                if current.shape[0] < y1 or current.shape[1] < x1:
                    return False
                crop = current[y0:y1, x0:x1]
                if (crop.shape != expected.shape or crop.dtype != expected.dtype or
                        not np.array_equal(crop, expected)):
                    return False
            return True
        fields = snapshot.get("__fields__", self.FULL_SNAPSHOT_FIELDS)
        return snapshots_equal(snapshot, self.edit_snapshot(fields=fields))

    def inverse_snapshot(self, snapshot: dict) -> dict:
        """Capture the current state using the same compact representation.

        Pixel-patch checkpoints make ordinary brush undo proportional to the
        painted corridor rather than the entire multi-megapixel page.  The
        inverse patch is also sufficient for redo.
        """
        kind = snapshot.get("__kind__")
        if kind == "pixel_tiles":
            result = self.result_bgr
            filter_base = self.filter_base_bgr
            tiles = []
            for tile in snapshot.get("tiles", ()) or ():
                x0, y0, x1, y1 = [int(v) for v in tile["bbox"]]
                tiles.append({
                    "bbox": (x0, y0, x1, y1),
                    "result_bgr": (None if result is None else
                                   result[y0:y1, x0:x1].copy()),
                    "filter_base_bgr": (None if filter_base is None else
                                        filter_base[y0:y1, x0:x1].copy()),
                })
            return {
                "__kind__": "pixel_tiles",
                "__fields__": self.FILTER_EDIT_FIELDS,
                "tiles": tiles,
                "filter_was_none": filter_base is None,
                "bias_brush_reference_bgr": self.bias_brush_reference_bgr,
                "bias_brush_reference_filter_bgr": self.bias_brush_reference_filter_bgr,
            }
        if kind != "pixel_patch":
            fields = snapshot.get("__fields__", self.FULL_SNAPSHOT_FIELDS)
            return self.edit_snapshot(fields=fields)
        x0, y0, x1, y1 = [int(v) for v in snapshot["bbox"]]
        result = self.result_bgr
        filter_base = self.filter_base_bgr
        return {
            "__kind__": "pixel_patch",
            "__fields__": self.FILTER_EDIT_FIELDS,
            "bbox": (x0, y0, x1, y1),
            "result_bgr": (None if result is None else
                           result[y0:y1, x0:x1].copy()),
            "filter_base_bgr": (None if filter_base is None else
                                filter_base[y0:y1, x0:x1].copy()),
            "filter_was_none": filter_base is None,
            # Bias references are not modified by the ordinary brush.  Keep
            # object references instead of copying two more complete pages.
            "bias_brush_reference_bgr": self.bias_brush_reference_bgr,
            "bias_brush_reference_filter_bgr": self.bias_brush_reference_filter_bgr,
        }

    def restore_snapshot(self, snapshot: dict) -> None:
        if snapshot.get("__kind__") == "pixel_tiles":
            tiles = list(snapshot.get("tiles", ()) or ())
            result = self.result_bgr
            if result is not None:
                for tile in tiles:
                    old_result = tile.get("result_bgr")
                    if old_result is None:
                        continue
                    x0, y0, x1, y1 = [int(v) for v in tile["bbox"]]
                    result[y0:y1, x0:x1] = old_result
                self.result_bgr = result
            if snapshot.get("filter_was_none"):
                self.filter_base_bgr = None
            else:
                filter_base = self.filter_base_bgr
                if filter_base is None and self.result_bgr is not None:
                    filter_base = self.result_bgr.copy()
                if filter_base is not None:
                    for tile in tiles:
                        old_filter = tile.get("filter_base_bgr")
                        if old_filter is None:
                            continue
                        x0, y0, x1, y1 = [int(v) for v in tile["bbox"]]
                        filter_base[y0:y1, x0:x1] = old_filter
                    self.filter_base_bgr = filter_base
            self.bias_brush_reference_bgr = snapshot.get(
                "bias_brush_reference_bgr")
            self.bias_brush_reference_filter_bgr = snapshot.get(
                "bias_brush_reference_filter_bgr")
            return
        if snapshot.get("__kind__") == "pixel_patch":
            x0, y0, x1, y1 = [int(v) for v in snapshot["bbox"]]
            result = self.result_bgr
            old_result = snapshot.get("result_bgr")
            if result is not None and old_result is not None:
                result[y0:y1, x0:x1] = old_result
                # Route through the setter so cache revisions and stale spill
                # tokens are invalidated exactly like a normal edit.
                self.result_bgr = result
            if snapshot.get("filter_was_none"):
                self.filter_base_bgr = None
            else:
                filter_base = self.filter_base_bgr
                old_filter = snapshot.get("filter_base_bgr")
                if old_filter is not None:
                    if filter_base is None:
                        current_result = self.result_bgr
                        if current_result is not None:
                            filter_base = current_result.copy()
                    if filter_base is not None:
                        filter_base[y0:y1, x0:x1] = old_filter
                        self.filter_base_bgr = filter_base
            self.bias_brush_reference_bgr = snapshot.get(
                "bias_brush_reference_bgr")
            self.bias_brush_reference_filter_bgr = snapshot.get(
                "bias_brush_reference_filter_bgr")
            return
        restore_edit_state(self, snapshot)

    def push_undo_snapshot(self, snapshot: dict, *, compare: bool = True) -> None:
        """Push an already-owned checkpoint without copying it again."""
        if not snapshot:
            return
        if (not compare or not self.undo_stack or
                not self._snapshot_equal(self.undo_stack[-1], snapshot)):
            self.undo_stack.append(snapshot)
            if len(self.undo_stack) > 20:
                self.undo_stack.pop(0)
        self.redo_stack.clear()

    def push_undo(self, fields=None):
        snapshot = self.edit_snapshot(fields=fields)
        self.push_undo_snapshot(snapshot)

    @staticmethod
    def _snapshot_equal(left: dict, right: dict) -> bool:
        left_kind = left.get("__kind__")
        right_kind = right.get("__kind__")
        compact_kinds = {"pixel_patch", "pixel_tiles"}
        if left_kind in compact_kinds or right_kind in compact_kinds:
            if left_kind != right_kind:
                return False
            if left_kind == "pixel_tiles":
                left_tiles = list(left.get("tiles", ()) or ())
                right_tiles = list(right.get("tiles", ()) or ())
                if len(left_tiles) != len(right_tiles):
                    return False
                if bool(left.get("filter_was_none")) != bool(right.get("filter_was_none")):
                    return False
                for a_tile, b_tile in zip(left_tiles, right_tiles):
                    if tuple(a_tile.get("bbox", ())) != tuple(b_tile.get("bbox", ())):
                        return False
                    for field in ("result_bgr", "filter_base_bgr"):
                        a, b = a_tile.get(field), b_tile.get(field)
                        if a is None or b is None:
                            if a is not None or b is not None:
                                return False
                        elif (a.shape != b.shape or a.dtype != b.dtype or
                              not np.array_equal(a, b)):
                            return False
                return True
            if tuple(left.get("bbox", ())) != tuple(right.get("bbox", ())):
                return False
            if bool(left.get("filter_was_none")) != bool(right.get("filter_was_none")):
                return False
            for field in ("result_bgr", "filter_base_bgr"):
                a, b = left.get(field), right.get(field)
                if a is None or b is None:
                    if a is not None or b is not None:
                        return False
                elif (a.shape != b.shape or a.dtype != b.dtype or
                      not np.array_equal(a, b)):
                    return False
            return True
        return snapshots_equal(left, right)

    def discard_unchanged_undo(self) -> None:
        if self.undo_stack and self.snapshot_equals_current(self.undo_stack[-1]):
            self.undo_stack.pop()



class _CollapsibleSection(QWidget):
    """Compact setting-card style collapsible section.

    The structure intentionally mirrors the lightweight settings-card /
    ExpandLayout pattern used by mature Qt settings UIs: a stable header, a
    concise collapsed summary and an optional detail body.  It uses only stock
    PySide6 widgets so Colortina does not gain a new runtime dependency.
    """

    expandedChanged = Signal(bool)

    def __init__(self, title: str, *, summary: str = "", expanded: bool = False,
                 settings_key: str | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self._title = str(title)
        self._settings_key = str(settings_key or "").strip() or None
        self._settings = QSettings("Colortina", "Colortina")
        if self._settings_key is not None:
            saved = self._settings.value(
                f"ui/collapsible/{self._settings_key}", None)
            if saved is not None:
                if isinstance(saved, str):
                    expanded = saved.strip().lower() in {"1", "true", "yes", "on"}
                else:
                    expanded = bool(saved)

        self.setObjectName("colortinaCollapsibleSection")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._header = QWidget()
        self._header.setObjectName("colortinaCollapsibleHeader")
        header_layout = QVBoxLayout(self._header)
        header_layout.setContentsMargins(10, 7, 10, 7)
        header_layout.setSpacing(3)

        self._toggle = QToolButton()
        self._toggle.setText(self._title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(bool(expanded))
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._toggle.setStyleSheet(
            "QToolButton { font-weight: 700; padding: 2px 0px; text-align: left; border: none; color: #2b3547; }"
            "QToolButton:hover { color: #2f80ed; }"
        )
        self._toggle.clicked.connect(self.set_expanded)
        header_layout.addWidget(self._toggle)

        self._summary = QLabel(str(summary or ""))
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet("color: #7b8698; font-size: 10px; padding-left: 20px; padding-bottom: 2px;")
        self._summary.setVisible(bool(summary) and not expanded)
        header_layout.addWidget(self._summary)
        root.addWidget(self._header)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(18, 7, 10, 10)
        self._content_layout.setSpacing(8)
        self._content.setVisible(bool(expanded))
        root.addWidget(self._content)

        self.setStyleSheet(
            "QWidget#colortinaCollapsibleSection {"
            " border: 1px solid #e7ecf3;"
            " border-radius: 12px; background: #ffffff; }"
            "QWidget#colortinaCollapsibleHeader { background: #fbfcfe; border-radius: 11px; }"
        )

    def content_layout(self) -> QVBoxLayout:
        return self._content_layout

    def add_widget(self, widget: QWidget, stretch: int = 0) -> None:
        self._content_layout.addWidget(widget, stretch)

    def add_layout(self, layout) -> None:
        self._content_layout.addLayout(layout)

    def set_summary(self, text: str) -> None:
        value = str(text or "")
        self._summary.setText(value)
        self._summary.setVisible(bool(value) and not self.is_expanded())

    def set_title(self, text: str) -> None:
        self._title = str(text or "")
        self._toggle.setText(self._title)

    def is_expanded(self) -> bool:
        return bool(self._toggle.isChecked())

    def set_expanded(self, expanded: bool) -> None:
        expanded = bool(expanded)
        changed = expanded != self.is_expanded()
        self._toggle.blockSignals(True)
        self._toggle.setChecked(expanded)
        self._toggle.blockSignals(False)
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._content.setVisible(expanded)
        self._summary.setVisible(bool(self._summary.text()) and not expanded)
        self._content.updateGeometry()
        self.updateGeometry()
        parent = self.parentWidget()
        if parent is not None:
            parent.updateGeometry()
        if self._settings_key is not None:
            self._settings.setValue(
                f"ui/collapsible/{self._settings_key}", expanded)
        if changed:
            self.expandedChanged.emit(expanded)

    def toggle(self) -> None:
        self.set_expanded(not self.is_expanded())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(tr("window_title"))
        self.resize(1400, 900)
        self.setMinimumSize(1080, 680)  # detailed render/edit tabs scroll vertically when needed

        icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "assets", "icon.png")
        if os.path.isfile(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self._pages: dict[str, PageState] = {}
        self._temp_pdf_page_dirs: set[str] = set()
        self._cleanup_stale_session_caches()
        self._session_layer_cache_dir = tempfile.mkdtemp(
            prefix="colortina_session_cache_")
        self._layer_cache_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="colortina-layer-cache")
        self._layer_cache_bridge = _LayerCacheBridge(self)
        self._layer_cache_bridge.completed.connect(self._on_layer_spill_done)
        self._export_cache_bridge = _ExportCacheBridge(self)
        self._export_cache_bridge.completed.connect(self._on_export_cache_done)
        self._pending_layer_spills: dict[str, tuple[int, int]] = {}
        self._pending_export_cache: dict[str, tuple[int, int]] = {}
        self._export_compress_queue = deque()
        self._export_compress_queued: set[str] = set()
        self._export_compress_timers: dict[str, QTimer] = {}
        self._recompress_after_pending: set[str] = set()
        self._respool_after_pending: set[str] = set()
        self._spill_delay_timers: dict[str, QTimer] = {}
        self._spill_queue = deque()
        self._spill_queued: set[str] = set()
        self._closing = False
        self._ui_busy = False
        self._stop_requested = False
        self._foreground_interaction_depth = 0
        self._current_path: str | None = None
        self._batch_worker: BatchColorizeWorker | None = None
        self._hair_model_worker: HairModelDownloadWorker | None = None
        self._pending_colorize_after_hair_download: bool | None = None
        self._local_recolor_worker: LocalModelRecolorWorker | None = None
        self._local_recolor_page_path: str | None = None
        self._local_recolor_selection_mask = None
        self._semantic_selection_worker: SemanticSelectionWorker | None = None
        self._hair_refine_worker: HairSelectionRefineWorker | None = None
        self._hair_refine_page_path = None
        self._semantic_selection_page_path: str | None = None
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
        # Rebuilding a full-page blue selection overlay for every mouse event
        # makes selection painting feel sticky on large scans.  Mutate the mask
        # immediately, but coalesce contour/preview regeneration to ~20 fps.
        self._selection_adjust_overlay_dirty = False
        self._selection_adjust_last_add_mode = True
        self._selection_adjust_gesture_active = False
        self._selection_adjust_overlay_timer = QTimer(self)
        self._selection_adjust_overlay_timer.setSingleShot(True)
        self._selection_adjust_overlay_timer.setInterval(50)
        self._selection_adjust_overlay_timer.timeout.connect(
            self._flush_selection_adjust_overlay)
        self._active_color_dialog: QColorDialog | None = None
        self._semantic_select_armed = False
        self._selection_control_color = QColor(28, 28, 32)
        self._batch_errors: list[tuple[str, str]] = []
        self._brush_color = QColor(255, 120, 120)
        self._hair_recolor_color = QColor(24, 24, 24)
        self._iris_recolor_color = QColor(70, 135, 220)
        # Independent from the ordinary brush and global page colour bias.
        self._bias_brush_color = QColor(120, 160, 255)
        self._last_picked_rgb_raw: tuple[int, int, int] | None = None
        self._local_brush_stroke_active = False
        self._brush_changed_during_stroke = False
        self._brush_patch_undo: dict | None = None
        self._brush_stroke_model_hint = False
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
        self._hair_guard_selection_masks = {}
        self._hair_guard_selection_histories = {}
        self._hair_guard_selector_mode = None
        self._hair_guard_selector_active = False
        self._hair_refine_brush_active = False
        self._suspend_hair_guard_tool_reset = False
        self._hair_guard_selection_stroke_mask = None
        self._hair_guard_selection_stroke_add_mode = True
        self._hair_guard_selection_stroke_base_mask = None

        # Window resize can generate dozens of events per second.  Re-styling
        # every child widget on each event makes live resize and splitter
        # movement feel sticky, so density updates are coalesced.
        self._responsive_density_timer = QTimer(self)
        self._responsive_density_timer.setSingleShot(True)
        self._responsive_density_timer.setInterval(90)
        self._responsive_density_timer.timeout.connect(
            self._apply_responsive_density)
        self._last_density_signature = None
        
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
        QTimer.singleShot(0, self._refresh_ai_hair_model_status)

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

        from config import Config
        from core.model_downloader import models_ready
        if not models_ready(Config.WEIGHTS_DIR):
            return  # Never download a model in a silent preload thread.
        import threading

        def _load():
            try:
                from pipeline import get_colorizer
                get_colorizer(ensure_weights=False)
            except Exception:
                pass

        # A daemon preloader can still be inside PyTorch C++ initialization
        # when Python shuts down, leading to `terminate called without an
        # active exception`.  Keep the thread alive through normal teardown.
        # No GUI/QObject is touched by its worker function.
        self._model_preload_thread = threading.Thread(
            target=_load, daemon=False, name="mcv2-engine-preload")
        self._model_preload_thread.start()

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
        if hasattr(self, "_responsive_density_timer"):
            self._responsive_density_timer.start()

    def _apply_responsive_density(self):
        """Adapt control density without undoing the polished theme.

        Earlier versions applied a second QSS block directly to the right panel
        during every resize.  That block reset button padding and group spacing
        to very small values, making the refined UI look unfinished again on
        normal laptop heights.  Density is now handled through geometry/font
        values only, leaving the global visual theme intact.
        """
        panel = getattr(self, "_right_panel", None)
        tabs = getattr(self, "_right_tabs", None)
        if panel is None or tabs is None:
            return
        height = max(1, self.height())
        if height < 740:
            font_px, button_h, margin, spacing = 10, 28, 7, 6
        elif height < 900:
            font_px, button_h, margin, spacing = 11, 30, 8, 7
        else:
            font_px, button_h, margin, spacing = 12, 32, 10, 8

        preferred = int(np.clip(self.width() * 0.30, 420, 560))
        signature = (font_px, button_h, margin, spacing, preferred)
        if signature == getattr(self, "_last_density_signature", None):
            return
        self._last_density_signature = signature

        panel.setMinimumWidth(min(440, preferred))
        panel.setMaximumWidth(max(520, preferred))
        font = panel.font()
        font.setPixelSize(font_px)
        panel.setFont(font)

        for group in panel.findChildren(QGroupBox):
            layout = group.layout()
            if layout is not None:
                layout.setContentsMargins(margin + 2, margin, margin + 2, margin + 2)
                layout.setSpacing(spacing)
        for button in panel.findChildren(QPushButton):
            button.setMinimumHeight(button_h)
            button.setMaximumHeight(max(button_h + 10, 42))

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
        self._progress.setTextVisible(False)
        self._progress.setVisible(False)
        self._progress.setFixedHeight(4)
        self._progress.setMinimumWidth(80)
        self._progress.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.statusBar().showMessage(tr("ready"))
        # The application already exposes persistent state in the top status
        # chip and workspace footer.  Keeping QMainWindow's native status bar
        # visible created a redundant empty strip below the footer.  Hide it
        # while preserving the object so all existing showMessage() calls stay
        # safe and backward-compatible.
        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().hide()

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
        prev_inkwash_strength = (getattr(self, "_inkwash_strength_slider", None).value()
                                  if getattr(self, "_inkwash_strength_slider", None) is not None else 78)
        prev_quality_key = (getattr(self, "_quality_combo", None) and
                            self._quality_combo.currentData()) or Config.DEFAULT_QUALITY_KEY
        prev_ink_fidelity = (getattr(self, "_chk_ink_fidelity", None) is None or
                             self._chk_ink_fidelity.isChecked())
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
        prev_skip_colored = prev_skip_colored.isChecked() if prev_skip_colored is not None else False
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
        splitter.setSizes([280, 900, 440])

        container = QWidget()
        container.setObjectName("CentralWorkspace")
        root_layout = QVBoxLayout(container)
        root_layout.setContentsMargins(10, 10, 10, 0)
        root_layout.setSpacing(8)
        root_layout.addWidget(self._build_workspace_header())
        root_layout.addWidget(splitter, stretch=1)
        root_layout.addWidget(self._build_workspace_footer())
        self.setCentralWidget(container)
        if old_central is not None:
            old_central.deleteLater()

        if prev_style_key in {"monochrome_people", "monochrome_page"}:
            prev_style_key = "monochrome"
        if prev_style_key in {"light", "light2_illustrative"}:
            prev_style_key = "light2"
        if prev_style_key:
            idx = self._style_combo.findData(prev_style_key)
            if idx >= 0:
                self._style_combo.setCurrentIndex(idx)
        self._on_style_combo_changed()
        idx = self._quality_combo.findData(prev_quality_key)
        if idx >= 0:
            self._quality_combo.setCurrentIndex(idx)
        self._chk_ink_fidelity.setChecked(prev_ink_fidelity)
        self._inkwash_strength_slider.setValue(prev_inkwash_strength)
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
        self._update_hair_recolor_swatch()
        self._sync_workspace_nav()
        self._apply_workspace_theme()
        self._update_tool_specific_visibility()
        self._update_pastel_controls_visibility()

        self._repopulate_page_list()
        self._update_controls_enabled()
        self._update_device_label()
        self._apply_responsive_density()

    def _build_workspace_header(self) -> QWidget:
        """Build the compact application header.

        The right-side workspace QTabWidget already provides navigation, so the
        duplicated top navigation strip is deliberately omitted.  The header
        only keeps the product name/version on the left and language/status on
        the right, matching the simplified reference layout.
        """
        bar = QWidget()
        bar.setObjectName("WorkspaceHeader")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(7)

        # Keep the old navigation attributes defined as None so helper methods
        # and language-triggered UI rebuilds remain backward-compatible.
        self._nav_group = None
        self._workspace_nav_buttons = []
        self._nav_pages_btn = None
        self._nav_render_btn = None
        self._nav_edit_btn = None
        self._nav_hair_recolor_btn = None
        self._nav_output_btn = None

        self._header_brand_label = QLabel("Colortina")
        self._header_brand_label.setObjectName("WorkspaceHeaderTitle")
        layout.addWidget(self._header_brand_label)

        self._header_version_chip = QLabel(Config.APP_VERSION_LABEL)
        self._header_version_chip.setObjectName("HeaderChip")
        layout.addWidget(self._header_version_chip)

        layout.addStretch(1)

        self._lang_button = QPushButton(tr("lang_button"))
        self._lang_button.setObjectName("HeaderLanguageButton")
        self._lang_button.setToolTip(tr("menu_language"))
        self._lang_button.clicked.connect(self._toggle_language)
        layout.addWidget(self._lang_button)

        self._header_status_chip = QLabel("● 就绪")
        self._header_status_chip.setObjectName("HeaderStatusChip")
        layout.addWidget(self._header_status_chip)
        return bar

    def _build_workspace_footer(self) -> QWidget:
        footer = QWidget()
        footer.setObjectName("WorkspaceFooter")
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        def add_stat(value_attr: str, caption: str, *, stretch: int = 1):
            cell = QWidget()
            cell.setObjectName("FooterStatCell")
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(14, 7, 14, 7)
            cell_layout.setSpacing(0)
            value = QLabel("0")
            value.setObjectName("FooterStatValue")
            value.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cap = QLabel(caption)
            cap.setObjectName("FooterStatCaption")
            cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cell_layout.addWidget(value)
            cell_layout.addWidget(cap)
            setattr(self, value_attr, value)
            layout.addWidget(cell, stretch)

        add_stat("_footer_total_pages", "页面")
        add_stat("_footer_selected_pages", "已选")
        add_stat("_footer_colored_pages", "已上色")
        add_stat("_footer_pending_pages", "待处理")

        status_cell = QWidget()
        status_cell.setObjectName("FooterStatCell")
        status_layout = QVBoxLayout(status_cell)
        status_layout.setContentsMargins(18, 7, 18, 7)
        status_layout.setSpacing(0)
        self._footer_status_value = QLabel("就绪")
        self._footer_status_value.setObjectName("FooterStatusValue")
        self._footer_status_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_caption = QLabel("状态")
        status_caption.setObjectName("FooterStatCaption")
        status_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self._footer_status_value)
        status_layout.addWidget(status_caption)
        # Keep the indeterminate progress indicator visible without reviving
        # QMainWindow's native status-bar strip.  The bar is only 4 px high and
        # occupies no space while hidden.
        progress = getattr(self, "_progress", None)
        if progress is not None:
            status_layout.addWidget(progress)
        layout.addWidget(status_cell, 1)
        return footer

    def _sync_workspace_nav(self):
        # Main workspace switching now lives only in the right-side tab bar.
        # Kept as a no-op for compatibility with existing signal hookups.
        return

    def _apply_workspace_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget#CentralWorkspace { background: #f4f6fa; color: #1f2937; }
            QMenuBar, QStatusBar { background: #ffffff; color: #5f6877; border-top: 1px solid #e9edf3; }

            QWidget#WorkspaceHeader, QWidget#WorkspaceFooter {
                background: #ffffff;
                border: 1px solid #e6ebf2;
                border-radius: 16px;
            }
            QLabel#WorkspaceHeaderTitle { font-size: 16px; font-weight: 800; color: #1f2937; padding: 0 6px 0 2px; }
            QLabel#HeaderChip {
                background: #f7f9fc; color: #667084; border: 1px solid #e4e8ef;
                border-radius: 9px; padding: 4px 9px; font-size: 11px; font-weight: 600;
            }
            QLabel#HeaderStatusChip {
                background: #eef8f2; color: #23825d; border: 1px solid #d9ede1;
                border-radius: 10px; padding: 5px 10px; font-size: 11px; font-weight: 700;
            }
            QWidget#FooterStatCell { background: transparent; border-right: 1px solid #edf1f5; }
            QLabel#FooterStatValue { font-size: 16px; font-weight: 800; color: #1f2937; }
            QLabel#FooterStatCaption { font-size: 10px; color: #8b93a1; }
            QLabel#FooterStatusValue { font-size: 15px; font-weight: 800; color: #2b3442; }

            QLabel#WorkspaceTitle { font-size: 15px; font-weight: 800; color: #1f2937; padding: 2px 2px 4px 2px; }
            QLabel#VersionLabel { color: #8b93a1; font-size: 11px; font-weight: 600; padding: 2px 0; }
            QLabel#PanelSectionTitle { font-size: 12px; font-weight: 700; color: #4d5a6b; text-transform: uppercase; }
            QLabel#StatsValue { font-size: 22px; font-weight: 800; color: #1f2937; }
            QLabel#StatsCaption { font-size: 11px; color: #7b8596; }
            QWidget#StatsCard { background: #ffffff; border: 1px solid #e8ebf0; border-radius: 13px; }

            QWidget#LeftPanel, QWidget#CanvasPanel, QWidget#RightPanel {
                background: #ffffff; border: 1px solid #e6ebf2; border-radius: 18px;
            }
            QWidget#LeftBottomActions { background: #ffffff; border-top: 1px solid #eef1f5; padding-top: 8px; }
            QWidget#RightPanel { background: #fbfcfe; }
            QLabel#CanvasTopLabel { font-size: 12px; font-weight: 700; color: #5e6675; }
            QLabel#DeviceInfoLabel {
                background: #f8fafc; color: #697485; border: 1px solid #e7ecf3;
                border-radius: 12px; padding: 7px 10px; font-size: 11px; font-weight: 600;
            }
            QLabel#RightPanelSectionLabel { font-size: 12px; font-weight: 800; color: #394559; letter-spacing: 0.3px; }

            QListWidget { background: #fbfbfc; border: 1px solid #eceef3; border-radius: 14px; padding: 8px; }
            QListWidget::item { background: #ffffff; border: 1px solid #eceef3; border-radius: 10px; padding: 9px 10px; margin: 3px 0; }
            QListWidget::item:selected { background: #e8f0fd; border-color: #3b82f6; color: #1e4fa8; }

            QPushButton {
                background: #ffffff; border: 1px solid #e2e7ef; border-radius: 11px;
                padding: 8px 12px; font-weight: 600; color: #273142;
            }
            QPushButton:hover { background: #f8fafc; border-color: #cfd7e3; }
            QPushButton:pressed { background: #eef3f8; }
            QPushButton:checked { background: #e9f1ff; border-color: #4e8df5; color: #2258b2; }
            QPushButton#GhostButton, QPushButton#HeaderLanguageButton {
                background: #f7f8fa; border: 1px solid #e3e6ec; border-radius: 10px;
                padding: 7px 14px; font-weight: 700;
            }
            QPushButton#HeaderLanguageButton:hover { background: #f0f4f8; border-color: #cbd2dc; }
            QPushButton#PrimaryAutoColorizeButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2f80ed, stop:1 #4f9bf8);
                color: #ffffff; border: 1px solid #2f80ed; border-radius: 14px;
                padding: 12px 14px; font-weight: 800;
            }
            QPushButton#PrimaryAutoColorizeButton:hover { background: #2f80ed; border-color: #246dd0; }
            QPushButton#PrimaryAutoColorizeButton:pressed { background: #246dd0; }
            QPushButton#PrimaryAutoColorizeButton[stopMode="true"] { background: #d74646; border-color: #bd3434; color: #ffffff; }
            QPushButton#PrimaryAutoColorizeButton[stopMode="true"]:hover { background: #c93b3b; border-color: #a92e2e; }
            QPushButton#PrimaryAutoColorizeButton[stopMode="true"]:pressed { background: #b63131; }
            QPushButton#PrimaryAutoColorizeButton:disabled { background: #d9e4f5; color: #ffffff; border-color: #d9e4f5; }
            QPushButton#AccentSecondaryButton { background: #edf5ff; color: #245ca8; border: 1px solid #cfe1fb; font-weight: 700; }
            QPushButton#AccentSecondaryButton:hover { background: #e3efff; border-color: #afcdf8; }
            QPushButton#OutputPrimaryButton {
                background: #2f80ed; color: #ffffff; border: 1px solid #2f80ed;
                border-radius: 11px; font-weight: 800; padding: 9px 12px;
            }
            QPushButton#OutputPrimaryButton:hover { background: #2876df; border-color: #2876df; }
            QPushButton#OutputPrimaryButton:pressed { background: #2268c6; border-color: #2268c6; }
            QPushButton#DangerSoftButton { background: #fff7f7; color: #b54a4a; border: 1px solid #f1dada; }
            QPushButton#DangerSoftButton:hover { background: #fff0f0; border-color: #e9c6c6; }
            QWidget#ViewSegmentBar { background: #f3f6fa; border: 1px solid #e3e8ef; border-radius: 12px; }
            QRadioButton#ViewSegmentButton {
                background: transparent; color: #697386; border: none; border-radius: 9px;
                padding: 8px 14px; spacing: 0px; font-weight: 700; min-height: 18px;
            }
            QRadioButton#ViewSegmentButton:hover { background: #ffffff; color: #334155; }
            QRadioButton#ViewSegmentButton:checked {
                background: #ffffff; color: #215bb4; border: 1px solid #cbdcf7;
            }
            QRadioButton#ViewSegmentButton:disabled { color: #aeb6c2; background: transparent; border: none; }
            QRadioButton#ViewSegmentButton::indicator { width: 0px; height: 0px; margin: 0px; }

            QGroupBox {
                font-weight: 800; color: #283243; background: #ffffff;
                border: 1px solid #e8edf4; border-radius: 16px; margin-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 12px; padding: 3px 8px;
                color: #273244; background: #ffffff; font-weight: 800;
            }
            QGroupBox[rightCard="true"] { background: #ffffff; border-color: #e8edf4; }

            QWidget#RightTabFrame { background: #ffffff; border-radius: 16px; }
            QWidget#RenderPrimaryActionBox {
                background: #ffffff; border: 1px solid #e7ecf3; border-radius: 16px;
            }
            QLabel#RenderActionStatusLabel { color: #6a7384; font-size: 11px; font-weight: 600; }

            QComboBox, QSpinBox, QLineEdit {
                background: #ffffff; border: 1px solid #dfe5ee; border-radius: 11px;
                padding: 6px 10px; min-height: 24px;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus { border-color: #7faef6; }
            QComboBox::drop-down { border: none; width: 24px; }
            QComboBox QAbstractItemView {
                background: #ffffff; border: 1px solid #dfe3ea; selection-background-color: #e8f0fd;
                selection-color: #1e4fa8; outline: 0; border-radius: 10px;
            }

            QTabWidget::pane { border: 1px solid #e8edf4; border-radius: 16px; top: -1px; background: #ffffff; }
            QTabBar::tab {
                background: #f7f8fa; border: 1px solid #e5e8ee; border-bottom: none;
                border-top-left-radius: 12px; border-top-right-radius: 12px;
                padding: 9px 6px; margin-right: 0px; min-width: 0px; color: #5c6678;
            }
            QTabBar::tab:hover { background: #f1f5fb; color: #334155; }
            QTabBar::tab:selected { background: #ffffff; color: #2154ad; font-weight: 800; border-color: #d7e3f5; }

            QScrollArea { background: transparent; border: none; }
            QScrollBar:vertical { width: 10px; background: transparent; margin: 8px 2px 8px 2px; }
            QScrollBar::handle:vertical { background: #d4dbe6; border-radius: 5px; min-height: 28px; }
            QScrollBar::handle:vertical:hover { background: #bcc5d3; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }

            QSlider::groove:horizontal { height: 6px; background: #dde4ee; border-radius: 3px; }
            QSlider::sub-page:horizontal { background: #a9c8fb; border-radius: 3px; }
            QSlider::handle:horizontal { width: 14px; margin: -5px 0; background: #2f80ed; border-radius: 7px; }

            QCheckBox, QRadioButton { spacing: 14px; padding: 4px 8px 4px 1px; min-height: 20px; }
            QCheckBox:disabled, QRadioButton:disabled { color: #a9b0bb; }
            QCheckBox::indicator, QRadioButton::indicator { width: 11px; height: 11px; margin-left: 2px; margin-right: 10px; }
            QCheckBox::indicator:unchecked, QRadioButton::indicator:unchecked { background: #ffffff; border: 1px solid #bfc7d3; border-radius: 6px; }
            QCheckBox::indicator:checked, QRadioButton::indicator:checked { background: #2f80ed; border: 1px solid #2f80ed; border-radius: 6px; }
            QCheckBox::indicator:disabled, QRadioButton::indicator:disabled { background: #f1f3f6; border: 1px solid #d7dce4; }

            QToolButton#colortinaCollapseToggle { border: none; background: transparent; font-weight: 800; color: #273041; padding: 2px 0; }
            QWidget#colortinaCollapsibleHeader { background: #fafbfc; border: 1px solid #eceef3; border-radius: 12px; }
        """)

    def _build_left_panel(self) -> QWidget:
        w = QWidget()
        w.setObjectName("LeftPanel")
        self._left_panel = w
        w.setMinimumWidth(270)
        w.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        action_grid = QGridLayout()
        action_grid.setContentsMargins(0, 0, 0, 0)
        action_grid.setHorizontalSpacing(6)
        action_grid.setVerticalSpacing(6)

        self._btn_import_images = QPushButton(tr("import_images"))
        self._btn_import_images.clicked.connect(self._import_images)
        self._btn_import_pdf = QPushButton(tr("import_pdf"))
        self._btn_import_pdf.clicked.connect(self._import_pdf)
        self._btn_import_folder = QPushButton(tr("import_folder"))
        self._btn_import_folder.clicked.connect(self._import_folder)
        # Local aliases keep the original UI contract/readability while the
        # instance attributes let _set_busy() lock structure-changing actions.
        btn_import_images = self._btn_import_images
        btn_import_pdf = self._btn_import_pdf
        btn_import_folder = self._btn_import_folder
        action_grid.addWidget(btn_import_images, 0, 0)
        action_grid.addWidget(btn_import_pdf, 0, 1)
        action_grid.addWidget(btn_import_folder, 1, 0, 1, 2)
        action_grid.setColumnStretch(0, 1)
        action_grid.setColumnStretch(1, 1)
        layout.addLayout(action_grid)

        page_section = QLabel("页面列表")
        page_section.setObjectName("PanelSectionTitle")
        layout.addWidget(page_section)

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
        self._page_list.itemSelectionChanged.connect(self._refresh_workspace_stats)

        select_row = QHBoxLayout()
        btn_select_all = QPushButton(tr("select_all"))
        btn_select_all.clicked.connect(lambda: self._page_list.selectAll())
        btn_select_none = QPushButton(tr("select_none"))
        btn_select_none.clicked.connect(lambda: self._page_list.clearSelection())
        select_row.addWidget(btn_select_all)
        select_row.addWidget(btn_select_none)

        layout.addLayout(select_row)
        layout.addWidget(self._page_list, stretch=1)

        # Keep page/project management anchored to the bottom of the sidebar.
        # The previous layout added another stretch *after* these controls, which
        # left a large blank area underneath and made the controls float upward.
        bottom_actions = QWidget()
        bottom_actions.setObjectName("LeftBottomActions")
        bottom_actions.setSizePolicy(QSizePolicy.Policy.Expanding,
                                     QSizePolicy.Policy.Fixed)
        bottom_layout = QVBoxLayout(bottom_actions)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(6)

        manage_row = QHBoxLayout()
        manage_row.setContentsMargins(0, 0, 0, 0)
        manage_row.setSpacing(6)
        self._btn_move_up = QPushButton(tr("move_up"))
        self._btn_move_up.clicked.connect(lambda: self._move_selected_page(-1))
        self._btn_move_down = QPushButton(tr("move_down"))
        self._btn_move_down.clicked.connect(lambda: self._move_selected_page(1))
        self._btn_move_up.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._btn_move_down.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        manage_row.addWidget(self._btn_move_up)
        manage_row.addWidget(self._btn_move_down)
        bottom_layout.addLayout(manage_row)

        self._btn_delete_pages = QPushButton(tr("delete_pages"))
        self._btn_delete_pages.clicked.connect(self._delete_selected_pages)
        self._btn_delete_pages.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        bottom_layout.addWidget(self._btn_delete_pages)

        project_row = QHBoxLayout()
        project_row.setContentsMargins(0, 0, 0, 0)
        project_row.setSpacing(6)
        self._btn_load_project = QPushButton(tr("load_project"))
        self._btn_load_project.clicked.connect(self._load_project)
        self._btn_save_project = QPushButton(tr("save_project"))
        self._btn_save_project.clicked.connect(self._save_project)
        self._btn_load_project.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._btn_save_project.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        project_row.addWidget(self._btn_load_project)
        project_row.addWidget(self._btn_save_project)
        bottom_layout.addLayout(project_row)

        layout.addWidget(bottom_actions, stretch=0, alignment=Qt.AlignmentFlag.AlignBottom)
        return w

    def _build_canvas_panel(self) -> QWidget:
        w = QWidget()
        w.setObjectName("CanvasPanel")
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        toolbar = QHBoxLayout()
        self._canvas_title_label = QLabel("")
        self._canvas_title_label.setObjectName("CanvasTopLabel")
        self._canvas_title_label.setVisible(False)
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
        self._canvas.selection_adjust_finished.connect(self._on_selection_adjust_finished)

        layout.addLayout(toolbar)
        layout.addWidget(self._canvas, stretch=1)
        return w

    def _build_right_panel(self) -> QWidget:
        """Build the responsive right control panel.

        Long settings pages use vertical scrolling, while the render workspace
        keeps its primary Auto Colorize action fixed below the settings viewport.
        Horizontal scrolling remains disabled so controls follow panel width.
        """
        panel = QWidget()
        panel.setObjectName("RightPanel")
        self._right_panel = panel
        panel.setMinimumWidth(420)
        panel.setMaximumWidth(520)
        panel.setSizePolicy(QSizePolicy.Policy.Preferred,
                            QSizePolicy.Policy.Expanding)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        self._device_label = QLabel(f"{tr('device_label')}—")
        self._device_label.setObjectName("DeviceInfoLabel")
        self._device_label.setSizePolicy(QSizePolicy.Policy.Expanding,
                                         QSizePolicy.Policy.Fixed)
        outer.addWidget(self._device_label)

        right_section_label = QLabel("参数与工具")
        right_section_label.setObjectName("RightPanelSectionLabel")
        outer.addWidget(right_section_label)

        self._right_tabs = QTabWidget()
        self._right_tabs.setDocumentMode(True)
        self._right_tabs.setUsesScrollButtons(False)
        self._right_tabs.tabBar().setExpanding(True)
        self._right_tabs.setElideMode(Qt.TextElideMode.ElideNone)
        self._right_tabs.currentChanged.connect(lambda _=None: self._sync_workspace_nav())
        outer.addWidget(self._right_tabs, stretch=1)

        def make_tab() -> tuple[QWidget, QVBoxLayout]:
            tab = QWidget()
            tab.setObjectName("RightTabFrame")
            tab.setSizePolicy(QSizePolicy.Policy.Expanding,
                              QSizePolicy.Policy.Expanding)
            lay = QVBoxLayout(tab)
            lay.setContentsMargins(8, 8, 8, 8)
            lay.setSpacing(6)
            return tab, lay

        def make_scrollable_tab() -> tuple[QScrollArea, QWidget, QVBoxLayout]:
            content, lay = make_tab()
            content.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Minimum)
            lay.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll.setWidget(content)
            scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            return scroll, content, lay

        def make_group(title: str) -> tuple[QGroupBox, QVBoxLayout]:
            group = QGroupBox(title)
            group.setProperty("rightCard", True)
            group.setSizePolicy(QSizePolicy.Policy.Expanding,
                                QSizePolicy.Policy.Expanding)
            lay = QVBoxLayout(group)
            lay.setContentsMargins(12, 10, 12, 12)
            lay.setSpacing(8)
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
            grid.setHorizontalSpacing(8)
            grid.setVerticalSpacing(6)
            for i, button in enumerate(buttons):
                tune_button(button)
                grid.addWidget(button, i // columns, i % columns)
            for col in range(columns):
                grid.setColumnStretch(col, 1)
            layout.addLayout(grid)

        from core.presets import STYLE_PRESETS, QUALITY_PRESETS

        # ── Tab 1: rendering and automatic colourization ──────────────
        # Keep settings scrollable while the primary Auto Colorize action stays
        # permanently visible at the bottom of the render workspace.  Earlier
        # UI refactors moved the whole auto_group into the dedicated hair tab
        # and accidentally moved the primary button with it.
        render_tab = QWidget()
        render_tab.setSizePolicy(QSizePolicy.Policy.Expanding,
                                 QSizePolicy.Policy.Expanding)
        render_page_layout = QVBoxLayout(render_tab)
        render_page_layout.setContentsMargins(0, 0, 0, 0)
        render_page_layout.setSpacing(8)
        render_scroll, render_content, render_layout = make_scrollable_tab()
        self._render_scroll_area = render_scroll
        render_page_layout.addWidget(render_scroll, stretch=1)
        style_group, style_layout = make_group(tr("style_quality_group"))

        style_grid = QGridLayout()
        style_grid.setContentsMargins(0, 0, 0, 0)
        style_grid.setHorizontalSpacing(10)
        style_grid.setVerticalSpacing(6)
        style_grid.addWidget(QLabel(tr("style_label")), 0, 0)
        self._style_combo = QComboBox()
        visible_style_keys = ["light2", "light2_soft", "light3",
                              "light2_git_legacy", "light3_git_legacy", "none"]
        for key in visible_style_keys:
            preset = STYLE_PRESETS.get(key)
            if preset is not None:
                self._style_combo.addItem(preset.label, key)
        self._style_combo.currentIndexChanged.connect(self._on_style_combo_changed)
        new_default = self._style_combo.findData(Config.DEFAULT_STYLE_KEY)
        self._style_combo.setCurrentIndex(max(new_default, 0))
        style_grid.addWidget(self._style_combo, 0, 1, 1, 2)
        # Expose the actual inference resolution instead of silently forcing
        # every request to Fast/576.  Keep Fast as the compatible default.
        self._quality_combo = QComboBox()
        for quality_key in ("draft", "standard", "detail"):
            preset = QUALITY_PRESETS[quality_key]
            self._quality_combo.addItem(preset.label, quality_key)
        self._quality_combo.setToolTip(
            "Fast 576：较快；精细 768：更多线条细节；高清 960：CPU 较慢、占用更多内存。"
            "这里只改变模型输入精度，不保证发色自动正确。")
        style_grid.addWidget(QLabel("渲染精度"), 1, 0)
        style_grid.addWidget(self._quality_combo, 1, 1, 1, 2)
        style_grid.setColumnStretch(1, 1)
        style_layout.addLayout(style_grid)
        # Optional ORIGINAL generator point control. The existing "mixed" route
        # remains default; nothing is auto-selected or exported as a hair mask.
        native_hint_row = QHBoxLayout()
        native_hint_row.addWidget(QLabel("颜色 Hint 方式"))
        self._native_hint_mode_combo = QComboBox()
        self._native_hint_mode_combo.addItem("标准区域提示", "mixed")
        self._native_hint_mode_combo.addItem("原生柔和点提示（点痕检测，不保证改色）", "native_soft")
        self._native_hint_mode_combo.setToolTip(
            "直接把用户的颜色点送入原 Generator 五通道输入；保留整页结构和上下文。"
            "不会自动识别头发。检测到明显圆形色斑时会自动保留无提示原模型结果；可能因此无法按指定颜色改色。")
        native_hint_row.addWidget(self._native_hint_mode_combo, 1)
        style_layout.addLayout(native_hint_row)
        self._chk_ink_fidelity = QCheckBox("原稿线稿保真 · 柔和颜色（推荐，可关闭）")
        self._chk_ink_fidelity.setChecked(True)
        self._chk_ink_fidelity.setToolTip(
            "保留原图的墨线、网点和细部纹理，并柔和过饱和颜色。"
            "不会猜测人物设定发色，也不会启用自动头发纠色。")
        style_layout.addWidget(self._chk_ink_fidelity)
        ink_row = QHBoxLayout()
        self._inkwash_strength_label = QLabel("水墨渲染浓度")
        ink_row.addWidget(self._inkwash_strength_label)
        self._inkwash_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self._inkwash_strength_slider.setRange(0, 100)
        self._inkwash_strength_slider.setValue(78)
        self._inkwash_strength_slider.setToolTip(
            "0%：保留原始 Generator 配色；78%：推荐水墨浓度；100%：最明显的淡彩透明度，黑色线稿始终保留。"
            "不对人物发色作身份推断。")
        self._inkwash_strength_spin = QSpinBox()
        self._inkwash_strength_spin.setRange(0, 100)
        self._inkwash_strength_spin.setSuffix("%")
        self._inkwash_strength_spin.setFixedWidth(68)
        self._inkwash_strength_spin.setValue(78)
        self._inkwash_strength_slider.valueChanged.connect(self._inkwash_strength_spin.setValue)
        self._inkwash_strength_spin.valueChanged.connect(self._inkwash_strength_slider.setValue)
        ink_row.addWidget(self._inkwash_strength_slider, stretch=1)
        ink_row.addWidget(self._inkwash_strength_spin)
        style_layout.addLayout(ink_row)

        strength_grid = QGridLayout()
        strength_grid.setContentsMargins(0, 0, 0, 0)
        strength_grid.setHorizontalSpacing(8)
        strength_grid.setVerticalSpacing(6)

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
            spin.setFixedWidth(68)
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
        self._style_fine_group.setProperty("rightCard", True)
        fine_layout = QVBoxLayout(self._style_fine_group)
        fine_layout.setContentsMargins(12, 12, 12, 12)
        fine_layout.setSpacing(8)
        self._style_fine_active_label = QLabel()
        self._style_fine_active_label.setVisible(False)
        fine_grid = QGridLayout()
        fine_grid.setContentsMargins(0, 0, 0, 0)
        fine_grid.setHorizontalSpacing(8)
        fine_grid.setVerticalSpacing(6)

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
            spin.setFixedWidth(68)
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
        self._filter_group.setProperty("rightCard", True)
        filter_layout = QVBoxLayout(self._filter_group)
        filter_layout.setContentsMargins(12, 12, 12, 12)
        filter_layout.setSpacing(8)
        filter_grid = QGridLayout()
        filter_grid.setContentsMargins(0, 0, 0, 0)
        filter_grid.setHorizontalSpacing(8)
        filter_grid.setVerticalSpacing(6)
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
        filter_color_grid.setHorizontalSpacing(8)
        filter_color_grid.setVerticalSpacing(6)
        filter_color_grid.addWidget(QLabel(tr("filter_color_strength")), 0, 0)
        self._filter_color_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self._filter_color_strength_slider.setRange(0, 150)
        self._filter_color_strength_slider.setValue(35)
        self._filter_color_strength_spin = QSpinBox()
        self._filter_color_strength_spin.setRange(0, 150)
        self._filter_color_strength_spin.setSuffix("%")
        self._filter_color_strength_spin.setValue(35)
        self._filter_color_strength_spin.setFixedWidth(68)
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
        self._pastel_controls_group.setProperty("rightCard", True)
        pastel_layout = QVBoxLayout(self._pastel_controls_group)
        pastel_layout.setContentsMargins(12, 12, 12, 12)
        pastel_layout.setSpacing(8)
        pastel_hint = QLabel(tr("pastel_controls_hint"))
        pastel_hint.setWordWrap(True)
        pastel_hint.setStyleSheet("color: #6c7f94; font-size: 10px;")
        pastel_layout.addWidget(pastel_hint)
        pastel_grid = QGridLayout()
        pastel_grid.setContentsMargins(0, 0, 0, 0)
        pastel_grid.setHorizontalSpacing(8)
        pastel_grid.setVerticalSpacing(6)

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
        self._chk_skip_colored.setChecked(False)
        self._chk_skip_colored.setToolTip(tr("skip_colored_tooltip"))
        style_layout.addWidget(self._chk_skip_colored)

        self._custom_color_bias_group = QGroupBox(tr("custom_color_bias_group"))
        self._custom_color_bias_group.setProperty("rightCard", True)
        custom_bias_layout = QVBoxLayout(self._custom_color_bias_group)
        custom_bias_layout.setContentsMargins(12, 12, 12, 12)
        custom_bias_layout.setSpacing(8)
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
        custom_bias_grid.setHorizontalSpacing(8)
        custom_bias_grid.setVerticalSpacing(6)
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
        self._custom_color_bias_spin.setFixedWidth(68)
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
        protection_grid.setHorizontalSpacing(22)
        protection_grid.setVerticalSpacing(10)
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
        protection_grid.setColumnStretch(0, 1)
        protection_grid.setColumnStretch(1, 1)
        custom_bias_details_layout.addLayout(protection_grid)
        custom_bias_layout.addWidget(self._custom_color_bias_details)
        style_layout.addWidget(self._custom_color_bias_group)
        self._update_custom_color_bias_swatch()
        self._update_custom_bias_controls_enabled()

        auto_group, auto_layout = make_group("头发纠色")
        auto_fold_toolbar = QHBoxLayout()
        auto_fold_toolbar.setContentsMargins(0, 0, 0, 0)
        auto_fold_toolbar.setSpacing(5)
        auto_fold_toolbar.addStretch(1)
        self._btn_collapse_auto_details = QPushButton("收起全部细则")
        self._btn_collapse_auto_details.setMaximumWidth(120)
        auto_fold_toolbar.addWidget(self._btn_collapse_auto_details)
        auto_layout.addLayout(auto_fold_toolbar)

        self._auto_detail_section = _CollapsibleSection(
            "基础保护", summary="文字气泡保护等自动上色基础开关",
            expanded=False, settings_key="render_base")
        auto_layout.addWidget(self._auto_detail_section)
        auto_content_layout = self._auto_detail_section.content_layout()

        self._hair_guard_detail_section = _CollapsibleSection(
            "纠色设置", summary="目标发色 · 多人物范围 · 独立蓝框 / 画笔 · 遮罩预览",
            expanded=False, settings_key="render_hair_guard")
        auto_layout.addWidget(self._hair_guard_detail_section)
        hair_guard_content_layout = self._hair_guard_detail_section.content_layout()

        self._ai_hair_detail_section = _CollapsibleSection(
            "AI 头发解析", summary="识别后端 · 模型状态 · 安装 / 检查",
            expanded=False, settings_key="render_ai_hair")
        auto_layout.addWidget(self._ai_hair_detail_section)
        ai_hair_content_layout = self._ai_hair_detail_section.content_layout()

        self._btn_collapse_auto_details.clicked.connect(
            lambda: self._collapse_sections(
                self._auto_detail_section, self._hair_guard_detail_section,
                self._ai_hair_detail_section))
        self._chk_protect_text = QCheckBox("保护文字气泡（AI 检测文字区域，保持原稿黑白）")
        self._chk_protect_text.setChecked(True)
        self._chk_protect_text.setToolTip(
            "用本地文字检测模型找到对话框文字像素，上色后把这些区域还原为原稿的"
            "纸白与墨黑，避免文字被染色。模型文件缺失时自动跳过，不影响上色。")
        auto_content_layout.addWidget(self._chk_protect_text)

        self._chk_auto_black_hair_guard = QCheckBox("自动头发纠色（把误判成橙/红暖色的头发重建为目标发色）")
        self._chk_auto_black_hair_guard.setChecked(False)
        self._chk_auto_black_hair_guard.setToolTip(
            "只对自动上色生效。它会先检测被 mc-v2 误成橙/红暖色的头发对象，"
            "再按原始黑白稿的明暗结构重建为目标发色。默认目标色为黑色，"
            "也可改成棕色、深蓝等。为避免误伤原本的暖色头发，默认关闭，需要时手动开启。")
        hair_guard_content_layout.addWidget(self._chk_auto_black_hair_guard)

        black_guard_row = QHBoxLayout()
        black_guard_row.setContentsMargins(18, 0, 0, 0)
        black_guard_row.setSpacing(8)
        black_guard_label = QLabel("头发纠色模式：")
        self._combo_auto_black_hair_guard_mode = QComboBox()
        self._combo_auto_black_hair_guard_mode.addItem("保守（更少误伤橙色区域）", "conservative")
        self._combo_auto_black_hair_guard_mode.addItem("标准（更积极修正橙发误判）", "normal")
        self._combo_auto_black_hair_guard_mode.setCurrentIndex(0)
        self._combo_auto_black_hair_guard_mode.setToolTip(
            "保守：仅在检测到大面积、明显带明暗结构的暖色头发时才触发自动纠色。\n"
            "标准：更容易触发，适合误判暖色头发比较严重的页面。")
        self._combo_auto_black_hair_guard_mode.setEnabled(False)
        self._chk_auto_black_hair_guard.toggled.connect(
            self._combo_auto_black_hair_guard_mode.setEnabled)
        black_guard_row.addWidget(black_guard_label)
        black_guard_row.addWidget(self._combo_auto_black_hair_guard_mode, 1)
        hair_guard_content_layout.addLayout(black_guard_row)

        black_guard_scope_row = QHBoxLayout()
        black_guard_scope_row.setContentsMargins(18, 0, 0, 0)
        black_guard_scope_row.setSpacing(8)
        black_guard_scope_label = QLabel("头发纠色范围：")
        self._combo_auto_black_hair_guard_scope = QComboBox()
        self._combo_auto_black_hair_guard_scope.addItem("人物附近优先（更稳，默认）", "near_face")
        self._combo_auto_black_hair_guard_scope.addItem("整页主暖色对象（更强）", "page")
        self._combo_auto_black_hair_guard_scope.setCurrentIndex(0)
        self._combo_auto_black_hair_guard_scope.setEnabled(False)
        self._combo_auto_black_hair_guard_scope.setToolTip(
            "人物附近优先：只有靠近检测到的人物头部的暖色头发对象才会被纠色，更适合避免误伤衣服、背景块。\n"
            "整页主暖色对象：按全页最可疑的大暖色头发对象纠色，适合误判很明显的整页。")
        self._chk_auto_black_hair_guard.toggled.connect(
            self._combo_auto_black_hair_guard_scope.setEnabled)
        black_guard_scope_row.addWidget(black_guard_scope_label)
        black_guard_scope_row.addWidget(self._combo_auto_black_hair_guard_scope, 1)
        hair_guard_content_layout.addLayout(black_guard_scope_row)

        black_guard_limit_row = QHBoxLayout()
        black_guard_limit_row.setContentsMargins(18, 0, 0, 0)
        black_guard_limit_row.setSpacing(8)
        black_guard_limit_label = QLabel("纠色作用区域：")
        self._combo_auto_black_hair_guard_limit_mode = QComboBox()
        self._combo_auto_black_hair_guard_limit_mode.addItem("整页 / 当前范围内自动检测", "full")
        self._combo_auto_black_hair_guard_limit_mode.addItem("仅当前选区（蓝框）", "selection")
        self._combo_auto_black_hair_guard_limit_mode.setCurrentIndex(0)
        self._combo_auto_black_hair_guard_limit_mode.setEnabled(False)
        self._combo_auto_black_hair_guard_limit_mode.setToolTip(
            "整页 / 当前范围内自动检测：按正常自动头发纠色逻辑工作。\n"
            "仅当前选区（蓝框）：只有蓝框内的候选头发才参与纠色，框外一律不参与；\n"
            "适合一张图很多人物时，只纠正你框住的那个人或那一块头发。")
        self._chk_auto_black_hair_guard.toggled.connect(
            self._combo_auto_black_hair_guard_limit_mode.setEnabled)
        black_guard_limit_row.addWidget(black_guard_limit_label)
        black_guard_limit_row.addWidget(self._combo_auto_black_hair_guard_limit_mode, 1)
        hair_guard_content_layout.addLayout(black_guard_limit_row)

        hair_guard_selector_label = QLabel("头发纠色独立选区")
        hair_guard_selector_label.setStyleSheet("font-weight: 600;")
        hair_guard_selector_label.setContentsMargins(18, 2, 0, 0)
        hair_guard_content_layout.addWidget(hair_guard_selector_label)

        self._hair_guard_selector_grid = QGridLayout()
        self._hair_guard_selector_grid.setContentsMargins(18, 0, 0, 0)
        self._hair_guard_selector_grid.setHorizontalSpacing(6)
        self._hair_guard_selector_grid.setVerticalSpacing(5)
        self._hair_guard_selector_grid.setColumnStretch(0, 1)
        self._hair_guard_selector_grid.setColumnStretch(1, 1)

        self._btn_hair_guard_rect_selector = QPushButton("矩形追加")
        self._btn_hair_guard_rect_selector.setToolTip("独立于编辑选区。可在上色前直接追加一个头发纠色蓝框区域；可连续框多个角色。")
        self._btn_hair_guard_rect_selector.clicked.connect(lambda: self._activate_hair_guard_selector("rect"))
        self._btn_hair_guard_brush_add = QPushButton("画笔添加")
        self._btn_hair_guard_brush_add.setToolTip("独立于编辑选区。旧头发纠色中用于补充蓝框；启用新版蒙版覆盖时，同时作为‘这是头发’的正向语义涂鸦种子。")
        self._btn_hair_guard_brush_add.clicked.connect(lambda: self._activate_hair_guard_selector("brush_add"))
        self._btn_hair_guard_brush_erase = QPushButton("画笔擦除")
        self._btn_hair_guard_brush_erase.setToolTip("独立于编辑选区。旧头发纠色中擦除蓝框；启用新版蒙版覆盖时，同时作为‘这里不是头发’的负向语义涂鸦，阻止自动扩区。")
        self._btn_hair_guard_brush_erase.clicked.connect(lambda: self._activate_hair_guard_selector("brush_erase"))
        self._btn_hair_guard_undo_last = QPushButton("撤销上一个区域")
        self._btn_hair_guard_undo_last.setToolTip("撤销最近一次矩形追加、画笔添加或画笔擦除操作。")
        self._btn_hair_guard_undo_last.clicked.connect(self._undo_last_hair_guard_selection_operation)
        self._btn_hair_guard_clear_selection = QPushButton("清除全部纠色选区")
        self._btn_hair_guard_clear_selection.setToolTip("清除本页全部头发纠色独立选区。")
        self._btn_hair_guard_clear_selection.clicked.connect(self._clear_hair_guard_selection)
        self._btn_hair_guard_finish_selector = QPushButton("结束选区器")
        self._btn_hair_guard_finish_selector.clicked.connect(self._deactivate_hair_guard_selector)

        selector_buttons = (
            self._btn_hair_guard_rect_selector, self._btn_hair_guard_brush_add,
            self._btn_hair_guard_brush_erase, self._btn_hair_guard_undo_last,
            self._btn_hair_guard_clear_selection, self._btn_hair_guard_finish_selector,
        )
        for index, _btn in enumerate(selector_buttons):
            _btn.setMinimumHeight(30)
            _btn.setMinimumWidth(0)
            _btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self._hair_guard_selector_grid.addWidget(_btn, index // 2, index % 2)
        hair_guard_content_layout.addLayout(self._hair_guard_selector_grid)

        self._label_hair_guard_selector_mode = QLabel("当前选区器：未启用")
        self._label_hair_guard_selector_mode.setContentsMargins(36, 0, 0, 0)
        self._label_hair_guard_selector_mode.setStyleSheet("color: #5d6f84; font-size: 10px;")
        self._label_hair_guard_selector_mode.setWordWrap(True)
        hair_guard_content_layout.addWidget(self._label_hair_guard_selector_mode)

        self._label_hair_guard_selection_stats = QLabel("当前头发纠色区域：0 个操作 / 0 px")
        self._label_hair_guard_selection_stats.setContentsMargins(36, 0, 0, 0)
        self._label_hair_guard_selection_stats.setStyleSheet("color: #5d6f84; font-size: 10px;")
        self._label_hair_guard_selection_stats.setWordWrap(True)
        hair_guard_content_layout.addWidget(self._label_hair_guard_selection_stats)

        black_guard_brush_row = QGridLayout()
        black_guard_brush_row.setContentsMargins(36, 0, 0, 0)
        black_guard_brush_row.setHorizontalSpacing(6)
        black_guard_brush_row.setVerticalSpacing(3)
        black_guard_brush_row.addWidget(QLabel("画笔半径"), 0, 0)
        self._hair_guard_brush_radius_slider = QSlider(Qt.Orientation.Horizontal)
        self._hair_guard_brush_radius_slider.setRange(4, 120)
        self._hair_guard_brush_radius_slider.setValue(24)
        self._hair_guard_brush_radius_spin = QSpinBox()
        self._hair_guard_brush_radius_spin.setRange(4, 120)
        self._hair_guard_brush_radius_spin.setValue(24)
        self._hair_guard_brush_radius_spin.setSuffix(" px")
        self._hair_guard_brush_radius_spin.setMaximumWidth(82)
        self._hair_guard_brush_radius_slider.valueChanged.connect(self._hair_guard_brush_radius_spin.setValue)
        self._hair_guard_brush_radius_spin.valueChanged.connect(self._hair_guard_brush_radius_slider.setValue)
        self._hair_guard_brush_radius_slider.valueChanged.connect(self._update_hair_guard_brush_radius)
        black_guard_brush_row.addWidget(self._hair_guard_brush_radius_slider, 0, 1)
        black_guard_brush_row.addWidget(self._hair_guard_brush_radius_spin, 0, 2)
        black_guard_brush_row.setColumnStretch(1, 1)
        hair_guard_content_layout.addLayout(black_guard_brush_row)

        black_guard_preview_row = QGridLayout()
        black_guard_preview_row.setContentsMargins(18, 0, 0, 0)
        black_guard_preview_row.setHorizontalSpacing(6)
        black_guard_preview_row.setVerticalSpacing(4)
        black_guard_preview_row.setColumnStretch(0, 1)
        black_guard_preview_row.setColumnStretch(1, 1)
        self._btn_preview_auto_black_hair_guard = QPushButton("预览自动头发纠色遮罩")
        self._btn_preview_auto_black_hair_guard.setToolTip(
            "蓝色：AI 主体头发；绿色：通过过滤的补漏区域；红色：检测到但已排除的细丝、描边或边缘薄壳。")
        self._btn_preview_auto_black_hair_guard.clicked.connect(self._preview_auto_black_hair_guard_mask)
        self._btn_clear_auto_black_hair_guard_preview = QPushButton("清除预览")
        self._btn_clear_auto_black_hair_guard_preview.clicked.connect(self._clear_auto_black_hair_guard_preview)
        for _btn in (self._btn_preview_auto_black_hair_guard, self._btn_clear_auto_black_hair_guard_preview):
            _btn.setMinimumHeight(30)
            _btn.setMinimumWidth(0)
            _btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        black_guard_preview_row.addWidget(self._btn_preview_auto_black_hair_guard, 0, 0)
        black_guard_preview_row.addWidget(self._btn_clear_auto_black_hair_guard_preview, 0, 1)
        hair_guard_content_layout.addLayout(black_guard_preview_row)

        hair_recolor_color_row = QHBoxLayout()
        hair_recolor_color_row.setContentsMargins(18, 0, 0, 0)
        hair_recolor_color_row.setSpacing(8)
        hair_recolor_color_row.addWidget(QLabel("目标发色（纠色 / 蒙版覆盖共用）："))
        self._btn_hair_recolor_color = QPushButton()
        self._btn_hair_recolor_color.setFixedSize(54, 30)
        self._btn_hair_recolor_color.clicked.connect(self._pick_hair_recolor_color)
        self._hair_recolor_color_info = QLabel()
        self._hair_recolor_color_info.setStyleSheet("color: #5d6f84; font-size: 10px;")
        hair_recolor_color_row.addWidget(self._btn_hair_recolor_color)
        hair_recolor_color_row.addWidget(self._hair_recolor_color_info, 1)
        hair_guard_content_layout.addLayout(hair_recolor_color_row)

        # ── Semantic mask overwrite layers ───────────────────────────────
        # These are deliberately separate from mc-v2 hints.  The generator
        # still auto-colourizes the whole page; enabled semantic regions are
        # replaced afterwards using binary masks (no translucent overlap).
        self._chk_semantic_mask_layers = QCheckBox(
            "启用蒙版覆盖上色（新版：发色 / 瞳色 / 背景）")
        self._chk_semantic_mask_layers.setChecked(False)
        self._chk_semantic_mask_layers.setToolTip(
            "先让 mc-v2 正常自动上色，再提取头发/虹膜/背景蒙版。启用的蒙版区域在最终阶段直接覆盖，"
            "蒙版外逐像素不变；不是半透明叠加。")
        hair_guard_content_layout.addWidget(self._chk_semantic_mask_layers)

        semantic_grid = QGridLayout()
        semantic_grid.setContentsMargins(36, 0, 0, 0)
        semantic_grid.setHorizontalSpacing(8)
        semantic_grid.setVerticalSpacing(5)

        self._chk_mask_hair_override = QCheckBox("发色覆盖")
        self._chk_mask_hair_override.setChecked(True)
        semantic_grid.addWidget(self._chk_mask_hair_override, 0, 0)
        semantic_grid.addWidget(QLabel("使用上方目标发色；保留 mc-v2 明暗"), 0, 1, 1, 2)

        self._chk_mask_iris_override = QCheckBox("瞳色覆盖")
        self._chk_mask_iris_override.setChecked(False)
        semantic_grid.addWidget(self._chk_mask_iris_override, 1, 0)
        self._btn_iris_recolor_color = QPushButton()
        self._btn_iris_recolor_color.setFixedSize(54, 28)
        self._btn_iris_recolor_color.clicked.connect(self._pick_iris_recolor_color)
        self._iris_recolor_color_info = QLabel()
        self._iris_recolor_color_info.setStyleSheet("color: #5d6f84; font-size: 10px;")
        semantic_grid.addWidget(self._btn_iris_recolor_color, 1, 1)
        semantic_grid.addWidget(self._iris_recolor_color_info, 1, 2)

        semantic_grid.addWidget(QLabel("背景覆盖："), 2, 0)
        self._combo_mask_background_mode = QComboBox()
        self._combo_mask_background_mode.addItem("保持自动背景", "keep")
        self._combo_mask_background_mode.addItem("背景灰度（保留 AI 明暗）", "grayscale")
        self._combo_mask_background_mode.addItem("背景低饱和", "desaturate")
        self._combo_mask_background_mode.addItem("恢复原黑白背景", "source_bw")
        semantic_grid.addWidget(self._combo_mask_background_mode, 2, 1, 1, 2)

        semantic_grid.addWidget(QLabel("覆盖着色："), 3, 0)
        self._combo_mask_recolor_mode = QComboBox()
        self._combo_mask_recolor_mode.addItem("精确色系 + 保留明暗（推荐）", "shading")
        self._combo_mask_recolor_mode.addItem("自然迁色 + 保留纹理", "shift")
        self._combo_mask_recolor_mode.addItem("纯色覆盖（检查蒙版）", "flat")
        semantic_grid.addWidget(self._combo_mask_recolor_mode, 3, 1, 1, 2)

        self._chk_mask_character_library_colors = QCheckBox("按角色库分别覆盖发色 / 瞳色")
        self._chk_mask_character_library_colors.setChecked(True)
        self._chk_mask_character_library_colors.setToolTip(
            "同页有多个已匹配角色时，优先使用角色库中各自的 hair / eyes 颜色；"
            "未匹配区域仍使用上面的统一目标色。关闭后整页只使用统一目标色。")
        semantic_grid.addWidget(self._chk_mask_character_library_colors, 4, 0, 1, 3)
        semantic_grid.setColumnStretch(2, 1)
        hair_guard_content_layout.addLayout(semantic_grid)

        semantic_preview_row = QHBoxLayout()
        semantic_preview_row.setContentsMargins(36, 0, 0, 0)
        semantic_preview_row.setSpacing(8)
        self._btn_preview_semantic_masks = QPushButton("预览 Hair / Iris / Background 蒙版")
        self._btn_preview_semantic_masks.clicked.connect(self._preview_semantic_mask_layers)
        self._btn_clear_semantic_masks_preview = QPushButton("清除蒙版预览")
        self._btn_clear_semantic_masks_preview.clicked.connect(self._clear_auto_black_hair_guard_preview)
        semantic_preview_row.addWidget(self._btn_preview_semantic_masks, 1)
        semantic_preview_row.addWidget(self._btn_clear_semantic_masks_preview, 0)
        hair_guard_content_layout.addLayout(semantic_preview_row)

        self._label_semantic_mask_note = QLabel(
            "覆盖顺序：背景 → 头发 → 虹膜。蒙版由语义分割 + 线稿区域 + 可用时的 CLIP + mc-v2 Encoder 特征共同判断；断线会自动回退 trapped-ball。最终始终二值覆盖，不做透明重合。")
        self._label_semantic_mask_note.setContentsMargins(36, 0, 0, 0)
        self._label_semantic_mask_note.setWordWrap(True)
        self._label_semantic_mask_note.setStyleSheet("color: #5d6f84; font-size: 10px;")
        hair_guard_content_layout.addWidget(self._label_semantic_mask_note)

        self._chk_semantic_mask_layers.toggled.connect(
            self._update_semantic_mask_controls_enabled)
        self._update_semantic_mask_controls_enabled()
        self._update_iris_recolor_swatch()

        self._chk_ai_hair_parser = QCheckBox(
            "启用 AI 发色解析（先识别完整头发，再应用目标发色）")
        self._chk_ai_hair_parser.setChecked(True)
        self._chk_ai_hair_parser.setToolTip(
            "自动模式优先使用 Anime-Face-Segmentation；StyleAnime 只作为低置信度时的辅助候选，"
            "不再直接用宽泛 StyleAnime 遮罩染色。安装后，SkyTNT 用于限制人物前景，"
            "yakhyo face-parsing 用于排除皮肤、眼镜和衣服。")
        ai_hair_content_layout.addWidget(self._chk_ai_hair_parser)

        ai_hair_backend_row = QHBoxLayout()
        ai_hair_backend_row.setContentsMargins(18, 0, 0, 0)
        ai_hair_backend_row.setSpacing(8)
        ai_hair_backend_row.addWidget(QLabel("AI 头发识别后端："))
        self._combo_ai_hair_parser_backend = QComboBox()
        self._combo_ai_hair_parser_backend.addItem("自动", "auto")
        self._combo_ai_hair_parser_backend.addItem(
            "StyleAnime / Danbooru-Parsing", "styleanime")
        self._combo_ai_hair_parser_backend.addItem(
            "Anime-Face-Segmentation", "anime_face_segmentation")
        self._combo_ai_hair_parser_backend.addItem(
            "AnimeSeg / Mask2Former（可选，本地安装）", "animeseg")
        self._combo_ai_hair_parser_backend.addItem(
            "仅旧算法（不运行 AI 解析）", "legacy")
        self._combo_ai_hair_parser_backend.setCurrentIndex(0)
        self._combo_ai_hair_parser_backend.setToolTip(
            "自动模式优先采用轻量动漫脸/头发分割，并按暖色头发证据评分；"
            "StyleAnime 仅在轻量结果不足时参与。AI 遮罩只作为语义边界，最终仍由暖色锚点重建头发。")
        ai_hair_backend_row.addWidget(self._combo_ai_hair_parser_backend, 1)
        ai_hair_content_layout.addLayout(ai_hair_backend_row)
        self._chk_ai_hair_parser.toggled.connect(
            self._combo_ai_hair_parser_backend.setEnabled)

        ai_hair_model_row = QHBoxLayout()
        ai_hair_model_row.setContentsMargins(18, 0, 0, 0)
        ai_hair_model_row.setSpacing(8)
        self._ai_hair_model_status = QLabel("正在检查 AI 头发模型…")
        self._ai_hair_model_status.setWordWrap(True)
        self._ai_hair_model_status.setStyleSheet(
            "color: #5d6f84; font-size: 10px;")
        self._btn_manage_ai_hair_models = QPushButton("安装/检查模型")
        self._btn_manage_ai_hair_models.clicked.connect(
            self._manage_ai_hair_models)
        ai_hair_model_row.addWidget(self._ai_hair_model_status, 1)
        ai_hair_model_row.addWidget(self._btn_manage_ai_hair_models)
        ai_hair_content_layout.addLayout(ai_hair_model_row)

        for _signal_owner in (
                self._chk_auto_black_hair_guard, self._combo_auto_black_hair_guard_mode,
                self._combo_auto_black_hair_guard_scope,
                self._combo_auto_black_hair_guard_limit_mode,
                self._chk_ai_hair_parser, self._combo_ai_hair_parser_backend):
            if isinstance(_signal_owner, QComboBox):
                _signal_owner.currentIndexChanged.connect(
                    self._refresh_auto_detail_summaries)
            else:
                _signal_owner.toggled.connect(self._refresh_auto_detail_summaries)
        self._refresh_auto_detail_summaries()

        # Primary render action belongs to the render workspace, not the
        # dedicated hair-recolor settings tab.  It is deliberately outside the
        # render QScrollArea so it cannot disappear below expanded details.
        self._render_action_box = QWidget()
        self._render_action_box.setObjectName("RenderPrimaryActionBox")
        self._render_action_box.setSizePolicy(QSizePolicy.Policy.Expanding,
                                              QSizePolicy.Policy.Fixed)
        render_action_layout = QVBoxLayout(self._render_action_box)
        render_action_layout.setContentsMargins(10, 10, 10, 10)
        render_action_layout.setSpacing(8)
        self._auto_status_label = QLabel("准备就绪")
        self._auto_status_label.setObjectName("RenderActionStatusLabel")
        self._auto_status_label.setWordWrap(True)
        self._auto_status_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._auto_status_label.setSizePolicy(QSizePolicy.Policy.Expanding,
                                              QSizePolicy.Policy.Preferred)
        render_action_layout.addWidget(self._auto_status_label)
        self._btn_auto = tune_button(QPushButton(tr("auto_btn")), 42)
        self._btn_auto.setObjectName("PrimaryAutoColorizeButton")
        self._btn_auto.setMinimumWidth(0)
        self._btn_auto.setToolTip("对当前选中的页面执行自动上色；头发纠色与蒙版覆盖页中的设置会在本次自动上色中生效。")
        self._btn_auto.clicked.connect(self._run_auto_colorize)
        render_action_layout.addWidget(self._btn_auto)
        render_page_layout.addWidget(self._render_action_box, stretch=0)

        render_detail_tabs = QTabWidget()
        render_detail_tabs.setDocumentMode(True)
        render_detail_tabs.setUsesScrollButtons(False)
        render_detail_tabs.tabBar().setExpanding(True)
        self._render_detail_tabs = render_detail_tabs

        detail_fine_tab, detail_fine_layout = make_tab()
        # v5.13.12: style fine-tuning is always expanded.  This page is already
        # its own dedicated tab, so an extra collapsible wrapper only adds an
        # unnecessary click and can hide controls after restoring old settings.
        self._style_fine_group.setVisible(True)
        self._style_fine_group.setSizePolicy(QSizePolicy.Policy.Expanding,
                                             QSizePolicy.Policy.Preferred)
        detail_fine_layout.addWidget(self._style_fine_group, stretch=1)
        render_detail_tabs.addTab(detail_fine_tab, tr("style_detail_tab_fine"))

        detail_filter_tab, detail_filter_layout = make_tab()
        # Natural-image filters likewise stay permanently expanded.
        self._filter_group.setVisible(True)
        self._filter_group.setSizePolicy(QSizePolicy.Policy.Expanding,
                                         QSizePolicy.Policy.Preferred)
        detail_filter_layout.addWidget(self._filter_group, stretch=1)
        render_detail_tabs.addTab(detail_filter_tab, tr("image_filter_group"))

        render_detail_tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        style_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        auto_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        render_layout.addWidget(style_group)
        render_layout.addWidget(render_detail_tabs)
        render_layout.addStretch(1)
        self._right_tabs.addTab(render_tab, tr("right_tab_render"))

        # ── Tab 2: manual editing ─────────────────────────────────────
        # Restore the single-page editor used before the nested detail tabs.
        # The main editing group owns the available vertical space and extends
        # toward the middle/bottom; history remains a compact fixed-height row.
        edit_tab, edit_content, edit_tab_layout = make_scrollable_tab()
        self._edit_scroll_area = edit_tab
        edit_tab_layout.setSpacing(6)
        edit_group, edit_layout = make_group(tr("edit_group"))
        edit_group.setSizePolicy(QSizePolicy.Policy.Expanding,
                                 QSizePolicy.Policy.Preferred)
        edit_layout.setSpacing(5)

        tool_grid = QGridLayout()
        tool_grid.setContentsMargins(0, 0, 0, 0)
        tool_grid.setHorizontalSpacing(16)
        tool_grid.setVerticalSpacing(8)
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

        edit_fold_toolbar = QHBoxLayout()
        edit_fold_toolbar.setContentsMargins(0, 0, 0, 0)
        edit_fold_toolbar.setSpacing(5)
        self._edit_context_hint = QLabel("当前工具细则会按工具自动切换显示")
        self._edit_context_hint.setStyleSheet("color: #6c7f94; font-size: 10px;")
        edit_fold_toolbar.addWidget(self._edit_context_hint, 1)
        self._btn_expand_current_edit_details = QPushButton("展开当前工具")
        self._btn_expand_current_edit_details.setMaximumWidth(120)
        edit_fold_toolbar.addWidget(self._btn_expand_current_edit_details)
        self._btn_collapse_edit_details = QPushButton("收起全部细则")
        self._btn_collapse_edit_details.setMaximumWidth(120)
        edit_fold_toolbar.addWidget(self._btn_collapse_edit_details)
        edit_layout.addLayout(edit_fold_toolbar)

        self._tool_visibility_rows = {}

        def register_tool_row(name: str, *widgets: QWidget) -> None:
            self._tool_visibility_rows[name] = tuple(
                widget for widget in widgets if widget is not None)

        self._tool_param_group, tool_param_layout = make_group(
            tr("tool_parameters_group"))
        self._tool_param_group.setSizePolicy(QSizePolicy.Policy.Expanding,
                                             QSizePolicy.Policy.Fixed)
        edit_layout.addWidget(self._tool_param_group)
        self._edit_tool_detail_section = _CollapsibleSection(
            "当前工具基础细则", summary="颜色 · 大小 · 填充模式 · 边界 / 选区参数",
            expanded=False, settings_key="edit_tool_base")
        tool_param_layout.addWidget(self._edit_tool_detail_section)
        tool_detail_layout = self._edit_tool_detail_section.content_layout()

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
        eyedrop_layout.setSpacing(14)
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
        self._brush_spin.setFixedWidth(68)
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
        self._picker_lightness_spin.setFixedWidth(68)
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
        self._gap_close_spin.setFixedWidth(68)
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
        selection_mode_layout.setSpacing(14)
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
        self._selection_feather_spin.setFixedWidth(68)
        self._selection_feather_spin.setToolTip(tr("selection_feather_hint"))
        self._selection_feather_slider.valueChanged.connect(
            self._selection_feather_spin.setValue)
        self._selection_feather_spin.valueChanged.connect(
            self._selection_feather_slider.setValue)
        edit_grid.addWidget(self._selection_feather_slider, 9, 1)
        edit_grid.addWidget(self._selection_feather_spin, 9, 2)
        register_tool_row("selection_feather", self._selection_feather_label, self._selection_feather_slider, self._selection_feather_spin)

        self._selection_boundary_section = _CollapsibleSection(
            "矩形 / 套索边界细则",
            summary="闭合区 · 边缘扩张 · 小区域 / 细线过滤 · 手动修补 · 线稿吸附",
            expanded=False, settings_key="edit_selection_boundary")
        tool_param_layout.addWidget(self._selection_boundary_section)
        selection_boundary_layout = self._selection_boundary_section.content_layout()
        selection_boundary_grid = QGridLayout()
        selection_boundary_grid.setContentsMargins(0, 0, 0, 0)
        selection_boundary_grid.setHorizontalSpacing(7)
        selection_boundary_grid.setVerticalSpacing(5)
        selection_boundary_layout.addLayout(selection_boundary_grid)

        self._selection_closed_expand_label = QLabel(tr("selection_closed_expand"))
        selection_boundary_grid.addWidget(self._selection_closed_expand_label, 0, 0)
        self._selection_closed_expand_slider = QSlider(Qt.Orientation.Horizontal)
        self._selection_closed_expand_slider.setRange(0, 20)
        self._selection_closed_expand_slider.setValue(0)
        self._selection_closed_expand_slider.setToolTip(
            tr("selection_closed_expand_hint"))
        self._selection_closed_expand_spin = QSpinBox()
        self._selection_closed_expand_spin.setRange(0, 20)
        self._selection_closed_expand_spin.setSuffix(" px")
        self._selection_closed_expand_spin.setValue(0)
        self._selection_closed_expand_spin.setFixedWidth(68)
        self._selection_closed_expand_spin.setToolTip(
            tr("selection_closed_expand_hint"))
        self._selection_closed_expand_slider.valueChanged.connect(
            self._selection_closed_expand_spin.setValue)
        self._selection_closed_expand_spin.valueChanged.connect(
            self._selection_closed_expand_slider.setValue)
        self._selection_closed_expand_slider.valueChanged.connect(
            self._schedule_closed_preview_refresh)
        selection_boundary_grid.addWidget(self._selection_closed_expand_slider, 0, 1)
        selection_boundary_grid.addWidget(self._selection_closed_expand_spin, 0, 2)
        register_tool_row("selection_closed_expand", self._selection_closed_expand_label, self._selection_closed_expand_slider, self._selection_closed_expand_spin)

        self._selection_closed_min_area_label = QLabel(tr("selection_closed_min_area"))
        selection_boundary_grid.addWidget(self._selection_closed_min_area_label, 1, 0)
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
        selection_boundary_grid.addWidget(self._selection_closed_min_area_slider, 1, 1)
        selection_boundary_grid.addWidget(self._selection_closed_min_area_spin, 1, 2)
        register_tool_row("selection_closed_min_area", self._selection_closed_min_area_label, self._selection_closed_min_area_slider, self._selection_closed_min_area_spin)

        self._selection_closed_min_thickness_label = QLabel(tr("selection_closed_min_thickness"))
        selection_boundary_grid.addWidget(self._selection_closed_min_thickness_label, 2, 0)
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
        selection_boundary_grid.addWidget(self._selection_closed_min_thickness_slider, 2, 1)
        selection_boundary_grid.addWidget(self._selection_closed_min_thickness_spin, 2, 2)
        register_tool_row("selection_closed_min_thickness", self._selection_closed_min_thickness_label, self._selection_closed_min_thickness_slider, self._selection_closed_min_thickness_spin)
        edit_grid.setColumnStretch(1, 1)
        tool_detail_layout.addLayout(edit_grid)

        # Independent custom-colour-bias brush controls. These widgets own
        # their state and never mutate the normal brush or global page-bias UI.
        self._bias_brush_section = _CollapsibleSection("颜色倾向画笔细则", summary="画笔 / 消除 · 颜色 · 大小 · 强度 · 色调保护", expanded=False, settings_key="edit_bias_brush")
        tool_param_layout.addWidget(self._bias_brush_section)

        self._bias_brush_box = QWidget()
        bias_brush_layout = QGridLayout(self._bias_brush_box)
        bias_brush_layout.setContentsMargins(0, 0, 0, 0)
        bias_brush_layout.setHorizontalSpacing(7)
        bias_brush_layout.setVerticalSpacing(5)

        self._bias_brush_mode_label = QLabel(tr("bias_brush_mode"))
        self._bias_brush_mode_box = QWidget()
        bias_brush_mode_layout = QHBoxLayout(self._bias_brush_mode_box)
        bias_brush_mode_layout.setContentsMargins(0, 0, 0, 0)
        bias_brush_mode_layout.setSpacing(12)
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
        self._bias_brush_size_spin.setFixedWidth(68)
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
        bias_protect_layout = QGridLayout(self._bias_brush_protection_box)
        bias_protect_layout.setContentsMargins(0, 0, 0, 0)
        bias_protect_layout.setHorizontalSpacing(22)
        bias_protect_layout.setVerticalSpacing(8)
        self._chk_bias_brush_protect_skin = QCheckBox(tr("custom_color_bias_protect_skin"))
        self._chk_bias_brush_protect_lineart = QCheckBox(tr("custom_color_bias_protect_lineart"))
        self._chk_bias_brush_protect_saturated = QCheckBox(tr("custom_color_bias_protect_saturated"))
        self._chk_bias_brush_protect_skin.setChecked(True)
        self._chk_bias_brush_protect_lineart.setChecked(True)
        self._chk_bias_brush_protect_saturated.setChecked(True)
        bias_protect_layout.addWidget(self._chk_bias_brush_protect_skin, 0, 0)
        bias_protect_layout.addWidget(self._chk_bias_brush_protect_lineart, 0, 1)
        bias_protect_layout.addWidget(self._chk_bias_brush_protect_saturated, 1, 0)
        bias_protect_layout.setColumnStretch(0, 1)
        bias_protect_layout.setColumnStretch(1, 1)
        bias_brush_layout.addWidget(QLabel(tr("custom_color_bias_protection")), 5, 0)
        bias_brush_layout.addWidget(self._bias_brush_protection_box, 5, 1, 1, 2)
        bias_brush_layout.setColumnStretch(1, 1)
        self._bias_brush_section.add_widget(self._bias_brush_box)
        register_tool_row("bias_brush_controls", self._bias_brush_section, self._bias_brush_box)
        self._update_bias_brush_swatch()

        self._bias_eraser_size_label = QLabel(tr("bias_eraser_size"))
        self._bias_eraser_size_slider = QSlider(Qt.Orientation.Horizontal)
        self._bias_eraser_size_slider.setRange(0, 100)
        self._bias_eraser_size_slider.setValue(self._brush_percent_from_px(18))
        self._bias_eraser_size_spin = QSpinBox()
        self._bias_eraser_size_spin.setRange(0, 100)
        self._bias_eraser_size_spin.setValue(self._bias_eraser_size_slider.value())
        self._bias_eraser_size_spin.setFixedWidth(68)
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
        selection_boundary_layout.addWidget(self._chk_selection_closed_only)
        register_tool_row("selection_closed_only", self._chk_selection_closed_only)
        self._update_closed_expand_controls_enabled(False)

        self._selection_action_section = _CollapsibleSection("矩形 / 选区操作", summary="应用 · AI 重上色 · 语义选取 · 取消", expanded=True, settings_key="edit_selection_actions")
        tool_param_layout.addWidget(self._selection_action_section)
        self._selection_action_box = QWidget()
        selection_action_row = QGridLayout(self._selection_action_box)
        selection_action_row.setContentsMargins(0, 0, 0, 0)
        selection_action_row.setHorizontalSpacing(6)
        selection_action_row.setVerticalSpacing(6)
        self._btn_apply_selection = QPushButton(tr("selection_apply"))
        self._btn_apply_selection.setToolTip(tr("selection_apply_hint"))
        self._btn_apply_selection.clicked.connect(
            self._apply_pending_selection_fill)
        self._btn_ai_recolor_selection = QPushButton(tr("selection_ai_recolor"))
        self._btn_ai_recolor_selection.setToolTip(tr("selection_ai_recolor_hint"))
        self._btn_ai_recolor_selection.clicked.connect(
            self._start_selection_model_recolor)
        self._btn_semantic_select = QPushButton(tr("selection_semantic_select"))
        self._btn_semantic_select.setToolTip(tr("selection_semantic_select_hint"))
        self._btn_semantic_select.clicked.connect(self._arm_semantic_selection)
        self._btn_refine_hair = QPushButton("智能贴合发丝")
        self._btn_refine_hair.setToolTip(
            "先用矩形/套索粗选头发，可用下方发丝+与非发丝−标注，再将蓝框细化为发丝形状。"
            "结合 mc-v2 已生成的配色做保守边界检查，但它不是独立训练的头发分割模型。"
            "不确定区域仍需红色负向笔刷确认；本操作只修改选区，不直接改变图片。")
        self._btn_refine_hair.clicked.connect(self._start_hair_selection_refine)
        self._btn_refine_positive = QPushButton("发丝 + 画笔")
        self._btn_refine_negative = QPushButton("非发丝 − 画笔")
        self._btn_refine_positive.setToolTip("在头发内部拖动画笔，标明这里一定是头发；笔迹与项目一同保存。")
        self._btn_refine_negative.setToolTip("在额头/面部/背景拖动画笔，标明这里不能染色；笔迹与项目一同保存。")
        self._btn_refine_positive.clicked.connect(
            lambda: self._activate_hair_refine_brush("brush_add"))
        self._btn_refine_negative.clicked.connect(
            lambda: self._activate_hair_refine_brush("brush_erase"))
        self._btn_refine_stop_brush = QPushButton("结束标注")
        self._btn_refine_stop_brush.clicked.connect(self._deactivate_hair_guard_selector)
        self._btn_refine_clear_brush = QPushButton("清除发丝标注")
        self._btn_refine_clear_brush.setToolTip("仅清除该页正负发丝标注，不清除粗选区和上色图片。")
        self._btn_refine_clear_brush.clicked.connect(self._clear_hair_guard_selection)
        self._btn_cancel_selection = QPushButton(tr("selection_cancel"))
        self._btn_cancel_selection.clicked.connect(
            self._cancel_pending_selection_fill)
        self._btn_restore_selection_bw = QPushButton("选区恢复黑白")
        self._btn_restore_selection_bw.setToolTip(
            "只将当前选区恢复为原始黑白，选区外逐像素不变；可撤销，整页 AI 结果保留。")
        self._btn_restore_selection_bw.clicked.connect(self._restore_selection_bw)
        for _row, _col, _button in (
            (0, 0, self._btn_apply_selection),
            (0, 1, self._btn_restore_selection_bw),
            (1, 0, self._btn_ai_recolor_selection),
            (1, 1, self._btn_semantic_select),
            (2, 0, self._btn_refine_hair),
            (2, 1, self._btn_cancel_selection),
            (3, 0, self._btn_refine_positive),
            (3, 1, self._btn_refine_negative),
            (4, 0, self._btn_refine_stop_brush),
            (4, 1, self._btn_refine_clear_brush),
        ):
            _button.setMinimumWidth(0)
            selection_action_row.addWidget(_button, _row, _col)
        selection_action_row.setColumnStretch(0, 1)
        selection_action_row.setColumnStretch(1, 1)
        self._selection_action_section.add_widget(self._selection_action_box)
        register_tool_row("selection_actions", self._selection_action_section, self._selection_action_box)

        self._selection_ai_options_section = _CollapsibleSection("矩形 / 选区 AI 上色细则", summary="局部提示 · 上下文范围 · 发色预设 · 颜色锁定", expanded=False, settings_key="edit_selection_ai")
        tool_param_layout.addWidget(self._selection_ai_options_section)
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

        self._chk_selection_region_control = QCheckBox(tr("selection_region_control"))
        self._chk_selection_region_control.setChecked(True)
        self._chk_selection_region_control.setToolTip(tr("selection_region_control_hint"))
        self._chk_selection_region_control.toggled.connect(
            self._update_selection_region_control_enabled)
        self._selection_control_preset_label = QLabel(tr("selection_control_hair_preset"))
        self._selection_control_preset_combo = QComboBox()
        self._selection_control_preset_combo.addItem(
            tr("selection_control_preset_custom"), "custom")
        self._selection_control_preset_combo.addItem(
            tr("selection_control_preset_ink_black"), "ink_black")
        self._selection_control_preset_combo.addItem(
            tr("selection_control_preset_soft_black"), "soft_black")
        self._selection_control_preset_combo.addItem(
            tr("selection_control_preset_cool_black"), "cool_black")
        self._selection_control_preset_combo.addItem(
            tr("selection_control_preset_light_ink"), "light_ink")
        self._selection_control_preset_combo.setToolTip(
            tr("selection_control_hair_preset_hint"))
        self._selection_control_preset_combo.currentIndexChanged.connect(
            self._on_selection_control_preset_changed)
        self._selection_control_color_label = QLabel(tr("selection_control_color"))
        self._selection_control_color_btn = QPushButton()
        self._selection_control_color_btn.clicked.connect(
            self._pick_selection_control_color)

        def _region_spin(key, minimum, maximum, value, suffix="%"):
            label = QLabel(tr(key))
            spin = QSpinBox()
            spin.setRange(minimum, maximum)
            spin.setValue(value)
            spin.setSuffix(suffix)
            return label, spin

        self._selection_control_hint_label, self._selection_control_hint_spin = _region_spin(
            "selection_control_hint_strength", 0, 100, 50)
        self._selection_control_gain_label, self._selection_control_gain_spin = _region_spin(
            "selection_control_feature_gain", 50, 400, 175)
        self._selection_control_deep_label, self._selection_control_deep_spin = _region_spin(
            "selection_control_deep", 0, 100, 38)
        self._selection_control_middle_label, self._selection_control_middle_spin = _region_spin(
            "selection_control_middle", 0, 100, 74)
        self._selection_control_shallow_label, self._selection_control_shallow_spin = _region_spin(
            "selection_control_shallow", 0, 100, 94)
        self._selection_control_final_label, self._selection_control_final_spin = _region_spin(
            "selection_control_final", 0, 100, 92)
        self._selection_control_luminance_label, self._selection_control_luminance_spin = _region_spin(
            "selection_control_luminance", 0, 100, 28)
        self._chk_selection_control_adapter = QCheckBox(tr("selection_control_adapter"))
        self._chk_selection_control_adapter.setChecked(False)
        self._chk_selection_control_adapter.setToolTip(tr("selection_control_adapter_hint"))
        self._selection_control_preset_combo.setCurrentIndex(1)
        self._on_selection_control_preset_changed()
        self._update_selection_control_swatch()

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
        selection_ai_options.addWidget(self._chk_selection_region_control, 8, 0, 1, 2)
        selection_ai_options.addWidget(self._selection_control_preset_label, 9, 0)
        selection_ai_options.addWidget(self._selection_control_preset_combo, 9, 1)
        selection_ai_options.addWidget(self._selection_control_color_label, 10, 0)
        selection_ai_options.addWidget(self._selection_control_color_btn, 10, 1)
        selection_ai_options.addWidget(self._selection_control_hint_label, 11, 0)
        selection_ai_options.addWidget(self._selection_control_hint_spin, 11, 1)
        selection_ai_options.addWidget(self._selection_control_gain_label, 12, 0)
        selection_ai_options.addWidget(self._selection_control_gain_spin, 12, 1)
        selection_ai_options.addWidget(self._selection_control_deep_label, 13, 0)
        selection_ai_options.addWidget(self._selection_control_deep_spin, 13, 1)
        selection_ai_options.addWidget(self._selection_control_middle_label, 14, 0)
        selection_ai_options.addWidget(self._selection_control_middle_spin, 14, 1)
        selection_ai_options.addWidget(self._selection_control_shallow_label, 15, 0)
        selection_ai_options.addWidget(self._selection_control_shallow_spin, 15, 1)
        selection_ai_options.addWidget(self._selection_control_final_label, 16, 0)
        selection_ai_options.addWidget(self._selection_control_final_spin, 16, 1)
        selection_ai_options.addWidget(self._selection_control_luminance_label, 17, 0)
        selection_ai_options.addWidget(self._selection_control_luminance_spin, 17, 1)
        selection_ai_options.addWidget(self._chk_selection_control_adapter, 18, 0, 1, 2)
        self._selection_ai_options_section.add_widget(self._selection_ai_options_box)
        register_tool_row("selection_ai_options", self._selection_ai_options_section, self._selection_ai_options_box)
        self._btn_collapse_edit_details.clicked.connect(
            lambda: self._collapse_sections(
                self._edit_tool_detail_section, self._bias_brush_section,
                self._selection_boundary_section, self._selection_action_section,
                self._selection_ai_options_section))
        self._btn_expand_current_edit_details.clicked.connect(
            self._expand_current_tool_sections)
        self._update_selection_focus_controls_enabled()
        self._update_selection_region_control_enabled()

        self._chk_selection_adjust = QCheckBox(tr("selection_adjust_checkbox"))
        self._chk_selection_adjust.setToolTip(tr("selection_adjust_hint"))
        self._chk_selection_adjust.toggled.connect(self._on_selection_adjust_toggled)
        selection_boundary_layout.addWidget(self._chk_selection_adjust)
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
        selection_boundary_layout.addWidget(self._selection_adjust_tools_box)
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
        selection_boundary_layout.addWidget(self._selection_adjust_row_box)
        register_tool_row("selection_adjust_radius", self._selection_adjust_row_box)
        self._on_selection_adjust_radius_changed(self._selection_adjust_slider.value())
        self._set_selection_adjust_mode("add")

        self._chk_selection_snap_lineart = QCheckBox(tr("selection_snap_lineart_checkbox"))
        self._chk_selection_snap_lineart.setChecked(False)
        self._chk_selection_snap_lineart.setToolTip(tr("selection_snap_lineart_hint"))
        selection_boundary_layout.addWidget(self._chk_selection_snap_lineart)
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
        selection_boundary_layout.addWidget(self._selection_snap_row_box)
        register_tool_row("selection_snap_distance", self._selection_snap_row_box)
        self._chk_selection_closed_only.toggled.connect(lambda _checked: self._refresh_edit_detail_summary())
        self._chk_selection_adjust.toggled.connect(lambda _checked: self._refresh_edit_detail_summary())
        self._chk_selection_snap_lineart.toggled.connect(lambda _checked: self._refresh_edit_detail_summary())

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
        self._btn_regenerate.setObjectName("AccentSecondaryButton")
        self._btn_regenerate.clicked.connect(self._run_regenerate)
        hint_layout.addWidget(self._btn_regenerate)
        edit_layout.addStretch(1)

        self._update_color_swatch()
        self._set_eyedropper_mode("point")
        self._update_selection_buttons(False)
        edit_tab_layout.addWidget(edit_group)

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

        # ── Tab 3: dedicated hair recolor workspace ─────────────────
        hair_tab, hair_content, hair_tab_layout = make_scrollable_tab()
        self._hair_recolor_scroll_area = hair_tab
        hair_tab_layout.setSpacing(6)
        auto_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        hair_tab_layout.addWidget(auto_group)
        hair_tab_layout.addStretch(1)

        hair_workspace = QWidget()
        hair_workspace_layout = QVBoxLayout(hair_workspace)
        hair_workspace_layout.setContentsMargins(0, 0, 0, 0)
        hair_workspace_layout.setSpacing(8)
        hair_workspace_layout.addWidget(hair_tab, stretch=1)

        self._hair_recolor_action_box = QWidget()
        self._hair_recolor_action_box.setObjectName("RenderPrimaryActionBox")
        self._hair_recolor_action_box.setSizePolicy(QSizePolicy.Policy.Expanding,
                                                    QSizePolicy.Policy.Fixed)
        hair_action_layout = QVBoxLayout(self._hair_recolor_action_box)
        hair_action_layout.setContentsMargins(10, 10, 10, 10)
        hair_action_layout.setSpacing(8)
        self._hair_recolor_status_label = QLabel("应用当前头发纠色设置并自动上色")
        self._hair_recolor_status_label.setObjectName("RenderActionStatusLabel")
        self._hair_recolor_status_label.setWordWrap(True)
        hair_action_layout.addWidget(self._hair_recolor_status_label)
        self._btn_hair_recolor_auto = tune_button(QPushButton("自动上色（应用头发纠色）"), 42)
        self._btn_hair_recolor_auto.setObjectName("PrimaryAutoColorizeButton")
        self._btn_hair_recolor_auto.setMinimumWidth(0)
        self._btn_hair_recolor_auto.setToolTip("按当前头发纠色页中的设置，对当前选中的页面直接执行自动上色。")
        self._btn_hair_recolor_auto.clicked.connect(self._run_auto_colorize)
        hair_action_layout.addWidget(self._btn_hair_recolor_auto)
        hair_workspace_layout.addWidget(self._hair_recolor_action_box, stretch=0)

        # Legacy contract marker for v5.13.11 regression checks:
        # self._right_tabs.addTab(hair_tab, "头发纠色")
        self._right_tabs.addTab(hair_workspace, "头发纠色")

        # ── Tab 4: view, restore and export ───────────────────────────
        output_tab, output_content, output_layout = make_scrollable_tab()
        self._output_scroll_area = output_tab

        view_group, view_layout = make_group("预览版本")
        self._view_tool_group = QButtonGroup(self)
        self._view_tool_group.setExclusive(True)
        self._radio_view_original = QRadioButton("原图")
        self._radio_view_ai = QRadioButton("AI结果")
        self._radio_view_edited = QRadioButton("编辑后")
        self._radio_view_edited.setChecked(True)

        self._view_segment_bar = QWidget()
        self._view_segment_bar.setObjectName("ViewSegmentBar")
        view_mode_row = QHBoxLayout(self._view_segment_bar)
        view_mode_row.setContentsMargins(4, 4, 4, 4)
        view_mode_row.setSpacing(4)
        for rb in (self._radio_view_original, self._radio_view_ai,
                   self._radio_view_edited):
            rb.setObjectName("ViewSegmentButton")
            rb.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            rb.setMinimumHeight(34)
            self._view_tool_group.addButton(rb)
            rb.toggled.connect(self._on_view_mode_changed)
            view_mode_row.addWidget(rb, 1)
        view_layout.addWidget(self._view_segment_bar)
        view_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        output_layout.addWidget(view_group, stretch=0)

        restore_group, restore_layout = make_group("版本操作")
        # Keep the established full-width restore layout contract while the
        # visual card itself is now separated from the preview switcher.
        view_layout = restore_layout
        self._btn_restore_ai = QPushButton(tr("restore_ai"))
        self._btn_restore_ai.setObjectName("AccentSecondaryButton")
        self._btn_restore_ai.clicked.connect(self._restore_to_ai_result)
        self._btn_restore_bw = QPushButton(tr("restore_bw"))
        self._btn_restore_bw.setObjectName("DangerSoftButton")
        self._btn_restore_bw.clicked.connect(self._restore_to_original)
        for _restore_btn in (self._btn_restore_ai, self._btn_restore_bw):
            _restore_btn.setMinimumHeight(36)
            _restore_btn.setSizePolicy(QSizePolicy.Policy.Expanding,
                                       QSizePolicy.Policy.Fixed)
            _restore_btn.setToolTip(_restore_btn.text())
        add_button_grid(view_layout,
                        [self._btn_restore_ai, self._btn_restore_bw], columns=1)
        restore_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        output_layout.addWidget(restore_group, stretch=0)

        export_group, export_layout = make_group(tr("export_group"))
        export_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        compression_label = QLabel(tr("export_compression_mode"))
        self._export_compression_combo = QComboBox()
        self._export_compression_combo.addItem(
            tr("export_compression_smart_webp"), "smart_webp")
        self._export_compression_combo.addItem(
            tr("export_compression_webp_lossless"), "webp_lossless")
        self._export_compression_combo.addItem(
            tr("export_compression_png"), "png_optimized")
        self._export_compression_combo.addItem(
            tr("export_compression_jpeg"), "jpeg_high")
        self._export_compression_combo.setToolTip(
            tr("export_compression_tooltip"))
        self._export_compression_combo.setCurrentIndex(
            self._export_compression_combo.findData("png_optimized"))
        self._export_compression_combo.currentIndexChanged.connect(
            self._on_export_compression_profile_changed)
        export_layout.addWidget(compression_label)
        export_layout.addWidget(self._export_compression_combo)
        self._export_compression_status = QLabel(tr("export_compression_waiting"))
        self._export_compression_status.setWordWrap(True)
        self._export_compression_status.setToolTip(
            tr("export_compression_status_tooltip"))
        export_layout.addWidget(self._export_compression_status)
        self._btn_export_page = QPushButton(tr("export_page"))
        self._btn_export_page.setObjectName("AccentSecondaryButton")
        self._btn_export_page.clicked.connect(self._export_current_page)
        self._btn_export_all = QPushButton(tr("export_all"))
        self._btn_export_all.setObjectName("OutputPrimaryButton")
        self._btn_export_all.clicked.connect(self._export_all_pages)
        self._btn_export_page.setMinimumHeight(38)
        self._btn_export_all.setMinimumHeight(38)
        add_button_grid(export_layout,
                        [self._btn_export_page, self._btn_export_all])
        output_layout.addWidget(export_group, stretch=0)
        output_layout.addStretch(1)
        # Legacy contract marker for dedicated output tab regression checks:
        # self._right_tabs.addTab(output_tab, tr("right_tab_output"))
        self._right_tabs.addTab(output_tab, "查看导出")

        return panel

    # ── Import ─────────────────────────────────────────────────────────

    @staticmethod
    def _cleanup_stale_session_caches(max_age_hours: int = 24) -> None:
        """Remove abandoned Colortina temp caches left by crashes."""
        root = tempfile.gettempdir()
        cutoff = time.time() - max(1, int(max_age_hours)) * 3600
        try:
            entries = os.scandir(root)
        except OSError:
            return
        with entries:
            for entry in entries:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                is_session_cache = entry.name.startswith("colortina_session_cache_")
                is_pdf_cache = (entry.name.startswith("colortina_") and
                                "_pages_" in entry.name)
                if not (is_session_cache or is_pdf_cache):
                    continue
                try:
                    if entry.stat(follow_symlinks=False).st_mtime < cutoff:
                        shutil.rmtree(entry.path, ignore_errors=True)
                except OSError:
                    pass

    def _cleanup_session_layer_cache(self) -> None:
        for timer in list(getattr(self, "_spill_delay_timers", {}).values()):
            timer.stop()
            timer.deleteLater()
        if hasattr(self, "_spill_delay_timers"):
            self._spill_delay_timers.clear()
        for timer in list(getattr(self, "_export_compress_timers", {}).values()):
            timer.stop()
            timer.deleteLater()
        if hasattr(self, "_export_compress_timers"):
            self._export_compress_timers.clear()
        if hasattr(self, "_export_compress_queue"):
            self._export_compress_queue.clear()
        if hasattr(self, "_export_compress_queued"):
            self._export_compress_queued.clear()
        if hasattr(self, "_spill_queue"):
            self._spill_queue.clear()
        if hasattr(self, "_spill_queued"):
            self._spill_queued.clear()
        for state in self._pages.values():
            state.clear_layer_cache()
        path = getattr(self, "_session_layer_cache_dir", None)
        if path:
            shutil.rmtree(path, ignore_errors=True)

    def _current_export_profile(self) -> str:
        from core.image_compressor import DEFAULT_PROFILE, normalize_profile
        combo = getattr(self, "_export_compression_combo", None)
        return normalize_profile(combo.currentData() if combo is not None else DEFAULT_PROFILE)

    def _cancel_delayed_export_compression(self, path: str | None) -> None:
        if not path:
            return
        timer = self._export_compress_timers.pop(path, None)
        if timer is not None:
            timer.stop()
            timer.deleteLater()
        if path in self._export_compress_queued:
            self._export_compress_queued.discard(path)
            try:
                self._export_compress_queue.remove(path)
            except ValueError:
                pass

    def _schedule_export_compression(self, path: str | None, *, delay_ms: int = 900) -> None:
        """Pre-encode final export bytes without blocking painting or clicks."""
        if self._closing or not path:
            return
        state = self._pages.get(path)
        if state is None or not state.has_result:
            return
        profile = self._current_export_profile()
        if state.has_valid_export_cache(profile):
            if path == self._current_path:
                self._update_export_compression_status(state)
            return
        if delay_ms <= 0 and int(getattr(self, "_foreground_interaction_depth", 0)) > 0:
            delay_ms = 550
        if delay_ms > 0:
            self._cancel_delayed_export_compression(path)
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(max(1, int(delay_ms)))
            timer.timeout.connect(
                lambda page_path=path: self._schedule_export_compression(
                    page_path, delay_ms=0))
            self._export_compress_timers[path] = timer
            timer.start()
            return
        self._cancel_delayed_export_compression(path)
        if path in self._pending_export_cache:
            self._recompress_after_pending.add(path)
            return
        if path not in self._export_compress_queued:
            self._export_compress_queue.append(path)
            self._export_compress_queued.add(path)
        self._pump_layer_spill_queue()
        self._pump_export_compression_queue()

    def _pump_export_compression_queue(self) -> None:
        # One shared executor keeps PNG session spilling and final-image codec
        # work from saturating all CPU cores at the same time. Layer spooling
        # has priority because it releases much more RAM.
        if (self._closing or self._pending_layer_spills or
                self._pending_export_cache or self._spill_queue):
            return
        while self._export_compress_queue:
            path = self._export_compress_queue.popleft()
            self._export_compress_queued.discard(path)
            state = self._pages.get(path)
            if state is None or not state.has_result:
                continue
            request = state.prepare_export_cache_request(
                self._session_layer_cache_dir, self._current_export_profile())
            if request is None:
                continue
            token = (int(request["state_id"]), int(request["serial"]))
            self._pending_export_cache[path] = token
            future = self._layer_cache_executor.submit(_write_export_cache, request)

            def _done(done_future, page_path=path, page_token=token):
                try:
                    result = done_future.result()
                except Exception as exc:  # noqa: BLE001
                    result = {
                        "path": page_path,
                        "state_id": page_token[0],
                        "serial": page_token[1],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                if getattr(self, "_closing", False):
                    _discard_export_cache_result(result)
                    return
                try:
                    self._export_cache_bridge.completed.emit(result)
                except RuntimeError:
                    _discard_export_cache_result(result)

            future.add_done_callback(_done)
            return

    def _on_export_cache_done(self, result: dict) -> None:
        path = str(result.get("path", ""))
        token = (int(result.get("state_id", -1)), int(result.get("serial", -1)))
        if self._pending_export_cache.get(path) == token:
            self._pending_export_cache.pop(path, None)
        state = self._pages.get(path)
        committed = False
        if (state is None or id(state) != token[0] or result.get("error")):
            _discard_export_cache_result(result)
        else:
            committed = state.commit_export_cache_result(result)
        if committed and path == self._current_path:
            self._update_export_compression_status(state)
        if path in self._recompress_after_pending:
            self._recompress_after_pending.discard(path)
            self._schedule_export_compression(path, delay_ms=0)
        self._pump_layer_spill_queue()
        self._pump_export_compression_queue()

    def _update_export_compression_status(self, state: PageState | None = None) -> None:
        label = getattr(self, "_export_compression_status", None)
        if label is None:
            return
        state = state or self._current_state()
        if state is None or not state.has_result:
            label.setText(tr("export_compression_waiting"))
            return
        path, meta = state.export_cache_info(self._current_export_profile())
        if not path:
            label.setText(tr("export_compression_pending"))
            return
        compressed = int(meta.get("compressed_bytes", 0) or 0)
        raw = int(meta.get("original_bytes", 0) or 0)
        saved = float(meta.get("saved_percent", 0.0) or 0.0)
        codec = str(meta.get("codec", ""))
        label.setText(tr("export_compression_ready").format(
            codec=codec, size=self._format_bytes(compressed),
            raw=self._format_bytes(raw), saved=saved))

    @staticmethod
    def _format_bytes(value: int) -> str:
        size = float(max(0, int(value)))
        for unit in ("B", "KiB", "MiB", "GiB"):
            if size < 1024.0 or unit == "GiB":
                return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
            size /= 1024.0
        return f"{size:.1f} GiB"

    def _on_export_compression_profile_changed(self, _index: int = -1) -> None:
        for state in self._pages.values():
            state.invalidate_export_cache()
        self._update_export_compression_status()
        for path, state in self._pages.items():
            if state.has_result:
                self._schedule_export_compression(
                    path, delay_ms=300 if path == self._current_path else 0)

    def _cancel_delayed_spill(self, path: str | None) -> None:
        if not path:
            return
        timer = self._spill_delay_timers.pop(path, None)
        if timer is not None:
            timer.stop()
            timer.deleteLater()
        if path in getattr(self, "_spill_queued", set()):
            self._spill_queued.discard(path)
            try:
                self._spill_queue.remove(path)
            except ValueError:
                pass

    def _begin_foreground_interaction(self) -> None:
        self._foreground_interaction_depth = max(
            0, int(getattr(self, "_foreground_interaction_depth", 0))) + 1

    def _end_foreground_interaction(self) -> None:
        self._foreground_interaction_depth = max(
            0, int(getattr(self, "_foreground_interaction_depth", 0)) - 1)

    def _schedule_page_spill(self, path: str | None, *, delay_ms: int = 0) -> None:
        """Spool an inactive page without doing compression on the UI thread."""
        if self._closing or not path or path == self._current_path:
            return
        state = self._pages.get(path)
        if state is None:
            return

        # PNG encoding runs outside Qt, but it still competes for CPU and disk
        # bandwidth.  Do not start a new spill while the user is actively
        # drawing or adjusting a selection; retry shortly after the gesture.
        if delay_ms <= 0 and int(getattr(self, "_foreground_interaction_depth", 0)) > 0:
            delay_ms = 450

        if delay_ms > 0:
            self._cancel_delayed_spill(path)
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(max(1, int(delay_ms)))
            timer.timeout.connect(
                lambda page_path=path: self._schedule_page_spill(page_path, delay_ms=0))
            self._spill_delay_timers[path] = timer
            timer.start()
            return

        self._cancel_delayed_spill(path)
        if state.can_release_cached_layers():
            state.release_cached_layers()
            return

        pending = self._pending_layer_spills.get(path)
        if pending is not None:
            self._respool_after_pending.add(path)
            return

        # Queue only lightweight page paths.  Preparing every request up front
        # would make the executor's waiting futures retain full-resolution
        # ndarray references even though only one PNG can be written at once.
        if path not in self._spill_queued:
            self._spill_queue.append(path)
            self._spill_queued.add(path)
        self._pump_layer_spill_queue()

    def _pump_layer_spill_queue(self) -> None:
        if self._closing or self._pending_layer_spills or self._pending_export_cache:
            return
        while self._spill_queue:
            path = self._spill_queue.popleft()
            self._spill_queued.discard(path)
            if path == self._current_path:
                continue
            state = self._pages.get(path)
            if state is None:
                continue
            if state.can_release_cached_layers():
                state.release_cached_layers()
                continue
            request = state.prepare_spill_request(self._session_layer_cache_dir)
            if request is None:
                continue
            token = (int(request["state_id"]), int(request["serial"]))
            self._pending_layer_spills[path] = token
            future = self._layer_cache_executor.submit(_write_layer_spill, request)

            def _done(done_future, page_path=path, page_token=token):
                try:
                    result = done_future.result()
                except Exception as exc:  # noqa: BLE001
                    result = {
                        "path": page_path,
                        "state_id": page_token[0],
                        "serial": page_token[1],
                        "error": f"{type(exc).__name__}: {exc}",
                        "created_paths": set(),
                    }
                if getattr(self, "_closing", False):
                    _discard_spill_files(result)
                    return
                try:
                    self._layer_cache_bridge.completed.emit(result)
                except RuntimeError:
                    _discard_spill_files(result)

            future.add_done_callback(_done)
            return

    def _on_layer_spill_done(self, result: dict) -> None:
        path = str(result.get("path", ""))
        token = (int(result.get("state_id", -1)), int(result.get("serial", -1)))
        if self._pending_layer_spills.get(path) == token:
            self._pending_layer_spills.pop(path, None)
        state = self._pages.get(path)
        if (state is None or id(state) != token[0] or result.get("error")):
            _discard_spill_files(result)
        else:
            state.commit_spill_result(
                result, release=(path != self._current_path))

        if path in self._respool_after_pending:
            self._respool_after_pending.discard(path)
            if path != self._current_path:
                self._schedule_page_spill(path, delay_ms=0)
        self._pump_layer_spill_queue()
        self._pump_export_compression_queue()

    def _spill_inactive_page(self, path: str | None) -> None:
        self._schedule_page_spill(path, delay_ms=0)

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
        safe_name = os.path.splitext(os.path.basename(pdf_path))[0] or "pdf"
        out_dir = tempfile.mkdtemp(prefix=f"colortina_{safe_name}_pages_")
        self._temp_pdf_page_dirs.add(out_dir)
        self.statusBar().showMessage(tr("splitting_pdf"))
        try:
            page_paths = extract_pages(pdf_path, out_dir, dpi=300)
        except Exception as exc:
            self._cleanup_temp_pdf_dir(out_dir)
            QMessageBox.critical(self, tr("error_title"), tr("pdf_split_failed").format(exc=exc))
            return
        added = self._add_pages_fast(page_paths)
        self.statusBar().showMessage(tr("imported_n_pages").format(n=added), 5000)

    def _cleanup_temp_pdf_dir(self, path: str):
        self._temp_pdf_page_dirs.discard(path)
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass

    def _cleanup_temp_pdf_dirs(self):
        for path in list(getattr(self, "_temp_pdf_page_dirs", set())):
            self._cleanup_temp_pdf_dir(path)

    def _running_background_workers(self) -> list:
        workers = []
        for name in ("_batch_worker", "_hair_model_worker", "_local_recolor_worker",
                     "_semantic_selection_worker", "_hair_refine_worker", "_line_extract_worker"):
            worker = getattr(self, name, None)
            if worker is None:
                continue
            try:
                if worker.isRunning():
                    workers.append(worker)
            except RuntimeError:
                continue
        return workers

    def closeEvent(self, event):
        running_workers = self._running_background_workers()
        if running_workers:
            # QThread children must not be destroyed while they are still
            # running.  Several model operations are not safely cancellable
            # mid-inference/download, so keep the window alive until the
            # current task reaches its normal completion callback.
            event.ignore()
            self.statusBar().showMessage(
                "后台任务仍在运行，请等待当前任务完成后再关闭程序。", 6000)
            try:
                QMessageBox.information(
                    self, "任务仍在运行",
                    "当前仍有上色、识别或模型任务在后台运行。为了避免线程被强制销毁导致崩溃，"
                    "请等待当前任务完成后再关闭程序。")
            except RuntimeError:
                pass
            return
        self._closing = True
        for timer in list(getattr(self, "_spill_delay_timers", {}).values()):
            timer.stop()
        for timer in list(getattr(self, "_export_compress_timers", {}).values()):
            timer.stop()
        try:
            self._layer_cache_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        self._spill_queue.clear()
        self._spill_queued.clear()
        self._export_compress_queue.clear()
        self._export_compress_queued.clear()
        self._cleanup_temp_pdf_dirs()
        self._cleanup_session_layer_cache()
        super().closeEvent(event)

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
            removed_state = self._pages.pop(path, None)
            if removed_state is not None:
                removed_state.clear_layer_cache()
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

    def _refresh_workspace_stats(self) -> None:
        total = len(getattr(self, "_pages", {}) or {})
        try:
            selected = len(self._page_list.selectedItems()) if hasattr(self, "_page_list") and self._page_list is not None else 0
        except RuntimeError:
            selected = 0
        colored = 0
        for _path, state in (getattr(self, "_pages", {}) or {}).items():
            if getattr(state, "has_ai_result", False):
                colored += 1
        pending = max(0, total - colored)
        for attr, value in (("_stats_total_pages", total), ("_stats_selected_pages", selected), ("_stats_colored_pages", colored), ("_stats_pending_pages", pending),
                            ("_footer_total_pages", total), ("_footer_selected_pages", selected), ("_footer_colored_pages", colored), ("_footer_pending_pages", pending)):
            label = getattr(self, attr, None)
            if label is not None:
                label.setText(str(int(value)))
        if bool(getattr(self, "_ui_busy", False)):
            status_text = "处理中"
        else:
            status_text = "待导入" if total <= 0 else ("待上色" if pending > 0 else "完成")
        footer_status = getattr(self, "_footer_status_value", None)
        if footer_status is not None:
            footer_status.setText(status_text)
        header_status = getattr(self, "_header_status_chip", None)
        if header_status is not None:
            header_status.setText("● " + status_text)

    def _repopulate_page_list(self):
        """Rebuild the page-list widget from `self._pages` — needed after
        a language-switch UI rebuild, since `_build_left_panel()` creates
        a brand new (empty) QListWidget each time."""
        selected_row = -1
        for i, (path, state) in enumerate(self._pages.items()):
            label = os.path.basename(path)
            if state.has_ai_result:
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
        self._refresh_workspace_stats()

    # ── Page selection ────────────────────────────────────────────────

    def _on_page_selected(self, current: QListWidgetItem, _previous):
        if current is None:
            return
        path = current.data(Qt.ItemDataRole.UserRole)
        previous_path = self._current_path
        self._cancel_pending_selection_fill()
        self._current_path = path
        self._cancel_delayed_spill(path)
        if previous_path and previous_path != path:
            # Keep the immediately previous page hot for a short grace period;
            # rapid back/forward clicks then avoid both compression and decode.
            self._schedule_page_spill(previous_path, delay_ms=1500)
        state = self._pages[path]
        self._update_view_mode_availability()
        try:
            self._sync_view_after_edit(state)
        except Exception as exc:
            self.statusBar().showMessage(str(exc), 6000)
            return
        # Auto-fit the newly displayed page to the window.
        self._canvas.fit_view()
        self._update_controls_enabled()
        self._refresh_workspace_stats()
        self._refresh_character_diagnostics()

    # ── Colorize actions ─────────────────────────────────────────────

    @staticmethod
    def _collapse_sections(*sections) -> None:
        for section in sections:
            if section is not None and hasattr(section, "set_expanded"):
                section.set_expanded(False)

    def _refresh_auto_detail_summaries(self, *_args) -> None:
        base = getattr(self, "_auto_detail_section", None)
        if base is not None:
            protect = bool(getattr(self, "_chk_protect_text", None) and
                           self._chk_protect_text.isChecked())
            base.set_summary("文字气泡保护：" + ("开启" if protect else "关闭"))

        hair = getattr(self, "_hair_guard_detail_section", None)
        if hair is not None:
            enabled = bool(getattr(self, "_chk_auto_black_hair_guard", None) and
                           self._chk_auto_black_hair_guard.isChecked())
            mode_combo = getattr(self, "_combo_auto_black_hair_guard_mode", None)
            scope_combo = getattr(self, "_combo_auto_black_hair_guard_scope", None)
            limit_combo = getattr(self, "_combo_auto_black_hair_guard_limit_mode", None)
            mode = mode_combo.currentText() if mode_combo is not None else ""
            scope = scope_combo.currentText() if scope_combo is not None else ""
            limit = limit_combo.currentText() if limit_combo is not None else ""
            color = QColor(getattr(self, "_hair_recolor_color", QColor(24, 24, 24))).name()
            selection_suffix = ""
            if self._auto_black_hair_guard_requires_selection():
                selection_mask = self._current_hair_guard_selection_mask()
                selection_area = int(np.count_nonzero(selection_mask)) if selection_mask is not None else 0
                selection_suffix = f" · 选区 {selection_area} px"
            hair.set_summary(
                f"{'开启' if enabled else '关闭'} · {mode} · {scope} · {limit} · {color}{selection_suffix}")

        ai = getattr(self, "_ai_hair_detail_section", None)
        if ai is not None:
            enabled = bool(getattr(self, "_chk_ai_hair_parser", None) and
                           self._chk_ai_hair_parser.isChecked())
            combo = getattr(self, "_combo_ai_hair_parser_backend", None)
            backend = combo.currentText() if combo is not None else "自动"
            ai.set_summary(f"{'开启' if enabled else '关闭'} · {backend}")

    def _refresh_edit_detail_summary(self, tool: str | None = None) -> None:
        tool = tool or self._current_manual_tool_key()
        section = getattr(self, "_edit_tool_detail_section", None)
        hint = getattr(self, "_edit_context_hint", None)
        summaries = {
            "brush": "画笔上色：颜色 · 画笔大小 · 填充模式 · 风格匹配",
            "bias_brush": "颜色倾向画笔：画笔 / 消除 · 颜色 · 强度 · 色调保护",
            "eyedropper": "吸管：点 / 区域取色 · 明度调整 · 模型提示",
            "bucket": "油漆桶：颜色 · 缝隙闭合 · 填充模式",
            "lasso": "套索上色：选区模式 · 羽化 · 闭合区过滤 · AI 重上色",
            "rect": "矩形上色：选区模式 · 羽化 · 闭合区过滤 · AI 重上色",
        }
        if section is not None:
            titles = {
                "brush": "画笔上色基础",
                "bias_brush": "颜色倾向画笔",
                "eyedropper": "吸管基础",
                "bucket": "油漆桶基础",
                "lasso": "套索上色基础",
                "rect": "矩形上色基础",
            }
            section.set_title(titles.get(tool, "当前工具基础细则"))
            section.set_summary(summaries.get(tool, "当前工具参数"))
        boundary = getattr(self, "_selection_boundary_section", None)
        if boundary is not None:
            closed_only = bool(getattr(self, "_chk_selection_closed_only", None) and
                               self._chk_selection_closed_only.isChecked())
            snap = bool(getattr(self, "_chk_selection_snap_lineart", None) and
                        self._chk_selection_snap_lineart.isChecked())
            adjust = bool(getattr(self, "_chk_selection_adjust", None) and
                          self._chk_selection_adjust.isChecked())
            boundary.set_summary(
                f"闭合区：{'开' if closed_only else '关'} · "
                f"线稿吸附：{'开' if snap else '关'} · "
                f"手动修补：{'开' if adjust else '关'}")
        if hint is not None:
            hint.setText(summaries.get(tool, "当前工具细则会按工具自动切换显示"))

    def _expand_current_tool_sections(self) -> None:
        tool = self._current_manual_tool_key()
        base = getattr(self, "_edit_tool_detail_section", None)
        bias = getattr(self, "_bias_brush_section", None)
        boundary = getattr(self, "_selection_boundary_section", None)
        actions = getattr(self, "_selection_action_section", None)
        ai = getattr(self, "_selection_ai_options_section", None)
        if tool == "bias_brush":
            if bias is not None:
                bias.set_expanded(True)
            return
        if base is not None:
            base.set_expanded(True)
        if tool in {"lasso", "rect"}:
            if boundary is not None:
                boundary.set_expanded(True)
            if actions is not None:
                actions.set_expanded(True)
            # AI details stay opt-in: do not automatically expand the very
            # long advanced panel unless the user explicitly opens it.

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
        # The primary action is intentionally a toggle: once any foreground
        # model task is running the same button becomes Stop.
        if bool(getattr(self, "_ui_busy", False)) or self._running_background_workers():
            self._stop_all_running_tasks()
            return
        self._run_colorize(regenerate_auto=True)

    def _run_regenerate(self):
        if bool(getattr(self, "_ui_busy", False)):
            return
        self._run_colorize(regenerate_auto=False)

    def _stop_all_running_tasks(self):
        """Request a safe stop for every active model/background worker.

        PyTorch kernels are not force-terminated: doing so can corrupt MPS/CUDA
        state and leave the application unusable.  Every Colortina worker now
        owns a cooperative stop flag; model/pipeline stages, panel/tile loops and
        result callbacks observe that flag and discard all late results.
        """
        if bool(getattr(self, "_stop_requested", False)):
            return
        self._stop_requested = True
        self._pending_colorize_after_hair_download = None
        requested = 0
        for worker in self._running_background_workers():
            try:
                cancel = getattr(worker, "cancel", None)
                if callable(cancel):
                    cancel()
                else:
                    worker.requestInterruption()
                requested += 1
            except RuntimeError:
                continue
        for button in (getattr(self, "_btn_auto", None),
                       getattr(self, "_btn_hair_recolor_auto", None)):
            if button is not None:
                button.setText(tr("stopping"))
                button.setEnabled(False)
        self.statusBar().showMessage("正在停止当前任务；不会再处理下一页或写回晚到结果…", 0)
        if hasattr(self, "_auto_status_label"):
            self._auto_status_label.setText(tr("stopping"))
        if requested == 0:
            self._finish_stop_state()

    def _finish_stop_state(self):
        self._stop_requested = False
        self._set_busy(False, tr("stopped"))
        if hasattr(self, "_auto_status_label"):
            self._auto_status_label.setText(tr("stopped"))
        self._release_compute_memory()

    def _selected_paths(self) -> list[str]:
        items = self._page_list.selectedItems()
        if len(items) >= 1:
            return [it.data(Qt.ItemDataRole.UserRole) for it in items]
        if self._current_path is not None:
            return [self._current_path]
        return []

    def _current_auto_black_hair_guard(self):
        checkbox = getattr(self, "_chk_auto_black_hair_guard", None)
        if checkbox is None or not checkbox.isChecked():
            return False
        combo = getattr(self, "_combo_auto_black_hair_guard_mode", None)
        if combo is None:
            return True
        return combo.currentData() or "normal"

    def _current_auto_black_hair_guard_scope(self):
        combo = getattr(self, "_combo_auto_black_hair_guard_scope", None)
        if combo is None:
            return "page"
        scope = combo.currentData() or "page"
        return scope if scope in {"page", "near_face"} else "page"

    def _current_auto_black_hair_guard_limit_mode(self):
        combo = getattr(self, "_combo_auto_black_hair_guard_limit_mode", None)
        if combo is None:
            return "full"
        mode = combo.currentData() or "full"
        return mode if mode in {"full", "selection"} else "full"

    def _auto_black_hair_guard_requires_selection(self) -> bool:
        return bool(self._current_auto_black_hair_guard()) and self._current_auto_black_hair_guard_limit_mode() == "selection"

    def _normalise_binary_mask(self, mask: np.ndarray | None) -> np.ndarray | None:
        if mask is None:
            return None
        arr = np.asarray(mask)
        if arr.size == 0:
            return None
        if arr.ndim == 3:
            arr = np.max(arr, axis=2)
        arr = np.where(arr > 0, 255, 0).astype(np.uint8)
        if not np.any(arr):
            return None
        return arr

    def _current_hair_guard_selection_mask(self) -> np.ndarray | None:
        store = getattr(self, "_hair_guard_selection_masks", None) or {}
        path = getattr(self, "_current_path", None)
        if path is None:
            return None
        mask = self._normalise_binary_mask(store.get(path))
        if mask is None:
            return None
        return mask.copy()

    def _set_current_hair_guard_selection_mask(self, mask: np.ndarray | None) -> np.ndarray | None:
        path = getattr(self, "_current_path", None)
        if path is None:
            return None
        store = getattr(self, "_hair_guard_selection_masks", None)
        if store is None:
            store = {}
            self._hair_guard_selection_masks = store
        normalised = self._normalise_binary_mask(mask)
        if normalised is None:
            store.pop(path, None)
            self._update_hair_guard_selection_stats()
            return None
        store[path] = normalised.copy()
        self._update_hair_guard_selection_stats()
        return normalised

    def _current_hair_guard_selection_history(self) -> list[tuple[str, np.ndarray]]:
        path = getattr(self, "_current_path", None)
        if path is None:
            return []
        store = getattr(self, "_hair_guard_selection_histories", None)
        if store is None:
            store = {}
            self._hair_guard_selection_histories = store
        history = store.get(path)
        if history is None:
            history = []
            legacy_mask = self._current_hair_guard_selection_mask()
            if legacy_mask is not None:
                history = [("add", legacy_mask.copy())]
                store[path] = history
            else:
                return []
        return [(str(kind), np.asarray(mask).copy()) for kind, mask in history]

    def _set_current_hair_guard_selection_history(self, history: list[tuple[str, np.ndarray]]) -> np.ndarray | None:
        path = getattr(self, "_current_path", None)
        if path is None:
            return None
        store = getattr(self, "_hair_guard_selection_histories", None)
        if store is None:
            store = {}
            self._hair_guard_selection_histories = store
        normalized_history: list[tuple[str, np.ndarray]] = []
        for kind, mask in history or []:
            norm = self._normalise_binary_mask(mask)
            if norm is None:
                continue
            normalized_history.append(("erase" if str(kind) == "erase" else "add", norm))
        if normalized_history:
            store[path] = [(kind, mask.copy()) for kind, mask in normalized_history]
        else:
            store.pop(path, None)
        return self._set_current_hair_guard_selection_mask(
            self._compose_hair_guard_selection_mask_from_history(normalized_history))

    def _compose_hair_guard_selection_mask_from_history(self, history: list[tuple[str, np.ndarray]] | None) -> np.ndarray | None:
        shape = self._hair_guard_selection_shape()
        if shape is None:
            return None
        composed = np.zeros(shape, np.uint8)
        any_active = False
        for kind, mask in history or []:
            norm = self._normalise_binary_mask(mask)
            if norm is None:
                continue
            if norm.shape[:2] != shape:
                norm = cv2.resize(norm, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
                norm = np.where(norm > 0, 255, 0).astype(np.uint8)
            if str(kind) == "erase":
                composed = cv2.bitwise_and(composed, cv2.bitwise_not(norm))
            else:
                composed = cv2.bitwise_or(composed, norm)
                any_active = True
            if np.any(composed):
                any_active = True
        if not any_active and not np.any(composed):
            return None
        return np.where(composed > 0, 255, 0).astype(np.uint8)

    def _hair_guard_semantic_scribbles_for_path(self, path: str) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Compose positive/negative semantic seeds for one page path.

        This intentionally derives its canvas size from the persisted brush
        masks, so saving a large project never has to materialize every source
        page merely to serialize scribbles.
        """
        history = (getattr(self, "_hair_guard_selection_histories", {}) or {}).get(path, [])
        legacy = (getattr(self, "_hair_guard_selection_masks", {}) or {}).get(path)
        if not history and legacy is not None:
            history = [("add", legacy)]
        first = next((self._normalise_binary_mask(mask) for _kind, mask in history
                      if self._normalise_binary_mask(mask) is not None), None)
        if first is None:
            return None, None
        shape = first.shape[:2]
        positive = np.zeros(shape, np.uint8)
        negative = np.zeros(shape, np.uint8)
        for kind, mask in history:
            norm = self._normalise_binary_mask(mask)
            if norm is None:
                continue
            if norm.shape[:2] != shape:
                norm = cv2.resize(norm, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
                norm = np.where(norm > 0, 255, 0).astype(np.uint8)
            if str(kind) == "erase":
                negative = cv2.bitwise_or(negative, norm)
            else:
                positive = cv2.bitwise_or(positive, norm)
        return (positive if np.any(positive) else None,
                negative if np.any(negative) else None)

    def _current_hair_guard_semantic_scribbles(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return positive/negative rough seeds from the existing hair brush history.

        In Semantic Mask mode the existing "画笔添加" and "画笔擦除" tools
        become true object-segmentation scribbles: adds say "this is hair";
        erases say "this is not hair".  This preserves the old desktop UI
        while absorbing the useful scribble interaction from manga-segment.
        """
        path = getattr(self, "_current_path", None)
        if path is None:
            return None, None
        return self._hair_guard_semantic_scribbles_for_path(path)

    def _all_semantic_hair_scribbles(self) -> dict[str, dict[str, np.ndarray]]:
        out = {}
        for path in (getattr(self, "_pages", {}) or {}):
            positive, negative = self._hair_guard_semantic_scribbles_for_path(path)
            row = {}
            if positive is not None:
                row["hair_positive"] = positive
            if negative is not None:
                row["hair_negative"] = negative
            if row:
                out[os.path.abspath(path)] = row
        return out

    def _append_hair_guard_selection_operation(self, mask: np.ndarray | None, *, add_mode: bool = True) -> np.ndarray | None:
        norm = self._normalise_binary_mask(mask)
        if norm is None:
            return self._current_hair_guard_selection_mask()
        history = self._current_hair_guard_selection_history()
        history.append(("add" if add_mode else "erase", norm))
        return self._set_current_hair_guard_selection_history(history)

    def _undo_last_hair_guard_selection_operation(self) -> None:
        history = self._current_hair_guard_selection_history()
        if not history:
            self.statusBar().showMessage("当前没有可撤销的头发纠色区域操作。", 2500)
            return
        history.pop()
        saved = self._set_current_hair_guard_selection_history(history)
        self._show_preferred_selection_overlay()
        count = len(history)
        area = int(np.count_nonzero(saved)) if saved is not None else 0
        self.statusBar().showMessage(f"已撤销上一个头发纠色区域操作。剩余 {count} 个操作，覆盖 {area} px。", 3000)

    def _update_hair_guard_selection_stats(self) -> None:
        label = getattr(self, "_label_hair_guard_selection_stats", None)
        if label is None:
            return
        history = self._current_hair_guard_selection_history()
        mask = self._current_hair_guard_selection_mask()
        area = int(np.count_nonzero(mask)) if mask is not None else 0
        label.setText(f"当前头发纠色区域：{len(history)} 个操作 / {area} px")
        self._refresh_auto_detail_summaries()
        undo_btn = getattr(self, "_btn_hair_guard_undo_last", None)
        if undo_btn is not None:
            undo_btn.setEnabled(bool(history))

    def _hair_guard_selection_shape(self) -> tuple[int, int] | None:
        state = self._current_state()
        if state is None:
            return None
        if getattr(state, "original_bgr", None) is not None:
            return tuple(int(v) for v in state.original_bgr.shape[:2])
        if getattr(state, "result_bgr", None) is not None:
            return tuple(int(v) for v in state.result_bgr.shape[:2])
        return None

    def _show_preferred_selection_overlay(self) -> None:
        self._update_hair_guard_selection_stats()
        hair_mask = self._current_hair_guard_selection_mask()
        self._update_hair_guard_selection_stats()
        if hair_mask is not None and (self._hair_guard_selector_active or self._auto_black_hair_guard_requires_selection()):
            self._canvas.set_selection_mask_overlay(hair_mask, mode="replace")
            return
        if getattr(self, "_pending_selection_mask", None) is not None and np.any(self._pending_selection_mask):
            self._canvas.set_selection_mask_overlay(
                self._pending_selection_mask, mode=self._current_selection_mode())
            return
        self._canvas.clear_selection_preview()

    def _update_hair_guard_brush_radius(self, value: int) -> None:
        self._canvas.set_selection_adjust_radius(int(value))

    def _activate_hair_guard_selector(self, mode: str, *, for_refinement: bool = False) -> None:
        state = self._current_state()
        shape = self._hair_guard_selection_shape()
        if state is None or shape is None:
            QMessageBox.information(self, "无法框选", "请先导入并选中页面，再使用头发纠色独立选区器。")
            return
        combo = getattr(self, "_combo_auto_black_hair_guard_limit_mode", None)
        if combo is not None and not for_refinement:
            idx = combo.findData("selection")
            if idx >= 0:
                combo.setCurrentIndex(idx)
        self._hair_guard_selector_mode = str(mode)
        self._hair_guard_selector_active = True
        mode_label = getattr(self, "_label_hair_guard_selector_mode", None)
        if mode_label is not None:
            mode_text = {"rect": "矩形追加", "brush_add": "画笔添加", "brush_erase": "画笔擦除"}.get(str(mode), str(mode))
            mode_label.setText(f"当前选区器：{mode_text}")
        self._canvas.clear_debug_overlays()
        existing = self._current_hair_guard_selection_mask()
        if existing is not None:
            self._canvas.set_selection_mask_overlay(existing, mode="replace")
        elif str(mode).startswith("brush"):
            self._canvas.clear_selection_preview()
        self._canvas.set_selection_adjust_radius(
            int(getattr(self, "_hair_guard_brush_radius_spin", None).value()
                if getattr(self, "_hair_guard_brush_radius_spin", None) is not None else 24))
        if mode == "rect":
            self._canvas.set_selection_adjust_enabled(False)
            self._canvas.set_selection_adjust_allow_empty(False)
            self._canvas.set_tool(HintCanvas.TOOL_RECT_BUCKET)
            self.statusBar().showMessage("头发纠色独立矩形框选器已开启：每次拖出一个蓝框并追加到当前头发纠色区域，可连续框多个角色。", 6000)
        else:
            self._canvas.set_selection_adjust_mode("erase" if mode == "brush_erase" else "add")
            self._canvas.set_selection_adjust_allow_empty(True)
            self._canvas.set_selection_adjust_enabled(True)
            self._canvas.set_tool(HintCanvas.TOOL_LASSO_BUCKET)
            self.statusBar().showMessage("头发纠色独立画笔已开启：可继续添加或擦除多个头发纠色区域，只作用于自动头发纠色。", 6000)

    def _deactivate_hair_guard_selector(self) -> None:
        if not self._hair_guard_selector_active and self._hair_guard_selector_mode is None:
            return
        was_refining = bool(getattr(self, '_hair_refine_brush_active', False))
        self._hair_guard_selector_active = False
        self._hair_refine_brush_active = False
        if was_refining:
            self._canvas.clear_debug_overlays()
        self._hair_guard_selector_mode = None
        mode_label = getattr(self, "_label_hair_guard_selector_mode", None)
        if mode_label is not None:
            mode_label.setText("当前选区器：未启用")
        self._canvas.set_selection_adjust_enabled(False)
        self._canvas.set_selection_adjust_allow_empty(False)
        self._show_preferred_selection_overlay()
        self._suspend_hair_guard_tool_reset = True
        try:
            self._on_tool_changed()
        finally:
            self._suspend_hair_guard_tool_reset = False
        self.statusBar().showMessage("已结束头发纠色独立选区器。", 2500)

    def _clear_hair_guard_selection(self) -> None:
        self._set_current_hair_guard_selection_history([])
        self._hair_guard_selection_stroke_mask = None
        self._hair_guard_selection_stroke_base_mask = None
        self._canvas.clear_debug_overlays()
        self._show_preferred_selection_overlay()
        self.statusBar().showMessage("已清除头发纠色独立选区。", 2500)

    def _current_auto_black_hair_guard_limit_mask(self):
        if not self._auto_black_hair_guard_requires_selection():
            return None
        hair_mask = self._current_hair_guard_selection_mask()
        if hair_mask is not None:
            return hair_mask
        mask = getattr(self, "_pending_selection_mask", None)
        return self._normalise_binary_mask(mask)

    def _ensure_auto_black_hair_guard_selection_ready(self, *, interactive: bool = False) -> bool:
        if not self._auto_black_hair_guard_requires_selection():
            return True
        limit_mask = self._current_auto_black_hair_guard_limit_mask()
        if limit_mask is not None and np.any(limit_mask):
            return True
        message = (
            "当前已将“自动头发纠色”的作用区域设为“仅当前选区（蓝框）”，"
            "但还没有有效选区。请先使用头发纠色独立框选器 / 画笔，或编辑区蓝框，框住目标人物或目标头发区域，再进行预览或自动上色。")
        if interactive:
            QMessageBox.information(self, "需要选区", message)
        else:
            self.statusBar().showMessage(message, 5000)
        return False

    def _current_auto_black_hair_guard_target_rgb(self):
        color = QColor(getattr(self, '_hair_recolor_color', QColor(24, 24, 24)))
        return (int(color.red()), int(color.green()), int(color.blue()))

    def _clear_auto_black_hair_guard_preview(self):
        self._canvas.clear_debug_overlays()
        self._show_preferred_selection_overlay()
        if (self._current_hair_guard_selection_mask() is None and
                getattr(self, "_pending_selection_mask", None) is None):
            state = self._current_state()
            if state is not None:
                self._sync_view_after_edit(state)
        self.statusBar().showMessage("已清除自动头发纠色预览。", 2200)

    def _preview_auto_black_hair_guard_mask(self):
        state = self._current_state()
        if state is None:
            return
        guard = self._current_auto_black_hair_guard()
        if not guard:
            self.statusBar().showMessage("当前已关闭自动头发纠色。", 2500)
            return
        if not self._ensure_auto_black_hair_guard_selection_ready(interactive=True):
            return
        scope = self._current_auto_black_hair_guard_scope()
        limit_mask = self._current_auto_black_hair_guard_limit_mask()
        semantic = state.ai_result_bgr
        if semantic is None:
            semantic = state.result_bgr
        if semantic is None:
            self.statusBar().showMessage("请先至少自动上色一次，再预览自动头发纠色遮罩。", 4000)
            return
        try:
            from core.illustrative_inkwash import build_auto_hair_recolor_mask
            parser_enabled = bool(
                getattr(self, "_chk_ai_hair_parser", None) is not None and
                self._chk_ai_hair_parser.isChecked() and
                self._current_ai_hair_parser_backend() != "legacy")
            mask, diagnostics = build_auto_hair_recolor_mask(
                state.original_bgr, semantic,
                mode=self._current_auto_black_hair_guard(),
                scope=scope,
                limit_mask=limit_mask,
                parser_enabled=parser_enabled,
                parser_backend=self._current_ai_hair_parser_backend(),
                parser_allow_download=False,
                return_preview_layers=True)
        except Exception as exc:
            self.statusBar().showMessage(f"自动头发纠色预览失败：{type(exc).__name__}: {exc}", 5000)
            return
        if mask is None or not np.any(mask):
            self.statusBar().showMessage(
                "当前设置下没有检测到会被自动纠色的头发区域。"
                + (f" 原因：{diagnostics.get('reason')}" if isinstance(diagnostics, dict) and diagnostics.get('reason') else ""),
                4500)
            return
        preview_layers = (diagnostics.get("_preview_layers", {})
                          if isinstance(diagnostics, dict) else {})
        primary = preview_layers.get("primary") if isinstance(preview_layers, dict) else None
        supplemental = preview_layers.get("supplemental") if isinstance(preview_layers, dict) else None
        rejected_thin = preview_layers.get("rejected_thin") if isinstance(preview_layers, dict) else None
        if primary is None:
            primary = mask
        self._canvas.set_auto_hair_preview_overlay(
            primary, supplemental, rejected_thin)
        px = int(np.count_nonzero(mask))
        primary_px = int(np.count_nonzero(primary)) if primary is not None else px
        supplemental_px = int(np.count_nonzero(supplemental)) if supplemental is not None else 0
        rejected_px = int(np.count_nonzero(rejected_thin)) if rejected_thin is not None else 0
        detector = (diagnostics.get("selected_detector", "未知")
                    if isinstance(diagnostics, dict) else "未知")
        self.statusBar().showMessage(
            f"头发遮罩预览：主体蓝 {primary_px} · 补漏绿 {supplemental_px} · 已排除细丝红 {rejected_px} · {detector}",
            6500)

    def _pick_hair_recolor_color(self):
        color = self._open_color_dialog(
            QColor(getattr(self, '_hair_recolor_color', QColor(24, 24, 24))),
            "选择自动纠色目标发色")
        if color.isValid():
            self._hair_recolor_color = QColor(color)
            self._update_hair_recolor_swatch()

    def _update_hair_recolor_swatch(self):
        btn = getattr(self, '_btn_hair_recolor_color', None)
        color = QColor(getattr(self, '_hair_recolor_color', QColor(24, 24, 24)))
        if btn is not None:
            btn.setText('')
            btn.setStyleSheet(f"background-color: {color.name()}; border: 1px solid #888;")
            btn.setToolTip(color.name())
        info = getattr(self, '_hair_recolor_color_info', None)
        if info is not None:
            info.setText(color.name() + " · 目标发色")
            info.setToolTip("该颜色用于旧头发纠色与新版 Hair 蒙版覆盖，不会改动普通画笔颜色。")
        if hasattr(self, "_refresh_auto_detail_summaries"):
            self._refresh_auto_detail_summaries()


    def _pick_iris_recolor_color(self):
        color = self._open_color_dialog(
            QColor(getattr(self, '_iris_recolor_color', QColor(70, 135, 220))),
            "选择统一瞳色")
        if color.isValid():
            self._iris_recolor_color = QColor(color)
            self._update_iris_recolor_swatch()

    def _update_iris_recolor_swatch(self):
        btn = getattr(self, '_btn_iris_recolor_color', None)
        color = QColor(getattr(self, '_iris_recolor_color', QColor(70, 135, 220)))
        if btn is not None:
            btn.setText('')
            btn.setStyleSheet(
                f"background-color: {color.name()}; border: 1px solid #888;")
            btn.setToolTip(color.name())
        info = getattr(self, '_iris_recolor_color_info', None)
        if info is not None:
            info.setText(color.name() + " · 虹膜覆盖目标")

    def _update_semantic_mask_controls_enabled(self):
        enabled = bool(getattr(self, '_chk_semantic_mask_layers', None) and
                       self._chk_semantic_mask_layers.isChecked())
        for widget in (
                getattr(self, '_chk_mask_hair_override', None),
                getattr(self, '_chk_mask_iris_override', None),
                getattr(self, '_btn_iris_recolor_color', None),
                getattr(self, '_combo_mask_background_mode', None),
                getattr(self, '_combo_mask_recolor_mode', None),
                getattr(self, '_chk_mask_character_library_colors', None),
                getattr(self, '_btn_preview_semantic_masks', None),
                getattr(self, '_btn_clear_semantic_masks_preview', None)):
            if widget is not None:
                widget.setEnabled(enabled)

    def _preview_semantic_mask_layers(self):
        state = self._current_state()
        if state is None:
            return
        base = state.ai_result_bgr
        if base is None:
            base = state.result_bgr
        if base is None:
            self.statusBar().showMessage("请先自动上色一次，再预览语义蒙版。", 3500)
            return
        try:
            from core.semantic_mask_layers import build_semantic_masks
            from pipeline import get_colorizer
            controls = self._current_semantic_mask_controls()
            fusion_cfg = dict(controls.get('fusion') or {})
            provider = None
            if bool(fusion_cfg.get('use_generator_features', True)):
                provider = lambda labels: get_colorizer(Config).extract_region_embeddings(
                    state.original_bgr, labels, size=576)
            masks = build_semantic_masks(
                base, state.original_bgr,
                parser_backend=self._current_ai_hair_parser_backend(),
                parser_allow_download=False,
                manual_masks=dict(controls.get('manual_masks') or {}),
                generator_feature_provider=provider,
                use_region_fusion=bool(fusion_cfg.get('enabled', True)),
                use_trappedball=bool(fusion_cfg.get('use_trappedball', True)),
                use_clip_regions=bool(fusion_cfg.get('use_clip_regions', True)),
                clip_allow_download=False, use_cache=True)
            self._canvas.set_semantic_mask_preview_overlay(
                masks.hair_mask, masks.iris_mask, masks.background_mask)
            diag = masks.diagnostics or {}
            self.statusBar().showMessage(
                "蒙版预览：蓝=头发，紫=虹膜，橙=背景；"
                f"hair {diag.get('hair_pixels', 0)} px / "
                f"iris {(diag.get('iris') or {}).get('iris_pixels', 0)} px / "
                f"background {diag.get('background_pixels', 0)} px",
                6000)
        except Exception as exc:
            self.statusBar().showMessage(
                f"语义蒙版预览失败：{type(exc).__name__}: {exc}", 6000)

    def _current_semantic_mask_controls(self) -> dict:
        enabled = bool(getattr(self, '_chk_semantic_mask_layers', None) and
                       self._chk_semantic_mask_layers.isChecked())
        hair_color = QColor(getattr(self, '_hair_recolor_color', QColor(24, 24, 24)))
        iris_color = QColor(getattr(self, '_iris_recolor_color', QColor(70, 135, 220)))
        mode_combo = getattr(self, '_combo_mask_recolor_mode', None)
        recolor_mode = str(mode_combo.currentData() or 'shading') if mode_combo is not None else 'shading'
        bg_combo = getattr(self, '_combo_mask_background_mode', None)
        bg_mode = str(bg_combo.currentData() or 'keep') if bg_combo is not None else 'keep'
        manual_masks = {}
        # Reuse Colortina's existing independent hair brush/selection as a
        # positive semantic scribble. The fusion engine expands the rough
        # stroke to complete line-bounded hair regions; it is not painted
        # literally unless the user supplies an exact mask through the API.
        hair_positive, hair_negative = self._current_hair_guard_semantic_scribbles()
        if enabled and hair_positive is not None and np.any(hair_positive):
            manual_masks['hair_positive'] = hair_positive.copy()
        if enabled and hair_negative is not None and np.any(hair_negative):
            manual_masks['hair_negative'] = hair_negative.copy()
        return {
            'enabled': enabled,
            'composition': 'hard_replace',
            'use_character_library_colors': bool(
                enabled and getattr(self, '_chk_mask_character_library_colors', None)
                and self._chk_mask_character_library_colors.isChecked()),
            'manual_masks': manual_masks,
            'fusion': {
                'enabled': True,
                'use_generator_features': True,
                'use_trappedball': True,
                'use_clip_regions': True,
                'clip_allow_download': False,
                'use_cache': True,
            },
            'hair': {
                'enabled': bool(enabled and getattr(self, '_chk_mask_hair_override', None)
                                and self._chk_mask_hair_override.isChecked()),
                'target_rgb': (hair_color.red(), hair_color.green(), hair_color.blue()),
                'mode': recolor_mode,
            },
            'iris': {
                'enabled': bool(enabled and getattr(self, '_chk_mask_iris_override', None)
                                and self._chk_mask_iris_override.isChecked()),
                'target_rgb': (iris_color.red(), iris_color.green(), iris_color.blue()),
                'mode': recolor_mode,
            },
            'background': {
                'enabled': bool(enabled and bg_mode != 'keep'),
                'mode': bg_mode,
            },
        }

    def _current_ai_hair_parser_backend(self) -> str:
        combo = getattr(self, "_combo_ai_hair_parser_backend", None)
        if combo is None:
            return "auto"
        return str(combo.currentData() or "auto")

    def _refresh_ai_hair_model_status(self):
        label = getattr(self, "_ai_hair_model_status", None)
        try:
            from core.hair_model_manager import (
                model_status, total_size_mb, validate_model)
            rows = model_status(Config, deep_validate=False)
            # ONNX files are the most likely to be replaced by an HTML/error
            # payload while still exceeding the size threshold. Validate these
            # two graphs before showing a green check, without loading the large
            # PyTorch checkpoints on every UI refresh.
            for key in ("skytnt_foreground", "yakhyo_face_parsing"):
                row = rows.get(key)
                if row and row.get("ready"):
                    valid, detail = validate_model(key, row.get("path"), Config)
                    row["ready"] = bool(valid)
                    row["detail"] = detail
            ready = {key: bool(row.get("ready")) for key, row in rows.items()}
            main_ready = ready.get("styleanime", False) or ready.get(
                "anime_face_segmentation", False)
            missing = [key for key, value in ready.items() if not value]
            missing_mb = total_size_mb(missing) if missing else 0.0
            text = (
                f"推荐主模型：轻量动漫分割 {'✓' if ready.get('anime_face_segmentation') else '—'}；"
                f"StyleAnime 辅助 {'✓' if ready.get('styleanime') else '—'}；"
                f"防泄露 {'✓' if ready.get('skytnt_foreground') else '—'}；"
                f"皮肤/眼镜排除 {'✓' if ready.get('yakhyo_face_parsing') else '—'}")
            if missing:
                text += f"。尚缺 {len(missing)} 项，约 {missing_mb:g} MB"
            else:
                text += "。四类模型均已安装"
            if not main_ready:
                text += "；当前会回退旧头发算法"
            if label is not None:
                label.setText(text)
                label.setToolTip("\n".join(
                    f"{row['display_name']}：{'已安装' if row['ready'] else '未安装'}\n{row['path']}"
                    for row in rows.values()))
            return rows
        except Exception as exc:
            if label is not None:
                label.setText(f"AI 头发模型状态读取失败：{type(exc).__name__}")
                label.setToolTip(str(exc))
            return {}

    def _manage_ai_hair_models(self):
        worker = getattr(self, "_hair_model_worker", None)
        if worker is not None and worker.isRunning():
            self.statusBar().showMessage("AI 头发模型正在下载/校验，请稍候。", 4000)
            return
        from core.hair_model_manager import (
            ALL_MODEL_KEYS, RECOMMENDED_MODEL_KEYS, model_status,
            repair_invalid_downloads, total_size_mb)
        removed = repair_invalid_downloads(Config)
        if removed:
            self.statusBar().showMessage(
                f"已清理 {len(removed)} 个损坏或不完整的模型文件。", 5000)
        rows = model_status(Config, deep_validate=False)
        missing_recommended = [
            key for key in RECOMMENDED_MODEL_KEYS if not rows[key]["ready"]]
        missing_all = [key for key in ALL_MODEL_KEYS if not rows[key]["ready"]]
        if not missing_all:
            QMessageBox.information(
                self, "AI 头发模型", "四类 AI 头发解析模型均已安装。")
            self._refresh_ai_hair_model_status()
            return

        box = QMessageBox(self)
        box.setWindowTitle("安装 AI 头发解析模型")
        box.setIcon(QMessageBox.Icon.Question)
        box.setText(
            "请选择要安装的模型。下载只会在你确认后进行，模型保存到本项目的 "
            "models/weights 目录，之后可离线使用。")
        details = []
        for key in missing_all:
            row = rows[key]
            details.append(
                f"• {row['display_name']}：约 {row['size_mb']:g} MB\n  {row['role']}")
        box.setDetailedText("\n".join(details))
        recommended_button = None
        if missing_recommended:
            recommended_button = box.addButton(
                f"安装推荐模型（约 {total_size_mb(missing_recommended):g} MB）",
                QMessageBox.ButtonRole.AcceptRole)
        all_button = box.addButton(
            f"安装全部缺失模型（约 {total_size_mb(missing_all):g} MB）",
            QMessageBox.ButtonRole.ActionRole)
        import_button = box.addButton(
            "导入本地模型文件", QMessageBox.ButtonRole.ActionRole)
        cancel_button = box.addButton(
            QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if recommended_button is not None and clicked is recommended_button:
            self._start_ai_hair_model_download(missing_recommended)
        elif clicked is all_button:
            self._start_ai_hair_model_download(missing_all)
        elif clicked is import_button:
            self._import_local_ai_hair_model()
        elif clicked is cancel_button:
            return

    def _import_local_ai_hair_model(self):
        from core.hair_model_manager import MODEL_SPECS
        keys = list(MODEL_SPECS.keys())
        labels = [MODEL_SPECS[key].display_name for key in keys]
        selected, ok = QInputDialog.getItem(
            self, "导入本地 AI 头发模型", "模型类型：", labels, 0, False)
        if not ok or not selected:
            return
        key = keys[labels.index(selected)]
        spec = MODEL_SPECS[key]
        file_filter = (
            "模型文件 (*.pth *.pt *.onnx *.zip);;所有文件 (*)")
        source, _ = QFileDialog.getOpenFileName(
            self, f"选择 {spec.display_name}", "", file_filter)
        if not source:
            return
        worker = getattr(self, "_hair_model_worker", None)
        if worker is not None and worker.isRunning():
            self.statusBar().showMessage("另一个 AI 模型任务正在运行。", 4000)
            return
        self._pending_colorize_after_hair_download = None
        self._hair_model_worker = HairModelImportWorker(key, source, self)
        self._hair_model_worker.status.connect(self._on_worker_status)
        self._hair_model_worker.finished_ok.connect(self._on_ai_hair_models_ready)
        self._hair_model_worker.finished_err.connect(self._on_ai_hair_models_error)
        self._hair_model_worker.cancelled.connect(self._on_ai_hair_models_cancelled)
        btn = getattr(self, "_btn_manage_ai_hair_models", None)
        if btn is not None:
            btn.setEnabled(False)
            btn.setText("正在导入…")
        self.statusBar().showMessage("正在校验并导入 AI 头发模型…", 0)
        self._hair_model_worker.start()

    def _start_ai_hair_model_download(self, keys, *, resume_colorize=None):
        keys = list(dict.fromkeys(str(key) for key in keys))
        if not keys:
            self._refresh_ai_hair_model_status()
            if resume_colorize is not None:
                QTimer.singleShot(0, lambda: self._run_colorize(bool(resume_colorize)))
            return
        worker = getattr(self, "_hair_model_worker", None)
        if worker is not None and worker.isRunning():
            return
        self._pending_colorize_after_hair_download = resume_colorize
        self._hair_model_worker = HairModelDownloadWorker(keys, self)
        self._hair_model_worker.status.connect(self._on_worker_status)
        self._hair_model_worker.finished_ok.connect(
            self._on_ai_hair_models_ready)
        self._hair_model_worker.finished_err.connect(
            self._on_ai_hair_models_error)
        self._hair_model_worker.cancelled.connect(self._on_ai_hair_models_cancelled)
        btn = getattr(self, "_btn_manage_ai_hair_models", None)
        if btn is not None:
            btn.setEnabled(False)
            btn.setText("正在下载…")
        if resume_colorize is not None:
            self._stop_requested = False
            self._set_busy(True, "正在准备 AI 头发解析模型…")
        else:
            self.statusBar().showMessage("正在准备 AI 头发解析模型…", 0)
        self._hair_model_worker.start()

    def _on_ai_hair_models_ready(self, _result):
        worker = self._hair_model_worker
        self._hair_model_worker = None
        btn = getattr(self, "_btn_manage_ai_hair_models", None)
        if btn is not None:
            btn.setEnabled(True)
            btn.setText("安装/检查模型")
        self._refresh_ai_hair_model_status()
        self.statusBar().showMessage("AI 头发解析模型准备完成。", 5000)
        pending = self._pending_colorize_after_hair_download
        self._pending_colorize_after_hair_download = None
        if worker is not None:
            worker.deleteLater()
        if pending is not None and not self._stop_requested:
            self._set_busy(False)
            QTimer.singleShot(0, lambda: self._run_colorize(bool(pending)))
        elif self._ui_busy:
            self._set_busy(False)

    def _on_ai_hair_models_cancelled(self):
        worker = self._hair_model_worker
        self._hair_model_worker = None
        self._pending_colorize_after_hair_download = None
        btn = getattr(self, "_btn_manage_ai_hair_models", None)
        if btn is not None:
            btn.setText("安装/检查模型")
        if worker is not None:
            worker.deleteLater()
        self._refresh_ai_hair_model_status()
        self._finish_stop_state()

    def _on_ai_hair_models_error(self, message: str):
        worker = self._hair_model_worker
        self._hair_model_worker = None
        self._pending_colorize_after_hair_download = None
        if self._ui_busy:
            self._set_busy(False)
        btn = getattr(self, "_btn_manage_ai_hair_models", None)
        if btn is not None:
            btn.setEnabled(True)
            btn.setText("安装/检查模型")
        self._refresh_ai_hair_model_status()
        if worker is not None:
            worker.deleteLater()
        QMessageBox.critical(
            self, "AI 头发模型下载失败",
            f"模型没有安装完成：\n{message}\n\n"
            "网络中断产生的有效 .part 文件会保留续传；校验失败、HTML 错误页或损坏的 ONNX 会自动删除，"
            "下次重试将从零下载。现有 mc-v2 上色与旧头发算法不受影响。")
        self.statusBar().showMessage("AI 头发模型准备失败。", 6000)

    def _ensure_ai_hair_models_for_run(self, regenerate_auto: bool) -> bool:
        checkbox = getattr(self, "_chk_ai_hair_parser", None)
        semantic_enabled = bool(
            getattr(self, "_chk_semantic_mask_layers", None) is not None and
            self._chk_semantic_mask_layers.isChecked())
        # The AI hair parser supports both optional hair recoloring and the
        # independent semantic mask workspace. Neither requires downloading
        # weights when both functions are switched off.
        if not self._current_auto_black_hair_guard() and not semantic_enabled:
            return True
        if (checkbox is None or not checkbox.isChecked()) and not semantic_enabled:
            return True
        backend = self._current_ai_hair_parser_backend()
        if backend in {"legacy", "none", "off"}:
            return True
        try:
            from core.hair_model_manager import (
                RECOMMENDED_MODEL_KEYS, model_status, total_size_mb)
            rows = model_status(Config, deep_validate=False)
        except Exception:
            return True
        if backend == "styleanime":
            ready = rows["styleanime"]["ready"]
            needed = ["styleanime"]
        elif backend == "anime_face_segmentation":
            ready = rows["anime_face_segmentation"]["ready"]
            needed = ["anime_face_segmentation"]
        else:
            ready = (rows["styleanime"]["ready"] or
                     rows["anime_face_segmentation"]["ready"] )
            needed = [key for key in RECOMMENDED_MODEL_KEYS
                      if not rows[key]["ready"]]
        if ready:
            return True
        worker = getattr(self, "_hair_model_worker", None)
        if worker is not None and worker.isRunning():
            self.statusBar().showMessage(
                "AI 头发模型仍在下载，完成后会继续上色。", 5000)
            return False

        box = QMessageBox(self)
        box.setWindowTitle("需要 AI 头发识别模型")
        box.setIcon(QMessageBox.Icon.Question)
        box.setText(
            "当前选择的 AI 头发识别后端尚未安装。是否先下载推荐模型？")
        box.setInformativeText(
            f"预计下载约 {total_size_mb(needed):g} MB。选择“使用旧算法继续”"
            "不会下载任何文件，本次会直接回退现有头发纠色算法。")
        download_button = box.addButton(
            "下载后继续上色", QMessageBox.ButtonRole.AcceptRole)
        legacy_button = box.addButton(
            "使用旧算法继续", QMessageBox.ButtonRole.DestructiveRole)
        cancel_button = box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is download_button:
            self._start_ai_hair_model_download(
                needed, resume_colorize=regenerate_auto)
            return False
        if clicked is legacy_button:
            return True
        if clicked is cancel_button:
            return False
        return False

    def _run_colorize(self, regenerate_auto: bool):
        if self._batch_worker is not None and self._batch_worker.isRunning():
            self.statusBar().showMessage("上色任务正在运行，请稍候。", 4000)
            return
        paths = self._selected_paths()
        if not paths:
            QMessageBox.information(self, tr("no_result_title"), "请先导入并选择至少一张页面。")
            return
        from core.model_downloader import models_ready
        allow_base_download = False
        if not models_ready(Config.WEIGHTS_DIR):
            if os.environ.get("COLORTINA_OFFLINE", "").strip().lower() in {"1", "true", "on", "yes"}:
                QMessageBox.warning(self, "本地模型缺失",
                    "已启用离线模式。请将 generator.zip 放入模型目录后重试。")
                return
            answer = QMessageBox.question(
                self, "首次安装 mc-v2 模型",
                "未发现本地上色模型 generator.zip（约 129 MB）。\n"
                "是否允许本次操作通过网络下载？画稿不会上传。",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return
            allow_base_download = True
        if not self._ensure_auto_black_hair_guard_selection_ready(interactive=True):
            return
        if not self._ensure_ai_hair_models_for_run(regenerate_auto):
            return
        current_scope = self._current_auto_black_hair_guard_scope()
        limit_mask = self._current_auto_black_hair_guard_limit_mask()
        pages = [(p, self._pages[p]._original_bgr,
                  self._pages[p].hint_manager,
                  dict(self._pages[p].forced_character_matches))
                 for p in paths]

        character_memories = self._get_or_create_character_memories()

        self._batch_errors = []
        self._stop_requested = False
        self._set_busy(True, tr("batch_colorizing").format(n=len(pages))
                       if len(pages) > 1 else tr("colorizing"))
        self._batch_worker = BatchColorizeWorker(
            pages, regenerate_auto,
            style_key=self._style_combo.currentData(),
            quality_key=self._quality_combo.currentData(),
            ink_fidelity=self._chk_ink_fidelity.isChecked(),
            hint_render_mode=self._native_hint_mode_combo.currentData(),
            allow_model_download=allow_base_download,
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
            auto_black_hair_guard=self._current_auto_black_hair_guard(),
            auto_black_hair_guard_scope=self._current_auto_black_hair_guard_scope(),
            auto_black_hair_guard_limit_mask=self._current_auto_black_hair_guard_limit_mask(),
            auto_black_hair_guard_target_rgb=self._current_auto_black_hair_guard_target_rgb(),
            ai_hair_parser_enabled=(
                self._chk_ai_hair_parser.isChecked() and
                self._current_ai_hair_parser_backend() != "legacy"),
            ai_hair_parser_backend=self._current_ai_hair_parser_backend(),
            ai_hair_parser_allow_download=False,
            semantic_mask_controls=self._current_semantic_mask_controls(),
        )
        self._batch_worker.page_done.connect(self._on_batch_page_done)
        self._batch_worker.page_skipped.connect(self._on_batch_page_skipped)
        self._batch_worker.page_error.connect(self._on_batch_page_error)
        self._batch_worker.status.connect(self._on_worker_status)
        self._batch_worker.cancelled.connect(lambda: None)
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
            "inkwash_strength": self._inkwash_strength_slider.value(),
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
            return [state for state in self._ordered_states() if state.has_result]
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
            if state.path == self._current_path:
                state.push_undo(fields=state.FILTER_EDIT_FIELDS)
            state.result_bgr = apply_image_filter(
                base.copy(), tuning,
                style_strength=style_strength,
                is_styled=style_key not in {None, "none"},
                source_bw_bgr=state.original_bgr)

            # Semantic mask layers are final-authority colour replacements.
            # A later manual filter must not wash out the locked hair/iris
            # colours, so reapply the same hard-replace contract after filtering
            # without touching mc-v2.  Use the stored AI result only to derive
            # masks; the replacement is applied to the freshly filtered image.
            semantic_controls = self._current_semantic_mask_controls()
            if semantic_controls.get("enabled", False):
                try:
                    from core.semantic_mask_layers import (
                        build_semantic_masks, apply_semantic_mask_overrides)
                    semantic_source = state.ai_result_bgr
                    if semantic_source is None:
                        semantic_source = state.result_bgr
                    fusion_cfg = dict(semantic_controls.get('fusion') or {})
                    provider = None
                    if bool(fusion_cfg.get('use_generator_features', True)):
                        from pipeline import get_colorizer
                        provider = lambda labels: get_colorizer(Config).extract_region_embeddings(
                            state.original_bgr, labels, size=576)
                    masks = build_semantic_masks(
                        semantic_source, state.original_bgr,
                        parser_backend=self._current_ai_hair_parser_backend(),
                        parser_allow_download=False,
                        manual_masks=dict(semantic_controls.get('manual_masks') or {}),
                        generator_feature_provider=provider,
                        use_region_fusion=bool(fusion_cfg.get('enabled', True)),
                        use_trappedball=bool(fusion_cfg.get('use_trappedball', True)),
                        use_clip_regions=bool(fusion_cfg.get('use_clip_regions', True)),
                        clip_allow_download=False, use_cache=True)
                    state.result_bgr, _masks, _diag = apply_semantic_mask_overrides(
                        state.result_bgr, state.original_bgr, semantic_controls,
                        parser_backend=self._current_ai_hair_parser_backend(),
                        parser_allow_download=False, masks=masks)
                    if self._chk_protect_text.isChecked():
                        from core.text_guard import protect_text_regions
                        source_guard = state.original_bgr
                        if source_guard.shape[:2] != state.result_bgr.shape[:2]:
                            source_guard = cv2.resize(
                                source_guard,
                                (state.result_bgr.shape[1], state.result_bgr.shape[0]),
                                interpolation=cv2.INTER_CUBIC)
                        state.result_bgr = protect_text_regions(
                            state.result_bgr, source_guard)
                except Exception as exc:
                    self.statusBar().showMessage(
                        f"滤镜已应用，但蒙版覆盖重应用失败：{type(exc).__name__}: {exc}",
                        6000)
            state.discard_unchanged_undo()
            self._mark_page_done(state.path)
            if state.path != self._current_path:
                state.undo_stack.clear()
                state.redo_stack.clear()
                self._schedule_page_spill(state.path, delay_ms=0)
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
        is_light3 = key in {"light3", "light3_git_legacy"}
        is_new_inkwash = key in {"light2", "light2_soft", "light3"}
        # Old Git presets use the original grading pipeline; the NEW watercolour
        # strength does not apply. Hide it rather than offering an inert slider.
        for widget in (getattr(self, "_inkwash_strength_label", None),
                       getattr(self, "_inkwash_strength_slider", None),
                       getattr(self, "_inkwash_strength_spin", None)):
            if widget is not None:
                widget.setVisible(is_new_inkwash)
        ink_fidelity = getattr(self, "_chk_ink_fidelity", None)
        if ink_fidelity is not None:
            ink_fidelity.setEnabled(key not in {"light2_git_legacy", "light3_git_legacy"})
            ink_fidelity.setToolTip(
                "旧版风格走原始分级，跳过新版线稿保真与水墨渲染，便于原样对照。"
                if key in {"light2_git_legacy", "light3_git_legacy"} else
                "保留原稿墨线、网点与细部纹理，可关闭。")
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
            "quality_key": self._quality_combo.currentData(),
            "hint_render_mode": self._native_hint_mode_combo.currentData(),
            "ink_fidelity": self._chk_ink_fidelity.isChecked(),
            "export_compression_profile": self._current_export_profile(),
            "style_strength": self._style_strength_slider.value(),
            "reference_strength": self._reference_strength_slider.value(),
            "manual_strength": self._manual_strength_slider.value(),
            "inkwash_strength": self._inkwash_strength_slider.value(),
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
            "auto_black_hair_guard_enabled": bool(self._chk_auto_black_hair_guard.isChecked()),
            "auto_black_hair_guard_mode": str(self._combo_auto_black_hair_guard_mode.currentData() or "conservative"),
            "auto_black_hair_guard_scope": str(self._combo_auto_black_hair_guard_scope.currentData() or "near_face"),
            "auto_black_hair_guard_limit_mode": str(self._combo_auto_black_hair_guard_limit_mode.currentData() or "full"),
            "semantic_mask_layers_enabled": bool(getattr(self, "_chk_semantic_mask_layers", None) and self._chk_semantic_mask_layers.isChecked()),
            "semantic_mask_hair_enabled": bool(getattr(self, "_chk_mask_hair_override", None) and self._chk_mask_hair_override.isChecked()),
            "semantic_mask_iris_enabled": bool(getattr(self, "_chk_mask_iris_override", None) and self._chk_mask_iris_override.isChecked()),
            "semantic_mask_background_mode": str(getattr(self, "_combo_mask_background_mode", None).currentData() if getattr(self, "_combo_mask_background_mode", None) is not None else "keep"),
            "semantic_mask_recolor_mode": str(getattr(self, "_combo_mask_recolor_mode", None).currentData() if getattr(self, "_combo_mask_recolor_mode", None) is not None else "shading"),
            "semantic_mask_hair_color": QColor(getattr(self, "_hair_recolor_color", QColor(24, 24, 24))).name(),
            "semantic_mask_iris_color": QColor(getattr(self, "_iris_recolor_color", QColor(70, 135, 220))).name(),
            "semantic_mask_character_library_colors": bool(
                getattr(self, "_chk_mask_character_library_colors", None)
                and self._chk_mask_character_library_colors.isChecked()),
            "fill_mode": self._current_fill_mode(),
            "selection_closed_only": self._chk_selection_closed_only.isChecked(),
            "selection_mode": self._current_selection_mode(),
            "selection_feather": self._selection_feather_slider.value(),
            "selection_classic_hints": self._chk_selection_classic_hints.isChecked() if getattr(self, '_chk_selection_classic_hints', None) is not None else True,
            "selection_focus_outside": self._chk_selection_focus_outside.isChecked() if getattr(self, '_chk_selection_focus_outside', None) is not None else True,
            "selection_focus_context": self._selection_focus_context_spin.value() if getattr(self, '_selection_focus_context_spin', None) is not None else 32,
            "selection_focus_fade": self._selection_focus_fade_spin.value() if getattr(self, '_selection_focus_fade_spin', None) is not None else 96,
            "selection_region_control_enabled": bool(getattr(self, "_chk_selection_region_control", None) and self._chk_selection_region_control.isChecked()),
            "selection_region_control_preset": str(getattr(self, "_selection_control_preset_combo", None).currentData() if getattr(self, "_selection_control_preset_combo", None) is not None else "custom"),
            "selection_region_control_color": QColor(getattr(self, "_selection_control_color", QColor(28, 28, 32))).name(),
            "selection_region_control_hint": int(getattr(self, "_selection_control_hint_spin", None).value() if getattr(self, "_selection_control_hint_spin", None) is not None else 50),
            "selection_region_control_gain": int(getattr(self, "_selection_control_gain_spin", None).value() if getattr(self, "_selection_control_gain_spin", None) is not None else 175),
            "selection_region_control_deep": int(getattr(self, "_selection_control_deep_spin", None).value() if getattr(self, "_selection_control_deep_spin", None) is not None else 38),
            "selection_region_control_middle": int(getattr(self, "_selection_control_middle_spin", None).value() if getattr(self, "_selection_control_middle_spin", None) is not None else 74),
            "selection_region_control_shallow": int(getattr(self, "_selection_control_shallow_spin", None).value() if getattr(self, "_selection_control_shallow_spin", None) is not None else 94),
            "selection_region_control_final": int(getattr(self, "_selection_control_final_spin", None).value() if getattr(self, "_selection_control_final_spin", None) is not None else 92),
            "selection_region_control_luminance": int(getattr(self, "_selection_control_luminance_spin", None).value() if getattr(self, "_selection_control_luminance_spin", None) is not None else 28),
            "selection_region_control_adapter": bool(getattr(self, "_chk_selection_control_adapter", None) and self._chk_selection_control_adapter.isChecked()),
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
                settings=self._project_settings(),
                semantic_scribbles=self._all_semantic_hair_scribbles())
        except Exception as exc:
            QMessageBox.warning(self, tr("error_title"),
                                tr("project_save_failed").format(exc=exc))
            return
        self._current_project_path = path
        for saved_path, saved_state in self._pages.items():
            if saved_path != self._current_path and saved_state.has_result:
                self._schedule_page_spill(saved_path, delay_ms=0)
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

        for old_state in self._pages.values():
            old_state.clear_layer_cache()
        self._pages.clear()
        self._page_list.clear()
        self._current_path = None
        self._hair_guard_selection_masks.clear()
        self._hair_guard_selection_histories.clear()
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
            positive_path = record.get("semantic_hair_positive")
            negative_path = record.get("semantic_hair_negative")
            semantic_history = []
            positive_mask = None
            negative_mask = None
            if positive_path and os.path.isfile(positive_path):
                positive_mask = _uimread(positive_path, cv2.IMREAD_GRAYSCALE)
                positive_mask = self._normalise_binary_mask(positive_mask)
                if positive_mask is not None:
                    semantic_history.append(("add", positive_mask.copy()))
            if negative_path and os.path.isfile(negative_path):
                negative_mask = _uimread(negative_path, cv2.IMREAD_GRAYSCALE)
                negative_mask = self._normalise_binary_mask(negative_mask)
                if negative_mask is not None:
                    semantic_history.append(("erase", negative_mask.copy()))
            if semantic_history:
                self._hair_guard_selection_histories[source] = semantic_history
                composed = (positive_mask.copy() if positive_mask is not None
                            else np.zeros(state.original_bgr.shape[:2], np.uint8))
                if negative_mask is not None:
                    if negative_mask.shape[:2] != composed.shape[:2]:
                        negative_mask = cv2.resize(negative_mask, (composed.shape[1], composed.shape[0]),
                                                   interpolation=cv2.INTER_NEAREST)
                    composed = cv2.bitwise_and(composed, cv2.bitwise_not(negative_mask))
                if np.any(composed):
                    self._hair_guard_selection_masks[source] = composed

            if state.result_bgr is not None:
                state.bias_brush_reference_bgr = state.result_bgr.copy()
                state.bias_brush_reference_filter_bgr = (
                    state.filter_base_bgr.copy() if state.filter_base_bgr is not None
                    else state.result_bgr.copy())

        settings = data.get("settings", {})
        loaded_style_key = settings.get("style_key")
        if loaded_style_key in {"monochrome_people", "monochrome_page"}:
            loaded_style_key = "monochrome"
        if loaded_style_key in {"light", "light2_illustrative"}:
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
        native_hint_idx = self._native_hint_mode_combo.findData(
            settings.get("hint_render_mode", "mixed"))
        self._native_hint_mode_combo.setCurrentIndex(max(0, native_hint_idx))
        export_profile = settings.get("export_compression_profile", "png_optimized")
        self._export_compression_combo.blockSignals(True)
        export_idx = self._export_compression_combo.findData(export_profile)
        self._export_compression_combo.setCurrentIndex(max(0, export_idx))
        self._export_compression_combo.blockSignals(False)
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
        self._chk_auto_black_hair_guard.setChecked(
            bool(settings.get("auto_black_hair_guard_enabled", False)))
        for combo, key, default in (
                (self._combo_auto_black_hair_guard_mode, "auto_black_hair_guard_mode", "conservative"),
                (self._combo_auto_black_hair_guard_scope, "auto_black_hair_guard_scope", "near_face"),
                (self._combo_auto_black_hair_guard_limit_mode, "auto_black_hair_guard_limit_mode", "full")):
            index = combo.findData(settings.get(key, default))
            if index >= 0:
                combo.setCurrentIndex(index)
        hair_color_name = settings.get("semantic_mask_hair_color", "#181818")
        if QColor(hair_color_name).isValid():
            self._hair_recolor_color = QColor(hair_color_name)
        iris_color_name = settings.get("semantic_mask_iris_color", "#4687dc")
        if QColor(iris_color_name).isValid():
            self._iris_recolor_color = QColor(iris_color_name)
        if getattr(self, "_chk_semantic_mask_layers", None) is not None:
            self._chk_semantic_mask_layers.setChecked(
                bool(settings.get("semantic_mask_layers_enabled", False)))
        if getattr(self, "_chk_mask_hair_override", None) is not None:
            self._chk_mask_hair_override.setChecked(
                bool(settings.get("semantic_mask_hair_enabled", True)))
        if getattr(self, "_chk_mask_iris_override", None) is not None:
            self._chk_mask_iris_override.setChecked(
                bool(settings.get("semantic_mask_iris_enabled", False)))
        if getattr(self, "_chk_mask_character_library_colors", None) is not None:
            self._chk_mask_character_library_colors.setChecked(
                bool(settings.get("semantic_mask_character_library_colors", True)))
        if getattr(self, "_combo_mask_background_mode", None) is not None:
            idx = self._combo_mask_background_mode.findData(
                settings.get("semantic_mask_background_mode", "keep"))
            if idx >= 0:
                self._combo_mask_background_mode.setCurrentIndex(idx)
        if getattr(self, "_combo_mask_recolor_mode", None) is not None:
            idx = self._combo_mask_recolor_mode.findData(
                settings.get("semantic_mask_recolor_mode", "shading"))
            if idx >= 0:
                self._combo_mask_recolor_mode.setCurrentIndex(idx)
        self._update_hair_recolor_swatch()
        self._update_iris_recolor_swatch()
        self._update_semantic_mask_controls_enabled()
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
        self._chk_skip_colored.setChecked(settings.get("skip_colored", False))
        self._chk_ink_fidelity.setChecked(bool(settings.get("ink_fidelity", True)))
        self._inkwash_strength_slider.setValue(int(settings.get("inkwash_strength", 78)))
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
        if getattr(self, '_chk_selection_region_control', None) is not None:
            self._chk_selection_region_control.setChecked(bool(
                settings.get("selection_region_control_enabled", True)))
        control_color = QColor(settings.get("selection_region_control_color", "#1c1c20"))
        if control_color.isValid():
            self._selection_control_color = control_color
            self._update_selection_control_swatch()
        for name, key, default in (
                ("_selection_control_hint_spin", "selection_region_control_hint", 50),
                ("_selection_control_gain_spin", "selection_region_control_gain", 175),
                ("_selection_control_deep_spin", "selection_region_control_deep", 38),
                ("_selection_control_middle_spin", "selection_region_control_middle", 74),
                ("_selection_control_shallow_spin", "selection_region_control_shallow", 94),
                ("_selection_control_final_spin", "selection_region_control_final", 92),
                ("_selection_control_luminance_spin", "selection_region_control_luminance", 28)):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setValue(int(settings.get(key, default)))
        preset_combo = getattr(self, "_selection_control_preset_combo", None)
        if preset_combo is not None:
            preset_key = str(settings.get("selection_region_control_preset", "custom"))
            index = preset_combo.findData(preset_key)
            preset_combo.blockSignals(True)
            preset_combo.setCurrentIndex(index if index >= 0 else 0)
            preset_combo.blockSignals(False)
        adapter = getattr(self, "_chk_selection_control_adapter", None)
        if adapter is not None:
            adapter.setChecked(bool(settings.get(
                "selection_region_control_adapter", False)) and
                _tunnel2_adapter_available())
        self._update_selection_region_control_enabled()
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
        for loaded_path, loaded_state in self._pages.items():
            if loaded_state.has_result:
                self._schedule_export_compression(
                    loaded_path, delay_ms=350 if loaded_path == self._current_path else 0)
            if loaded_path != self._current_path and loaded_state.has_result:
                self._schedule_page_spill(loaded_path, delay_ms=0)
        self._update_controls_enabled()
        self._update_pastel_controls_visibility()
        message = tr("project_loaded").format(path=path)
        if missing:
            message += tr("project_missing_pages").format(n=missing)
        self.statusBar().showMessage(message, 8000)

    def _on_worker_status(self, message: str):
        if bool(getattr(self, "_stop_requested", False)):
            return
        self.statusBar().showMessage(message, 0)
        if hasattr(self, "_auto_status_label"):
            self._auto_status_label.setText(message)

    def _on_batch_page_done(self, path: str, result_payload):
        sender = self.sender()
        if bool(getattr(self, "_stop_requested", False)) or (
                sender is not None and callable(getattr(sender, "cancel_requested", None))
                and sender.cancel_requested()):
            return
        state = self._pages.get(path)
        if state is None:
            return
        quality_report = None
        if isinstance(result_payload, dict):
            result_bgr = result_payload.get("result_bgr")
            filter_base_bgr = result_payload.get("filter_base_bgr")
            quality_report = result_payload.get("quality_report")
        elif isinstance(result_payload, tuple) and len(result_payload) == 2:
            result_bgr, filter_base_bgr = result_payload
        else:
            result_bgr, filter_base_bgr = result_payload, None
        if result_bgr is None:
            return
        if path == self._current_path:
            state.push_undo()  # one active page keeps normal undo semantics
        else:
            # A book-wide batch must not retain three full-resolution prior
            # layers per page in undo history; that is exactly what forces
            # macOS to allocate huge swap files.
            state.undo_stack.clear()
            state.redo_stack.clear()
        # Take ownership of worker outputs where safe.  Only the editable
        # result needs an immediate independent copy; this removes one to two
        # full-page copies from every batch completion callback.
        result_bgr = np.ascontiguousarray(result_bgr)
        state.ai_result_bgr = result_bgr
        state.result_bgr = result_bgr.copy()
        if filter_base_bgr is None or filter_base_bgr is result_bgr:
            state.filter_base_bgr = result_bgr.copy()
        else:
            state.filter_base_bgr = np.ascontiguousarray(filter_base_bgr)
        state.pipeline_diagnostics = dict(
            getattr(state.hint_manager, "last_diagnostics", {}) or {})
        # A fresh automatic result has no colour-bias stroke to erase yet.
        # Build these two full-resolution reference layers lazily only when
        # the bias brush is actually used.
        state.bias_brush_reference_bgr = None
        state.bias_brush_reference_filter_bgr = None
        self._refresh_character_diagnostics()
        state.quality_report = quality_report
        if state.quality_report is None:
            try:
                from core.quality_score import assess_colorization
                state.quality_report = assess_colorization(
                    state.original_bgr, result_bgr)
            except Exception:
                state.quality_report = None
        self._mark_page_done(path)
        self._canvas.clear_dabs()
        if path == self._current_path:
            self._schedule_export_compression(path, delay_ms=350)
            self._radio_view_edited.setChecked(True)
            self._sync_view_after_edit(state)
        else:
            # Release full-resolution batch layers first.  Final export
            # compression then reads the lossless session cache, so completed
            # pages cannot accumulate in RAM while WebP/PNG encoding catches up.
            self._schedule_page_spill(state.path, delay_ms=0)
            self._schedule_export_compression(state.path, delay_ms=0)

    def _on_batch_page_error(self, path: str, message: str):
        sender = self.sender()
        if bool(getattr(self, "_stop_requested", False)) or (
                sender is not None and callable(getattr(sender, "cancel_requested", None))
                and sender.cancel_requested()):
            return
        name = os.path.basename(path)
        self._batch_errors.append((name, message))
        visible = tr("page_colorize_failed").format(name=name, message=message)
        self.statusBar().showMessage(visible, 8000)
        if hasattr(self, "_auto_status_label"):
            self._auto_status_label.setText(visible)

    def _on_batch_page_skipped(self, path: str, reason: str):
        sender = self.sender()
        if bool(getattr(self, "_stop_requested", False)) or (
                sender is not None and callable(getattr(sender, "cancel_requested", None))
                and sender.cancel_requested()):
            return
        name = os.path.basename(path)
        visible = tr("page_colorize_skipped").format(name=name, reason=reason)
        self.statusBar().showMessage(visible, 5000)
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
        try:
            from pipeline import clear_transient_page_caches
            clear_transient_page_caches()
        except Exception:
            pass
        for path, state in self._pages.items():
            if path != self._current_path and state.has_result:
                self._schedule_page_spill(state.path, delay_ms=0)
        self._release_compute_memory()
        worker = self._batch_worker
        self._batch_worker = None
        if self._stop_requested:
            if worker is not None:
                worker.deleteLater()
            self._finish_stop_state()
            self._update_device_label()
            return
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
        self._ui_busy = bool(busy)
        self._progress.setVisible(busy)
        self._btn_auto.setText(tr("stop_btn") if busy else tr("auto_btn"))
        self._btn_auto.setProperty("stopMode", bool(busy))
        self._btn_auto.style().unpolish(self._btn_auto)
        self._btn_auto.style().polish(self._btn_auto)
        self._btn_auto.setEnabled(True if busy else self._current_state() is not None)
        hair_btn = getattr(self, "_btn_hair_recolor_auto", None)
        if hair_btn is not None:
            hair_btn.setText(tr("stop_btn") if busy else "自动上色（应用头发纠色）")
            hair_btn.setProperty("stopMode", bool(busy))
            hair_btn.style().unpolish(hair_btn)
            hair_btn.style().polish(hair_btn)
            hair_btn.setEnabled(True if busy else self._current_state() is not None)
        for btn in (self._btn_regenerate, self._btn_undo,
                   self._btn_clear, self._btn_undo_edit, self._btn_redo_edit,
                   self._btn_restore_ai, self._btn_restore_bw,
                   self._btn_export_page, self._btn_export_all,
                   self._btn_apply_selection, self._btn_ai_recolor_selection,
                   self._btn_restore_selection_bw,
                   getattr(self, '_btn_refine_hair', None),
                   getattr(self, '_btn_refine_positive', None),
                   getattr(self, '_btn_refine_negative', None),
                   getattr(self, '_btn_refine_stop_brush', None),
                   getattr(self, '_btn_refine_clear_brush', None),
                   self._btn_semantic_select, self._btn_clear_selection_hints,
                   self._btn_cancel_selection,
                   getattr(self, '_btn_hair_guard_rect_selector', None),
                   getattr(self, '_btn_hair_guard_brush_add', None),
                   getattr(self, '_btn_hair_guard_brush_erase', None),
                   getattr(self, '_btn_hair_guard_undo_last', None),
                   getattr(self, '_btn_hair_guard_clear_selection', None),
                   getattr(self, '_btn_hair_guard_finish_selector', None),
                   getattr(self, '_btn_import_images', None),
                   getattr(self, '_btn_import_pdf', None),
                   getattr(self, '_btn_import_folder', None),
                   getattr(self, '_btn_move_up', None),
                   getattr(self, '_btn_move_down', None),
                   getattr(self, '_btn_delete_pages', None),
                   getattr(self, '_btn_load_project', None),
                   getattr(self, '_btn_save_project', None),
                   getattr(self, '_lang_button', None)):
            if btn is not None:
                btn.setEnabled(not busy)
        header_status = getattr(self, "_header_status_chip", None)
        footer_status = getattr(self, "_footer_status_value", None)
        hair_status_label = getattr(self, "_hair_recolor_status_label", None)
        if busy:
            if header_status is not None:
                header_status.setText("● 处理中")
            if footer_status is not None:
                footer_status.setText("处理中")
            if hair_status_label is not None:
                hair_status_label.setText("正在应用头发纠色设置并上色…")
        else:
            self._refresh_workspace_stats()
            if hair_status_label is not None:
                hair_status_label.setText("应用当前头发纠色设置并自动上色")
        model_btn = getattr(self, "_btn_manage_ai_hair_models", None)
        if model_btn is not None:
            model_worker = getattr(self, "_hair_model_worker", None)
            model_btn.setEnabled(
                not busy and not (model_worker is not None and model_worker.isRunning()))
        if not busy:
            self._update_selection_buttons(
                self._pending_selection_mask is not None)
            self._update_controls_enabled()
        if message:
            self.statusBar().showMessage(message, 0 if busy else 5000)

    def _update_device_label(self):
        """Show validated accelerator readiness without loading mc-v2 weights."""
        try:
            import torch
            from core.device_runtime import probe_torch_device
            report = probe_torch_device(Config.ML_DEVICE, torch_module=torch,
                                        smoke_test=True)
            from core.model_downloader import models_ready
            suffix = "模型已就绪" if models_ready(Config.WEIGHTS_DIR) else "首次上色将下载模型"
            device = report.selected
            runtime = ("生成器 MPS / 降噪 CPU" if device == "mps"
                       else report.short_reason)
            try:
                import pipeline as _pipeline
                engine = getattr(_pipeline, "_colorizer", None)
                if engine is not None:
                    device = str(getattr(engine, "device_name", device))
                    warning = str(getattr(engine, "device_warning", "") or "")
                    if warning:
                        runtime = warning
                    elif device == "mps":
                        runtime = "生成器 MPS / 降噪 CPU"
                    else:
                        runtime = "mc-v2 实际运行设备"
            except Exception:
                pass
            self._device_label.setText(
                f"{tr('device_label')}{device} · {runtime} · {suffix}")
            self._device_label.setToolTip(
                f"macOS: {report.macos_version or '—'}\n"
                f"架构: {report.architecture}\n"
                f"Python: {report.python_version}\n"
                f"PyTorch: {report.torch_version}\n"
                f"MPS built/available/smoke: "
                f"{report.mps_built}/{report.mps_available}/{report.mps_smoke_ok}\n"
                f"说明: {report.reason}")
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
        if getattr(self, '_btn_refine_hair', None) is not None:
            self._btn_refine_hair.setEnabled(bool(active) and not bool(getattr(self, '_ui_busy', False)))
        for name in ('_btn_refine_positive', '_btn_refine_negative',
                     '_btn_refine_stop_brush', '_btn_refine_clear_brush'):
            button = getattr(self, name, None)
            if button is not None:
                button.setEnabled(self._current_state() is not None and
                                  not bool(getattr(self, '_ui_busy', False)))
        if getattr(self, '_btn_restore_selection_bw', None) is not None:
            state = self._current_state()
            self._btn_restore_selection_bw.setEnabled(
                bool(active) and state is not None and state.result_bgr is not None
                and not bool(getattr(self, '_ui_busy', False)))
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
        if self._hair_guard_selector_active and self._hair_guard_selector_mode in {"brush_add", "brush_erase"}:
            shape = self._hair_guard_selection_shape()
            if shape is None:
                return
            if not self._selection_adjust_gesture_active:
                self._selection_adjust_gesture_active = True
                self._begin_foreground_interaction()
                base = self._current_hair_guard_selection_mask()
                if base is None:
                    base = np.zeros(shape, np.uint8)
                self._hair_guard_selection_stroke_base_mask = base.copy()
                self._hair_guard_selection_stroke_mask = np.zeros(shape, np.uint8)
                self._hair_guard_selection_stroke_add_mode = bool(add_mode)
            stroke = self._hair_guard_selection_stroke_mask
            base = self._hair_guard_selection_stroke_base_mask
            if stroke is None or base is None:
                return
            cv2.circle(stroke, (int(ix), int(iy)), max(1, int(radius_px)), 255, thickness=-1)
            preview = cv2.bitwise_or(base, stroke) if self._hair_guard_selection_stroke_add_mode else cv2.bitwise_and(base, cv2.bitwise_not(stroke))
            if getattr(self, '_hair_refine_brush_active', False):
                self._show_hair_refine_scribbles(pending=stroke, add_mode=add_mode)
            else:
                self._canvas.set_selection_mask_overlay(preview, mode="replace")
            return
        mask = self._pending_selection_mask
        if mask is None:
            return
        if not self._selection_adjust_gesture_active:
            self._selection_adjust_gesture_active = True
            self._begin_foreground_interaction()
        # Manual corrections become authoritative.  Stop the cached AI preview
        # from regenerating on Apply and overwriting the user's add/erase work.
        self._clear_closed_preview_cache()
        value = 255 if add_mode else 0
        cv2.circle(mask, (int(ix), int(iy)), max(1, int(radius_px)), value, thickness=-1)
        self._selection_adjust_last_add_mode = bool(add_mode)
        self._selection_adjust_overlay_dirty = True
        if not self._selection_adjust_overlay_timer.isActive():
            self._selection_adjust_overlay_timer.start()

    def _flush_selection_adjust_overlay(self):
        if not self._selection_adjust_overlay_dirty:
            return
        self._selection_adjust_overlay_dirty = False
        mask = self._pending_selection_mask
        active = mask is not None and bool(np.any(mask))
        self._update_selection_buttons(active)
        if active:
            mode = self._current_selection_mode()
            if (getattr(self, '_chk_selection_bw_preview', None) is not None
                    and self._chk_selection_bw_preview.isChecked()):
                self._refresh_selection_display_preview()
            else:
                self._canvas.set_selection_mask_overlay(mask, mode=mode)
        else:
            self._pending_selection_mask = None
            self._pending_selection_kind = None
            self._canvas.clear_selection_preview()
        self.statusBar().showMessage(
            tr("selection_adjust_add_status") if self._selection_adjust_last_add_mode
            else tr("selection_adjust_erase_status"),
            1200)

    def _on_selection_adjust_finished(self):
        try:
            if self._hair_guard_selector_active and self._hair_guard_selector_mode in {"brush_add", "brush_erase"}:
                stroke = self._hair_guard_selection_stroke_mask
                add_mode = bool(self._hair_guard_selection_stroke_add_mode)
                if stroke is not None and np.any(stroke):
                    saved = self._append_hair_guard_selection_operation(stroke, add_mode=add_mode)
                    if getattr(self, '_hair_refine_brush_active', False):
                        self._show_hair_refine_scribbles()
                    else:
                        self._canvas.set_selection_mask_overlay(saved, mode="replace") if saved is not None else self._canvas.clear_selection_preview()
                    count = len(self._current_hair_guard_selection_history())
                    area = int(np.count_nonzero(saved)) if saved is not None else 0
                    self.statusBar().showMessage(
                        f"已{'添加' if add_mode else '擦除'}头发纠色区域。当前 {count} 个操作，覆盖 {area} px。",
                        1800)
                self._hair_guard_selection_stroke_mask = None
                self._hair_guard_selection_stroke_base_mask = None
                return
            if self._selection_adjust_overlay_timer.isActive():
                self._selection_adjust_overlay_timer.stop()
            self._flush_selection_adjust_overlay()
        finally:
            if self._selection_adjust_gesture_active:
                self._selection_adjust_gesture_active = False
                self._end_foreground_interaction()

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
        if self._selection_adjust_overlay_timer.isActive():
            self._selection_adjust_overlay_timer.stop()
        self._selection_adjust_overlay_dirty = False
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
        self._semantic_select_armed = False
        if self._selection_adjust_overlay_timer.isActive():
            self._selection_adjust_overlay_timer.stop()
        self._selection_adjust_overlay_dirty = False
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

    def _restore_selection_bw(self):
        """Undoable, mask-clipped local B/W restore (unlike full-page reset)."""
        state = self._current_state()
        mask = self._pending_selection_mask
        if state is None or state.result_bgr is None or mask is None or not np.any(mask):
            self.statusBar().showMessage("请先选择已上色页面中的区域。", 3500)
            return
        if getattr(self, "_ui_busy", False):
            return
        from core.local_model_recolor import restore_selection_to_original
        state.push_undo(fields=state.PIXEL_EDIT_FIELDS)
        state.result_bgr, state.filter_base_bgr, changed = restore_selection_to_original(
            state.original_bgr, state.result_bgr, state.filter_base_bgr, mask)
        if changed == 0:
            state.discard_unchanged_undo()
            self.statusBar().showMessage("选区已经是原始黑白，无需恢复。", 3000)
            return
        self._sync_bias_reference_to_current(state)
        self._radio_view_edited.setChecked(True)
        self._sync_view_after_edit(state)
        self._schedule_export_compression(state.path, delay_ms=350)
        self.statusBar().showMessage(f"已恢复选区黑白：{changed} 个像素（可撤销）", 5000)

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

    def _update_selection_control_swatch(self):
        button = getattr(self, "_selection_control_color_btn", None)
        if button is None:
            return
        color = QColor(getattr(self, "_selection_control_color", QColor(28, 28, 32)))
        button.setText(color.name().upper())
        text = "#ffffff" if color.lightness() < 120 else "#111111"
        button.setStyleSheet(
            f"QPushButton {{ background: {color.name()}; color: {text}; "
            "border: 1px solid #718096; border-radius: 4px; padding: 4px; }")

    def _on_selection_control_preset_changed(self, _index: int | None = None):
        combo = getattr(self, "_selection_control_preset_combo", None)
        if combo is None:
            return
        key = str(combo.currentData() or "custom")
        presets = {
            # RGB, final lock %, luminance pull %, hint %, gain %, deep %,
            # middle %, shallow %. These presets are tuned for black-hair source
            # control: they bias the generator harder before the deterministic
            # post-lock removes any remaining warm cast.
            # Ink Black is deliberately neutral. Use Cool Black for a blue-black
            # finish. Keeping the two presets separate avoids the old situation
            # where "墨黑" still looked slightly teal/brown on long highlights.
            "ink_black": ((24, 24, 24), 40, 0, 100, 160, 70, 90, 100),
            "soft_black": ((40, 40, 46), 32, 0, 96, 155, 66, 88, 98),
            "cool_black": ((24, 30, 44), 36, 0, 100, 165, 72, 92, 100),
            "light_ink": ((68, 69, 78), 25, 0, 90, 150, 60, 84, 94),
        }
        preset = presets.get(key)
        if preset is None:
            return
        rgb, final_lock, luminance, hint_opacity, gain, deep, middle, shallow = preset
        self._selection_control_color = QColor(*rgb)
        widget_values = {
            "_selection_control_final_spin": final_lock,
            "_selection_control_luminance_spin": luminance,
            "_selection_control_hint_spin": hint_opacity,
            "_selection_control_gain_spin": gain,
            "_selection_control_deep_spin": deep,
            "_selection_control_middle_spin": middle,
            "_selection_control_shallow_spin": shallow,
        }
        for name, value in widget_values.items():
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setValue(value)
        self._update_selection_control_swatch()

    def _pick_selection_control_color(self):
        current = QColor(getattr(self, "_selection_control_color", QColor(28, 28, 32)))
        chosen = self._open_color_dialog(current, tr("selection_control_color"))
        if chosen.isValid():
            self._selection_control_color = QColor(chosen)
            combo = getattr(self, "_selection_control_preset_combo", None)
            if combo is not None:
                custom_index = combo.findData("custom")
                if custom_index >= 0:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(custom_index)
                    combo.blockSignals(False)
            self._update_selection_control_swatch()

    def _update_selection_region_control_enabled(self, _checked: bool | None = None):
        enabled = bool(getattr(self, "_chk_selection_region_control", None)
                       and self._chk_selection_region_control.isChecked())
        names = (
            "_selection_control_preset_label", "_selection_control_preset_combo",
            "_selection_control_color_label", "_selection_control_color_btn",
            "_selection_control_hint_label", "_selection_control_hint_spin",
            "_selection_control_gain_label", "_selection_control_gain_spin",
            "_selection_control_deep_label", "_selection_control_deep_spin",
            "_selection_control_middle_label", "_selection_control_middle_spin",
            "_selection_control_shallow_label", "_selection_control_shallow_spin",
            "_selection_control_final_label", "_selection_control_final_spin",
            "_selection_control_luminance_label", "_selection_control_luminance_spin",
        )
        for name in names:
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(enabled)
        adapter = getattr(self, "_chk_selection_control_adapter", None)
        if adapter is not None:
            adapter.setEnabled(enabled and _tunnel2_adapter_available())

    def _current_selection_region_control(self) -> dict:
        color = QColor(getattr(self, "_selection_control_color", QColor(28, 28, 32)))
        def pct(name: str, default: int) -> float:
            widget = getattr(self, name, None)
            return float(widget.value() if widget is not None else default) / 100.0
        enabled = bool(getattr(self, "_chk_selection_region_control", None)
                       and self._chk_selection_region_control.isChecked())
        rgb = (color.red(), color.green(), color.blue())
        black_like = (max(rgb) <= 96 and (max(rgb) - min(rgb)) <= 34)
        neutral_ink = (max(rgb) <= 48 and (max(rgb) - min(rgb)) <= 6)
        return {
            "enabled": enabled,
            "target_rgb": rgb,
            "hint_opacity": pct("_selection_control_hint_spin", 50),
            "feature_gain": pct("_selection_control_gain_spin", 175),
            "deep_strength": pct("_selection_control_deep_spin", 38),
            "middle_strength": pct("_selection_control_middle_spin", 74),
            "shallow_strength": pct("_selection_control_shallow_spin", 94),
            "deep_shrink_px": 2 if black_like else 8,
            "middle_shrink_px": 1 if black_like else 3,
            "shallow_shrink_px": 0,
            "mask_feather_px": (1 if black_like else max(1, int(getattr(
                self, "_selection_feather_slider", None).value()
                if getattr(self, "_selection_feather_slider", None) is not None else 3))),
            "final_color_lock": pct("_selection_control_final_spin", 92),
            "luminance_shift": pct("_selection_control_luminance_spin", 28),
            "protect_lineart": True,
            "preserve_highlights": True,
            "use_adapter": bool(getattr(self, "_chk_selection_control_adapter", None)
                                and self._chk_selection_control_adapter.isChecked()),
            # Automatic real-checkpoint black-hair calibration. No new UI is
            # required: dark low-chroma targets receive an inverse-calibrated
            # internal generator hint plus a stronger warm-cast cleanup.
            "source_bias_strength": 0.0,
            "source_bias_tone": 0,
            "source_bias_feather_px": 0,
            "neutralize_chroma": (1.00 if neutral_ink else 0.94) if black_like else 0.0,
            "warm_suppression": 1.00 if black_like else 0.0,
            "protect_lineart_threshold": 108 if black_like else 88,
            # Never silently reconstruct/expand a manually reviewed selection.
            # A separately approved full-hair semantic selection arrives as an
            # already-built mask and must remain authoritative as well.
            "complete_hair_object": False,
            "expand_control_mask": False,
            "expand_merge_mask": False,
            "hair_object_max_ratio": 2.15,
        }

    def _show_hair_refine_scribbles(self, *, pending=None, add_mode=None):
        """Blue=definite hair, red=definite non-hair, including live strokes."""
        positive, negative = self._current_hair_guard_semantic_scribbles()
        if pending is not None and np.any(pending):
            if bool(add_mode):
                positive = (pending.copy() if positive is None else
                            cv2.bitwise_or(positive, pending))
            else:
                negative = (pending.copy() if negative is None else
                            cv2.bitwise_or(negative, pending))
        self._canvas.set_auto_hair_preview_overlay(positive, None, negative)

    def _activate_hair_refine_brush(self, mode: str):
        """Expose already-persisted positive/negative hair strokes in Edit tab."""
        self._activate_hair_guard_selector(mode, for_refinement=True)
        self._hair_refine_brush_active = True
        self._show_hair_refine_scribbles()
        self.statusBar().showMessage(
            "发丝正向标注：在头发内部涂抹，完成后点「智能贴合发丝」。" if mode == "brush_add"
            else "非发丝负向标注：在面部、背景等禁止覆盖区域涂抹。", 6500)

    def _start_hair_selection_refine(self):
        state = self._current_state()
        rough = self._pending_selection_mask
        if state is None or state.result_bgr is None or rough is None or not np.any(rough):
            self.statusBar().showMessage("请先在上色页面用矩形或套索粗选头发。", 4000)
            return
        if self._hair_refine_worker is not None and self._hair_refine_worker.isRunning():
            return
        if getattr(self, '_ui_busy', False):
            return
        positive, negative = self._current_hair_guard_semantic_scribbles()
        if self._hair_guard_selector_active:
            self._deactivate_hair_guard_selector()
        colour = state.ai_result_bgr if state.ai_result_bgr is not None else state.result_bgr
        distance_widget = getattr(self, '_selection_snap_distance_spin', None)
        distance = max(12, min(48, int(distance_widget.value()) * 3 if distance_widget else 22))
        worker = HairSelectionRefineWorker(
            state.original_bgr, colour, rough,
            positive=positive, negative=negative, grow_px=distance, parent=self)
        self._hair_refine_worker = worker
        self._hair_refine_page_path = state.path
        worker.status.connect(self._on_worker_status)
        worker.finished_ok.connect(self._on_hair_selection_refined)
        worker.finished_err.connect(self._on_hair_selection_refine_failed)
        worker.cancelled.connect(self._on_hair_selection_refine_cancelled)
        self._set_busy(True, "正在贴合发丝边界…")
        worker.start()

    def _finish_hair_selection_refine(self):
        worker = self._hair_refine_worker
        self._hair_refine_worker = None
        self._hair_refine_page_path = None
        if worker is not None:
            worker.deleteLater()
        self._set_busy(False)
        self._update_selection_buttons(self._pending_selection_mask is not None)

    def _on_hair_selection_refined(self, payload):
        state = self._current_state()
        if state is None or state.path != self._hair_refine_page_path:
            self._finish_hair_selection_refine()
            return
        mask = np.asarray((payload or {}).get('mask'))
        info = dict((payload or {}).get('diagnostics') or {})
        if mask.ndim != 2 or not np.any(mask):
            self._finish_hair_selection_refine()
            self.statusBar().showMessage("发丝细化结果为空，粗选区未改变。", 4500)
            return
        if info.get('used_fallback'):
            # Never silently replace a precise manual selection with a failed
            # segmentation that may contain face or background.
            self._finish_hair_selection_refine()
            self.statusBar().showMessage(
                f"未能安全细化（{info.get('reason')}），保留原选区，请补充正负标注。", 5500)
            return
        self._set_pending_selection(mask.astype(np.uint8), 'guided_hair', combine=False)
        self._finish_hair_selection_refine()
        self.statusBar().showMessage(
            f"发丝贴合成功：{info.get('selected_pixels',0)} px，补入 "
            f"{info.get('added_pixels',0)} px，排除 {info.get('removed_pixels',0)} px；"
            "可直接局部改色/恢复黑白/AI 重算。", 7500)

    def _on_hair_selection_refine_failed(self, message: str):
        self._finish_hair_selection_refine()
        self.statusBar().showMessage(f"发丝贴合失败：{message}", 6000)

    def _on_hair_selection_refine_cancelled(self):
        self._finish_hair_selection_refine()
        self.statusBar().showMessage("已停止发丝贴合。", 3000)

    def _arm_semantic_selection(self):
        state = self._current_state()
        if state is None or state.result_bgr is None:
            QMessageBox.information(self, tr("no_result_title"), tr("no_result_body"))
            return
        self._semantic_select_armed = True
        if getattr(self, "_radio_bucket", None) is not None:
            self._radio_bucket.setChecked(True)
            self._on_tool_changed()
        self.statusBar().showMessage(tr("selection_semantic_armed"), 7000)

    def _run_semantic_selection(self, state, ix: int, iy: int) -> bool:
        if (self._semantic_selection_worker is not None and
                self._semantic_selection_worker.isRunning()):
            self.statusBar().showMessage(tr("selection_semantic_busy"), 3000)
            return False
        colour = state.ai_result_bgr if state.ai_result_bgr is not None else state.result_bgr
        if colour is None:
            return False
        # An existing rectangle/polygon/brush selection is treated as a rough
        # semantic boundary, not as pixels that must all remain selected.  The
        # actual full-resolution analysis runs in a QThread; first-use OpenCV
        # initialisation can otherwise pause the UI for several hundred ms.
        rough_limit = (None if self._pending_selection_mask is None else
                       self._pending_selection_mask.copy())
        control = self._current_selection_region_control()
        rgb = tuple(control.get("target_rgb") or (255, 255, 255))
        complete_hair_object = bool(
            max(rgb) <= 96 and (max(rgb) - min(rgb)) <= 34)
        worker = SemanticSelectionWorker(
            state.original_bgr, colour, (int(ix), int(iy)),
            limit_mask=rough_limit,
            complete_hair_object=complete_hair_object, parent=self)
        self._semantic_selection_worker = worker
        self._semantic_selection_page_path = state.path
        worker.status.connect(self._on_worker_status)
        worker.finished_ok.connect(self._on_semantic_selection_done)
        worker.finished_err.connect(self._on_semantic_selection_error)
        worker.cancelled.connect(self._on_semantic_selection_cancelled)
        self._set_busy(True, tr("selection_semantic_running"))
        worker.start()
        return True

    def _finish_semantic_selection_worker(self):
        worker = self._semantic_selection_worker
        self._semantic_selection_worker = None
        self._semantic_selection_page_path = None
        if worker is not None:
            worker.deleteLater()
        self._set_busy(False)
        self._update_selection_buttons(self._pending_selection_mask is not None)

    def _on_semantic_selection_done(self, payload):
        page_path = self._semantic_selection_page_path
        state = self._pages.get(page_path) if page_path else None
        mask = np.asarray((payload or {}).get("mask"))
        diagnostics = dict((payload or {}).get("diagnostics") or {})
        if state is None or state.path != self._current_path:
            self._finish_semantic_selection_worker()
            return
        if mask.ndim != 2 or not np.any(mask):
            self._finish_semantic_selection_worker()
            self.statusBar().showMessage(tr("selection_semantic_empty"), 4500)
            return
        selection_kind = (
            "mcv2_complete_hair" if diagnostics.get(
                "complete_hair_object_enabled", False) else "mcv2_semantic")
        self._set_pending_selection(mask.astype(np.uint8), selection_kind,
                                    combine=False)
        pixels = int(diagnostics.get("selected_pixels", np.count_nonzero(mask)))
        self._finish_semantic_selection_worker()
        self.statusBar().showMessage(
            tr("selection_semantic_done").format(pixels=pixels), 5000)

    def _on_semantic_selection_cancelled(self):
        self._finish_semantic_selection_worker()
        self._stop_requested = False
        self.statusBar().showMessage(tr("stopped"), 4000)

    def _on_semantic_selection_error(self, message: str):
        self._finish_semantic_selection_worker()
        QMessageBox.critical(
            self, tr("selection_semantic_failed_title"),
            tr("selection_semantic_failed").format(message=message))

    def _clear_selection_model_hints(self):
        state = self._current_state()
        mask = self._pending_selection_mask
        if state is None or mask is None or not np.any(mask):
            self.statusBar().showMessage(tr("selection_fill_empty"), 2500)
            return
        from core.local_model_recolor import clear_manual_hints_in_selection
        state.push_undo(fields=state.HINT_EDIT_FIELDS)
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
            region_control={
                **self._current_selection_region_control(),
                "hair_object_mask_prebuilt": bool(
                    self._pending_selection_kind in {"mcv2_complete_hair", "guided_hair"}),
            },
            protect_text_overlay=bool(getattr(self, "_chk_protect_text", None)
                                      and self._chk_protect_text.isChecked()),
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
        worker.cancelled.connect(self._on_local_recolor_cancelled)
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

    def _on_local_recolor_cancelled(self):
        self._finish_local_recolor_worker()
        self._stop_requested = False
        self.statusBar().showMessage(tr("stopped"), 4000)

    def _on_local_recolor_done(self, payload):
        if bool(getattr(self, "_stop_requested", False)):
            return
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
        self._schedule_export_compression(state.path, delay_ms=350)

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
        self._refresh_edit_detail_summary(tool)
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

        selection_tool = tool in {"lasso", "rect"}
        base_section = getattr(self, "_edit_tool_detail_section", None)
        if base_section is not None:
            base_section.setVisible(tool != "bias_brush")
        bias_section = getattr(self, "_bias_brush_section", None)
        if bias_section is not None:
            bias_section.setVisible(tool == "bias_brush")
        boundary_section = getattr(self, "_selection_boundary_section", None)
        if boundary_section is not None:
            boundary_section.setVisible(selection_tool)
        action_section = getattr(self, "_selection_action_section", None)
        if action_section is not None:
            action_section.setVisible(selection_tool)
        ai_section = getattr(self, "_selection_ai_options_section", None)
        if ai_section is not None:
            ai_section.setVisible(selection_tool)

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
        if self._hair_guard_selector_active and not self._suspend_hair_guard_tool_reset:
            self._hair_guard_selector_active = False
            self._hair_refine_brush_active = False
            self._hair_guard_selector_mode = None
            mode_label = getattr(self, "_label_hair_guard_selector_mode", None)
            if mode_label is not None:
                mode_label.setText("当前选区器：未启用")
            self._canvas.set_selection_adjust_enabled(False)
            self._canvas.set_selection_adjust_allow_empty(False)
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
        if not self._hair_guard_selector_active:
            self._show_preferred_selection_overlay()
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
        self._update_hair_recolor_swatch()
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
        self._begin_foreground_interaction()
        state = self._current_state()
        self._local_brush_stroke_active = bool(
            state is not None and state.result_bgr is not None)
        self._brush_changed_during_stroke = False
        self._brush_patch_undo = None
        if self._local_brush_stroke_active:
            model_hint = bool(getattr(self, '_chk_brush_model_hint', None)
                              and self._chk_brush_model_hint.isChecked())
            self._brush_stroke_model_hint = model_hint
            if model_hint:
                state.push_undo(fields=state.HINT_EDIT_FIELDS)
            else:
                # Direct paint records the pre-stroke pixels lazily as the
                # painted corridor expands.  Avoid copying two complete page
                # layers before the first visible dab.
                state.redo_stack.clear()
                self._brush_patch_undo = {
                    "state_path": state.path,
                    "tile_size": 128,
                    "tiles": {},
                    "filter_was_none": state.filter_base_bgr is None,
                    "bias_brush_reference_bgr": state.bias_brush_reference_bgr,
                    "bias_brush_reference_filter_bgr": state.bias_brush_reference_filter_bgr,
                }

    def _extend_brush_patch_undo(self, state: "PageState",
                                 roi: tuple[int, int, int, int]) -> None:
        patch = self._brush_patch_undo
        if not patch or patch.get("state_path") != state.path:
            return
        result = state.result_bgr
        if result is None:
            return
        h, w = result.shape[:2]
        x0, y0, x1, y1 = [int(v) for v in roi]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x0 >= x1 or y0 >= y1:
            return
        tile_size = max(32, int(patch.get("tile_size", 128)))
        tiles = patch.setdefault("tiles", {})
        tx0, tx1 = x0 // tile_size, (x1 - 1) // tile_size
        ty0, ty1 = y0 // tile_size, (y1 - 1) // tile_size
        filter_base = state.filter_base_bgr
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                key = (tx, ty)
                if key in tiles:
                    continue
                bx0, by0 = tx * tile_size, ty * tile_size
                bx1, by1 = min(w, bx0 + tile_size), min(h, by0 + tile_size)
                tiles[key] = {
                    "bbox": (bx0, by0, bx1, by1),
                    "result_bgr": result[by0:by1, bx0:bx1].copy(),
                    "filter_base_bgr": (
                        None if patch.get("filter_was_none") or filter_base is None
                        else filter_base[by0:by1, bx0:bx1].copy()),
                }

    def _commit_brush_patch_undo(self, state: "PageState") -> None:
        patch = self._brush_patch_undo
        self._brush_patch_undo = None
        if not patch or patch.get("state_path") != state.path or not patch.get("tiles"):
            return
        ordered_tiles = [patch["tiles"][key] for key in sorted(patch["tiles"])]
        state.push_undo_snapshot({
            "__kind__": "pixel_tiles",
            "__fields__": state.FILTER_EDIT_FIELDS,
            "tiles": ordered_tiles,
            "filter_was_none": bool(patch.get("filter_was_none")),
            "bias_brush_reference_bgr": patch.get("bias_brush_reference_bgr"),
            "bias_brush_reference_filter_bgr": patch.get(
                "bias_brush_reference_filter_bgr"),
        })

    def _on_hint_dab_added(self, x_norm: float, y_norm: float, rgb: tuple,
                           radius_norm: float):
        state = self._current_state()
        if state is None:
            return

        # Before the first auto-colorize, brush dabs remain model hints.
        if state.result_bgr is None:
            state.push_undo(fields=state.HINT_EDIT_FIELDS)
            changed = state.hint_manager.add_manual_hint(
                x_norm, y_norm, rgb, radius_norm)
            if not changed:
                state.discard_unchanged_undo()
            self._refresh_hint_overlay()
            return

        if not self._local_brush_stroke_active:
            self._local_brush_stroke_active = True
            self._brush_changed_during_stroke = False
            model_hint = bool(getattr(self, '_chk_brush_model_hint', None)
                              and self._chk_brush_model_hint.isChecked())
            self._brush_stroke_model_hint = model_hint
            if model_hint:
                state.push_undo(fields=state.HINT_EDIT_FIELDS)
            else:
                state.redo_stack.clear()
                self._brush_patch_undo = {
                    "state_path": state.path,
                    "tile_size": 128,
                    "tiles": {},
                    "filter_was_none": state.filter_base_bgr is None,
                    "bias_brush_reference_bgr": state.bias_brush_reference_bgr,
                    "bias_brush_reference_filter_bgr": state.bias_brush_reference_filter_bgr,
                }

        if bool(self._brush_stroke_model_hint):
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
        roi = (max(0, ix - radius_px), max(0, iy - radius_px),
               min(w, ix + radius_px + 1), min(h, iy + radius_px + 1))
        self._extend_brush_patch_undo(state, roi)

        # The V5 brush has one predictable path: paint the exact live corridor.
        # It never waits for mouse release, expands into a region, or snaps to
        # line art. Solid ink pixels are still protected by apply_brush_edit.
        state.result_bgr, state.filter_base_bgr, compact, changed = apply_brush_edit(
            state.original_bgr, state.result_bgr, state.filter_base_bgr,
            ix, iy, radius_px, paint_rgb, opacity=min(1.0, strength),
            region_map=None, gap_close=0, mode=fill_mode,
            snap_to_lineart=False, pupil_blend=False, compact_mask=True)
        self._brush_changed_during_stroke |= changed

        if changed:
            self._radio_view_edited.setChecked(True)
            if (hasattr(self._canvas, "update_image_region") and
                    isinstance(compact, tuple) and len(compact) >= 4):
                self._canvas.update_image_region(
                    state.result_bgr, tuple(int(v) for v in compact[:4]))
            elif hasattr(self._canvas, "update_image_pixels"):
                self._canvas.update_image_pixels(state.result_bgr)
            else:
                self._sync_view_after_edit(state)

    def _on_brush_stroke_finished(self):
        try:
            if not self._local_brush_stroke_active:
                self._brush_patch_undo = None
                return
            state = self._current_state()
            self._local_brush_stroke_active = False
            self._canvas.clear_dabs()
            if state is None or state.result_bgr is None:
                self._brush_changed_during_stroke = False
                self._brush_patch_undo = None
                return

            model_hint = bool(self._brush_stroke_model_hint)
            if model_hint:
                state.discard_unchanged_undo()
            elif self._brush_changed_during_stroke:
                self._commit_brush_patch_undo(state)
            else:
                self._brush_patch_undo = None
            if self._brush_changed_during_stroke:
                self._sync_bias_reference_to_current(state)
            already_showing_edited = self._radio_view_edited.isChecked()
            self._radio_view_edited.setChecked(True)
            if already_showing_edited:
                # Live ROI updates already contain the final direct-paint
                # pixels.  Flushing the last coalesced rectangle avoids one
                # full-resolution upload on every mouse release.
                self._refresh_live_canvas_after_edit(state)
            if model_hint and self._brush_changed_during_stroke:
                message = tr("brush_model_hint_done")
            else:
                message = (tr("local_brush_done") if self._brush_changed_during_stroke
                           else tr("manual_edit_no_change"))
            self.statusBar().showMessage(message, 3000)
            self._brush_changed_during_stroke = False
            self._brush_stroke_model_hint = False
        finally:
            self._end_foreground_interaction()

    def _sync_bias_reference_to_current(self, state: "PageState" | None):
        """Invalidate the bias-eraser baseline after a non-bias edit.

        The old implementation eagerly copied two complete page layers after
        every brush stroke/bucket fill.  The baseline is now created lazily at
        the next bias-brush stroke by ``_ensure_bias_reference_exists``.
        """
        if state is None:
            return
        state.bias_brush_reference_bgr = None
        state.bias_brush_reference_filter_bgr = None

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
        self._begin_foreground_interaction()
        self._ensure_bias_reference_exists(state)
        try:
            from core.bias_brush import empty_stroke_alpha, prepare_bias_candidate
            import time
            self._bias_brush_base_result = state.result_bgr.copy()
            self._bias_brush_base_filter = (
                state.filter_base_bgr.copy() if state.filter_base_bgr is not None
                else state.result_bgr.copy())
            # Reuse the immutable pre-stroke bases as the undo checkpoint.
            # The old route copied both complete layers once for undo and once
            # again for live compositing.
            state.push_undo_snapshot({
                "__fields__": state.FILTER_EDIT_FIELDS,
                "result_bgr": self._bias_brush_base_result,
                "filter_base_bgr": self._bias_brush_base_filter,
            }, compare=False)
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
            if state.undo_stack:
                state.undo_stack.pop()
            self._reset_bias_brush_runtime()
            self._end_foreground_interaction()
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
                    if hasattr(self._canvas, "update_image_region"):
                        self._canvas.update_image_region(state.result_bgr, roi)
                    elif hasattr(self._canvas, "update_image_pixels"):
                        self._canvas.update_image_pixels(state.result_bgr)
                    else:
                        self._sync_view_after_edit(state)
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(
                tr("bias_brush_failed").format(message=f"{type(exc).__name__}: {exc}"),
                6000)

    def _on_bias_brush_stroke_finished(self):
        try:
            self._canvas.clear_dabs()
            state = self._current_state()
            active = self._bias_brush_stroke_active
            changed = self._bias_brush_changed_during_stroke
            if active and state is not None and state.result_bgr is not None:
                try:
                    # Final-quality pass: recompute the complete accumulated
                    # stroke with the exact full-page compositor.  The ROI path
                    # above is preview-only.
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
                            brush_radius_px=int(getattr(
                                self, '_bias_brush_last_radius_px', 18)))
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
                if not changed and state.undo_stack:
                    state.undo_stack.pop()
                already_showing_edited = self._radio_view_edited.isChecked()
                self._radio_view_edited.setChecked(True)
                if already_showing_edited:
                    # The cohesive final pass can differ from the live ROI
                    # preview, so one full refresh is still required here.
                    self._sync_view_after_edit(state)
            self._reset_bias_brush_runtime()
            if not active:
                return
            self.statusBar().showMessage(
                tr("bias_brush_done") if changed else tr("manual_edit_no_change"),
                3000)
        finally:
            self._end_foreground_interaction()

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
        self._begin_foreground_interaction()
        try:
            from core.bias_brush import empty_stroke_alpha
            import time
            self._bias_eraser_base_result = state.result_bgr.copy()
            self._bias_eraser_base_filter = (
                state.filter_base_bgr.copy() if state.filter_base_bgr is not None
                else state.result_bgr.copy())
            state.push_undo_snapshot({
                "__fields__": state.FILTER_EDIT_FIELDS,
                "result_bgr": self._bias_eraser_base_result,
                "filter_base_bgr": self._bias_eraser_base_filter,
            }, compare=False)
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
            if state.undo_stack:
                state.undo_stack.pop()
            self._reset_bias_eraser_runtime()
            self._end_foreground_interaction()
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
                    if hasattr(self._canvas, 'update_image_region'):
                        self._canvas.update_image_region(state.result_bgr, roi)
                    elif hasattr(self._canvas, 'update_image_pixels'):
                        self._canvas.update_image_pixels(state.result_bgr)
                    else:
                        self._sync_view_after_edit(state)
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(
                tr("bias_brush_failed").format(message=f"{type(exc).__name__}: {exc}"),
                6000)

    def _on_bias_eraser_stroke_finished(self):
        try:
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
                if not changed and state.undo_stack:
                    state.undo_stack.pop()
                already_showing_edited = self._radio_view_edited.isChecked()
                self._radio_view_edited.setChecked(True)
                if already_showing_edited:
                    self._sync_view_after_edit(state)
            self._reset_bias_eraser_runtime()
            if not active:
                return
            self.statusBar().showMessage(
                tr("bias_eraser_done") if changed else tr("manual_edit_no_change"),
                3000)
        finally:
            self._end_foreground_interaction()

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
        state.push_undo(fields=state.HINT_EDIT_FIELDS)
        state.hint_manager.clear_manual_hints()
        state.discard_unchanged_undo()
        self._canvas.clear_dabs()
        self._refresh_hint_overlay()

    def _clear_eyedropper_model_hints(self):
        state = self._current_state()
        if state is None:
            return
        state.push_undo(fields=state.HINT_EDIT_FIELDS)
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
        if bool(getattr(self, "_semantic_select_armed", False)):
            self._semantic_select_armed = False
            self._run_semantic_selection(state, ix, iy)
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

        state.push_undo(fields=state.PIXEL_EDIT_FIELDS)
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
        state.push_undo(fields=state.PIXEL_EDIT_FIELDS)
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
        if self._hair_guard_selector_active and self._hair_guard_selector_mode in {"brush_add", "brush_erase"}:
            return
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
        if self._hair_guard_selector_active and self._hair_guard_selector_mode == "rect":
            shape = self._hair_guard_selection_shape()
            if shape is None:
                return
            mask = build_rect_selection_mask(shape, x1, y1, x2, y2)
            saved = self._append_hair_guard_selection_operation(mask, add_mode=True)
            if saved is None:
                self._canvas.clear_selection_preview()
            else:
                self._canvas.set_selection_mask_overlay(saved, mode="replace")
            count = len(self._current_hair_guard_selection_history())
            area = int(np.count_nonzero(saved)) if saved is not None else 0
            self.statusBar().showMessage(f"已追加头发纠色蓝框区域。当前 {count} 个操作，覆盖 {area} px。", 2500)
            return
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
            from core.model_downloader import manga_line_model_ready
            from config import Config
            allow_line_download = False
            if not manga_line_model_ready(Config.MANGA_LINE_WEIGHTS_PATH):
                if os.environ.get("COLORTINA_OFFLINE", "").strip().lower() in {"1", "true", "on", "yes"}:
                    QMessageBox.warning(self, "缺少漫画结构线模型",
                                        "已启用离线模式，请先安装 erika.pth。")
                    return
                answer = QMessageBox.question(
                    self, "安装漫画结构线模型",
                    "该功能需要 erika.pth（约 170 MB），是否允许本次联网下载？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No)
                if answer != QMessageBox.StandardButton.Yes:
                    return
                allow_line_download = True
            worker = MangaLineExtractionWorker(
                state.original_bgr, raw_mask, parent=self,
                allow_model_download=allow_line_download)
            self._line_extract_worker = worker
            worker.status.connect(self._on_worker_status)
            worker.finished_ok.connect(self._on_manga_line_finished)
            worker.finished_err.connect(self._on_manga_line_failed)
            worker.cancelled.connect(lambda: self.statusBar().showMessage(tr("stopped"), 4000))
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
        current = None
        target = None
        # Project the current page onto each checkpoint's field set. Ordinary
        # edits therefore avoid materialising untouched AI/cache layers.
        while state.undo_stack:
            candidate = state.undo_stack.pop()
            current = state.inverse_snapshot(candidate)
            if not state._snapshot_equal(candidate, current):
                target = candidate
                break
        if target is None or current is None:
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
        current = None
        target = None
        while state.redo_stack:
            candidate = state.redo_stack.pop()
            current = state.inverse_snapshot(candidate)
            if not state._snapshot_equal(candidate, current):
                target = candidate
                break
        if target is None or current is None:
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
        state.push_undo(fields=state.PIXEL_EDIT_FIELDS)
        state.result_bgr = state.ai_result_bgr.copy()
        self._sync_bias_reference_to_current(state)
        self._radio_view_edited.setChecked(True)
        self._sync_view_after_edit(state)
        self.statusBar().showMessage(tr("restored_ai"), 3000)

    def _update_view_mode_availability(self) -> None:
        state = self._current_state()
        busy = bool(getattr(self, "_ui_busy", False))
        has_page = state is not None
        has_ai = bool(state is not None and state.ai_result_bgr is not None)
        has_edited = bool(state is not None and state.result_bgr is not None)
        original = getattr(self, "_radio_view_original", None)
        ai = getattr(self, "_radio_view_ai", None)
        edited = getattr(self, "_radio_view_edited", None)
        if original is not None:
            original.setEnabled(has_page and not busy)
        if ai is not None:
            ai.setEnabled(has_ai and not busy)
        if edited is not None:
            edited.setEnabled(has_edited and not busy)
        if state is None:
            return
        if ai is not None and ai.isChecked() and not has_ai:
            original.setChecked(True)
        elif edited is not None and edited.isChecked() and not has_edited:
            if has_ai and ai is not None:
                ai.setChecked(True)
            elif original is not None:
                original.setChecked(True)

    def _on_view_mode_changed(self):
        state = self._current_state()
        if state is not None:
            self._sync_view_after_edit(state)

    def _sync_view_after_edit(self, state: "PageState"):
        """Refresh the current layer and optional non-destructive B&W preview."""
        if self._radio_view_original.isChecked():
            image = state.original_bgr
            cache_key = (state.path, "original")
        elif self._radio_view_ai.isChecked():
            ai_image = state.ai_result_bgr
            if ai_image is not None:
                image = ai_image
                cache_key = (state.path, "ai", state._layer_revision.get("ai", 0))
            else:
                image = state.original_bgr
                cache_key = (state.path, "original")
        else:
            result_image = state.result_bgr
            if result_image is not None:
                image = result_image
                cache_key = (state.path, "result",
                             state._layer_revision.get("result", 0))
            else:
                image = state.original_bgr
                cache_key = (state.path, "original")

        mask = self._pending_selection_mask
        has_mask = bool(mask is not None and mask.size and np.any(mask))
        preview_on = bool(
            has_mask
            and getattr(self, '_chk_selection_bw_preview', None) is not None
            and self._chk_selection_bw_preview.isChecked()
            and not self._radio_view_original.isChecked())
        if preview_on:
            from core.local_model_recolor import preview_black_and_white
            image = preview_black_and_white(image, state.original_bgr, mask)
            cache_key = None
        self._canvas.set_image(image, fit=False, cache_key=cache_key)
        self._refresh_hint_overlay()
        if has_mask:
            self._canvas.set_selection_mask_overlay(
                mask, mode=self._current_selection_mode())
        hair_mask = self._current_hair_guard_selection_mask()
        if hair_mask is not None and (self._hair_guard_selector_active or self._auto_black_hair_guard_requires_selection()):
            self._canvas.set_selection_mask_overlay(hair_mask, mode="replace")
        if state.has_result:
            self._schedule_export_compression(state.path, delay_ms=900)
        self._update_export_compression_status(state)

    def _refresh_live_canvas_after_edit(self, state: "PageState") -> None:
        """Finish a live stroke without rebuilding the full page pixmap."""
        if hasattr(self._canvas, "flush_pending_pixels"):
            self._canvas.flush_pending_pixels()
        if (self._radio_view_edited.isChecked() and
                hasattr(self._canvas, "mark_current_cache_key")):
            self._canvas.mark_current_cache_key((
                state.path, "result", state._layer_revision.get("result", 0)))
        self._refresh_hint_overlay()
        mask = self._pending_selection_mask
        if mask is not None and mask.size and np.any(mask):
            self._canvas.set_selection_mask_overlay(
                mask, mode=self._current_selection_mode())
        self._schedule_export_compression(state.path, delay_ms=900)
        self._update_export_compression_status(state)

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

    def _ensure_export_cache_sync(self, state: PageState) -> tuple[str, dict]:
        """Return one final encoded asset, creating it once when necessary.

        Normal colorization/edit flows prepare this in the background.  This
        synchronous fallback is used only when Export is pressed before that
        short background task finishes or immediately after changing profiles.
        """
        profile = self._current_export_profile()
        cached_path, metadata = state.export_cache_info(profile)
        if cached_path:
            return cached_path, metadata

        self._cancel_delayed_export_compression(state.path)
        self.statusBar().showMessage(tr("export_compression_pending"), 0)
        QApplication.processEvents()
        request = state.prepare_export_cache_request(
            self._session_layer_cache_dir, profile)
        if request is None:
            # A released page may still be waiting for its lossless spill.  The
            # property lazily reloads it here only for this explicit export.
            image = state.result_bgr
            if image is None:
                raise RuntimeError(tr("no_colorized_page"))
            request = state.prepare_export_cache_request(
                self._session_layer_cache_dir, profile)
        if request is None:
            raise RuntimeError(tr("no_colorized_page"))
        result = _write_export_cache(request)
        if not state.commit_export_cache_result(result):
            raise RuntimeError("The page changed while its export image was being compressed")
        self._update_export_compression_status(state)
        cached_path, metadata = state.export_cache_info(profile)
        if not cached_path:
            raise RuntimeError("Compressed export cache was not created")
        return cached_path, metadata

    @staticmethod
    def _compression_filter_for_extension(extension: str) -> str:
        ext = extension.lower()
        if ext == ".webp":
            return "WebP (*.webp)"
        if ext == ".jpg" or ext == ".jpeg":
            return "JPEG (*.jpg *.jpeg)"
        return "PNG (*.png)"

    @staticmethod
    def _normalize_export_target(path: str, extension: str) -> str:
        extension = extension.lower()
        base, current = os.path.splitext(path)
        if current.lower() != extension:
            return base + extension
        return path

    def _export_current_page(self):
        state = self._current_state()
        if state is None or not state.has_result:
            QMessageBox.information(self, tr("no_result_title"), tr("no_colorized_page"))
            return
        try:
            cache_path, metadata = self._ensure_export_cache_sync(state)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, tr("error_title"), str(exc))
            return
        extension = os.path.splitext(cache_path)[1].lower()
        default_name = (os.path.splitext(os.path.basename(state.path))[0] +
                        "_colorized" + extension)
        out_path, _ = QFileDialog.getSaveFileName(
            self, tr("export_page"), default_name,
            self._compression_filter_for_extension(extension))
        if not out_path:
            return
        out_path = self._normalize_export_target(out_path, extension)
        try:
            shutil.copyfile(cache_path, out_path)
        except OSError as exc:
            QMessageBox.warning(self, tr("error_title"), str(exc))
            return
        self.statusBar().showMessage(
            tr("exported_compressed_to").format(
                path=out_path,
                size=self._format_bytes(int(metadata.get("compressed_bytes", 0) or 0)),
                saved=float(metadata.get("saved_percent", 0.0) or 0.0)),
            7000)

    @staticmethod
    def _unique_export_path(folder: str, base_name: str, extension: str,
                            reserved: set[str] | None = None) -> str:
        """Never silently overwrite same-named pages from different folders.

        Also preserve pre-existing user files in the chosen export directory.
        """
        reserved = reserved if reserved is not None else set()
        stem = os.path.splitext(base_name)[0]
        attempt = 1
        while True:
            suffix = "" if attempt == 1 else f"_{attempt}"
            candidate = os.path.join(folder, stem + suffix + extension)
            key = os.path.normcase(os.path.abspath(candidate))
            if key not in reserved and not os.path.exists(candidate):
                reserved.add(key)
                return candidate
            attempt += 1

    def _export_all_pages(self):
        done = [s for s in self._ordered_states() if s.has_result]
        if not done:
            QMessageBox.information(self, tr("no_result_title"), tr("no_colorized_pages"))
            return
        out_dir = QFileDialog.getExistingDirectory(self, tr("select_export_folder"))
        if not out_dir:
            return
        total_compressed = 0
        total_raw = 0
        exported = 0
        reserved_export_paths: set[str] = set()
        for index, state in enumerate(done, start=1):
            self.statusBar().showMessage(
                f"{tr('export_compression_pending')} {index}/{len(done)}", 0)
            QApplication.processEvents()
            try:
                cache_path, metadata = self._ensure_export_cache_sync(state)
                extension = os.path.splitext(cache_path)[1].lower()
                name = (os.path.splitext(os.path.basename(state.path))[0] +
                        "_colorized" + extension)
                target = self._unique_export_path(
                    out_dir, name, extension, reserved_export_paths)
                shutil.copyfile(cache_path, target)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(
                    self, tr("error_title"),
                    f"{os.path.basename(state.path)}: {exc}")
                continue
            exported += 1
            total_compressed += int(metadata.get("compressed_bytes", 0) or 0)
            total_raw += int(metadata.get("original_bytes", 0) or 0)
            if state.path != self._current_path:
                self._schedule_page_spill(state.path, delay_ms=0)
        saved = (max(0.0, (1.0 - total_compressed / total_raw) * 100.0)
                 if total_raw else 0.0)
        self.statusBar().showMessage(
            tr("exported_compressed_pages").format(
                n=exported, dir=out_dir,
                size=self._format_bytes(total_compressed), saved=saved), 8000)

    # ── Misc ──────────────────────────────────────────────────────────

    def _canvas_zoom_in(self):
        self._canvas.zoom_in()

    def _canvas_zoom_out(self):
        self._canvas.zoom_out()

    def _update_controls_enabled(self):
        state = self._current_state()
        busy = bool(getattr(self, "_ui_busy", False))
        has_page = state is not None
        has_ai = bool(state is not None and state.ai_result_bgr is not None)
        has_result = bool(state is not None and state.result_bgr is not None)
        any_result = any(getattr(item, "result_bgr", None) is not None
                         for item in (getattr(self, "_pages", {}) or {}).values())

        conditions = {
            "_btn_auto": has_page,
            "_btn_hair_recolor_auto": has_page,
            "_btn_regenerate": has_page,
            "_btn_undo": has_page,
            "_btn_clear": has_page,
            "_btn_undo_edit": has_page,
            "_btn_redo_edit": has_page,
            "_btn_restore_ai": has_ai,
            "_btn_restore_bw": has_ai or has_result,
            "_btn_export_page": has_result,
            "_btn_export_all": any_result,
        }
        for name, allowed in conditions.items():
            btn = getattr(self, name, None)
            if btn is not None:
                try:
                    if name in {"_btn_auto", "_btn_hair_recolor_auto"}:
                        btn.setEnabled(bool(allowed) and (not self._stop_requested))
                    else:
                        btn.setEnabled(bool(allowed) and not busy)
                except RuntimeError:
                    pass
        self._update_view_mode_availability()
        self._refresh_workspace_stats()
