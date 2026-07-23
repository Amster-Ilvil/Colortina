"""Canvas widget — shows the current page and captures manual hint input.

Tools:
- Brush: click/drag performs a strictly local edit when a result exists;
  before the first model run it stores local model hints.
- Eyedropper: click samples a color from the currently displayed image.

Coordinates: the QGraphicsScene is sized exactly to the image's pixel
dimensions, so `mapToScene()` gives image-pixel coordinates directly —
no separate transform bookkeeping needed. Zoom/pan just scale the view,
never the scene.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QImage, QPixmap, QColor, QPainter, QPen, QBrush
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsEllipseItem


def bgr_to_qpixmap(image_bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
    # .copy() — QImage doesn't own the numpy buffer, and it must outlive it
    return QPixmap.fromImage(qimg.copy())


class HintCanvas(QGraphicsView):
    # Emitted for every brush dab: (x_norm, y_norm, (r, g, b), radius_norm)
    hint_dab_added = Signal(float, float, tuple, float)
    brush_stroke_started = Signal()
    brush_stroke_finished = Signal()
    # Emitted when the eyedropper samples a pixel: (r, g, b)
    color_picked = Signal(tuple)
    # Emitted by the region-fill (paint bucket) tool: full-res pixel coords
    region_fill_requested = Signal(int, int)

    TOOL_BRUSH = "brush"
    TOOL_EYEDROPPER = "eyedropper"
    TOOL_BUCKET = "bucket"
    TOOL_PAN = "pan"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self._pixmap_item = None
        self._current_bgr: np.ndarray | None = None
        self._image_w = 0
        self._image_h = 0

        self._tool = self.TOOL_BRUSH
        self._brush_color = QColor(255, 0, 0)
        self._brush_radius = 12  # image pixels
        self._painting = False
        self._dab_items: list[QGraphicsEllipseItem] = []
        self._debug_items: list = []
        self._last_dab_pos: tuple[int, int] | None = None

    # ── Image display ──────────────────────────────────────────────────

    def set_image(self, image_bgr: np.ndarray, fit: bool = True) -> None:
        self._current_bgr = image_bgr
        self._image_h, self._image_w = image_bgr.shape[:2]
        pixmap = bgr_to_qpixmap(image_bgr)

        self._scene.clear()
        self._dab_items = []
        self._debug_items = []
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(0, 0, self._image_w, self._image_h))

        if fit:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def clear_dabs(self) -> None:
        """Remove the visual brush-dab overlay (call after a re-colorize,
        once the new result already reflects those hints)."""
        for item in self._dab_items:
            self._scene.removeItem(item)
        self._dab_items = []

    def undo_last_dab(self) -> None:
        """Remove just the most recent dab overlay (pair with
        HintManager.undo_last_manual() to keep data and view in sync)."""
        if self._dab_items:
            item = self._dab_items.pop()
            self._scene.removeItem(item)


    def clear_debug_overlays(self) -> None:
        for item in self._debug_items:
            try:
                self._scene.removeItem(item)
            except RuntimeError:
                pass
        self._debug_items = []

    def set_hint_overlay(self, hints, labels: np.ndarray | None = None,
                         *, show_regions: bool = False, context=None) -> None:
        """Preview stored auto/manual hints and optional region boundaries."""
        self.clear_debug_overlays()
        if self._pixmap_item is None:
            return
        if show_regions and labels is not None and labels.shape == (self._image_h, self._image_w):
            edges = np.zeros(labels.shape, dtype=bool)
            edges[1:, :] |= labels[1:, :] != labels[:-1, :]
            edges[:, 1:] |= labels[:, 1:] != labels[:, :-1]
            rgba = np.zeros((self._image_h, self._image_w, 4), dtype=np.uint8)
            rgba[edges, 0] = 0
            rgba[edges, 1] = 210
            rgba[edges, 2] = 255
            rgba[edges, 3] = 115
            qimg = QImage(rgba.data, self._image_w, self._image_h, rgba.strides[0],
                          QImage.Format.Format_RGBA8888)
            item = self._scene.addPixmap(QPixmap.fromImage(qimg.copy()))
            item.setZValue(5)
            self._debug_items.append(item)

        if show_regions and context is not None:
            for instance in getattr(context, "character_instances", []) or []:
                bbox = instance.head_bbox or instance.body_bbox
                if not bbox:
                    continue
                x, y, w, h = bbox
                if instance.lock_allowed:
                    color = QColor(60, 210, 100)       # green: safe to lock
                elif instance.matched_character_id is not None:
                    color = QColor(245, 190, 50)       # yellow: ambiguous
                else:
                    color = QColor(210, 70, 70)        # red: unmatched/conflict
                rect = self._scene.addRect(x, y, w, h, QPen(color, 2))
                rect.setZValue(12)
                self._debug_items.append(rect)
                char_name = (f"C{instance.matched_character_id}"
                             if instance.matched_character_id is not None else "?")
                score = float(instance.top1_score or 0.0)
                margin = float(instance.margin or 0.0)
                label_item = self._scene.addSimpleText(
                    f"{char_name}  {score:.2f} / Δ{margin:.2f}")
                label_item.setBrush(QBrush(color))
                label_item.setPos(x, max(0, y - 18))
                label_item.setZValue(13)
                self._debug_items.append(label_item)

        for hint in hints or []:
            x = float(hint.x_norm) * self._image_w
            y = float(hint.y_norm) * self._image_h
            radius = max(3.0, float(hint.radius_norm) * self._image_w)
            item = QGraphicsEllipseItem(x - radius, y - radius, radius * 2, radius * 2)
            color = QColor(*hint.color)
            item.setBrush(QBrush(color))
            source = getattr(hint, "source", "auto_instance")
            pen_color = Qt.GlobalColor.white if source == "manual" else Qt.GlobalColor.black
            item.setPen(QPen(pen_color, 1))
            opacity = {"manual": 0.78, "character_identity": 0.58,
                       "scene_palette": 0.38, "auto_instance": 0.32}.get(source, 0.25)
            item.setOpacity(opacity)
            item.setZValue(10 if hint.priority >= 100 else 8)
            self._scene.addItem(item)
            self._debug_items.append(item)

    # ── Tool state ─────────────────────────────────────────────────────

    def set_tool(self, tool: str) -> None:
        self._tool = tool

    def set_brush_color(self, color: QColor) -> None:
        self._brush_color = color

    def set_brush_radius(self, radius_px: int) -> None:
        self._brush_radius = max(1, radius_px)

    def zoom_in(self) -> None:
        self.scale(1.25, 1.25)

    def zoom_out(self) -> None:
        self.scale(0.8, 0.8)

    def fit_view(self) -> None:
        if self._pixmap_item is not None:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    # ── Mouse handling ────────────────────────────────────────────────

    def set_eyedropper_mode(self, mode: str) -> None:
        """'point' samples the single pixel under the cursor; 'region'
        samples the median color of the enclosed area around the click
        (tolerance flood fill) — much more robust on screentone,
        gradients and JPEG noise."""
        self._eyedropper_mode = mode if mode in ("point", "region") else "point"

    def _robust_patch_color(self, ix: int, iy: int, radius: int = 2) -> tuple[int, int, int] | None:
        if self._current_bgr is None:
            return None
        x1 = max(0, ix - radius)
        y1 = max(0, iy - radius)
        x2 = min(self._image_w, ix + radius + 1)
        y2 = min(self._image_h, iy + radius + 1)
        patch = self._current_bgr[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        flat = patch.reshape(-1, 3).astype(np.float32)
        brightness = flat.mean(axis=1)
        usable = flat[brightness > 20.0]
        if usable.shape[0] >= max(4, flat.shape[0] // 3):
            flat = usable

        # Prefer the dominant local cluster near the click instead of a plain
        # median across unrelated neighbouring colours. This keeps the sampled
        # colour closer to what the user sees under the cursor.
        center = self._current_bgr[iy, ix].astype(np.float32)
        d = np.linalg.norm(flat - center[None, :], axis=1)
        near = flat[d <= max(12.0, float(np.percentile(d, 45)))]
        if near.shape[0] >= max(3, flat.shape[0] // 5):
            flat = near
        b, g, r = np.median(flat, axis=0)
        return (int(np.clip(r, 0, 255)), int(np.clip(g, 0, 255)), int(np.clip(b, 0, 255)))

    def _sample_color_at(self, ix: int, iy: int) -> tuple[int, int, int] | None:
        if self._current_bgr is None:
            return None
        if not (0 <= ix < self._image_w and 0 <= iy < self._image_h):
            return None
        if getattr(self, "_eyedropper_mode", "point") == "region":
            color = self._sample_region_color(ix, iy)
            if color is not None:
                return color
        return self._robust_patch_color(ix, iy, radius=1)

    def _sample_region_color(self, ix: int, iy: int) -> tuple[int, int, int] | None:
        """Representative color of the tolerance-flood-filled area at (ix, iy)."""
        import numpy as np
        import cv2
        img = self._current_bgr
        h, w = img.shape[:2]
        smooth = cv2.medianBlur(img, 5)
        mask = np.zeros((h + 2, w + 2), np.uint8)
        flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        try:
            cv2.floodFill(smooth, mask, (ix, iy), 0,
                          loDiff=(14, 14, 14), upDiff=(14, 14, 14), flags=flags)
        except cv2.error:
            return self._robust_patch_color(ix, iy, radius=2)
        m = mask[1:-1, 1:-1] > 0
        if m.sum() < 8:
            return self._robust_patch_color(ix, iy, radius=2)
        pixels = img[m].reshape(-1, 3).astype(np.float32)
        brightness = pixels.mean(axis=1)
        usable = pixels[brightness > 20.0]
        if usable.shape[0] >= max(8, pixels.shape[0] // 4):
            pixels = usable
        center = img[iy, ix].astype(np.float32)
        d = np.linalg.norm(pixels - center[None, :], axis=1)
        near = pixels[d <= max(14.0, float(np.percentile(d, 40)))]
        if near.shape[0] >= max(6, pixels.shape[0] // 6):
            pixels = near
        b, g, r = np.median(pixels, axis=0)
        return (int(np.clip(r, 0, 255)), int(np.clip(g, 0, 255)), int(np.clip(b, 0, 255)))

    def _drop_dab(self, ix: int, iy: int) -> None:
        if self._last_dab_pos is not None:
            dx = ix - self._last_dab_pos[0]
            dy = iy - self._last_dab_pos[1]
            min_step = max(2, int(self._brush_radius * 0.45))
            if dx * dx + dy * dy < min_step * min_step:
                return
        self._last_dab_pos = (ix, iy)
        color = (self._brush_color.red(), self._brush_color.green(),
                 self._brush_color.blue())
        x_norm = ix / self._image_w
        y_norm = iy / self._image_h
        # Brush radius is in image pixels (matches the visual overlay
        # below) — convert to a fraction of image width so it survives
        # the resize into the model's working resolution unchanged.
        radius_norm = self._brush_radius / self._image_w

        item = QGraphicsEllipseItem(ix - self._brush_radius, iy - self._brush_radius,
                                    self._brush_radius * 2, self._brush_radius * 2)
        item.setBrush(QBrush(self._brush_color))
        item.setPen(QPen(Qt.GlobalColor.white, max(1, self._brush_radius // 6)))
        item.setOpacity(0.85)
        self._scene.addItem(item)
        self._dab_items.append(item)

        self.hint_dab_added.emit(x_norm, y_norm, color, radius_norm)

    def mousePressEvent(self, event):
        if self._pixmap_item is None:
            return super().mousePressEvent(event)

        if event.button() == Qt.MouseButton.LeftButton:
            pos = self.mapToScene(event.pos())
            ix, iy = int(pos.x()), int(pos.y())

            if self._tool == self.TOOL_BRUSH:
                if 0 <= ix < self._image_w and 0 <= iy < self._image_h:
                    self._painting = True
                    self._last_dab_pos = None
                    self.brush_stroke_started.emit()
                    self._drop_dab(ix, iy)
                return
            elif self._tool == self.TOOL_EYEDROPPER:
                color = self._sample_color_at(ix, iy)
                if color is not None:
                    self.color_picked.emit(color)
                return
            elif self._tool == self.TOOL_BUCKET:
                if 0 <= ix < self._image_w and 0 <= iy < self._image_h:
                    self.region_fill_requested.emit(ix, iy)
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._painting and self._tool == self.TOOL_BRUSH:
            pos = self.mapToScene(event.pos())
            ix, iy = int(pos.x()), int(pos.y())
            if 0 <= ix < self._image_w and 0 <= iy < self._image_h:
                self._drop_dab(ix, iy)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            was_painting = self._painting
            self._painting = False
            self._last_dab_pos = None
            if was_painting:
                self.brush_stroke_finished.emit()
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        # Ctrl+wheel to zoom; plain wheel scrolls (standard QGraphicsView behavior)
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)
        else:
            super().wheelEvent(event)
