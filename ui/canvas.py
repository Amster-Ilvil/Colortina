"""Canvas widget — shows the current page and captures manual hint input.

Two tools:
- Brush: click/drag lays down color-hint dabs (fed into HintManager).
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

    # ── Image display ──────────────────────────────────────────────────

    def set_image(self, image_bgr: np.ndarray, fit: bool = True) -> None:
        self._current_bgr = image_bgr
        self._image_h, self._image_w = image_bgr.shape[:2]
        pixmap = bgr_to_qpixmap(image_bgr)

        self._scene.clear()
        self._dab_items = []
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

    def _sample_color_at(self, ix: int, iy: int) -> tuple[int, int, int] | None:
        if self._current_bgr is None:
            return None
        if not (0 <= ix < self._image_w and 0 <= iy < self._image_h):
            return None
        b, g, r = self._current_bgr[iy, ix]
        return (int(r), int(g), int(b))

    def _drop_dab(self, ix: int, iy: int) -> None:
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
                self._painting = True
                if 0 <= ix < self._image_w and 0 <= iy < self._image_h:
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
            self._painting = False
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        # Ctrl+wheel to zoom; plain wheel scrolls (standard QGraphicsView behavior)
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)
        else:
            super().wheelEvent(event)
