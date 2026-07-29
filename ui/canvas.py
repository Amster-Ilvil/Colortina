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
from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import QImage, QPixmap, QColor, QPainter, QPen, QBrush, QPainterPath
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsPathItem, QGraphicsRectItem


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
    # Completely independent custom-colour-bias brush channel.
    bias_brush_dab_added = Signal(float, float, float)
    bias_brush_stroke_started = Signal()
    bias_brush_stroke_finished = Signal()
    bias_eraser_dab_added = Signal(float, float, float)
    bias_eraser_stroke_started = Signal()
    bias_eraser_stroke_finished = Signal()
    # Emitted when the eyedropper samples a pixel: ((r, g, b), x_norm, y_norm)
    color_picked = Signal(tuple, float, float)
    # Emitted by the region-fill (paint bucket) tool: full-res pixel coords
    region_fill_requested = Signal(int, int)
    # Emitted by selection-recolor tools. Polygon points are image pixel coords.
    polygon_fill_requested = Signal(object)
    rect_fill_requested = Signal(int, int, int, int)
    selection_preview_active = Signal(bool)
    # Emitted when the user manually tweaks the pending blue selection mask: (ix, iy, radius_px, add_mode)
    selection_adjust_dab = Signal(int, int, int, bool)

    TOOL_BRUSH = "brush"
    TOOL_BIAS_BRUSH = "bias_brush"
    TOOL_BIAS_ERASER = "bias_eraser"
    TOOL_EYEDROPPER = "eyedropper"
    TOOL_BUCKET = "bucket"
    TOOL_LASSO_BUCKET = "lasso_bucket"
    TOOL_RECT_BUCKET = "rect_bucket"
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
        self._bias_brush_color = QColor(120, 160, 255)
        self._bias_brush_radius = 18
        self._bias_eraser_color = QColor(210, 210, 210)
        self._bias_eraser_radius = 18
        self._painting = False
        self._dab_items: list[QGraphicsEllipseItem] = []
        self._debug_items: list = []
        self._last_dab_pos: tuple[int, int] | None = None
        self._selection_points: list[tuple[int, int]] = []
        self._selection_item: QGraphicsPathItem | None = None
        self._selection_rect_item: QGraphicsRectItem | None = None
        self._pending_selection_item: QGraphicsPathItem | None = None
        self._selection_combine_mode = "replace"
        self._selection_start: tuple[int, int] | None = None
        self._selection_rect: tuple[int, int, int, int] | None = None
        self._selection_adjust_enabled = False
        self._selection_adjust_radius = 18
        self._selection_adjust_active = False
        self._selection_adjust_add_mode = True
        self._selection_adjust_mode = "add"

    # ── Image display ──────────────────────────────────────────────────

    def set_image(self, image_bgr: np.ndarray, fit: bool = True) -> None:
        self._current_bgr = image_bgr
        self._image_h, self._image_w = image_bgr.shape[:2]
        pixmap = bgr_to_qpixmap(image_bgr)

        self._scene.clear()
        self._dab_items = []
        self._debug_items = []
        self._selection_points = []
        self._selection_item = None
        self._selection_rect_item = None
        self._pending_selection_item = None
        self._selection_start = None
        self._selection_rect = None
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(0, 0, self._image_w, self._image_h))

        if fit:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def update_image_pixels(self, image_bgr: np.ndarray) -> None:
        """Refresh only the displayed pixels without rebuilding the scene.

        Brush edits use this during a stroke so the recolor is visible
        immediately. Keeping the scene intact avoids losing mouse capture or
        temporary overlays while dragging.
        """
        if image_bgr is None:
            return
        h, w = image_bgr.shape[:2]
        if self._pixmap_item is None or (w, h) != (self._image_w, self._image_h):
            self.set_image(image_bgr, fit=False)
            return
        self._current_bgr = image_bgr
        self._pixmap_item.setPixmap(bgr_to_qpixmap(image_bgr))

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
            if getattr(hint, "source", "") == "eyedropper_hint":
                continue
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

    def set_bias_brush_color(self, color: QColor) -> None:
        self._bias_brush_color = QColor(color)

    def set_bias_brush_radius(self, radius_px: int) -> None:
        self._bias_brush_radius = max(1, int(radius_px))

    def set_bias_eraser_radius(self, radius_px: int) -> None:
        self._bias_eraser_radius = max(1, int(radius_px))

    def zoom_in(self) -> None:
        self.scale(1.25, 1.25)

    def zoom_out(self) -> None:
        self.scale(0.8, 0.8)

    def fit_view(self) -> None:
        if self._pixmap_item is not None:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    # ── Mouse handling ────────────────────────────────────────────────
    def set_selection_adjust_enabled(self, enabled: bool) -> None:
        self._selection_adjust_enabled = bool(enabled)

    def set_selection_adjust_radius(self, radius_px: int) -> None:
        self._selection_adjust_radius = max(1, int(radius_px))

    def set_selection_adjust_mode(self, mode: str) -> None:
        self._selection_adjust_mode = "erase" if str(mode) == "erase" else "add"

    def _can_adjust_pending_selection(self) -> bool:
        return bool(self._selection_adjust_enabled and self._pending_selection_item is not None
                    and self._tool in (self.TOOL_RECT_BUCKET, self.TOOL_LASSO_BUCKET))

    def _emit_selection_adjust_dab(self, ix: int, iy: int) -> None:
        if 0 <= ix < self._image_w and 0 <= iy < self._image_h:
            self.selection_adjust_dab.emit(
                int(ix), int(iy), int(self._selection_adjust_radius),
                bool(self._selection_adjust_add_mode))

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
        radius_norm = self._brush_radius / self._image_w

        item = QGraphicsEllipseItem(ix - self._brush_radius, iy - self._brush_radius,
                                    self._brush_radius * 2, self._brush_radius * 2)
        item.setBrush(QBrush(self._brush_color))
        item.setPen(QPen(Qt.GlobalColor.white, max(1, self._brush_radius // 6)))
        item.setOpacity(0.85)
        self._scene.addItem(item)
        self._dab_items.append(item)

        self.hint_dab_added.emit(x_norm, y_norm, color, radius_norm)

    def _drop_bias_dab(self, ix: int, iy: int) -> None:
        def emit_bias_point(px: int, py: int) -> None:
            x_norm = px / max(1, self._image_w)
            y_norm = py / max(1, self._image_h)
            radius_norm = self._bias_brush_radius / max(1, self._image_w)
            item = QGraphicsEllipseItem(
                px - self._bias_brush_radius, py - self._bias_brush_radius,
                self._bias_brush_radius * 2, self._bias_brush_radius * 2)
            preview = QColor(self._bias_brush_color)
            item.setBrush(QBrush(preview))
            item.setPen(QPen(QColor(255, 255, 255, 210),
                             max(1, self._bias_brush_radius // 8)))
            item.setOpacity(0.42)
            item.setZValue(12)
            self._scene.addItem(item)
            self._dab_items.append(item)
            self.bias_brush_dab_added.emit(x_norm, y_norm, radius_norm)

        if self._last_dab_pos is None:
            self._last_dab_pos = (ix, iy)
            emit_bias_point(ix, iy)
            return

        lx, ly = self._last_dab_pos
        dx = ix - lx
        dy = iy - ly
        min_step = max(2, int(self._bias_brush_radius * 0.34))
        distance = float(np.hypot(dx, dy))
        if distance < float(min_step):
            return
        steps = max(1, int(np.ceil(distance / float(min_step))))
        for step in range(1, steps + 1):
            t = step / float(steps)
            px = int(round(lx + dx * t))
            py = int(round(ly + dy * t))
            emit_bias_point(px, py)
        self._last_dab_pos = (ix, iy)


    def _drop_bias_eraser_dab(self, ix: int, iy: int) -> None:
        def emit_eraser_point(px: int, py: int) -> None:
            x_norm = px / max(1, self._image_w)
            y_norm = py / max(1, self._image_h)
            radius_norm = self._bias_eraser_radius / max(1, self._image_w)
            item = QGraphicsEllipseItem(
                px - self._bias_eraser_radius, py - self._bias_eraser_radius,
                self._bias_eraser_radius * 2, self._bias_eraser_radius * 2)
            preview = QColor(self._bias_eraser_color)
            item.setBrush(QBrush(QColor(preview.red(), preview.green(), preview.blue(), 40)))
            item.setPen(QPen(QColor(255, 255, 255, 220),
                             max(1, self._bias_eraser_radius // 8)))
            item.setOpacity(0.55)
            item.setZValue(12)
            self._scene.addItem(item)
            self._dab_items.append(item)
            self.bias_eraser_dab_added.emit(x_norm, y_norm, radius_norm)

        if self._last_dab_pos is None:
            self._last_dab_pos = (ix, iy)
            emit_eraser_point(ix, iy)
            return

        lx, ly = self._last_dab_pos
        dx = ix - lx
        dy = iy - ly
        min_step = max(2, int(self._bias_eraser_radius * 0.34))
        distance = float(np.hypot(dx, dy))
        if distance < float(min_step):
            return
        steps = max(1, int(np.ceil(distance / float(min_step))))
        for step in range(1, steps + 1):
            t = step / float(steps)
            px = int(round(lx + dx * t))
            py = int(round(ly + dy * t))
            emit_eraser_point(px, py)
        self._last_dab_pos = (ix, iy)

    def set_selection_combine_mode(self, mode: str) -> None:
        self._selection_combine_mode = mode if mode in {"replace", "add", "subtract"} else "replace"

    def _selection_preview_color(self, mode: str | None = None) -> QColor:
        mode = mode or self._selection_combine_mode
        if mode == "add":
            return QColor(0, 185, 100)
        if mode == "subtract":
            return QColor(225, 70, 70)
        return QColor(0, 150, 245)

    def set_selection_mask_overlay(self, mask: np.ndarray | None,
                                   mode: str | None = None) -> None:
        self._clear_drawing_selection_overlay()
        self._clear_pending_selection_overlay()
        if mask is None or mask.size == 0 or not np.any(mask):
            self.selection_preview_active.emit(False)
            return
        binary = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.selection_preview_active.emit(False)
            return
        path = QPainterPath()
        for contour in contours:
            if len(contour) < 2:
                continue
            first = contour[0][0]
            path.moveTo(float(first[0]), float(first[1]))
            for pt in contour[1:]:
                x, y = pt[0]
                path.lineTo(float(x), float(y))
            path.closeSubpath()
        path.setFillRule(Qt.FillRule.OddEvenFill)
        color = self._selection_preview_color(mode)
        item = QGraphicsPathItem(path)
        item.setPen(QPen(color, 2))
        item.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 36)))
        item.setZValue(19)
        self._scene.addItem(item)
        self._pending_selection_item = item
        self.selection_preview_active.emit(True)

    def clear_selection_preview(self) -> None:
        self._selection_points = []
        self._selection_start = None
        self._selection_rect = None
        self._clear_selection_overlay()
        self.selection_preview_active.emit(False)

    def _clear_pending_selection_overlay(self) -> None:
        item = self._pending_selection_item
        if item is not None:
            try:
                self._scene.removeItem(item)
            except RuntimeError:
                pass
            self._pending_selection_item = None

    def _clear_drawing_selection_overlay(self) -> None:
        for attr in ("_selection_item", "_selection_rect_item"):
            item = getattr(self, attr, None)
            if item is not None:
                try:
                    self._scene.removeItem(item)
                except RuntimeError:
                    pass
                setattr(self, attr, None)

    def _clear_selection_overlay(self) -> None:
        self._clear_drawing_selection_overlay()
        self._clear_pending_selection_overlay()

    def _update_polygon_overlay(self, closed: bool = False) -> None:
        self._clear_drawing_selection_overlay()
        if len(self._selection_points) < 2:
            return
        path = QPainterPath(QPointF(*self._selection_points[0]))
        for x, y in self._selection_points[1:]:
            path.lineTo(float(x), float(y))
        if closed and len(self._selection_points) >= 3:
            path.closeSubpath()
        color = self._selection_preview_color()
        item = QGraphicsPathItem(path)
        item.setPen(QPen(color, 2))
        item.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 40)))
        item.setZValue(20)
        self._scene.addItem(item)
        self._selection_item = item

    def _update_rect_overlay(self, x1: int, y1: int, x2: int, y2: int) -> None:
        self._clear_drawing_selection_overlay()
        self._selection_rect = (x1, y1, x2, y2)
        rx1, rx2 = sorted((x1, x2))
        ry1, ry2 = sorted((y1, y2))
        color = self._selection_preview_color()
        item = QGraphicsRectItem(rx1, ry1, max(1, rx2-rx1), max(1, ry2-ry1))
        item.setPen(QPen(color, 2))
        item.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 40)))
        item.setZValue(20)
        self._scene.addItem(item)
        self._selection_rect_item = item

    def _event_scene_pos(self, event):
        try:
            viewport_pos = event.position().toPoint()
        except Exception:
            viewport_pos = event.pos()
        return self.mapToScene(viewport_pos)

    def mousePressEvent(self, event):
        if self._pixmap_item is None:
            return super().mousePressEvent(event)

        pos = self._event_scene_pos(event)
        ix, iy = int(pos.x()), int(pos.y())

        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            if self._can_adjust_pending_selection():
                self._selection_adjust_active = True
                configured_add = self._selection_adjust_mode != "erase"
                self._selection_adjust_add_mode = (configured_add if event.button() == Qt.MouseButton.LeftButton else not configured_add)
                self._emit_selection_adjust_dab(ix, iy)
                return

        if event.button() == Qt.MouseButton.LeftButton:

            if self._tool == self.TOOL_BRUSH:
                if 0 <= ix < self._image_w and 0 <= iy < self._image_h:
                    self._painting = True
                    self._last_dab_pos = None
                    self.brush_stroke_started.emit()
                    self._drop_dab(ix, iy)
                return
            elif self._tool == self.TOOL_BIAS_BRUSH:
                if 0 <= ix < self._image_w and 0 <= iy < self._image_h:
                    self._painting = True
                    self._last_dab_pos = None
                    self.bias_brush_stroke_started.emit()
                    self._drop_bias_dab(ix, iy)
                return
            elif self._tool == self.TOOL_BIAS_ERASER:
                if 0 <= ix < self._image_w and 0 <= iy < self._image_h:
                    self._painting = True
                    self._last_dab_pos = None
                    self.bias_eraser_stroke_started.emit()
                    self._drop_bias_eraser_dab(ix, iy)
                return
            elif self._tool == self.TOOL_EYEDROPPER:
                color = self._sample_color_at(ix, iy)
                if color is not None:
                    x_norm = ix / max(1, self._image_w - 1)
                    y_norm = iy / max(1, self._image_h - 1)
                    self.color_picked.emit(color, float(x_norm), float(y_norm))
                return
            elif self._tool == self.TOOL_BUCKET:
                if 0 <= ix < self._image_w and 0 <= iy < self._image_h:
                    self.region_fill_requested.emit(ix, iy)
                return
            elif self._tool == self.TOOL_LASSO_BUCKET:
                if 0 <= ix < self._image_w and 0 <= iy < self._image_h:
                    self._selection_points = [(ix, iy)]
                    self._painting = True
                    self._update_polygon_overlay()
                return
            elif self._tool == self.TOOL_RECT_BUCKET:
                if 0 <= ix < self._image_w and 0 <= iy < self._image_h:
                    self._selection_start = (ix, iy)
                    self._painting = True
                    self._update_rect_overlay(ix, iy, ix, iy)
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._selection_adjust_active and self._can_adjust_pending_selection():
            pos = self._event_scene_pos(event)
            ix, iy = int(pos.x()), int(pos.y())
            self._emit_selection_adjust_dab(ix, iy)
            return
        if self._painting and self._tool == self.TOOL_BRUSH:
            pos = self._event_scene_pos(event)
            ix, iy = int(pos.x()), int(pos.y())
            if 0 <= ix < self._image_w and 0 <= iy < self._image_h:
                self._drop_dab(ix, iy)
            return
        if self._painting and self._tool == self.TOOL_BIAS_BRUSH:
            pos = self._event_scene_pos(event)
            ix, iy = int(pos.x()), int(pos.y())
            if 0 <= ix < self._image_w and 0 <= iy < self._image_h:
                self._drop_bias_dab(ix, iy)
            return
        if self._painting and self._tool == self.TOOL_BIAS_ERASER:
            pos = self._event_scene_pos(event)
            ix, iy = int(pos.x()), int(pos.y())
            if 0 <= ix < self._image_w and 0 <= iy < self._image_h:
                self._drop_bias_eraser_dab(ix, iy)
            return
        if self._painting and self._tool == self.TOOL_LASSO_BUCKET:
            pos = self._event_scene_pos(event)
            ix, iy = int(pos.x()), int(pos.y())
            if 0 <= ix < self._image_w and 0 <= iy < self._image_h:
                if not self._selection_points or (ix - self._selection_points[-1][0]) ** 2 + (iy - self._selection_points[-1][1]) ** 2 >= 9:
                    self._selection_points.append((ix, iy))
                    self._update_polygon_overlay()
            return
        if self._painting and self._tool == self.TOOL_RECT_BUCKET:
            pos = self._event_scene_pos(event)
            ix, iy = int(pos.x()), int(pos.y())
            if self._selection_start is not None:
                sx, sy = self._selection_start
                ix = min(max(ix, 0), self._image_w - 1)
                iy = min(max(iy, 0), self._image_h - 1)
                self._update_rect_overlay(sx, sy, ix, iy)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton) and self._selection_adjust_active:
            self._selection_adjust_active = False
            return
        if event.button() == Qt.MouseButton.LeftButton:
            was_painting = self._painting
            active_tool = self._tool
            self._painting = False
            self._last_dab_pos = None
            if was_painting and active_tool == self.TOOL_BRUSH:
                self.brush_stroke_finished.emit()
            elif was_painting and active_tool == self.TOOL_BIAS_BRUSH:
                self.bias_brush_stroke_finished.emit()
            elif was_painting and active_tool == self.TOOL_BIAS_ERASER:
                self.bias_eraser_stroke_finished.emit()
            elif was_painting and active_tool == self.TOOL_LASSO_BUCKET:
                points = list(self._selection_points)
                if len(points) >= 3:
                    self._selection_points = points
                    self._update_polygon_overlay(closed=True)
                    self.selection_preview_active.emit(True)
                    self.polygon_fill_requested.emit(points)
                else:
                    self.clear_selection_preview()
            elif was_painting and active_tool == self.TOOL_RECT_BUCKET:
                start = self._selection_start
                pos = self._event_scene_pos(event)
                ix, iy = int(pos.x()), int(pos.y())
                ix = min(max(ix, 0), self._image_w - 1)
                iy = min(max(iy, 0), self._image_h - 1)
                if start is not None:
                    self._update_rect_overlay(start[0], start[1], ix, iy)
                    self.selection_preview_active.emit(True)
                    self.rect_fill_requested.emit(start[0], start[1], ix, iy)
                else:
                    self.clear_selection_preview()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self._pixmap_item is None:
            return super().mouseDoubleClickEvent(event)
        if event.button() == Qt.MouseButton.LeftButton and self._tool == self.TOOL_LASSO_BUCKET and self._painting:
            pos = self._event_scene_pos(event)
            ix, iy = int(pos.x()), int(pos.y())
            if 0 <= ix < self._image_w and 0 <= iy < self._image_h:
                if not self._selection_points or self._selection_points[-1] != (ix, iy):
                    self._selection_points.append((ix, iy))
            self._painting = False
            points = list(self._selection_points)
            if len(points) >= 3:
                self._selection_points = points
                self._update_polygon_overlay(closed=True)
                self.selection_preview_active.emit(True)
                self.polygon_fill_requested.emit(points)
            else:
                self.clear_selection_preview()
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        # Ctrl+wheel to zoom; plain wheel scrolls (standard QGraphicsView behavior)
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)
        else:
            super().wheelEvent(event)
