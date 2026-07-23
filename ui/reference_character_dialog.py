"""Manual character identity enrolment from one colour reference image.

The dialog is deliberately explicit: select the whole head, then sample hair,
skin, iris and optional clothing colours.  This is safer than accepting noisy
automatic detections on dense covers or rotated/occluded artwork.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QPoint, QRect
from PySide6.QtGui import QColor, QPainter, QPen, QBrush
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QDialogButtonBox, QLineEdit, QGroupBox, QMessageBox, QSizePolicy,
)

from core.manual_reference import sample_hex_at
from ui.canvas import bgr_to_qpixmap
from ui.i18n import tr


_SAMPLE_COLORS = {
    "hair": QColor(255, 196, 55),
    "skin": QColor(70, 220, 110),
    "eyes": QColor(70, 205, 255),
    "clothing": QColor(205, 95, 255),
}


class ReferenceCharacterCanvas(QLabel):
    selection_changed = Signal()
    sample_changed = Signal(str, str)

    def __init__(self, image_bgr: np.ndarray, parent=None):
        super().__init__(parent)
        self._image = image_bgr
        self._base = bgr_to_qpixmap(image_bgr)
        self._mode = "select"
        self._selection: tuple[int, int, int, int] | None = None
        self._samples: dict[str, tuple[int, int, str]] = {}
        self._drag_start: tuple[int, int] | None = None
        self._drag_current: tuple[int, int] | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(560, 520)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self._render()

    @property
    def selection(self):
        return self._selection

    @property
    def colors(self) -> dict[str, str]:
        return {key: item[2] for key, item in self._samples.items()}

    def set_mode(self, mode: str):
        self._mode = mode
        if mode == "select":
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.PointingHandCursor)

    def clear(self):
        self._selection = None
        self._samples.clear()
        self._drag_start = None
        self._drag_current = None
        self._render()
        self.selection_changed.emit()

    def _display_geometry(self) -> tuple[int, int, int, int]:
        if self._base.isNull():
            return 0, 0, 1, 1
        pix = self._base.scaled(
            max(1, self.width() - 12), max(1, self.height() - 12),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        x0 = int((self.width() - pix.width()) / 2)
        y0 = int((self.height() - pix.height()) / 2)
        return x0, y0, pix.width(), pix.height()

    def _widget_to_image(self, pos) -> tuple[int, int] | None:
        x0, y0, dw, dh = self._display_geometry()
        x = float(pos.x()) - x0
        y = float(pos.y()) - y0
        if not (0 <= x < dw and 0 <= y < dh):
            return None
        h, w = self._image.shape[:2]
        ix = int(np.clip(round(x / max(1, dw - 1) * (w - 1)), 0, w - 1))
        iy = int(np.clip(round(y / max(1, dh - 1) * (h - 1)), 0, h - 1))
        return ix, iy

    def _image_to_display(self, x: int, y: int,
                          pix_w: int, pix_h: int) -> tuple[int, int]:
        h, w = self._image.shape[:2]
        return (int(round(x / max(1, w - 1) * (pix_w - 1))),
                int(round(y / max(1, h - 1) * (pix_h - 1))))

    def _render(self):
        if self.width() <= 0 or self.height() <= 0 or self._base.isNull():
            return
        pix = self._base.scaled(
            max(1, self.width() - 12), max(1, self.height() - 12),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        selection = self._selection
        if self._drag_start is not None and self._drag_current is not None:
            x1, y1 = self._drag_start
            x2, y2 = self._drag_current
            selection = (min(x1, x2), min(y1, y2),
                         abs(x2 - x1), abs(y2 - y1))
        if selection is not None:
            x, y, w, h = selection
            dx, dy = self._image_to_display(x, y, pix.width(), pix.height())
            dx2, dy2 = self._image_to_display(x + w, y + h,
                                               pix.width(), pix.height())
            painter.setPen(QPen(QColor(255, 70, 70), 4))
            painter.setBrush(QBrush(QColor(255, 70, 70, 30)))
            painter.drawRect(QRect(QPoint(dx, dy), QPoint(dx2, dy2)))

        for key, (x, y, _hex) in self._samples.items():
            dx, dy = self._image_to_display(x, y, pix.width(), pix.height())
            color = _SAMPLE_COLORS.get(key, QColor(255, 255, 255))
            painter.setPen(QPen(Qt.GlobalColor.white, 4))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(QPoint(dx, dy), 8, 8)
            painter.setPen(QPen(Qt.GlobalColor.black, 1))
            painter.drawText(dx + 10, dy - 8, tr(f"manual_sample_{key}"))
        painter.end()
        self.setPixmap(pix)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        point = self._widget_to_image(event.position())
        if point is None:
            return
        if self._mode == "select":
            self._drag_start = point
            self._drag_current = point
            self._render()
            return
        key = self._mode
        value = sample_hex_at(self._image, point[0], point[1], radius_px=9)
        self._samples[key] = (point[0], point[1], value)
        self.sample_changed.emit(key, value)
        self._render()

    def mouseMoveEvent(self, event):
        if self._mode == "select" and self._drag_start is not None:
            point = self._widget_to_image(event.position())
            if point is not None:
                self._drag_current = point
                self._render()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if (event.button() == Qt.MouseButton.LeftButton and
                self._mode == "select" and self._drag_start is not None):
            point = self._widget_to_image(event.position()) or self._drag_current
            if point is not None:
                x1, y1 = self._drag_start
                x2, y2 = point
                x, y = min(x1, x2), min(y1, y2)
                w, h = abs(x2 - x1), abs(y2 - y1)
                if w >= 24 and h >= 24:
                    self._selection = (x, y, w, h)
            self._drag_start = None
            self._drag_current = None
            self._render()
            self.selection_changed.emit()
            return
        super().mouseReleaseEvent(event)


class ReferenceCharacterDialog(QDialog):
    def __init__(self, image_bgr: np.ndarray, parent=None,
                 default_name: str = ""):
        super().__init__(parent)
        self.image_bgr = image_bgr.copy()
        self.rotation = 0
        self.setWindowTitle(tr("manual_character_title"))
        self.resize(1120, 780)
        self.setMinimumSize(900, 650)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        instruction = QLabel(tr("manual_character_instruction"))
        instruction.setWordWrap(True)
        root.addWidget(instruction)

        body = QHBoxLayout()
        self._body_layout = body
        body.setSpacing(8)
        self.canvas = ReferenceCharacterCanvas(self.image_bgr)
        body.addWidget(self.canvas, 1)

        controls = QGroupBox(tr("manual_character_controls"))
        controls.setMinimumWidth(260)
        controls.setMaximumWidth(330)
        grid = QGridLayout(controls)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(5)
        grid.setVerticalSpacing(6)

        grid.addWidget(QLabel(tr("manual_character_name")), 0, 0)
        self.name_edit = QLineEdit(default_name)
        grid.addWidget(self.name_edit, 0, 1, 1, 2)

        select_btn = QPushButton(tr("manual_select_head"))
        select_btn.clicked.connect(lambda: self.canvas.set_mode("select"))
        grid.addWidget(select_btn, 1, 0, 1, 3)

        self._swatches: dict[str, QLabel] = {}
        row = 2
        for key in ("hair", "skin", "eyes", "clothing"):
            button = QPushButton(tr(f"manual_sample_{key}"))
            button.clicked.connect(
                lambda _checked=False, k=key: self.canvas.set_mode(k))
            swatch = QLabel("—")
            swatch.setAlignment(Qt.AlignmentFlag.AlignCenter)
            swatch.setMinimumWidth(72)
            swatch.setStyleSheet("border: 1px solid #999; border-radius: 3px;")
            grid.addWidget(button, row, 0, 1, 2)
            grid.addWidget(swatch, row, 2)
            self._swatches[key] = swatch
            row += 1

        rotate_left = QPushButton(tr("manual_rotate_left"))
        rotate_right = QPushButton(tr("manual_rotate_right"))
        rotate_left.clicked.connect(lambda: self._rotate(-90))
        rotate_right.clicked.connect(lambda: self._rotate(90))
        grid.addWidget(rotate_left, row, 0, 1, 1)
        grid.addWidget(rotate_right, row, 1, 1, 2)
        row += 1

        clear = QPushButton(tr("manual_clear_selection"))
        clear.clicked.connect(self.canvas.clear)
        grid.addWidget(clear, row, 0, 1, 3)
        row += 1
        note = QLabel(tr("manual_character_note"))
        note.setWordWrap(True)
        note.setStyleSheet("color: #777; font-size: 11px;")
        grid.addWidget(note, row, 0, 1, 3)
        grid.setRowStretch(row + 1, 1)
        body.addWidget(controls)
        root.addLayout(body, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._validate_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.canvas.sample_changed.connect(self._update_swatch)
        self.canvas.set_mode("select")

    @property
    def head_bbox(self):
        return self.canvas.selection

    @property
    def colors(self):
        return self.canvas.colors

    @property
    def character_name(self):
        return self.name_edit.text().strip()

    def _update_swatch(self, key: str, value: str):
        label = self._swatches[key]
        label.setText(value.upper())
        label.setStyleSheet(
            f"background: {value}; border: 1px solid #777; border-radius: 3px;"
            + (" color: white;" if key == "hair" else ""))

    def _rotate(self, delta: int):
        # Rotate the actual reference image.  Coordinates returned to the core
        # are therefore always in the displayed orientation and need no hidden
        # transform on save/load.
        if delta > 0:
            self.image_bgr = cv2.rotate(self.image_bgr, cv2.ROTATE_90_CLOCKWISE)
        else:
            self.image_bgr = cv2.rotate(self.image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.rotation = (self.rotation + delta) % 360
        old = self.canvas
        self.canvas = ReferenceCharacterCanvas(self.image_bgr)
        self.canvas.sample_changed.connect(self._update_swatch)
        self.canvas.set_mode("select")
        self._body_layout.replaceWidget(old, self.canvas)
        old.deleteLater()
        for label in self._swatches.values():
            label.setText("—")
            label.setStyleSheet("border: 1px solid #999; border-radius: 3px;")

    def _validate_accept(self):
        if self.head_bbox is None:
            QMessageBox.information(self, tr("manual_character_title"),
                                    tr("manual_character_need_head"))
            return
        if "hair" not in self.colors:
            QMessageBox.information(self, tr("manual_character_title"),
                                    tr("manual_character_need_hair"))
            return
        if not self.character_name:
            QMessageBox.information(self, tr("manual_character_title"),
                                    tr("manual_character_need_name"))
            return
        self.accept()
