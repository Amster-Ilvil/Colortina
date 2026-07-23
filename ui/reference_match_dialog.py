"""Dual-image point correspondence editor for reference colour transfer."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QFont
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDialogButtonBox, QScrollArea,
)

from ui.canvas import bgr_to_qpixmap
from ui.i18n import get_language


class PointImageLabel(QLabel):
    point_clicked = Signal(float, float)

    def __init__(self, image_bgr, parent=None):
        super().__init__(parent)
        self._base = bgr_to_qpixmap(image_bgr)
        self._points: list[tuple[float, float]] = []
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 420)
        self.setMouseTracking(True)
        self._render()

    def set_points(self, points):
        self._points = list(points)
        self._render()

    def _scaled(self):
        return self._base.scaled(
            max(1, self.width() - 8), max(1, self.height() - 8),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)

    def _render(self):
        if self.width() <= 0 or self.height() <= 0:
            return
        pix = self._scaled()
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(QFont("Sans", 11, QFont.Weight.Bold))
        for index, (xn, yn) in enumerate(self._points, start=1):
            x = int(xn * pix.width())
            y = int(yn * pix.height())
            painter.setPen(QPen(Qt.GlobalColor.white, 4))
            painter.setBrush(QColor(240, 70, 70, 210))
            painter.drawEllipse(QPoint(x, y), 8, 8)
            painter.setPen(QPen(Qt.GlobalColor.black, 1))
            painter.drawText(x + 10, y - 7, str(index))
        painter.end()
        self.setPixmap(pix)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()

    def mousePressEvent(self, event):
        pix = self.pixmap()
        if event.button() != Qt.MouseButton.LeftButton or pix is None:
            return super().mousePressEvent(event)
        x0 = (self.width() - pix.width()) / 2
        y0 = (self.height() - pix.height()) / 2
        x = event.position().x() - x0
        y = event.position().y() - y0
        if 0 <= x < pix.width() and 0 <= y < pix.height():
            self.point_clicked.emit(float(x / pix.width()), float(y / pix.height()))
            return
        super().mousePressEvent(event)


class ReferenceMatchDialog(QDialog):
    def __init__(self, reference_bgr, target_bgr, parent=None):
        super().__init__(parent)
        self._pairs: list[tuple[float, float, float, float]] = []
        self._pending_ref: tuple[float, float] | None = None
        zh = get_language() == "zh"
        self.setWindowTitle("参考图对应点" if zh else "Reference Correspondence Points")
        self.resize(980, 680)

        layout = QVBoxLayout(self)
        self._instruction = QLabel()
        self._instruction.setWordWrap(True)
        layout.addWidget(self._instruction)

        row = QHBoxLayout()
        left_box = QVBoxLayout()
        left_box.addWidget(QLabel("彩色参考图" if zh else "Color Reference"))
        self.reference_view = PointImageLabel(reference_bgr)
        left_box.addWidget(self.reference_view, 1)
        right_box = QVBoxLayout()
        right_box.addWidget(QLabel("当前目标页" if zh else "Current Target Page"))
        self.target_view = PointImageLabel(target_bgr)
        right_box.addWidget(self.target_view, 1)
        row.addLayout(left_box, 1)
        row.addLayout(right_box, 1)
        layout.addLayout(row, 1)

        tools = QHBoxLayout()
        self._count = QLabel()
        undo = QPushButton("撤销上一对" if zh else "Undo Last Pair")
        undo.clicked.connect(self._undo)
        clear = QPushButton("清空" if zh else "Clear")
        clear.clicked.connect(self._clear)
        tools.addWidget(self._count)
        tools.addStretch(1)
        tools.addWidget(undo)
        tools.addWidget(clear)
        layout.addLayout(tools)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.reference_view.point_clicked.connect(self._reference_clicked)
        self.target_view.point_clicked.connect(self._target_clicked)
        self._refresh()

    @property
    def pairs(self):
        return list(self._pairs)

    def _reference_clicked(self, x, y):
        self._pending_ref = (x, y)
        self._refresh()

    def _target_clicked(self, x, y):
        if self._pending_ref is None:
            return
        self._pairs.append((*self._pending_ref, x, y))
        self._pending_ref = None
        self._refresh()

    def _undo(self):
        if self._pending_ref is not None:
            self._pending_ref = None
        elif self._pairs:
            self._pairs.pop()
        self._refresh()

    def _clear(self):
        self._pairs.clear()
        self._pending_ref = None
        self._refresh()

    def _refresh(self):
        zh = get_language() == "zh"
        ref_points = [(p[0], p[1]) for p in self._pairs]
        target_points = [(p[2], p[3]) for p in self._pairs]
        if self._pending_ref is not None:
            ref_points.append(self._pending_ref)
        self.reference_view.set_points(ref_points)
        self.target_view.set_points(target_points)
        self._count.setText((f"已完成 {len(self._pairs)} 对" if zh
                             else f"{len(self._pairs)} pair(s) completed"))
        if self._pending_ref is None:
            text = ("先点击左侧彩色参考图，再点击右侧目标页的对应区域。"
                    if zh else
                    "Click a color on the left reference, then its matching region on the target.")
        else:
            text = ("已记录参考点，请点击右侧目标页对应位置。"
                    if zh else
                    "Reference point recorded; click the corresponding target position.")
        self._instruction.setText(text)
