"""Small editor for the book-level character palette library."""

from __future__ import annotations

import re

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QDialogButtonBox, QMessageBox, QLabel,
)

from ui.i18n import get_language

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
_ATTRS = ("hair", "skin", "eyes", "clothing")


class CharacterLibraryDialog(QDialog):
    def __init__(self, library, parent=None):
        super().__init__(parent)
        self._library = library
        zh = get_language() == "zh"
        self.setWindowTitle("角色配色管理" if zh else "Character Palette Manager")
        self.resize(760, 420)

        layout = QVBoxLayout(self)
        label = QLabel(
            "可修改角色名称和绝对配色。颜色格式为 #RRGGBB。"
            if zh else "Edit character names and identity colors. Use #RRGGBB values.")
        label.setWordWrap(True)
        layout.addWidget(label)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["ID", "名称" if zh else "Name", "头发" if zh else "Hair",
             "皮肤" if zh else "Skin", "眼睛" if zh else "Eyes",
             "衣服主色" if zh else "Clothing Main",
             "服装槽（逗号分隔）" if zh else "Clothing Slots (comma-separated)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table, 1)
        self._populate()

        tools = QHBoxLayout()
        remove = QPushButton("删除选中角色" if zh else "Remove Selected")
        remove.clicked.connect(self._remove_selected)
        tools.addWidget(remove)
        tools.addStretch(1)
        layout.addLayout(tools)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save |
            QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _populate(self):
        self.table.setRowCount(0)
        for ch in self._library.characters:
            row = self.table.rowCount()
            self.table.insertRow(row)
            id_item = QTableWidgetItem(str(ch.char_id))
            id_item.setFlags(id_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, id_item)
            self.table.setItem(row, 1, QTableWidgetItem(ch.name or f"Character {ch.char_id}"))
            for col, attr in enumerate(_ATTRS, start=2):
                self.table.setItem(row, col, QTableWidgetItem(ch.colors.get(attr, "")))
            clothing_slots = list(getattr(ch, "color_slots", {}).get("clothing", []))[:3]
            self.table.setItem(row, 6, QTableWidgetItem(", ".join(clothing_slots)))

    def _remove_selected(self):
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)

    def _save(self):
        staged = []
        for row in range(self.table.rowCount()):
            char_id = int(self.table.item(row, 0).text())
            name = self.table.item(row, 1).text().strip()
            colors = {}
            for col, attr in enumerate(_ATTRS, start=2):
                value = (self.table.item(row, col).text().strip()
                         if self.table.item(row, col) else "")
                if value:
                    if not _HEX.match(value):
                        QMessageBox.warning(
                            self, "Invalid color",
                            f"{attr}: {value} is not a #RRGGBB color")
                        return
                    colors[attr] = value.lower()
            slots_text = (self.table.item(row, 6).text().strip()
                          if self.table.item(row, 6) else "")
            clothing_slots = []
            for value in re.split(r"[,;，；\s]+", slots_text):
                value = value.strip()
                if not value:
                    continue
                if not _HEX.match(value):
                    QMessageBox.warning(
                        self, "Invalid color",
                        f"clothing slot: {value} is not a #RRGGBB color")
                    return
                value = value.lower()
                if value not in clothing_slots:
                    clothing_slots.append(value)
                if len(clothing_slots) >= 3:
                    break
            staged.append((char_id, name, colors, clothing_slots))

        kept_ids = {item[0] for item in staged}
        self._library.characters = [
            ch for ch in self._library.characters if ch.char_id in kept_ids]
        for char_id, name, colors, clothing_slots in staged:
            slot_payload = {"clothing": clothing_slots} if clothing_slots else None
            self._library.update_character(
                char_id, name=name, colors=colors, color_slots=slot_payload)
        self.accept()
