"""Page-local character binding editor.

The dialog does not recolour anything directly.  It records an explicit policy
for every detected hair/head anchor:

- automatic: use Top-2 confidence/margin gates;
- do not lock: never apply identity colour on this page instance;
- a concrete character: forced binding, treated as an intentional user rule.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QLabel, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QHeaderView,
)

from ui.i18n import tr


class CharacterMatchDialog(QDialog):
    def __init__(self, instances: list, character_library,
                 current: dict[int, int] | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("bind_characters_title"))
        self.resize(720, 380)
        self._instances = list(instances or [])
        self._library = character_library
        self._current = {int(k): int(v) for k, v in (current or {}).items()}
        self._combos: list[tuple[int, QComboBox]] = []

        layout = QVBoxLayout(self)
        note = QLabel(tr("bind_characters_help"))
        note.setWordWrap(True)
        layout.addWidget(note)

        table = QTableWidget(len(self._instances), 5, self)
        table.setHorizontalHeaderLabels([
            tr("instance_col"), tr("auto_match_col"), tr("score_col"),
            tr("margin_col"), tr("binding_col"),
        ])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.Stretch)

        characters = list(getattr(character_library, "characters", []) or [])
        for row, instance in enumerate(self._instances):
            hair_id = int(instance.hair_regions[0]) if instance.hair_regions else -1
            table.setItem(row, 0, QTableWidgetItem(str(instance.instance_id)))
            matched_id = instance.matched_character_id
            matched = next((c for c in characters if c.char_id == matched_id), None)
            matched_name = ((matched.name or f"#{matched.char_id}") if matched is not None
                            else tr("unmatched"))
            table.setItem(row, 1, QTableWidgetItem(matched_name))
            table.setItem(row, 2, QTableWidgetItem(
                f"{float(instance.top1_score or 0.0):.3f}"))
            table.setItem(row, 3, QTableWidgetItem(
                f"{float(instance.margin or 0.0):.3f}"))

            combo = QComboBox(table)
            combo.addItem(tr("binding_auto"), None)
            combo.addItem(tr("binding_disable"), -1)
            for character in characters:
                label = character.name or tr("character_default_name").format(
                    id=character.char_id)
                combo.addItem(label, int(character.char_id))
            selected = self._current.get(hair_id, None)
            index = combo.findData(selected)
            combo.setCurrentIndex(max(0, index))
            table.setCellWidget(row, 4, combo)
            self._combos.append((hair_id, combo))

        layout.addWidget(table)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def bindings(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for region_id, combo in self._combos:
            value = combo.currentData()
            if region_id >= 0 and value is not None:
                out[int(region_id)] = int(value)
        return out
