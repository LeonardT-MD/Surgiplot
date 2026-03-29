from __future__ import annotations

from typing import List, Optional, Tuple
from PySide6 import QtWidgets, QtCore

_SOURCE_CHOICES = ["navigation", "photogrammetry", "scanner", "database"]


class LabelEditorDialog(QtWidgets.QDialog):
    def __init__(self, rows: List[dict], source: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Surgiplot — Review imported points")
        self.resize(900, 500)
        self._rows = [dict(r) for r in rows]
        self._source = (source or "").strip().lower()

        layout = QtWidgets.QVBoxLayout(self)

        # Source selection row
        src_layout = QtWidgets.QHBoxLayout()
        src_label = QtWidgets.QLabel("Source of data:")
        self.src_combo = QtWidgets.QComboBox()
        self.src_combo.addItems(_SOURCE_CHOICES)
        if self._source in _SOURCE_CHOICES:
            self.src_combo.setCurrentText(self._source)
        else:
            self.src_combo.setCurrentIndex(0)
        src_layout.addWidget(src_label)
        src_layout.addWidget(self.src_combo)
        src_layout.addStretch(1)
        layout.addLayout(src_layout)

        # Table
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["name", "x", "y", "z", "labels"])
        self.table.setRowCount(len(self._rows))
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked
            | QtWidgets.QAbstractItemView.EditKeyPressed
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)

        for i, r in enumerate(self._rows):
            self._set_item(i, 0, r.get("name", ""), editable=False)
            self._set_item(i, 1, r.get("x", ""), editable=False)
            self._set_item(i, 2, r.get("y", ""), editable=False)
            self._set_item(i, 3, r.get("z", ""), editable=False)
            self._set_item(i, 4, r.get("labels", ""), editable=True)

        layout.addWidget(self.table)

        hint = QtWidgets.QLabel(
            "Tip: in 'labels' you can add comma-separated aliases "
            "(e.g., clinoid, ACP_medial, target).\n"
            "After OK, you can call points by either their canonical name or any alias."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _set_item(self, row: int, col: int, value, editable: bool):
        item = QtWidgets.QTableWidgetItem(str(value))
        if not editable:
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
        self.table.setItem(row, col, item)

    def get_rows_and_source(self) -> Tuple[List[dict], str]:
        out = []
        for r in range(self.table.rowCount()):
            name = self.table.item(r, 0).text().strip()
            x = float(self.table.item(r, 1).text())
            y = float(self.table.item(r, 2).text())
            z = float(self.table.item(r, 3).text())
            labels = self.table.item(r, 4).text().strip()
            out.append({"name": name, "x": x, "y": y, "z": z, "labels": labels})
        src = self.src_combo.currentText().strip().lower()
        return out, src


def run_label_editor(rows: List[dict], source: str = "", parent=None) -> Optional[Tuple[List[dict], str]]:
    """Run the labeling grid (Qt). Returns (rows, source) or None if cancelled."""
    app = QtWidgets.QApplication.instance()
    owns_app = False
    if app is None:
        app = QtWidgets.QApplication([])
        owns_app = True

    dlg = LabelEditorDialog(rows=rows, source=source, parent=parent)
    res = dlg.exec()
    if res == QtWidgets.QDialog.Accepted:
        out = dlg.get_rows_and_source()
    else:
        out = None

    if owns_app:
        app.quit()
    return out
