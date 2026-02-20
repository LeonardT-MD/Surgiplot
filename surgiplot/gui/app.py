from __future__ import annotations

from PySide6 import QtWidgets, QtCore
from surgiplot.core.io.loaders import load_dataset, load_points_from_manual
from surgiplot.gui.label_editor import LabelEditorDialog
from surgiplot.metrics import VOM_VOA, AOA_SF, AOE


class StartWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Surgiplot")
        self.resize(720, 420)

        layout = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel(
            "<h2>Surgiplot</h2><p>Load points, label them, then run metrics.</p>"
        )
        title.setTextFormat(QtCore.Qt.RichText)
        layout.addWidget(title)

        # Mode selection
        mode_group = QtWidgets.QGroupBox("Input mode")
        mode_layout = QtWidgets.QVBoxLayout(mode_group)
        self.rb_file = QtWidgets.QRadioButton(
            "Load annotation file (txt/csv/tsv/space-delimited)"
        )
        self.rb_manual = QtWidgets.QRadioButton("Manual entry (paste coordinates)")
        self.rb_file.setChecked(True)
        mode_layout.addWidget(self.rb_file)
        mode_layout.addWidget(self.rb_manual)
        layout.addWidget(mode_group)

        # Source selection
        src_layout = QtWidgets.QHBoxLayout()
        src_layout.addWidget(QtWidgets.QLabel("Source of data:"))
        self.source = QtWidgets.QComboBox()
        self.source.addItems(["navigation", "photogrammetry", "scanner"])
        src_layout.addWidget(self.source)
        src_layout.addStretch(1)
        layout.addLayout(src_layout)

        # File chooser
        file_layout = QtWidgets.QHBoxLayout()
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setPlaceholderText("Select an annotation file...")
        btn_browse = QtWidgets.QPushButton("Browse")
        btn_browse.clicked.connect(self._browse)
        file_layout.addWidget(self.path_edit)
        file_layout.addWidget(btn_browse)
        layout.addLayout(file_layout)

        # Manual text
        self.manual_text = QtWidgets.QPlainTextEdit()
        self.manual_text.setPlaceholderText(
            "Manual entry format (one point per line):\n"
            "name x y z\n"
            "OR\n"
            "x y z (names auto-assigned)"
        )
        self.manual_text.setVisible(False)
        layout.addWidget(self.manual_text)

        # Continue
        btn = QtWidgets.QPushButton("Continue → Label points")
        btn.clicked.connect(self._continue)
        layout.addWidget(btn)

        self.rb_file.toggled.connect(self._toggle_mode)

    def _toggle_mode(self):
        is_file = self.rb_file.isChecked()
        self.path_edit.setVisible(is_file)
        self.manual_text.setVisible(not is_file)

    def _browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select annotation file",
            "",
            "Text/CSV (*.txt *.csv *.tsv);;All files (*)",
        )
        if path:
            self.path_edit.setText(path)

    def _continue(self):
        src = self.source.currentText()

        if self.rb_file.isChecked():
            path = self.path_edit.text().strip()
            if not path:
                QtWidgets.QMessageBox.warning(
                    self, "Missing file", "Please select an annotation file."
                )
                return
            try:
                ds = load_dataset(path, source=src, alias_points=True)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Load error", str(e))
                return
        else:
            raw = self.manual_text.toPlainText().strip()
            if not raw:
                QtWidgets.QMessageBox.warning(
                    self, "Missing input", "Please paste coordinates for manual entry."
                )
                return
            try:
                ds = self._parse_manual(raw, source=src)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Parse error", str(e))
                return

        # Open label editor
        dlg = LabelEditorDialog(
            ds.to_table_rows(),
            source=ds.meta.get("source", ""),
            parent=self,
        )
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return

        rows, src2 = dlg.get_rows_and_source()
        if src2:
            ds.meta["source"] = src2
        ds.apply_labels_from_table(rows, overwrite=True)

        # Go to metric setup
        self.metric_window = MetricWindow(ds)   # keep reference alive
        self.metric_window.show()
        self.close()

    def _parse_manual(self, raw: str, source: str):
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        pts = []
        names = []
        for ln in lines:
            parts = ln.split()
            if len(parts) == 4:
                names.append(parts[0])
                pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif len(parts) == 3:
                pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
            else:
                raise ValueError(f"Bad line: '{ln}'")

        if names:
            return load_points_from_manual(
                pts, names=names, source=source, alias_points=True
            )
        return load_points_from_manual(
            pts, names=None, source=source, alias_points=False
        )


class MetricWindow(QtWidgets.QWidget):
    def __init__(self, ds):
        super().__init__()
        self.ds = ds
        self.setWindowTitle("Surgiplot — Metrics")
        self.resize(820, 520)

        layout = QtWidgets.QVBoxLayout(self)

        layout.addWidget(
            QtWidgets.QLabel(
                f"<b>Dataset loaded</b> — source: {self.ds.meta.get('source', '')}"
            )
        )

        # Metric selector
        self.metric = QtWidgets.QComboBox()
        self.metric.addItems(["VOM_VOA", "AOA_SF", "AOE"])
        layout.addWidget(self.metric)

        # Inputs
        form = QtWidgets.QFormLayout()

        self.entry_edit = QtWidgets.QLineEdit()
        self.entry_edit.setPlaceholderText(
            "Entry points: comma-separated names (e.g., point_1, point_2, point_3, point_4)"
        )
        form.addRow("Entry", self.entry_edit)

        self.target_edit = QtWidgets.QLineEdit()
        self.target_edit.setPlaceholderText(
            "Target points (polygon) or pivot point (AOE/AOA_SF)"
        )
        form.addRow("Target", self.target_edit)

        self.constraints_edit = QtWidgets.QLineEdit()
        self.constraints_edit.setPlaceholderText(
            "Optional constraints (AOA_SF): comma-separated"
        )
        form.addRow("Constraints", self.constraints_edit)

        self.stand_dist = QtWidgets.QDoubleSpinBox()
        self.stand_dist.setRange(0.1, 1000.0)
        self.stand_dist.setValue(10.0)
        self.stand_dist.setSuffix(" mm")
        form.addRow("stand_dist (sVOM)", self.stand_dist)

        self.A_edit = QtWidgets.QLineEdit()
        self.A_edit.setPlaceholderText("A point name")
        self.B_edit = QtWidgets.QLineEdit()
        self.B_edit.setPlaceholderText("B pivot name")
        self.C_edit = QtWidgets.QLineEdit()
        self.C_edit.setPlaceholderText("C point name")
        form.addRow("AOE A", self.A_edit)
        form.addRow("AOE B", self.B_edit)
        form.addRow("AOE C", self.C_edit)

        layout.addLayout(form)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()

        btn_run = QtWidgets.QPushButton("Run")
        btn_run.clicked.connect(self._run)
        btn_row.addWidget(btn_run)

        btn_export = QtWidgets.QPushButton("Export labeled dataset")
        btn_export.clicked.connect(self._export_dataset)
        btn_row.addWidget(btn_export)

        btn_snip = QtWidgets.QPushButton("Copy Python snippet")
        btn_snip.clicked.connect(self._copy_snippet)
        btn_row.addWidget(btn_snip)

        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        # Output
        self.out = QtWidgets.QPlainTextEdit()
        self.out.setReadOnly(True)
        layout.addWidget(self.out)

        self.metric.currentTextChanged.connect(self._update_visibility)
        self._update_visibility(self.metric.currentText())

    def _update_visibility(self, metric: str):
        is_vom = metric == "VOM_VOA"
        is_aoa = metric == "AOA_SF"
        is_aoe = metric == "AOE"

        self.entry_edit.setVisible(is_vom or is_aoa)
        self.target_edit.setVisible(is_vom or is_aoa)
        self.constraints_edit.setVisible(is_aoa)
        self.stand_dist.setVisible(is_vom)

        self.A_edit.setVisible(is_aoe)
        self.B_edit.setVisible(is_aoe)
        self.C_edit.setVisible(is_aoe)

    def _split_names(self, s: str):
        return [x.strip() for x in (s or "").split(",") if x.strip()]

    def _run(self):
        m = self.metric.currentText()
        try:
            if m == "VOM_VOA":
                entry = self._split_names(self.entry_edit.text())
                target = self._split_names(self.target_edit.text())
                res = VOM_VOA(
                    data=self.ds,
                    entry=entry,
                    target=target,
                    stand_dist=float(self.stand_dist.value()),
                    return_debug=True,
                )
                self.out.setPlainText(
                    f"VoA (deg): {res.voa_deg:.3f}\n"
                    f"VOM (mm^3): {res.vom_mm3:.3f}\n"
                    f"sVOM (mm^3): {res.svom_mm3:.3f}\n\n"
                    f"Debug:\n{res.debug}"
                )

            elif m == "AOA_SF":
                entry = self._split_names(self.entry_edit.text())
                target = self._split_names(self.target_edit.text())
                if len(target) != 1:
                    raise ValueError("AOA_SF expects target as a single pivot point name.")
                constraints = self._split_names(self.constraints_edit.text()) or None
                res = AOA_SF(
                    data=self.ds,
                    entry=entry,
                    target=target[0],
                    constraints=constraints,
                    return_debug=True,
                )
                self.out.setPlainText(
                    f"AoA vertical (deg): {res.aoa_vertical_deg:.3f}\n"
                    f"AoA horizontal (deg): {res.aoa_horizontal_deg:.3f}\n"
                    f"SF proxy area (mm^2): {res.sf_proxy_mm2:.3f}\n\n"
                    f"Debug:\n{res.debug}"
                )

            else:
                A = self.A_edit.text().strip()
                B = self.B_edit.text().strip()
                C = self.C_edit.text().strip()
                res = AOE(data=self.ds, A=A, B=B, C=C, return_debug=True)
                self.out.setPlainText(
                    f"AoE (deg): {res.aoe_deg:.3f}\n\n"
                    f"Debug:\n{res.debug}"
                )

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

    def _current_metric_snippet(self) -> str:
        m = self.metric.currentText()
        src = self.ds.meta.get("source", "")

        header = (
            "from surgiplot import load_dataset\n"
            "from surgiplot.metrics import VOM_VOA, AOA_SF, AOE\n\n"
            f"# Source of data: {src}\n"
        )

        path = self.ds.meta.get("path", "<PATH_TO_ANNOTATION_FILE>")
        header += f'ds = load_dataset(r"{path}", source="{src}")\n'
        header += "ds = ds.edit_labels()  # optional: opens labeling grid\n\n"

        if m == "VOM_VOA":
            entry = ", ".join([f'"{x}"' for x in self._split_names(self.entry_edit.text())])
            target = ", ".join([f'"{x}"' for x in self._split_names(self.target_edit.text())])
            sd = float(self.stand_dist.value())
            return header + (
                f"res = VOM_VOA(data=ds, entry=[{entry}], target=[{target}], stand_dist={sd})\n"
                "print(res)\n"
            )

        if m == "AOA_SF":
            entry = ", ".join([f'"{x}"' for x in self._split_names(self.entry_edit.text())])
            target_list = self._split_names(self.target_edit.text())
            target = target_list[0] if target_list else "<TARGET_PIVOT>"

            cons_list = self._split_names(self.constraints_edit.text())
            if cons_list:
                cons = ", ".join([f'"{x}"' for x in cons_list])
                cons_clause = f", constraints=[{cons}]"
            else:
                cons_clause = ""

            return header + (
                f'res = AOA_SF(data=ds, entry=[{entry}], target="{target}"{cons_clause})\n'
                "print(res)\n"
            )

        A = self.A_edit.text().strip() or "<A>"
        B = self.B_edit.text().strip() or "<B>"
        C = self.C_edit.text().strip() or "<C>"
        return header + (
            f'res = AOE(data=ds, A="{A}", B="{B}", C="{C}")\n'
            "print(res)\n"
        )

    def _copy_snippet(self):
        snip = self._current_metric_snippet()
        QtWidgets.QApplication.clipboard().setText(snip)
        QtWidgets.QMessageBox.information(self, "Copied", "Python snippet copied to clipboard.")

    def _export_dataset(self):
        fmt, ok = QtWidgets.QInputDialog.getItem(
            self, "Export format", "Choose export format:", ["CSV", "JSON"], 0, False
        )
        if not ok:
            return

        if fmt == "CSV":
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save CSV", "surgiplot_points.csv", "CSV (*.csv)"
            )
            if not path:
                return
            self.ds.export_csv(path)
        else:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save JSON", "surgiplot_dataset.json", "JSON (*.json)"
            )
            if not path:
                return
            self.ds.export_json(path)

        QtWidgets.QMessageBox.information(self, "Exported", f"Saved to:\n{path}")


def main():
    app = QtWidgets.QApplication([])
    w = StartWindow()
    w.show()
    app.exec()
