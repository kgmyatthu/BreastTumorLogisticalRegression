from __future__ import annotations
import json, os, sys, traceback
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QListWidget, QPushButton, QLabel, QFileDialog, QMessageBox,
    QTableWidget, QTableWidgetItem, QScrollArea, QFormLayout, QLineEdit,
    QSizePolicy, QStatusBar
)

def load_model_bundle(path: str = "tumor_model.joblib"):
    import joblib
    bundle = joblib.load(path)
    model = bundle["model"]
    feature_names = bundle.get("feature_names")
    target_names = bundle.get("target_names", ["malignant", "benign"])
    if not feature_names or len(feature_names) != 30:
        try:
            from sklearn.datasets import load_breast_cancer
            feature_names = list(load_breast_cancer().feature_names)
        except Exception as e:
            raise RuntimeError(f"Could not determine 30 feature names: {e}")
    return model, feature_names, target_names

def load_heldout_json(path: str = "heldout_test.json"):
    with open(path, "r") as f:
        return json.load(f)


class MainWindow(QMainWindow):
    def __init__(self, model_path="tumor_model.joblib", json_path="heldout_test.json"):
        super().__init__()
        self.setWindowTitle("Tumor Classifier Demo (PyQt5, FNA features)")
        self.resize(1200, 720)

        try:
            self.model, self.feature_names, self.target_names = load_model_bundle(model_path)
        except Exception as e:
            QMessageBox.critical(self, "Model load error", f"Failed to load model:\n{e}")
            raise

        self.heldout = {"heldout_samples": []}
        self.json_path = json_path
        if os.path.exists(json_path):
            try:
                self.heldout = load_heldout_json(json_path)
            except Exception as e:
                QMessageBox.warning(self, "JSON load warning",
                                    f"Could not load '{json_path}':\n{e}\nContinuing without it.")

        self._build_ui(model_path, json_path)


    def _build_ui(self, model_path, json_path):
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        self.setCentralWidget(splitter)

        left = QWidget()
        left.setMinimumWidth(430)
        l_layout = QVBoxLayout(left)
        title_l = QLabel("<b>Held-out (non-training) samples</b>")
        l_layout.addWidget(title_l)

        row = QHBoxLayout()
        self.btn_load_json = QPushButton("Load JSON…")
        self.btn_predict_selected = QPushButton("Predict Selected")
        row.addStretch(1)
        row.addWidget(self.btn_predict_selected)
        row.addWidget(self.btn_load_json)
        l_layout.addLayout(row)

        self.listbox = QListWidget()
        self.listbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        l_layout.addWidget(self.listbox, 3)

        self.info_left = QLabel("Select a sample to view features.")
        l_layout.addWidget(self.info_left)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Feature", "Value"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        l_layout.addWidget(self.table, 5)

        self.pred_left = QLabel("")
        self.pred_left.setStyleSheet("font-weight: bold;")
        l_layout.addWidget(self.pred_left)

        splitter.addWidget(left)

        right = QWidget()
        r_layout = QVBoxLayout(right)
        title_r = QLabel("<b>Manual input (30 FNA features)</b>")
        r_layout.addWidget(title_r)

        btn_row = QHBoxLayout()
        self.btn_fill = QPushButton("Fill from Selected")
        self.btn_clear = QPushButton("Clear")
        self.btn_predict_manual = QPushButton("Predict (Manual)")
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_predict_manual)
        btn_row.addWidget(self.btn_fill)
        btn_row.addWidget(self.btn_clear)
        r_layout.addLayout(btn_row)

        # Scroll area with form
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        inner = QWidget()
        self.form = QFormLayout(inner)
        self.inputs: list[QLineEdit] = []
        for name in self.feature_names:
            line = QLineEdit()
            line.setPlaceholderText("number")
            self.inputs.append(line)
            self.form.addRow(name, line)
        self.scroll.setWidget(inner)
        r_layout.addWidget(self.scroll, 1)

        self.pred_right = QLabel("")
        self.pred_right.setStyleSheet("font-weight: bold;")
        r_layout.addWidget(self.pred_right)

        splitter.addWidget(right)
        splitter.setSizes([480, 720])

        sb = QStatusBar()
        self.setStatusBar(sb)
        sb.showMessage(f"Model: {os.path.basename(model_path)} | "
                       f"Held-out: {os.path.basename(json_path) if os.path.exists(json_path) else '(none)'}")

        self._populate_list()

        self.listbox.currentRowChanged.connect(self.on_select)
        self.btn_load_json.clicked.connect(self.on_load_json)
        self.btn_predict_selected.clicked.connect(self.on_predict_selected)
        self.btn_fill.clicked.connect(self.on_fill_from_selected)
        self.btn_clear.clicked.connect(self.on_clear_manual)
        self.btn_predict_manual.clicked.connect(self.on_predict_manual)


    def _populate_list(self):
        self.listbox.clear()
        samples = self.heldout.get("heldout_samples", [])
        for i, rec in enumerate(samples):
            rid = rec.get("id", i)
            true_name = rec.get("true_label_name", str(rec.get("true_label", "?")))
            pred_name = rec.get("pred_label_name", "?")
            p_b = rec.get("prob_benign", None)
            if p_b is not None:
                text = f"#{rid} | true={true_name} "
            else:
                text = f"#{rid} | true={true_name}"
            self.listbox.addItem(text)

    def _selected_record(self):
        row = self.listbox.currentRow()
        if row < 0:
            return None
        return self.heldout.get("heldout_samples", [])[row]

    def _features_to_vector(self, feats_dict: dict[str, float]) -> np.ndarray:
        try:
            return np.array([feats_dict[name] for name in self.feature_names], dtype=np.float64).reshape(1, -1)
        except KeyError as e:
            raise ValueError(f"Missing feature: {e}")


    def on_load_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open held-out JSON", "", "JSON (*.json);;All files (*)")
        if not path:
            return
        try:
            self.heldout = load_heldout_json(path)
            self.json_path = path
            self._populate_list()
            self.info_left.setText("Loaded held-out JSON. Select a sample.")
            self.table.setRowCount(0)
            self.pred_left.setText("")
            self.statusBar().showMessage(f"Model: {os.path.basename(sys.argv[1]) if len(sys.argv)>1 else 'tumor_model.joblib'} | Held-out: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Load error", f"Failed to load JSON:\n{e}")

    def on_select(self, row: int):
        rec = self._selected_record()
        if not rec:
            return
        feats = rec.get("features", {})
        self.table.setRowCount(0)
        for i, name in enumerate(self.feature_names):
            self.table.insertRow(i)
            self.table.setItem(i, 0, QTableWidgetItem(name))
            val = feats.get(name, None)
            self.table.setItem(i, 1, QTableWidgetItem("—" if val is None else f"{val:.6f}"))
        true_name = rec.get("true_label_name", str(rec.get("true_label", "?")))
        rid = rec.get("id", row)
        self.info_left.setText(f"Sample #{rid} | true={true_name}")
        self.pred_left.setText("")

    def on_predict_selected(self):
        rec = self._selected_record()
        if not rec:
            QMessageBox.information(self, "Select", "Please select a held-out sample first.")
            return
        feats = rec.get("features", {})
        try:
            x = self._features_to_vector(feats)
            probs = self.model.predict_proba(x)[0]  # [P(malignant), P(benign)]
            pred = int(probs[1] >= 0.5)
            label = self.target_names[pred]
            self.pred_left.setText(f"Predicted: {label} | P[benign]={probs[1]:.4f}  P[malignant]={probs[0]:.4f}")
        except Exception as e:
            QMessageBox.critical(self, "Predict error", f"Failed to predict:\n{e}\n\n{traceback.format_exc()}")

    def on_fill_from_selected(self):
        rec = self._selected_record()
        if not rec:
            QMessageBox.information(self, "Select", "Please select a held-out sample first.")
            return
        feats = rec.get("features", {})
        for i, name in enumerate(self.feature_names):
            self.inputs[i].setText("" if name not in feats else str(feats[name]))

    def on_clear_manual(self):
        for line in self.inputs:
            line.clear()
        self.pred_right.setText("")

    def on_predict_manual(self):
        try:
            vals = []
            for i, name in enumerate(self.feature_names):
                txt = self.inputs[i].text().strip()
                if txt == "":
                    raise ValueError(f"Missing value for '{name}'")
                vals.append(float(txt))
            x = np.array(vals, dtype=np.float64).reshape(1, -1)
            probs = self.model.predict_proba(x)[0]  # [P(malignant), P(benign)]
            pred = int(probs[1] >= 0.5)
            label = self.target_names[pred]
            self.pred_right.setText(f"Predicted: {label} | P[benign]={probs[1]:.4f}  P[malignant]={probs[0]:.4f}")
        except Exception as e:
            QMessageBox.critical(self, "Predict error", f"Failed to predict from manual inputs:\n{e}")

def main():
    model_path = "tumor_model.joblib"
    json_path = "heldout_test.json"
    if len(sys.argv) >= 2:
        model_path = sys.argv[1]
    if len(sys.argv) >= 3:
        json_path = sys.argv[2]

    app = QApplication(sys.argv)
    w = MainWindow(model_path=model_path, json_path=json_path)
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
