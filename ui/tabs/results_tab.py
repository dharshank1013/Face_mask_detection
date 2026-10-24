"""Results tab for inspecting model evaluation metrics, confusion matrix, and exporting reports."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QGroupBox, QGridLayout, QTextEdit,
    QScrollArea, QFrame, QFileDialog, QMessageBox
)


class ResultsTab(QWidget):
    """Results viewer displaying test accuracy, confusion matrix, and export utilities."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.refresh_results()

    def init_ui(self) -> None:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(18)

        # Title
        title_label = QLabel("Evaluation Metrics & Confusion Matrix")
        title_label.setStyleSheet("font-size: 22px; font-weight: 800; color: #38bdf8;")
        layout.addWidget(title_label)

        # 1. Top Metrics Cards
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(14)

        self.acc_card = self._create_stat_card("Test Accuracy", "--%", "#10b981")
        self.loss_card = self._create_stat_card("Test Loss", "--", "#38bdf8")
        self.samples_card = self._create_stat_card("Test Samples", "--", "#f59e0b")

        metrics_grid.addWidget(self.acc_card, 0, 0)
        metrics_grid.addWidget(self.loss_card, 0, 1)
        metrics_grid.addWidget(self.samples_card, 0, 2)
        layout.addLayout(metrics_grid)

        # 2. Main Content: Confusion Matrix & Classification Report
        content_grid = QGridLayout()
        content_grid.setSpacing(16)

        # Left: Confusion Matrix Heatmap
        cm_box = QGroupBox("Confusion Matrix Heatmap")
        cm_l = QVBoxLayout(cm_box)
        self.cm_image_label = QLabel("Confusion matrix plot will appear after evaluation.")
        self.cm_image_label.setAlignment(Qt.AlignCenter)
        self.cm_image_label.setFixedSize(380, 320)
        self.cm_image_label.setStyleSheet("border: 1px dashed #334155; border-radius: 6px; background-color: #0f172a;")
        cm_l.addWidget(self.cm_image_label)
        content_grid.addWidget(cm_box, 0, 0)

        # Right: Classification Report (Precision, Recall, F1)
        rep_box = QGroupBox("Classification Report (Precision, Recall, F1)")
        rep_l = QVBoxLayout(rep_box)
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setFixedHeight(320)
        rep_l.addWidget(self.report_text)
        content_grid.addWidget(rep_box, 0, 1)

        layout.addLayout(content_grid)

        # 3. Actions Bar
        actions_card = QFrame()
        actions_card.setStyleSheet("background-color: #1e293b; border-radius: 8px; padding: 12px;")
        act_l = QHBoxLayout(actions_card)
        act_l.setSpacing(12)

        refresh_btn = QPushButton("Refresh Metrics")
        refresh_btn.setObjectName("btn-secondary")
        refresh_btn.clicked.connect(self.refresh_results)

        open_folder_btn = QPushButton("Open Outputs Folder")
        open_folder_btn.clicked.connect(self.open_output_folder)

        export_btn = QPushButton("Export Summary (CSV / Text)")
        export_btn.setObjectName("btn-success")
        export_btn.clicked.connect(self.export_summary)

        act_l.addWidget(refresh_btn)
        act_l.addWidget(open_folder_btn)
        act_l.addWidget(export_btn)
        act_l.addStretch()

        layout.addWidget(actions_card)
        layout.addStretch()

        scroll.setWidget(container)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

    def _create_stat_card(self, title: str, default_val: str, color: str) -> QFrame:
        card = QFrame()
        card.setStyleSheet("background-color: #1e293b; border-radius: 8px; padding: 14px;")
        l = QVBoxLayout(card)
        t = QLabel(title)
        t.setStyleSheet("font-size: 12px; color: #94a3b8; font-weight: bold;")
        v = QLabel(default_val)
        v.setStyleSheet(f"font-size: 24px; font-weight: 800; color: {color}; margin-top: 4px;")
        v.setObjectName("value_label")
        l.addWidget(t)
        l.addWidget(v)
        return card

    def refresh_results(self) -> None:
        """Load metrics JSON, confusion matrix image, and classification report."""
        metrics_file = Path("outputs/metrics/test_metrics.json")
        cm_file = Path("outputs/plots/confusion_matrix.png")
        report_file = Path("outputs/metrics/classification_report.txt")

        # Load Metrics JSON
        if metrics_file.exists():
            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                acc = data.get("test_accuracy", 0.0) * 100.0
                loss = data.get("test_loss", 0.0)
                n_samples = data.get("total_samples", 0)

                self.acc_card.findChild(QLabel, "value_label").setText(f"{acc:.2f}%")
                self.loss_card.findChild(QLabel, "value_label").setText(f"{loss:.4f}")
                self.samples_card.findChild(QLabel, "value_label").setText(f"{n_samples}")
            except Exception as e:
                pass

        # Load Confusion Matrix
        if cm_file.exists():
            pix = QPixmap(str(cm_file))
            if not pix.isNull():
                scaled = pix.scaled(370, 310, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.cm_image_label.setPixmap(scaled)

        # Load Report Text
        if report_file.exists():
            try:
                with open(report_file, "r", encoding="utf-8") as f:
                    self.report_text.setPlainText(f.read())
            except Exception:
                pass
        else:
            self.report_text.setPlainText("No evaluation report generated yet.\nRun training or evaluation to compute metrics.")

    def open_output_folder(self) -> None:
        """Open outputs directory in Windows File Explorer."""
        out_dir = Path("outputs").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(out_dir))
        except Exception as e:
            QMessageBox.warning(self, "Open Folder Error", f"Could not open directory: {e}")

    def export_summary(self) -> None:
        """Export combined metrics report to chosen destination."""
        dest_file, _ = QFileDialog.getSaveFileName(
            self,
            "Export Summary Report",
            "outputs/exports/evaluation_summary.txt",
            "Text Files (*.txt);;All Files (*.*)",
        )
        if not dest_file:
            return

        report_content = self.report_text.toPlainText()
        acc = self.acc_card.findChild(QLabel, "value_label").text()
        loss = self.loss_card.findChild(QLabel, "value_label").text()
        samples = self.samples_card.findChild(QLabel, "value_label").text()

        summary_text = (
            "=====================================================\n"
            "   FACE MASK DETECTION - EVALUATION SUMMARY REPORT\n"
            "=====================================================\n"
            f"Test Accuracy : {acc}\n"
            f"Test Loss     : {loss}\n"
            f"Total Samples : {samples}\n"
            "-----------------------------------------------------\n"
            "Classification Metrics Breakdown:\n\n"
            f"{report_content}\n"
            "=====================================================\n"
        )

        try:
            Path(dest_file).parent.mkdir(parents=True, exist_ok=True)
            with open(dest_file, "w", encoding="utf-8") as f:
                f.write(summary_text)
            QMessageBox.information(self, "Export Successful", f"Saved summary report to:\n{dest_file}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Could not save file: {e}")
