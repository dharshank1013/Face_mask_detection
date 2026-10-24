"""Training tab for configuring hyperparameters, launching background training, and monitoring progress."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QGroupBox, QGridLayout, QSpinBox,
    QDoubleSpinBox, QComboBox, QLineEdit, QProgressBar,
    QTextEdit, QScrollArea, QFrame, QMessageBox
)

from verify_hardware import get_device, get_hardware_info
from ui.threads import TrainingWorker


class TrainingTab(QWidget):
    """Training management interface with background worker thread and live metrics."""

    model_trained = Signal(str)  # Emits path to newly trained checkpoint

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker: Optional[TrainingWorker] = None
        self.init_ui()
        self.refresh_checkpoints()

    def init_ui(self) -> None:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(18)

        # Title
        title_label = QLabel("MobileNetV2 Transfer Learning & Training")
        title_label.setStyleSheet("font-size: 22px; font-weight: 800; color: #38bdf8;")
        layout.addWidget(title_label)

        # 1. Config and Hyperparameters Group
        cfg_group = QGroupBox("Training Configuration & Hyperparameters")
        cfg_layout = QGridLayout(cfg_group)
        cfg_layout.setSpacing(12)

        # Epochs
        cfg_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100)
        self.epochs_spin.setValue(10)
        cfg_layout.addWidget(self.epochs_spin, 0, 1)

        # Batch Size
        cfg_layout.addWidget(QLabel("Batch Size:"), 0, 2)
        self.batch_combo = QComboBox()
        self.batch_combo.addItems(["16", "32", "64"])
        self.batch_combo.setCurrentText("32")
        cfg_layout.addWidget(self.batch_combo, 0, 3)

        # Learning Rate
        cfg_layout.addWidget(QLabel("Learning Rate:"), 1, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(5)
        self.lr_spin.setRange(0.00001, 0.05)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.0003)
        cfg_layout.addWidget(self.lr_spin, 1, 1)

        # Model Save Name
        cfg_layout.addWidget(QLabel("Checkpoint Filename:"), 1, 2)
        self.model_name_edit = QLineEdit("mobilenet_v2_mask_best.pt")
        cfg_layout.addWidget(self.model_name_edit, 1, 3)

        # Existing Checkpoint Selector (for inspecting / resuming)
        cfg_layout.addWidget(QLabel("Existing Checkpoint:"), 2, 0)
        self.checkpoint_combo = QComboBox()
        cfg_layout.addWidget(self.checkpoint_combo, 2, 1, 1, 2)

        refresh_chk_btn = QPushButton("Refresh List")
        refresh_chk_btn.setObjectName("btn-secondary")
        refresh_chk_btn.clicked.connect(self.refresh_checkpoints)
        cfg_layout.addWidget(refresh_chk_btn, 2, 3)

        layout.addWidget(cfg_group)

        # 2. Control Buttons & Hardware Status
        ctrl_card = QFrame()
        ctrl_card.setStyleSheet("background-color: #1e293b; border-radius: 8px; padding: 12px;")
        ctrl_layout = QHBoxLayout(ctrl_card)

        self.start_btn = QPushButton("Start Training")
        self.start_btn.setObjectName("btn-success")
        self.start_btn.setFixedHeight(40)
        self.start_btn.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.start_btn.clicked.connect(self.start_training)

        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.setObjectName("btn-danger")
        self.stop_btn.setFixedHeight(40)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.stop_btn.clicked.connect(self.stop_training)

        self.device_indicator = QLabel("Target Device: DETECTING...")
        self.device_indicator.setStyleSheet("font-weight: bold; color: #38bdf8; margin-left: 10px;")

        ctrl_layout.addWidget(self.start_btn)
        ctrl_layout.addWidget(self.stop_btn)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.device_indicator)
        layout.addWidget(ctrl_card)

        # 3. Live Progress and Metrics Monitor
        prog_group = QGroupBox("Live Training Progress & Epoch Metrics")
        prog_layout = QVBoxLayout(prog_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        prog_layout.addWidget(self.progress_bar)

        # Live metric badges row
        metric_row = QHBoxLayout()
        self.lbl_epoch = self._create_metric_chip("Current Epoch: --/--", "#38bdf8")
        self.lbl_train_loss = self._create_metric_chip("Train Loss: --", "#94a3b8")
        self.lbl_train_acc = self._create_metric_chip("Train Acc: --", "#10b981")
        self.lbl_val_loss = self._create_metric_chip("Val Loss: --", "#f59e0b")
        self.lbl_val_acc = self._create_metric_chip("Val Acc: --", "#10b981")

        metric_row.addWidget(self.lbl_epoch)
        metric_row.addWidget(self.lbl_train_loss)
        metric_row.addWidget(self.lbl_train_acc)
        metric_row.addWidget(self.lbl_val_loss)
        metric_row.addWidget(self.lbl_val_acc)
        prog_layout.addLayout(metric_row)

        layout.addWidget(prog_group)

        # 4. Training Console Log & Result Chart
        bottom_grid = QGridLayout()
        bottom_grid.setSpacing(14)

        # Console
        console_box = QGroupBox("Training Console Output")
        con_l = QVBoxLayout(console_box)
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setFixedHeight(230)
        con_l.addWidget(self.log_console)
        bottom_grid.addWidget(console_box, 0, 0)

        # Curve Chart Preview
        chart_box = QGroupBox("Loss & Accuracy Curves")
        chart_l = QVBoxLayout(chart_box)
        self.chart_label = QLabel("Curves will appear here after training.")
        self.chart_label.setAlignment(Qt.AlignCenter)
        self.chart_label.setFixedHeight(230)
        self.chart_label.setStyleSheet("border: 1px dashed #334155; border-radius: 6px; color: #94a3b8;")
        chart_l.addWidget(self.chart_label)
        bottom_grid.addWidget(chart_box, 0, 1)

        layout.addLayout(bottom_grid)
        layout.addStretch()

        scroll.setWidget(container)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

        self.update_device_indicator()
        self.load_existing_curve()

    def _create_metric_chip(self, text: str, color: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"background-color: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 6px 12px; font-weight: bold; color: {color}; font-size: 12px;")
        lbl.setAlignment(Qt.AlignCenter)
        return lbl

    def update_device_indicator(self) -> None:
        device = get_device()
        hw = get_hardware_info()
        dev_type = device.type.upper()
        if dev_type == "XPU":
            self.device_indicator.setText(f"Target Device: INTEL ARC GPU (XPU: {hw['xpu_device_name']})")
            self.device_indicator.setStyleSheet("font-weight: bold; color: #10b981;")
        elif dev_type == "CUDA":
            self.device_indicator.setText(f"Target Device: NVIDIA CUDA ({hw['cuda_device_name']})")
            self.device_indicator.setStyleSheet("font-weight: bold; color: #10b981;")
        else:
            self.device_indicator.setText("Target Device: CPU Fallback")
            self.device_indicator.setStyleSheet("font-weight: bold; color: #38bdf8;")

    def refresh_checkpoints(self) -> None:
        chk_dir = Path("checkpoints")
        self.checkpoint_combo.clear()
        if chk_dir.exists():
            files = list(chk_dir.glob("*.pt"))
            if files:
                for f in sorted(files, key=lambda x: x.name):
                    self.checkpoint_combo.addItem(f.name, str(f))
            else:
                self.checkpoint_combo.addItem("No checkpoints found", "")
        else:
            self.checkpoint_combo.addItem("No checkpoints found", "")

    def load_existing_curve(self) -> None:
        plot_path = Path("outputs/plots/loss_accuracy_curve.png")
        if plot_path.exists():
            pix = QPixmap(str(plot_path))
            if not pix.isNull():
                scaled = pix.scaled(480, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.chart_label.setPixmap(scaled)

    def start_training(self) -> None:
        """Launch background worker thread."""
        epochs = self.epochs_spin.value()
        batch_size = int(self.batch_combo.currentText())
        lr = self.lr_spin.value()
        save_name = self.model_name_edit.text().strip() or "mobilenet_v2_mask_best.pt"

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_console.clear()
        self.log_console.append(f">>> Initializing training: {epochs} epochs, batch size {batch_size}, LR {lr}...")

        self.worker = TrainingWorker(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            model_save_name=save_name,
            parent=self,
        )
        self.worker.progress_updated.connect(self.on_progress_update)
        self.worker.log_received.connect(self.on_log_message)
        self.worker.training_finished.connect(self.on_training_finished)
        self.worker.training_failed.connect(self.on_training_failed)
        self.worker.start()

    def stop_training(self) -> None:
        if self.worker and self.worker.isRunning():
            self.log_console.append(">>> Aborting training... waiting for current epoch to stop safely.")
            self.worker.stop()
            self.stop_btn.setEnabled(False)

    def on_progress_update(self, data: dict) -> None:
        epoch = data["epoch"]
        total = data["total_epochs"]
        percent = int((epoch / total) * 100)
        self.progress_bar.setValue(percent)

        self.lbl_epoch.setText(f"Current Epoch: {epoch}/{total}")
        self.lbl_train_loss.setText(f"Train Loss: {data['train_loss']:.4f}")
        self.lbl_train_acc.setText(f"Train Acc: {data['train_acc'] * 100:.1f}%")
        self.lbl_val_loss.setText(f"Val Loss: {data['val_loss']:.4f}")
        self.lbl_val_acc.setText(f"Val Acc: {data['val_acc'] * 100:.1f}%")

    def on_log_message(self, msg: str) -> None:
        self.log_console.append(msg)

    def on_training_finished(self, results: dict) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        self.log_console.append("\n[✓] Training complete!")
        best_path = results.get("best_checkpoint_path", "")
        self.refresh_checkpoints()
        self.load_existing_curve()

        if best_path:
            self.model_trained.emit(best_path)

        QMessageBox.information(
            self,
            "Training Complete",
            f"Training finished successfully!\n\n"
            f"Final Test Accuracy: {results.get('test_results', {}).get('accuracy', 0) * 100:.2f}%\n"
            f"Saved Best Checkpoint: {best_path}"
        )

    def on_training_failed(self, err_msg: str) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_console.append(f"\n[!] Training Failed: {err_msg}")
        QMessageBox.critical(self, "Training Error", f"An error occurred during training:\n{err_msg}")
