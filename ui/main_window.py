"""Main application window uniting all tabs and shared inference engine."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QVBoxLayout, QWidget,
    QStatusBar, QLabel
)

from verify_hardware import get_device, get_hardware_info
from inference import MaskInferencePipeline
from ui.styles import DARK_THEME_QSS
from ui.tabs.home_tab import HomeTab
from ui.tabs.dataset_tab import DatasetTab
from ui.tabs.training_tab import TrainingTab
from ui.tabs.camera_tab import CameraTab
from ui.tabs.results_tab import ResultsTab
from ui.tabs.about_tab import AboutTab

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main desktop window for Face Mask Detection System."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Mask Detection System - Intel Arc XPU & MobileNetV2")
        self.resize(1120, 800)
        self.setMinimumSize(950, 680)

        # Apply dark theme styling
        self.setStyleSheet(DARK_THEME_QSS)

        # Initialize shared inference pipeline
        self.device = get_device()
        self.pipeline = MaskInferencePipeline(device=self.device)

        # Attempt auto-loading existing checkpoint
        self._auto_load_existing_checkpoint()

        self.init_ui()
        self.init_status_bar()

    def _auto_load_existing_checkpoint(self) -> None:
        """Load best existing checkpoint if available."""
        chk_dir = Path("checkpoints")
        if chk_dir.exists():
            best_chk = chk_dir / "mobilenet_v2_mask_best.pt"
            if best_chk.exists():
                self.pipeline.load_model(best_chk)
            else:
                pts = list(chk_dir.glob("*.pt"))
                if pts:
                    self.pipeline.load_model(pts[0])

    def init_ui(self) -> None:
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(10, 10, 10, 10)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        # Instantiate tabs
        self.home_tab = HomeTab()
        self.dataset_tab = DatasetTab()
        self.training_tab = TrainingTab()
        self.camera_tab = CameraTab(pipeline=self.pipeline)
        self.results_tab = ResultsTab()
        self.about_tab = AboutTab()

        # Connect inter-tab signals
        self.home_tab.navigate_to_tab.connect(self.tabs.setCurrentIndex)
        self.training_tab.model_trained.connect(self.on_model_trained)
        self.tabs.currentChanged.connect(self.on_tab_changed)

        # Add tabs
        self.tabs.addTab(self.home_tab, " Home ")
        self.tabs.addTab(self.dataset_tab, " Dataset ")
        self.tabs.addTab(self.training_tab, " Training ")
        self.tabs.addTab(self.camera_tab, " Live Camera ")
        self.tabs.addTab(self.results_tab, " Results ")
        self.tabs.addTab(self.about_tab, " About ")

        layout.addWidget(self.tabs)
        self.setCentralWidget(central_widget)

    def init_status_bar(self) -> None:
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)

        hw = get_hardware_info()
        dev_label = f"Device: {hw['selected_device'].upper()}"
        if hw["selected_device"] == "xpu":
            dev_label += f" ({hw['xpu_device_name']})"

        lbl_device = QLabel(dev_label)
        lbl_device.setStyleSheet("color: #10b981; font-weight: bold; padding-right: 15px;")
        status_bar.addPermanentWidget(lbl_device)

        lbl_torch = QLabel(f"PyTorch: {hw['pytorch_version']}")
        lbl_torch.setStyleSheet("color: #94a3b8; padding-right: 10px;")
        status_bar.addPermanentWidget(lbl_torch)

        status_bar.showMessage("Ready. Select a tab to explore datasets, train, or start live webcam inference.")

    def on_model_trained(self, checkpoint_path: str) -> None:
        """Handle new model checkpoint training completion."""
        logger.info(f"Reloading active pipeline with newly trained model: {checkpoint_path}")
        self.pipeline.load_model(checkpoint_path)
        self.camera_tab.update_model_status()
        self.home_tab.refresh_status()
        self.results_tab.refresh_results()
        self.statusBar().showMessage(f"Active model updated to {Path(checkpoint_path).name}", 5000)

    def on_tab_changed(self, index: int) -> None:
        """Stop camera when switching away from the Camera tab for resource efficiency."""
        # Index 3 is Camera Tab
        if index != 3 and self.camera_tab.worker and self.camera_tab.worker.isRunning():
            self.camera_tab.stop_camera()
        elif index == 0:
            self.home_tab.refresh_status()
        elif index == 4:
            self.results_tab.refresh_results()

    def closeEvent(self, event) -> None:
        """Clean shutdown when closing application window."""
        if self.camera_tab.worker and self.camera_tab.worker.isRunning():
            self.camera_tab.stop_camera()
        if self.training_tab.worker and self.training_tab.worker.isRunning():
            self.training_tab.stop_training()
        event.accept()
