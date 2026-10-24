"""Live camera tab for real-time Face Mask Detection with OpenCV and PySide6."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QGroupBox, QComboBox, QSlider,
    QCheckBox, QFrame, QMessageBox
)

import cv2
from inference import MaskInferencePipeline
from ui.threads import CameraWorker

logger = logging.getLogger(__name__)


class CameraTab(QWidget):
    """Real-time webcam inference view with bounding boxes, live FPS, and controls."""

    def __init__(self, pipeline: MaskInferencePipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.worker: Optional[CameraWorker] = None
        self.init_ui()

    def scan_available_cameras(self) -> None:
        """Scan available camera devices (indices 0 to 3) and auto-select active color camera."""
        current_selection = self.cam_combo.currentData() if self.cam_combo.count() > 0 else None
        self.cam_combo.clear()

        found = []
        best_active_idx = None

        for i in range(4):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(i)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                ret, frame = cap.read()
                cap.release()

                is_black = (ret and frame is not None and frame.mean() < 2.0)
                if not is_black and best_active_idx is None:
                    best_active_idx = i

                status_tag = " [Black / Privacy Shutter]" if is_black else " [Active Webcam]"
                label = f"Camera {i} ({w}x{h}){status_tag}"
                found.append((i, label, is_black))

        if not found:
            found.append((0, "Camera 0 (Default)", False))

        for idx, name, _ in found:
            self.cam_combo.addItem(name, userData=idx)

        # Prefer previous user selection, otherwise auto-select the active webcam
        target_idx = current_selection if current_selection is not None else best_active_idx
        if target_idx is not None:
            for i in range(self.cam_combo.count()):
                if self.cam_combo.itemData(i) == target_idx:
                    self.cam_combo.setCurrentIndex(i)
                    break

    def init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 16, 20, 16)
        main_layout.setSpacing(12)

        # 1. Top Controls Bar
        ctrl_card = QFrame()
        ctrl_card.setStyleSheet("background-color: #1e293b; border-radius: 8px; padding: 10px;")
        ctrl_layout = QHBoxLayout(ctrl_card)
        ctrl_layout.setSpacing(14)

        # Camera Index Dropdown & Detect Button
        ctrl_layout.addWidget(QLabel("Camera:"))
        self.cam_combo = QComboBox()
        self.scan_available_cameras()
        ctrl_layout.addWidget(self.cam_combo)

        self.detect_cam_btn = QPushButton("Detect")
        self.detect_cam_btn.setToolTip("Scan system for connected webcams")
        self.detect_cam_btn.setFixedWidth(65)
        self.detect_cam_btn.clicked.connect(self.scan_available_cameras)
        ctrl_layout.addWidget(self.detect_cam_btn)

        # Start / Stop Buttons
        self.start_btn = QPushButton("Start Camera")
        self.start_btn.setObjectName("btn-success")
        self.start_btn.setFixedHeight(36)
        self.start_btn.clicked.connect(self.start_camera)

        self.stop_btn = QPushButton("Stop Camera")
        self.stop_btn.setObjectName("btn-danger")
        self.stop_btn.setFixedHeight(36)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_camera)

        ctrl_layout.addWidget(self.start_btn)
        ctrl_layout.addWidget(self.stop_btn)

        # Confidence Slider
        ctrl_layout.addWidget(QLabel("Confidence:"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 95)
        self.conf_slider.setValue(int(self.pipeline.confidence_threshold * 100))
        self.conf_slider.setFixedWidth(120)
        self.conf_slider.valueChanged.connect(self.on_conf_changed)

        self.conf_label = QLabel(f"{int(self.pipeline.confidence_threshold * 100)}%")
        self.conf_label.setStyleSheet("font-weight: bold; color: #38bdf8; min-width: 35px;")

        ctrl_layout.addWidget(self.conf_slider)
        ctrl_layout.addWidget(self.conf_label)

        # Privacy Logging Checkbox
        self.log_check = QCheckBox("Privacy-friendly Event Logging")
        self.log_check.setToolTip("When enabled, appends [Timestamp, Class, Confidence] to outputs/logs/ without saving any images.")
        self.log_check.stateChanged.connect(self.on_logging_toggled)
        ctrl_layout.addWidget(self.log_check)

        ctrl_layout.addStretch()
        main_layout.addWidget(ctrl_card)

        # 2. Main Live Video Preview
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet(
            "background-color: #090d16; border: 2px solid #334155; border-radius: 8px; color: #64748b; font-size: 16px;"
        )
        self.video_label.setText("Camera is currently stopped.\nClick 'Start Camera' to begin live detection.")
        main_layout.addWidget(self.video_label, 1)

        # 3. Status Footer Bar
        footer = QFrame()
        footer.setStyleSheet("background-color: #1e293b; border-radius: 6px; padding: 8px 14px;")
        f_layout = QHBoxLayout(footer)

        self.model_status_lbl = QLabel(self._get_model_status_text())
        self.model_status_lbl.setStyleSheet("font-weight: bold; font-size: 12px;")

        self.fps_lbl = QLabel("Live FPS: 0.0")
        self.fps_lbl.setStyleSheet("font-weight: bold; color: #06b6d4; font-size: 12px; margin-left: 20px;")

        self.faces_lbl = QLabel("Faces: 0")
        self.faces_lbl.setStyleSheet("font-weight: bold; color: #38bdf8; font-size: 12px; margin-left: 15px;")

        # Legend badges
        legend_layout = QHBoxLayout()
        legend_layout.setSpacing(8)

        leg_green = QLabel("● With Mask")
        leg_green.setStyleSheet("color: #10b981; font-weight: bold; font-size: 11px;")
        leg_orange = QLabel("● Incorrect Mask")
        leg_orange.setStyleSheet("color: #f59e0b; font-weight: bold; font-size: 11px;")
        leg_red = QLabel("● Without Mask")
        leg_red.setStyleSheet("color: #ef4444; font-weight: bold; font-size: 11px;")

        legend_layout.addWidget(leg_green)
        legend_layout.addWidget(leg_orange)
        legend_layout.addWidget(leg_red)

        f_layout.addWidget(self.model_status_lbl)
        f_layout.addWidget(self.fps_lbl)
        f_layout.addWidget(self.faces_lbl)
        f_layout.addStretch()
        f_layout.addLayout(legend_layout)

        main_layout.addWidget(footer)

    def _get_model_status_text(self) -> str:
        if self.pipeline.is_model_loaded:
            return "● Model: Active (MobileNetV2)"
        else:
            return "▲ Model: Not Loaded (Face detection only)"

    def update_model_status(self) -> None:
        if self.pipeline.is_model_loaded:
            self.model_status_lbl.setText("● Model: Active (MobileNetV2)")
            self.model_status_lbl.setStyleSheet("color: #10b981; font-weight: bold; font-size: 12px;")
        else:
            self.model_status_lbl.setText("▲ Model: Not Loaded (Train or select a model first)")
            self.model_status_lbl.setStyleSheet("color: #f59e0b; font-weight: bold; font-size: 12px;")

    def on_conf_changed(self, val: int) -> None:
        conf = val / 100.0
        self.conf_label.setText(f"{val}%")
        self.pipeline.confidence_threshold = conf

    def on_logging_toggled(self, state: int) -> None:
        enabled = (state == Qt.Checked or state == 2)
        self.pipeline.enable_logging = enabled

    def start_camera(self) -> None:
        """Start background camera thread."""
        cam_idx = self.cam_combo.currentData()
        if cam_idx is None:
            cam_idx = self.cam_combo.currentIndex()
            if cam_idx < 0:
                cam_idx = 0

        self.update_model_status()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.cam_combo.setEnabled(False)
        self.detect_cam_btn.setEnabled(False)

        self.worker = CameraWorker(pipeline=self.pipeline, camera_index=cam_idx, parent=self)
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.camera_error.connect(self.on_camera_error)
        self.worker.start()

    def stop_camera(self) -> None:
        """Stop background camera thread."""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker = None

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.cam_combo.setEnabled(True)
        self.detect_cam_btn.setEnabled(True)
        self.video_label.setText("Camera is currently stopped.")
        self.fps_lbl.setText("Live FPS: 0.0")
        self.faces_lbl.setText("Faces: 0")

    def on_frame_ready(self, q_img: QImage, fps: float, face_count: int) -> None:
        """Render frame on video label."""
        if not q_img.isNull():
            pix = QPixmap.fromImage(q_img)
            scaled = pix.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.video_label.setPixmap(scaled)
            self.fps_lbl.setText(f"Live FPS: {fps:.1f}")
            self.faces_lbl.setText(f"Faces: {face_count}")

    def on_camera_error(self, err_msg: str) -> None:
        self.stop_camera()
        QMessageBox.warning(self, "Camera Error", err_msg)
