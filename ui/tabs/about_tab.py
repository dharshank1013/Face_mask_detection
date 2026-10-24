"""About tab explaining architecture, system design, limitations, and educational scope."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox,
    QScrollArea, QFrame
)


class AboutTab(QWidget):
    """About & Documentation view providing student-friendly explanations and limitations."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self) -> None:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(18)

        # Title
        title_label = QLabel("About Face Mask Detection System")
        title_label.setStyleSheet("font-size: 22px; font-weight: 800; color: #38bdf8;")
        layout.addWidget(title_label)

        # 1. Pipeline Explanation Card
        pipe_box = QGroupBox("System Pipeline & Working Principle")
        pipe_l = QVBoxLayout(pipe_box)
        pipe_desc = QLabel(
            "<b>How it works:</b><br><br>"
            "1. <b>Stage 1 - Face Localization:</b> OpenCV scans each incoming webcam frame and locates all human faces in real time.<br>"
            "2. <b>Stage 2 - Crop & Preprocessing:</b> Each detected face bounding box is extracted with a slight contextual margin, "
            "converted to RGB, resized to 224 × 224 pixels, and normalized using standard ImageNet statistics.<br>"
            "3. <b>Stage 3 - MobileNetV2 Classification:</b> The lightweight, highly optimized MobileNetV2 neural network inspects the face crop and outputs class probabilities.<br>"
            "4. <b>Stage 4 - Smoothing & Overlay:</b> Softmax confidence scores are smoothed across recent frames to eliminate flicker. "
            "Bounding boxes and badges are drawn dynamically in real time:<br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;• <font color='#10b981'><b>Green</b></font>: With Mask<br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;• <font color='#ef4444'><b>Red</b></font>: Without Mask<br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;• <font color='#f59e0b'><b>Orange</b></font>: Incorrect Mask (e.g. worn below nose or mouth)"
        )
        pipe_desc.setStyleSheet("font-size: 13px; color: #e2e8f0; line-height: 1.6;")
        pipe_desc.setWordWrap(True)
        pipe_l.addWidget(pipe_desc)
        layout.addWidget(pipe_box)

        # 2. Limitations Card
        lim_box = QGroupBox("Known Limitations & Environmental Factors")
        lim_l = QVBoxLayout(lim_box)
        lim_desc = QLabel(
            "Like any computer vision system, performance depends on real-world capture conditions:<br><br>"
            "• <b>Lighting Conditions:</b> Extreme shadows, backlighting, or severe underexposure may prevent OpenCV from detecting faces.<br>"
            "• <b>Head Poses:</b> Severe profile (side) views or steep tilted angles reduce face detector sensitivity.<br>"
            "• <b>Unusual Mask Styles:</b> Masks with complex graphics, face prints, scarves, or transparent shields may differ from training distributions.<br>"
            "• <b>Dataset Bias:</b> Neural networks inherit patterns from their training photos; accuracy can vary across camera resolutions and skin tones."
        )
        lim_desc.setStyleSheet("font-size: 13px; color: #cbd5e1; line-height: 1.5;")
        lim_desc.setWordWrap(True)
        lim_l.addWidget(lim_desc)
        layout.addWidget(lim_box)

        # 3. Educational & Privacy Disclaimer
        disc_box = QGroupBox("Educational Scope & Privacy Disclaimer")
        disc_l = QVBoxLayout(disc_box)
        disc_desc = QLabel(
            "<b>Academic Learning Project:</b><br>"
            "This application is created solely for computer vision coursework, study, and educational demonstration.<br><br>"
            "<b>Strict Privacy Guarantee:</b><br>"
            "• No facial images, video recordings, or biometric templates are stored or transmitted.<br>"
            "• This system does <b>not</b> identify individuals or perform facial recognition.<br>"
            "• It is strictly not designed, intended, or authorized for surveillance, policing, or automated enforcement."
        )
        disc_desc.setStyleSheet("font-size: 13px; color: #94a3b8; line-height: 1.5;")
        disc_desc.setWordWrap(True)
        disc_l.addWidget(disc_desc)
        layout.addWidget(disc_box)

        layout.addStretch()

        scroll.setWidget(container)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)
