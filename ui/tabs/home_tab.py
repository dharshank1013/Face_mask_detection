"""Home tab for Face Mask Detection System."""

from __future__ import annotations

from pathlib import Path
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QGroupBox, QGridLayout, QFrame
)

from verify_hardware import get_hardware_info, get_device
from dataset_tools import validate_dataset1, analyze_dataset2


class HomeTab(QWidget):
    """Home overview page displaying pipeline summary, hardware, and model status."""

    navigate_to_tab = Signal(int)  # 0: Home, 1: Dataset, 2: Training, 3: Camera, 4: Results, 5: About

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.refresh_status()

    def init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(18)

        # Header Title Banner
        header_card = QFrame()
        header_card.setStyleSheet("background-color: #1e293b; border-radius: 10px; padding: 18px;")
        header_layout = QVBoxLayout(header_card)

        title_label = QLabel("Face Mask Detection System")
        title_label.setStyleSheet("font-size: 26px; font-weight: 800; color: #38bdf8;")
        header_layout.addWidget(title_label)

        desc_label = QLabel(
            "An end-to-end real-time computer vision desktop application powered by PyTorch, MobileNetV2, "
            "and OpenCV. Designed with Intel Arc GPU (XPU) hardware acceleration and safe CPU fallback."
        )
        desc_label.setStyleSheet("font-size: 14px; color: #94a3b8; line-height: 1.4;")
        desc_label.setWordWrap(True)
        header_layout.addWidget(desc_label)

        # Pipeline Flow Diagram Card
        flow_card = QFrame()
        flow_card.setStyleSheet("background-color: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 10px; margin-top: 10px;")
        flow_layout = QHBoxLayout(flow_card)
        flow_layout.setSpacing(10)

        steps = [
            ("1. Input", "Webcam Video Feed"),
            ("2. Detect", "OpenCV Face Detector"),
            ("3. Preprocess", "224x224 RGB Crop"),
            ("4. Classify", "MobileNetV2 (3-Class)"),
            ("5. Smooth & Render", "Overlays & Live FPS"),
        ]
        for step_title, step_desc in steps:
            step_box = QFrame()
            step_box.setStyleSheet("background-color: #1e293b; border-radius: 6px; padding: 6px 12px;")
            box_l = QVBoxLayout(step_box)
            box_l.setContentsMargins(4, 4, 4, 4)
            st = QLabel(step_title)
            st.setStyleSheet("font-weight: bold; color: #38bdf8; font-size: 11px;")
            sd = QLabel(step_desc)
            sd.setStyleSheet("color: #cbd5e1; font-size: 11px;")
            box_l.addWidget(st)
            box_l.addWidget(sd)
            flow_layout.addWidget(step_box)

        header_layout.addWidget(flow_card)
        main_layout.addWidget(header_card)

        # Status Cards Grid (Hardware, Model, Dataset)
        grid = QGridLayout()
        grid.setSpacing(16)

        # 1. Hardware Status Card
        hw_box = QGroupBox("Hardware & Compute Device")
        hw_layout = QVBoxLayout(hw_box)
        self.device_badge = QLabel("DETECTING...")
        self.device_badge.setStyleSheet("font-size: 16px; font-weight: bold; color: #10b981;")
        self.hw_details = QLabel("Gathering device specs...")
        self.hw_details.setStyleSheet("color: #94a3b8; font-size: 12px;")
        self.hw_details.setWordWrap(True)

        refresh_hw_btn = QPushButton("Re-verify Hardware")
        refresh_hw_btn.setObjectName("btn-secondary")
        refresh_hw_btn.clicked.connect(self.refresh_status)

        hw_layout.addWidget(self.device_badge)
        hw_layout.addWidget(self.hw_details)
        hw_layout.addStretch()
        hw_layout.addWidget(refresh_hw_btn)
        grid.addWidget(hw_box, 0, 0)

        # 2. Model Status Card
        model_box = QGroupBox("Active Classifier Model")
        model_layout = QVBoxLayout(model_box)
        self.model_status_badge = QLabel("CHECKING...")
        self.model_status_badge.setStyleSheet("font-size: 15px; font-weight: bold; color: #f59e0b;")
        self.model_details = QLabel("Scanning checkpoints folder...")
        self.model_details.setStyleSheet("color: #94a3b8; font-size: 12px;")
        self.model_details.setWordWrap(True)

        train_nav_btn = QPushButton("Go to Training Page")
        train_nav_btn.clicked.connect(lambda: self.navigate_to_tab.emit(2))

        model_layout.addWidget(self.model_status_badge)
        model_layout.addWidget(self.model_details)
        model_layout.addStretch()
        model_layout.addWidget(train_nav_btn)
        grid.addWidget(model_box, 0, 1)

        # 3. Dataset Discovery Card
        ds_box = QGroupBox("Dataset Discovery")
        ds_layout = QVBoxLayout(ds_box)
        self.dataset_status_badge = QLabel("CHECKING...")
        self.dataset_status_badge.setStyleSheet("font-size: 15px; font-weight: bold; color: #38bdf8;")
        self.dataset_details = QLabel("Inspecting dataset 1 and dataset 2...")
        self.dataset_details.setStyleSheet("color: #94a3b8; font-size: 12px;")
        self.dataset_details.setWordWrap(True)

        ds_nav_btn = QPushButton("Inspect Datasets")
        ds_nav_btn.setObjectName("btn-secondary")
        ds_nav_btn.clicked.connect(lambda: self.navigate_to_tab.emit(1))

        ds_layout.addWidget(self.dataset_status_badge)
        ds_layout.addWidget(self.dataset_details)
        ds_layout.addStretch()
        ds_layout.addWidget(ds_nav_btn)
        grid.addWidget(ds_box, 0, 2)

        main_layout.addLayout(grid)

        # Quick Actions Bar
        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(14)

        open_cam_btn = QPushButton("Start Real-time Live Camera")
        open_cam_btn.setObjectName("btn-success")
        open_cam_btn.setFixedHeight(44)
        open_cam_btn.setStyleSheet("font-size: 14px; font-weight: bold;")
        open_cam_btn.clicked.connect(lambda: self.navigate_to_tab.emit(3))

        view_results_btn = QPushButton("View Model Metrics & Results")
        view_results_btn.setObjectName("btn-secondary")
        view_results_btn.setFixedHeight(44)
        view_results_btn.clicked.connect(lambda: self.navigate_to_tab.emit(4))

        actions_layout.addWidget(open_cam_btn)
        actions_layout.addWidget(view_results_btn)
        main_layout.addLayout(actions_layout)

        main_layout.addStretch()

    def refresh_status(self) -> None:
        """Query hardware, checkpoints, and datasets."""
        # 1. Hardware
        try:
            hw = get_hardware_info()
            dev_str = hw["selected_device"].upper()
            if dev_str == "XPU":
                self.device_badge.setText("Active Device: INTEL ARC GPU (XPU)")
                self.device_badge.setStyleSheet("font-size: 15px; font-weight: bold; color: #10b981;")
                self.hw_details.setText(
                    f"PyTorch: {hw['pytorch_version']}\n"
                    f"XPU Device: {hw['xpu_device_name']}\n"
                    f"Native PyTorch XPU acceleration is active."
                )
            elif dev_str == "CUDA":
                self.device_badge.setText("Active Device: NVIDIA GPU (CUDA)")
                self.device_badge.setStyleSheet("font-size: 15px; font-weight: bold; color: #10b981;")
                self.hw_details.setText(
                    f"PyTorch: {hw['pytorch_version']}\n"
                    f"CUDA Device: {hw['cuda_device_name']}"
                )
            else:
                self.device_badge.setText("Active Device: CPU Fallback")
                self.device_badge.setStyleSheet("font-size: 15px; font-weight: bold; color: #38bdf8;")
                self.hw_details.setText(
                    f"PyTorch: {hw['pytorch_version']}\n"
                    f"XPU: Not active | CUDA: False\n"
                    f"Universal CPU compute fallback is active."
                )
        except Exception as e:
            self.device_badge.setText("Hardware Detection Error")
            self.hw_details.setText(str(e))

        # 2. Checkpoints
        chk_dir = Path("checkpoints")
        checkpoints = list(chk_dir.glob("*.pt")) if chk_dir.exists() else []
        if checkpoints:
            best_chks = [c for c in checkpoints if "best" in c.name]
            active_chk = best_chks[0] if best_chks else checkpoints[0]
            self.model_status_badge.setText(f"Model Ready: {active_chk.name}")
            self.model_status_badge.setStyleSheet("font-size: 14px; font-weight: bold; color: #10b981;")
            self.model_details.setText(
                f"Location: {active_chk}\n"
                f"Total saved checkpoints: {len(checkpoints)}\n"
                f"Ready for real-time live webcam inference."
            )
        else:
            self.model_status_badge.setText("No Model Trained Yet")
            self.model_status_badge.setStyleSheet("font-size: 14px; font-weight: bold; color: #f59e0b;")
            self.model_details.setText(
                "No trained .pt checkpoint found in checkpoints/.\n"
                "Please go to the Training tab to train the MobileNetV2 classifier."
            )

        # 3. Dataset
        d1 = validate_dataset1("dataset 1", verify_images=False)
        d2 = analyze_dataset2("dataset 2")
        if d1["valid"]:
            self.dataset_status_badge.setText(f"Dataset 1 Ready ({d1['total_images']} images)")
            self.dataset_status_badge.setStyleSheet("font-size: 14px; font-weight: bold; color: #10b981;")
            d2_str = f"Dataset 2: {d2['total_xml_files']} XMLs, {d2['total_image_files']} images" if d2["exists"] else "Dataset 2 not detected"
            self.dataset_details.setText(
                f"Classes: Incorrect ({d1['classes']['mask_weared_incorrect']['count']}), "
                f"With Mask ({d1['classes']['with_mask']['count']}), "
                f"Without Mask ({d1['classes']['without_mask']['count']})\n"
                f"{d2_str}"
            )
        else:
            self.dataset_status_badge.setText("Dataset Issue Detected")
            self.dataset_status_badge.setStyleSheet("font-size: 14px; font-weight: bold; color: #ef4444;")
            self.dataset_details.setText("; ".join(d1.get("errors", ["Directory missing"])))
