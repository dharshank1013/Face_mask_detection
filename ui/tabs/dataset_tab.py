"""Dataset tab for viewing, inspecting, and validating Dataset 1 and Dataset 2."""

from __future__ import annotations

from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QGroupBox, QGridLayout, QTextEdit,
    QScrollArea, QFrame, QMessageBox
)

from dataset_tools import (
    validate_dataset1,
    analyze_dataset2,
    get_sample_images_dataset1,
    DISPLAY_NAME_MAP,
)


class DatasetTab(QWidget):
    """Dataset page presenting image counts, sample previews, and validation diagnostics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.load_dataset_stats()
        self.refresh_sample_previews()

    def init_ui(self) -> None:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(18)

        # Title
        title_label = QLabel("Dataset Exploration & Health Diagnostics")
        title_label.setStyleSheet("font-size: 22px; font-weight: 800; color: #38bdf8;")
        layout.addWidget(title_label)

        # 1. Summary Cards
        summary_grid = QGridLayout()
        summary_grid.setSpacing(14)

        # Dataset 1 card
        self.d1_card = QGroupBox("Dataset 1: Classifier Source")
        d1_layout = QVBoxLayout(self.d1_card)
        self.d1_label = QLabel("Loading counts...")
        self.d1_label.setStyleSheet("font-size: 13px; color: #f8fafc;")
        d1_layout.addWidget(self.d1_label)
        summary_grid.addWidget(self.d1_card, 0, 0)

        # Dataset 2 card
        self.d2_card = QGroupBox("Dataset 2: Detection Annotations")
        d2_layout = QVBoxLayout(self.d2_card)
        self.d2_label = QLabel("Loading counts...")
        self.d2_label.setStyleSheet("font-size: 13px; color: #f8fafc;")
        d2_layout.addWidget(self.d2_label)
        summary_grid.addWidget(self.d2_card, 0, 1)

        layout.addLayout(summary_grid)

        # 2. Preview Gallery
        preview_group = QGroupBox("Dataset 1 Class Image Previews")
        preview_layout = QVBoxLayout(preview_group)

        self.preview_grid = QGridLayout()
        self.preview_grid.setSpacing(16)

        self.preview_labels = {}
        self.caption_labels = {}

        classes = ["mask_weared_incorrect", "with_mask", "without_mask"]
        for col_idx, cls in enumerate(classes):
            cell_frame = QFrame()
            cell_frame.setStyleSheet("background-color: #0f172a; border-radius: 6px; padding: 10px;")
            cell_l = QVBoxLayout(cell_frame)
            cell_l.setAlignment(Qt.AlignCenter)

            img_label = QLabel()
            img_label.setFixedSize(160, 160)
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setStyleSheet("border: 1px solid #334155; border-radius: 4px; background-color: #000000;")

            caption = QLabel(DISPLAY_NAME_MAP.get(cls, cls))
            caption.setAlignment(Qt.AlignCenter)
            caption.setStyleSheet("font-weight: bold; color: #38bdf8; margin-top: 6px;")

            cell_l.addWidget(img_label)
            cell_l.addWidget(caption)

            self.preview_labels[cls] = img_label
            self.caption_labels[cls] = caption
            self.preview_grid.addWidget(cell_frame, 0, col_idx)

        preview_layout.addLayout(self.preview_grid)

        btn_row = QHBoxLayout()
        shuffle_btn = QPushButton("Shuffle Image Previews")
        shuffle_btn.setObjectName("btn-secondary")
        shuffle_btn.clicked.connect(self.refresh_sample_previews)
        btn_row.addWidget(shuffle_btn)
        btn_row.addStretch()
        preview_layout.addLayout(btn_row)

        layout.addWidget(preview_group)

        # 3. Validation and Diagnostics Section
        diag_group = QGroupBox("Dataset Validation & Diagnostics")
        diag_layout = QVBoxLayout(diag_group)

        btn_bar = QHBoxLayout()
        self.validate_btn = QPushButton("Run Deep Dataset Validation")
        self.validate_btn.clicked.connect(self.run_validation)

        self.summary_d2_btn = QPushButton("Generate Dataset 2 Annotation Summary")
        self.summary_d2_btn.setObjectName("btn-secondary")
        self.summary_d2_btn.clicked.connect(self.run_d2_summary)

        btn_bar.addWidget(self.validate_btn)
        btn_bar.addWidget(self.summary_d2_btn)
        btn_bar.addStretch()
        diag_layout.addLayout(btn_bar)

        self.diag_console = QTextEdit()
        self.diag_console.setReadOnly(True)
        self.diag_console.setFixedHeight(180)
        diag_layout.addWidget(self.diag_console)

        layout.addWidget(diag_group)
        layout.addStretch()

        scroll.setWidget(container)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

    def load_dataset_stats(self) -> None:
        """Load and display high-level dataset statistics."""
        d1 = validate_dataset1("dataset 1", verify_images=False)
        if d1["valid"]:
            c_inc = d1["classes"].get("mask_weared_incorrect", {}).get("count", 0)
            c_with = d1["classes"].get("with_mask", {}).get("count", 0)
            c_without = d1["classes"].get("without_mask", {}).get("count", 0)
            self.d1_label.setText(
                f"Total Images: <b>{d1['total_images']}</b><br>"
                f"• Incorrect Mask: {c_inc}<br>"
                f"• With Mask: {c_with}<br>"
                f"• Without Mask: {c_without}<br>"
                f"Status: <font color='#10b981'><b>Balanced & Ready</b></font>"
            )
        else:
            self.d1_label.setText(f"<font color='#ef4444'>Errors: {'; '.join(d1['errors'])}</font>")

        d2 = analyze_dataset2("dataset 2")
        if d2["exists"]:
            self.d2_label.setText(
                f"Total XML Files: <b>{d2['total_xml_files']}</b><br>"
                f"Total Images: <b>{d2['total_image_files']}</b><br>"
                f"Matched Pairs: <b>{d2['matched_pairs']}</b><br>"
                f"Annotated Objects: <b>{d2['total_objects']}</b><br>"
                f"Status: <font color='#10b981'><b>PASCAL VOC Available</b></font>"
            )
        else:
            self.d2_label.setText("Dataset 2 not found.")

    def refresh_sample_previews(self) -> None:
        """Fetch random image for each class and display pixmap."""
        samples = get_sample_images_dataset1("dataset 1")
        for cls, img_path in samples.items():
            lbl = self.preview_labels.get(cls)
            if lbl is None:
                continue
            if img_path and img_path.exists():
                pix = QPixmap(str(img_path))
                if not pix.isNull():
                    scaled = pix.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    lbl.setPixmap(scaled)
                    lbl.setToolTip(f"{img_path.name}\nSize: {pix.width()}x{pix.height()}")
                else:
                    lbl.setText("Corrupt")
            else:
                lbl.setText("Not found")

    def run_validation(self) -> None:
        """Execute comprehensive validation."""
        self.diag_console.clear()
        self.diag_console.append(">>> Starting Deep Dataset 1 Validation (verifying file integrity)...")
        res = validate_dataset1("dataset 1", verify_images=True, sample_verify_limit=100)

        self.diag_console.append(f"Base Directory   : {res['base_path']}")
        self.diag_console.append(f"Validation Result: {'PASSED' if res['valid'] else 'FAILED'}")
        self.diag_console.append(f"Total Valid Imgs : {res['total_images']}")
        self.diag_console.append(f"Balance Ratio    : {res['class_balance_ratio']:.2f}")

        for cls, info in res["classes"].items():
            self.diag_console.append(
                f"  - {info['display_name']} ({cls}): {info['count']} images (Status: {info['status']})"
            )

        if res["corrupt_files"]:
            self.diag_console.append(f"\n[!] Corrupt images detected: {len(res['corrupt_files'])}")
            for fpath, err in res["corrupt_files"][:5]:
                self.diag_console.append(f"    {fpath}: {err}")
        else:
            self.diag_console.append("\n[✓] No corrupt images found in verified sample.")

        if res["warnings"]:
            self.diag_console.append("\nWarnings:")
            for w in res["warnings"]:
                self.diag_console.append(f"  ! {w}")

        self.diag_console.append("\n>>> Validation Complete.")

    def run_d2_summary(self) -> None:
        """Parse Dataset 2 XML files and display detailed breakdown."""
        self.diag_console.clear()
        self.diag_console.append(">>> Parsing Dataset 2 Pascal VOC Annotations...")
        d2 = analyze_dataset2("dataset 2")
        self.diag_console.append(f"Annotations XMLs : {d2['total_xml_files']}")
        self.diag_console.append(f"Corresponding Imgs: {d2['total_image_files']}")
        self.diag_console.append(f"Matched Pairs    : {d2['matched_pairs']}")
        self.diag_console.append(f"Total Objects    : {d2['total_objects']}")
        self.diag_console.append("\nBounding Box Breakdown by Label:")
        for label, count in d2["object_counts"].items():
            self.diag_console.append(f"  • {label:22s} : {count} boxes")
        self.diag_console.append("\n>>> Dataset 2 Summary Complete.")
