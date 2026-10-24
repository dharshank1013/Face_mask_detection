# Real-Time Face Mask Detection System

[![CI Pipeline](https://github.com/dharshank1013/Face_mask_detection/actions/workflows/ci.yml/badge.svg)](https://github.com/dharshank1013/Face_mask_detection/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-MobileNetV2-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Intel%20Arc-XPU%20Accelerated-0071C5?logo=intel&logoColor=white)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html)
[![GUI](https://img.shields.io/badge/PySide6-Qt%20Desktop-41CD52?logo=qt&logoColor=white)](https://doc.qt.io/qtforpython/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A complete, production-grade desktop application for real-time Face Mask Detection and Classification built with **PyTorch**, **MobileNetV2**, **OpenCV**, and **PySide6**.

Engineered for native **Intel Arc GPU (XPU)** hardware acceleration with automatic, seamless fallback to CPU or NVIDIA CUDA.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Exact Dataset Folder Layout](#2-exact-dataset-folder-layout)
3. [Virtual Environment Setup (PowerShell)](#3-virtual-environment-setup-powershell)
4. [Installing Dependencies](#4-installing-dependencies)
5. [Hardware Verification & Intel Arc XPU Support](#5-hardware-verification--intel-arc-xpu-support)
6. [Running Dataset Validation](#6-running-dataset-validation)
7. [Training the Classifier](#7-training-the-classifier)
8. [Running the Desktop Application](#8-running-the-desktop-application)
9. [Using the Live Camera Page](#9-using-the-live-camera-page)
10. [Troubleshooting & Common Fixes](#10-troubleshooting--common-fixes)
11. [Metrics & Evaluation Guide](#11-metrics--evaluation-guide)

---

## 1. Project Overview

The system implements a robust two-stage real-time vision pipeline:

```text
Webcam Frame ──> OpenCV Face Detector ──> Face Crop (RGB) ──> 224x224 Resize ──> MobileNetV2 Classifier ──> Smoothed Overlay
```

### The 3 Target Classes:
- **With Mask**: Face mask properly worn covering both nose and mouth (Green bounding box).
- **Without Mask**: No face mask detected on the face (Red bounding box).
- **Incorrect Mask**: Face mask present but worn improperly, such as under the nose or below the chin (Orange bounding box).

### Key Technical Highlights:
- **Architecture**: MobileNetV2 with custom linear classifier head trained via ImageNet transfer learning.
- **Hardware Priority**: Strict `xpu` (Intel Arc) ➔ `cuda` (NVIDIA) ➔ `cpu` (Universal fallback).
- **Flicker-Free Real-Time Inference**: Rolling window prediction smoothing over temporal frames.
- **Privacy Guarantee**: Does not save facial imagery; optional privacy-friendly detection logs store only timestamp, label, and confidence.
- **Modern Native UI**: Dark-themed PySide6 desktop GUI with live training monitor, camera feed, and interactive metrics.

---

## 2. Exact Dataset Folder Layout

The project uses two local datasets located directly in `C:\Users\kdhar\Documents\FOCV`:

```text
C:\Users\kdhar\Documents\FOCV\
├── dataset 1\                      # Primary classifier training dataset (8,982 images)
│   ├── mask_weared_incorrect\      # 2,994 images
│   ├── with_mask\                  # 2,994 images
│   └── without_mask\               # 2,994 images
│
├── dataset 2\                      # Supplementary detection dataset (853 images + XMLs)
│   ├── annotations\                # 853 PASCAL VOC XML annotation files
│   └── images\                     # 853 matching full scene images
```

> **Important**: Dataset 1 is the primary source for training the MobileNetV2 classifier. Dataset 2 contains multi-face scene photos and PASCAL VOC annotations used for validation and analysis.

---

## 3. Virtual Environment Setup (PowerShell)

On Windows PowerShell, create and activate an isolated virtual environment:

```powershell
# Navigate to the project root
cd C:\Users\kdhar\Documents\FOCV

# Create virtual environment with Python 3.10 or 3.11
py -3.10 -m venv venv

# If PowerShell blocks script execution, temporarily allow execution:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Activate the virtual environment
.\venv\Scripts\Activate.ps1
```

Once activated, your terminal prompt will be prefixed with `(venv)`.

---

## 4. Installing Dependencies

Install the required packages using `pip`:

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install project requirements
pip install -r requirements.txt
```

### `requirements.txt` includes:
- `torch>=2.1.0` & `torchvision>=0.16.0` (Deep learning framework & MobileNetV2)
- `opencv-python>=4.8.0` (Face detection & video capture)
- `PySide6>=6.5.0` (Native Qt desktop application UI)
- `numpy>=1.24.0` (Numerical matrix processing)
- `scikit-learn>=1.3.0` (Stratified splits, classification metrics)
- `matplotlib>=3.8.0` (Loss curves, confusion matrix plotting)
- `pillow>=10.0.0` (Image I/O)
- `lxml>=4.9.0` (PASCAL VOC XML parsing)
- `pyyaml>=6.0` (YAML configuration)
- `pytest>=7.0.0` (Automated testing suite)

---

## 5. Hardware Verification & Intel Arc XPU Support

Run the built-in hardware verification utility:

```powershell
python verify_hardware.py
```

### Device Selection Hierarchy:
1. `xpu` if `torch.xpu.is_available()` is True (Intel Arc GPU).
2. `cuda` if `torch.cuda.is_available()` is True (NVIDIA GPU).
3. `cpu` safe universal fallback.

### Enabling Intel Arc GPU (XPU) on Windows:
To use Intel Arc GPUs natively with PyTorch XPU:
1. Ensure the latest **Intel Graphics Driver** for Intel Arc is installed from the [Intel Driver & Support Assistant](https://www.intel.com/content/www/us/en/support/detect.html).
2. Install the official PyTorch XPU build:
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
   ```
3. Re-run `python verify_hardware.py` to confirm that `XPU Available: True` and the device name (e.g. *Intel(R) Arc(TM) Graphics*) appears.

---

## 6. Running Dataset Validation

Verify the folder structure, image readability, and class balance before training:

```powershell
python dataset_tools.py
```

To run a deep byte-level integrity verification across all images:

```powershell
python dataset_tools.py --verify-all
```

To run the automated test suite:

```powershell
pytest tests/test_system.py -v
```

---

## 7. Training the Classifier

You can train the model either through the **Desktop UI (Training Tab)** or directly via the command line:

```powershell
# Standard training (10 epochs, batch size 32, AdamW lr 0.0003)
python train.py --epochs 10 --batch-size 32 --lr 0.0003

# Resume an interrupted training run
python train.py --epochs 15 --resume checkpoints/mobilenet_v2_mask_latest.pt
```

### Training Features:
- **Stratified Split**: 70% Train (6,287 images), 15% Validation (1,347 images), 15% Test (1,348 images).
- **Manifest**: Saved automatically to `data/splits/split_manifest.csv`.
- **Augmentation**: Random horizontal flip, modest rotation, color jitter, and affine translation (training set only).
- **Early Stopping**: Tracks validation loss and saves the best model checkpoint to `checkpoints/mobilenet_v2_mask_best.pt`.
- **Artifacts Generated**:
  - `outputs/metrics/training_metrics.json`
  - `outputs/metrics/test_metrics.json`
  - `outputs/metrics/classification_report.txt`
  - `outputs/plots/loss_accuracy_curve.png`
  - `outputs/plots/confusion_matrix.png`

---

## 8. Running the Desktop Application

Launch the native PySide6 desktop GUI:

```powershell
python app.py
```

### Application Pages:
1. **Home**: High-level overview, compute hardware card, active model status, and quick-launch buttons.
2. **Dataset**: Class counts, random image preview gallery with shuffle button, and live diagnostics console.
3. **Training**: Hyperparameter controls, non-blocking background worker thread, live epoch progress bar, real-time loss/acc monitors, and training curve display.
4. **Live Camera**: Real-time webcam feed with bounding boxes, color badges, FPS counter, confidence slider, and privacy logging.
5. **Results**: Test accuracy, loss, confusion matrix heatmap, per-class classification report, and export tools.
6. **About**: Educational system architecture, limitations, and privacy disclaimer.

---

## 9. Using the Live Camera Page

1. Open the application with `python app.py`.
2. Click on the **Live Camera** tab.
3. Select your camera index (default: `Camera 0`).
4. Click **Start Camera**.
5. Position your face in front of the webcam:
   - Green badge: **With Mask**
   - Red badge: **Without Mask**
   - Orange badge: **Incorrect Mask**
6. Adjust the **Confidence Slider** (e.g. 60%) to filter out uncertain predictions.
7. Check **Privacy-friendly Event Logging** to record detection timestamps to `outputs/logs/detection_events.csv` without capturing personal imagery.
8. Click **Stop Camera** before exiting or switching tabs.

---

## 10. Troubleshooting & Common Fixes

### 1. Intel Arc / XPU Not Detected
- **Cause**: PyTorch was installed from standard PyPI which defaults to CPU on Windows, or Intel Arc driver is missing.
- **Solution**: Install Intel Arc driver from Intel's website, then install PyTorch XPU wheel:
  ```powershell
  pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
  python verify_hardware.py
  ```

### 2. CPU Fallback Notice
- **Cause**: XPU and CUDA are both unavailable on the system.
- **Normal Behavior**: The software automatically switches to CPU. All training, inference, and UI features work normally on CPU.

### 3. Camera Unavailable
- **Cause**: Webcam is in use by another application (e.g., Zoom, Teams, Lenovo Vantage) or Windows Camera privacy settings are blocking access.
- **Solution**:
  - Close other camera apps.
  - Check Windows Settings ➔ Privacy & Security ➔ Camera ➔ Ensure *Let desktop apps access your camera* is **ON**.
  - Try selecting `Camera 1` or `Camera 2` in the dropdown.

### 4. Missing Class Folders
- **Cause**: Dataset 1 folder structure was renamed or moved.
- **Solution**: Ensure `dataset 1/` contains `mask_weared_incorrect`, `with_mask`, and `without_mask`. Run `python dataset_tools.py` to diagnose.

### 5. Missing Model Checkpoint
- **Cause**: No training run has been executed yet.
- **Solution**: Go to the **Training** tab in the desktop app and click **Start Training**, or run `python train.py --epochs 3` from PowerShell.

---

## 11. Metrics & Evaluation Guide

- **Accuracy**: The overall percentage of correctly classified face crops out of all test crops.
- **Precision**: Of all faces labeled as *Class X*, how many were truly *Class X*? (Minimizes false alarms).
- **Recall**: Of all actual *Class X* faces present, how many did the model correctly identify? (Minimizes missed detections).
- **F1-Score**: Harmonic mean of Precision and Recall ($2 \times \frac{P \times R}{P + R}$). Best metric for balanced performance.
- **Confusion Matrix**: A 3×3 grid comparing true labels against predicted labels to highlight where any misclassifications occur.
- **FPS (Frames Per Second)**: Real-time rendering speed of the video feed. A higher FPS (e.g. 20–30+ FPS) ensures smooth video playback without lag.

### Evaluation Plots:
- **Loss & Accuracy Curves**: View training vs. validation curves in `outputs/plots/loss_accuracy_curve.png`.
- **Confusion Matrix**: View test set multi-class confusion matrix in `outputs/plots/confusion_matrix.png`.

---

## 12. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
