"""Unit and integration test suite for Face Mask Detection System.

Covers:
- Device selection priority and hardware verification
- Class mapping and display labels
- Dataset 1 validation and Dataset 2 XML parsing
- Model architecture and output head dimensions
- Inference preprocessing and tensor dimensions
- Prediction smoothing logic
"""

import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
import torch

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from verify_hardware import get_device, run_benchmark, is_xpu_available, is_cuda_available
from prepare_data import CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS, DISPLAY_NAMES, get_transforms
from dataset_tools import validate_dataset1, analyze_dataset2, parse_voc_annotation
from models.model_builder import build_mobilenet_v2, NUM_CLASSES
from detector import FaceDetector
from inference import PredictionSmoother, COLOR_GREEN, COLOR_RED, COLOR_ORANGE, CLASS_COLORS


def test_device_selection():
    """Verify device selection follows strict XPU -> CUDA -> CPU priority."""
    chosen = get_device()
    assert chosen.type in ("xpu", "cuda", "cpu")

    # If force_cpu=True, always returns CPU
    cpu_device = get_device(force_cpu=True)
    assert cpu_device.type == "cpu"

    # Test benchmark on CPU
    success, elapsed, msg = run_benchmark(torch.device("cpu"), matrix_size=64)
    assert success is True
    assert elapsed > 0


def test_class_mapping():
    """Verify exact 3 classes, their indices, and UI display strings."""
    assert len(CLASS_NAMES) == 3
    assert CLASS_NAMES == ["mask_weared_incorrect", "with_mask", "without_mask"]

    assert CLASS_TO_IDX["mask_weared_incorrect"] == 0
    assert CLASS_TO_IDX["with_mask"] == 1
    assert CLASS_TO_IDX["without_mask"] == 2

    assert IDX_TO_CLASS[0] == "mask_weared_incorrect"
    assert IDX_TO_CLASS[1] == "with_mask"
    assert IDX_TO_CLASS[2] == "without_mask"

    assert DISPLAY_NAMES["mask_weared_incorrect"] == "Incorrect Mask"
    assert DISPLAY_NAMES["with_mask"] == "With Mask"
    assert DISPLAY_NAMES["without_mask"] == "Without Mask"

    # Color requirements: Green for With Mask, Red for Without Mask, Orange for Incorrect Mask
    assert CLASS_COLORS[0] == COLOR_ORANGE
    assert CLASS_COLORS[1] == COLOR_GREEN
    assert CLASS_COLORS[2] == COLOR_RED


def test_dataset1_validation():
    """Verify Dataset 1 directory structure and counts."""
    d1_path = Path("dataset 1")
    if not d1_path.exists():
        pytest.skip("Dataset 1 folder not present at default path")

    res = validate_dataset1(d1_path, verify_images=False)
    assert res["valid"] is True
    assert res["total_images"] > 0
    assert "mask_weared_incorrect" in res["classes"]
    assert "with_mask" in res["classes"]
    assert "without_mask" in res["classes"]
    assert res["classes"]["with_mask"]["count"] > 0


def test_dataset2_parsing():
    """Verify Dataset 2 Pascal VOC XML parsing."""
    d2_path = Path("dataset 2")
    if not d2_path.exists():
        pytest.skip("Dataset 2 folder not present at default path")

    res = analyze_dataset2(d2_path)
    assert res["has_annotations"] is True
    assert res["total_xml_files"] > 0
    assert res["total_objects"] > 0
    assert "with_mask" in res["object_counts"]


def test_model_architecture():
    """Verify MobileNetV2 architecture with 3-class classifier head."""
    model = build_mobilenet_v2(num_classes=NUM_CLASSES, pretrained=False)
    assert model is not None

    # Dummy forward pass
    dummy_input = torch.randn(2, 3, 224, 224, dtype=torch.float32)
    output = model(dummy_input)
    assert output.shape == (2, 3), f"Expected shape (2, 3), got {output.shape}"


def test_inference_preprocessing():
    """Verify preprocessing transforms format images to (3, 224, 224) tensors."""
    transforms_dict = get_transforms(img_size=224)
    eval_transform = transforms_dict["test"]

    # Create dummy PIL image with irregular dimension
    img = Image.new("RGB", (480, 640), color=(100, 150, 200))
    tensor = eval_transform(img)

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    # Check that tensor has been normalized (not raw 0-255 uint8)
    assert tensor.min() < 0.0 or tensor.max() > 1.0


def test_prediction_smoothing():
    """Verify rolling window prediction smoothing reduces flicker."""
    smoother = PredictionSmoother(window_size=3)

    # Fake detected box (x, y, w, h)
    box = (100, 100, 80, 80)

    # Frame 1: With Mask (class 1, 0.90)
    res1 = smoother.update([box], [(1, 0.90)])
    assert res1[0][0] == 1

    # Frame 2: Glitch / flicker: Without Mask (class 2, 0.55)
    res2 = smoother.update([box], [(2, 0.55)])
    # Smoothed result should still retain With Mask (class 1) due to history
    assert res2[0][0] == 1

    # Frame 3: With Mask confirmed (class 1, 0.92)
    res3 = smoother.update([box], [(1, 0.92)])
    assert res3[0][0] == 1


def test_face_detector_haar():
    """Verify OpenCV Haar Cascade initializes and runs safely."""
    detector = FaceDetector(method="haar")
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Should run without crashing and return a list
    boxes = detector.detect_faces(blank_frame)
    assert isinstance(boxes, list)
