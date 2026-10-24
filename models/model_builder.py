"""MobileNetV2 Model Builder and Checkpoint Manager for 3-class Face Mask Detection.

Classes:
0: mask_weared_incorrect (Incorrect Mask)
1: with_mask             (With Mask)
2: without_mask          (Without Mask)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

logger = logging.getLogger(__name__)

NUM_CLASSES = 3
DEFAULT_CLASS_NAMES = [
    "mask_weared_incorrect",
    "with_mask",
    "without_mask",
]


def build_mobilenet_v2(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """Build MobileNetV2 with custom classifier head for Face Mask Detection.

    Args:
        num_classes: Number of output classes (default 3).
        pretrained: If True, load ImageNet pretrained weights.
        dropout: Classifier dropout rate (default 0.2).

    Returns:
        torch.nn.Module instance.
    """
    weights = None
    if pretrained:
        try:
            weights = MobileNet_V2_Weights.DEFAULT
        except Exception as e:
            logger.warning(f"Could not load ImageNet weights automatically: {e}. Building without weights.")
            weights = None

    try:
        model = mobilenet_v2(weights=weights)
    except Exception as e:
        logger.warning(f"Failed to fetch remote weights ({e}). Initializing randomly.")
        model = mobilenet_v2(weights=None)

    # In MobileNetV2: model.classifier is Sequential(Dropout, Linear(last_channel, 1000))
    in_features = model.last_channel  # Typically 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )

    return model


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: Dict[str, Any],
    checkpoint_path: str | Path,
    class_names: Optional[list[str]] = None,
    scheduler: Optional[Any] = None,
) -> Path:
    """Save model checkpoint with full training state and metadata.

    Args:
        model: Trained PyTorch model.
        optimizer: Optimizer state (optional).
        epoch: Current epoch number.
        metrics: Dictionary containing loss, accuracy, etc.
        checkpoint_path: Destination file path.
        class_names: List of class names.
        scheduler: LR scheduler state (optional).

    Returns:
        Path to saved checkpoint.
    """
    dest = Path(checkpoint_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    # Ensure model is on CPU for clean serialization
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    payload = {
        "architecture": "mobilenet_v2",
        "num_classes": len(class_names),
        "class_names": class_names,
        "epoch": epoch,
        "state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "metrics": metrics,
        "torch_version": torch.__version__,
    }

    torch.save(payload, str(dest))
    logger.info(f"Checkpoint saved to {dest} (Epoch {epoch})")
    return dest


def load_checkpoint(
    checkpoint_path: str | Path,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load model and metadata from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        device: Device to map model to. If None, stays on CPU.

    Returns:
        Tuple of (model, metadata_dict).
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    target_device = device if device is not None else torch.device("cpu")
    checkpoint = torch.load(str(path), map_location=target_device, weights_only=False)

    num_classes = checkpoint.get("num_classes", NUM_CLASSES)
    model = build_mobilenet_v2(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(target_device)
    model.eval()

    metadata = {
        "epoch": checkpoint.get("epoch", 0),
        "class_names": checkpoint.get("class_names", DEFAULT_CLASS_NAMES),
        "metrics": checkpoint.get("metrics", {}),
        "architecture": checkpoint.get("architecture", "mobilenet_v2"),
    }

    return model, metadata
