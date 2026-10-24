"""Evaluation module for Face Mask Detection.

Computes test accuracy, per-class precision, recall, F1-score, confusion matrix,
and exports plots and metrics to outputs/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from prepare_data import CLASS_NAMES, DISPLAY_NAMES, get_data_loaders
from models.model_builder import load_checkpoint

logger = logging.getLogger(__name__)


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: Optional[list[str]] = None,
    output_dir: str | Path = "outputs",
) -> Dict[str, Any]:
    """Run full evaluation on a test dataset DataLoader.

    Args:
        model: Trained PyTorch model.
        test_loader: Test DataLoader.
        device: Device to run evaluation on.
        class_names: List of class names.
        output_dir: Directory to save plots and metrics.

    Returns:
        Dict with accuracy, per-class metrics, confusion matrix, and report dict.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    model.eval()
    model.to(device)

    all_preds: list[int] = []
    all_targets: list[int] = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    n_samples = len(all_targets)
    avg_loss = total_loss / n_samples if n_samples > 0 else 0.0
    correct = sum(p == t for p, t in zip(all_preds, all_targets))
    accuracy = correct / n_samples if n_samples > 0 else 0.0

    target_names = [DISPLAY_NAMES.get(c, c) for c in class_names]

    # Sklearn classification report
    report_dict = classification_report(
        all_targets,
        all_preds,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        all_targets,
        all_preds,
        target_names=target_names,
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)

    # Save artifacts
    out_path = Path(output_dir)
    plots_dir = out_path / "plots"
    metrics_dir = out_path / "metrics"
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    cm_path = plots_dir / "confusion_matrix.png"
    plot_and_save_confusion_matrix(cm, target_names, cm_path)

    metrics_payload = {
        "test_loss": float(avg_loss),
        "test_accuracy": float(accuracy),
        "total_samples": int(n_samples),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
    }

    metrics_file = metrics_dir / "test_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    report_file = metrics_dir / "classification_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    return {
        "accuracy": accuracy,
        "loss": avg_loss,
        "confusion_matrix": cm,
        "confusion_matrix_path": str(cm_path),
        "classification_report": report_dict,
        "classification_report_text": report_text,
        "metrics_file": str(metrics_file),
    }


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    class_labels: list[str],
    save_path: str | Path,
) -> None:
    """Plot and save confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_labels,
        yticklabels=class_labels,
        title="Face Mask Detection - Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold",
            )

    fig.tight_layout()
    plt.savefig(str(save_path), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    from verify_hardware import get_device

    parser = argparse.ArgumentParser(description="Evaluate trained Face Mask classifier checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/mobilenet_v2_mask_best.pt", help="Path to checkpoint")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    chk_path = Path(args.checkpoint)
    if not chk_path.exists():
        print(f"Error: Checkpoint {chk_path} not found. Please train a model first.")
        exit(1)

    device = get_device()
    print(f"Evaluating {chk_path} on {device}...")

    model, meta = load_checkpoint(chk_path, device=device)
    _, _, test_loader, counts = get_data_loaders(batch_size=args.batch_size)
    results = evaluate_model(model, test_loader, device=device)

    print("\n" + "=" * 65)
    print("   MODEL EVALUATION RESULTS")
    print("=" * 65)
    print(f"Test Accuracy: {results['accuracy'] * 100:.2f}%")
    print(f"Test Loss    : {results['loss']:.4f}")
    print("\nClassification Report:")
    print(results["classification_report_text"])
    print(f"Confusion Matrix saved to: {results['confusion_matrix_path']}")
    print(f"Metrics saved to         : {results['metrics_file']}")
    print("=" * 65)
