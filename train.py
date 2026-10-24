"""Training pipeline for Face Mask Detection using MobileNetV2.

Supports:
- Device priority: XPU (Intel Arc) -> CUDA -> CPU fallback
- AdamW optimizer & learning rate scheduling
- Early stopping based on validation loss
- Checkpoint saving (best model & last state for resuming)
- Training & validation loss/accuracy tracking and Matplotlib plotting
- Cancellation hook for PySide6 background worker thread
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
import time
from typing import Dict, Any, Optional, Callable
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import yaml

from verify_hardware import get_device, get_hardware_info
from prepare_data import get_data_loaders, CLASS_NAMES
from models.model_builder import build_mobilenet_v2, save_checkpoint, load_checkpoint
from evaluate import evaluate_model

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def plot_training_history(
    history: Dict[str, list[float]],
    save_path: str | Path,
) -> None:
    """Plot and save loss and accuracy progression curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    # Loss subplot
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4")
    ax1.plot(epochs, history["val_loss"], "s-", label="Val Loss", color="#ff7f0e")
    ax1.set_title("Training & Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CrossEntropy Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Accuracy subplot
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", label="Train Accuracy", color="#2ca02c")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "s-", label="Val Accuracy", color="#d62728")
    ax2.set_title("Training & Validation Accuracy (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), bbox_inches="tight")
    plt.close(fig)


def train_model(
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 0.0003,
    weight_decay: float = 0.01,
    early_stopping_patience: int = 4,
    img_size: int = 224,
    checkpoint_dir: str | Path = "checkpoints",
    model_save_name: str = "mobilenet_v2_mask_best.pt",
    resume_checkpoint: Optional[str | Path] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    stop_requested: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Execute training loop for MobileNetV2 Face Mask Classifier.

    Args:
        epochs: Number of epochs to train.
        batch_size: Mini-batch size.
        learning_rate: Initial AdamW learning rate.
        weight_decay: L2 penalty factor.
        early_stopping_patience: Epochs without val loss improvement before stopping.
        img_size: Image dimension (224).
        checkpoint_dir: Folder to store checkpoints.
        model_save_name: Filename for best checkpoint.
        resume_checkpoint: Optional path to checkpoint to resume training from.
        progress_callback: Callback passing status dict to UI thread.
        log_callback: Callback passing log string to UI console.
        stop_requested: Callable returning True if training was aborted.

    Returns:
        Dict with comprehensive training and evaluation metrics.
    """
    def log(msg: str) -> None:
        logger.info(msg)
        if log_callback:
            log_callback(msg)

    # 1. Hardware selection
    hw_info = get_hardware_info()
    device = get_device()
    log(f"Starting training on device: {device.type.upper()} ({hw_info.get('xpu_device_name', 'N/A') if device.type == 'xpu' else 'Generic'})")

    # 2. Setup DataLoaders
    log("Loading and preparing dataset splits...")
    train_loader, val_loader, test_loader, counts = get_data_loaders(
        batch_size=batch_size,
        img_size=img_size,
    )
    log(f"Dataset split sizes: Train={counts['train']}, Val={counts['val']}, Test={counts['test']}")

    # 3. Model setup
    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0
    history: Dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    if resume_checkpoint and Path(resume_checkpoint).exists():
        log(f"Resuming training from checkpoint: {resume_checkpoint}")
        model, meta = load_checkpoint(resume_checkpoint, device=device)
        start_epoch = meta.get("epoch", 0) + 1
        prev_metrics = meta.get("metrics", {})
        best_val_loss = prev_metrics.get("best_val_loss", float("inf"))
    else:
        log("Instantiating MobileNetV2 with ImageNet pretrained weights...")
        model = build_mobilenet_v2(num_classes=len(CLASS_NAMES), pretrained=True)
        model.to(device)

    # 4. Criterion, Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    chk_dir = Path(checkpoint_dir)
    chk_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = chk_dir / model_save_name
    latest_model_path = chk_dir / "mobilenet_v2_mask_latest.pt"

    start_time = time.time()
    best_model_weights = copy.deepcopy(model.state_dict())

    log(f"Beginning training for {epochs} epochs (Batch Size={batch_size}, LR={learning_rate})...")

    for epoch in range(start_epoch, epochs + 1):
        if stop_requested and stop_requested():
            log("Training halted by user request.")
            break

        epoch_start = time.time()

        # ================= Training Phase =================
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            if stop_requested and stop_requested():
                break

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += images.size(0)

        scheduler.step()

        epoch_train_loss = running_train_loss / train_total if train_total > 0 else 0.0
        epoch_train_acc = train_correct / train_total if train_total > 0 else 0.0

        # ================= Validation Phase =================
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)

        epoch_val_loss = running_val_loss / val_total if val_total > 0 else 0.0
        epoch_val_acc = val_correct / val_total if val_total > 0 else 0.0

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        epoch_time = time.time() - epoch_start
        status_line = (
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) | "
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc * 100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc * 100:.2f}%"
        )
        log(status_line)

        # Notify UI worker thread
        if progress_callback:
            progress_callback({
                "epoch": epoch,
                "total_epochs": epochs,
                "train_loss": epoch_train_loss,
                "train_acc": epoch_train_acc,
                "val_loss": epoch_val_loss,
                "val_acc": epoch_val_acc,
                "best_val_loss": best_val_loss,
                "status_text": status_line,
            })

        # Save latest checkpoint for resumption
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics={"train_loss": epoch_train_loss, "val_loss": epoch_val_loss, "val_acc": epoch_val_acc},
            checkpoint_path=latest_model_path,
            scheduler=scheduler,
        )

        # Early stopping & best model tracking
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_model_weights = copy.deepcopy(model.state_dict())
            log(f"  -> Validation loss improved to {best_val_loss:.4f}. Saving best model checkpoint.")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={
                    "best_val_loss": best_val_loss,
                    "val_loss": epoch_val_loss,
                    "val_acc": epoch_val_acc,
                    "history": history,
                },
                checkpoint_path=best_model_path,
                scheduler=scheduler,
            )
        else:
            patience_counter += 1
            log(f"  -> Validation loss did not improve. Early stopping counter: {patience_counter}/{early_stopping_patience}")
            if patience_counter >= early_stopping_patience:
                log(f"Early stopping triggered at epoch {epoch} due to no validation improvement.")
                break

    total_training_time = time.time() - start_time
    log(f"Training loop completed in {total_training_time:.2f} seconds.")

    # Save loss/acc plot
    plot_path = Path("outputs/plots/loss_accuracy_curve.png")
    plot_training_history(history, plot_path)
    log(f"Saved training history curves to {plot_path}")

    # Load best weights for final test evaluation
    model.load_state_dict(best_model_weights)
    log("Running final evaluation on unseen Test Set...")
    eval_results = evaluate_model(model, test_loader, device=device, output_dir="outputs")

    log(f"Final Test Accuracy: {eval_results['accuracy'] * 100:.2f}%")
    log(f"Final Test Loss: {eval_results['loss']:.4f}")

    summary_payload = {
        "hardware_info": hw_info,
        "selected_device": str(device),
        "total_training_time_seconds": total_training_time,
        "epochs_completed": len(history["train_loss"]),
        "best_val_loss": best_val_loss,
        "history": history,
        "test_results": {
            "accuracy": eval_results["accuracy"],
            "loss": eval_results["loss"],
            "classification_report": eval_results["classification_report"],
        },
        "best_checkpoint_path": str(best_model_path),
    }

    metrics_save_path = Path("outputs/metrics/training_metrics.json")
    metrics_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    log(f"Saved full training run summary to {metrics_save_path}")
    return summary_payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Face Mask Detection MobileNetV2 Classifier.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint path")
    args = parser.parse_args()

    results = train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        resume_checkpoint=args.resume,
    )
