"""Data preparation, stratified splitting, and PyTorch dataset definitions.

Splits Dataset 1 into Train (70%), Validation (15%), and Test (15%) splits
with a fixed random seed, saving a manifest CSV to data/splits/split_manifest.csv.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)

CLASS_NAMES = [
    "mask_weared_incorrect",
    "with_mask",
    "without_mask",
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

DISPLAY_NAMES = {
    "mask_weared_incorrect": "Incorrect Mask",
    "with_mask": "With Mask",
    "without_mask": "Without Mask",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(img_size: int = 224) -> Dict[str, transforms.Compose]:
    """Build data transformations.

    Training augmentation only: horizontal flip, modest rotation,
    color jitter, slight affine translation/scale.
    Validation/test: resize and normalisation only.
    """
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return {
        "train": train_transforms,
        "val": eval_transforms,
        "test": eval_transforms,
    }


class FaceMaskDataset(Dataset):
    """PyTorch Dataset loading image files with error handling."""

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        transform: Optional[transforms.Compose] = None,
    ):
        """Args:

        samples: List of (file_path_str, class_idx)
        transform: torchvision transformation pipeline
        """
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        file_path, label = self.samples[idx]
        try:
            with Image.open(file_path) as img:
                img_rgb = img.convert("RGB")
                if self.transform is not None:
                    tensor = self.transform(img_rgb)
                else:
                    tensor = transforms.ToTensor()(img_rgb)
                return tensor, label
        except Exception as e:
            logger.warning(f"Error reading image {file_path}: {e}. Returning blank placeholder.")
            # Return neutral tensor to prevent training crashes on corrupted items
            blank = torch.zeros((3, 224, 224), dtype=torch.float32)
            return blank, label


def create_or_load_splits(
    dataset_dir: str | Path = "dataset 1",
    splits_csv_path: str | Path = "data/splits/split_manifest.csv",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    force_recreate: bool = False,
) -> Dict[str, List[Tuple[str, int]]]:
    """Create or load stratified train/val/test splits manifest.

    Args:
        dataset_dir: Path to Dataset 1 folder.
        splits_csv_path: Path to CSV storing split metadata.
        train_ratio: Fraction for training (default 0.70).
        val_ratio: Fraction for validation (default 0.15).
        test_ratio: Fraction for testing (default 0.15).
        random_seed: Seed for reproducible stratification.
        force_recreate: Re-split even if manifest exists.

    Returns:
        Dict mapping 'train', 'val', 'test' to lists of (filepath, class_idx).
    """
    splits_file = Path(splits_csv_path)
    dataset_path = Path(dataset_dir)

    if splits_file.exists() and not force_recreate:
        logger.info(f"Loading existing split manifest from {splits_file}")
        splits_data: Dict[str, List[Tuple[str, int]]] = {"train": [], "val": [], "test": []}
        with open(splits_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                split_name = row["split"]
                fpath = row["filepath"]
                cidx = int(row["class_idx"])
                if split_name in splits_data:
                    splits_data[split_name].append((fpath, cidx))

        if (
            len(splits_data["train"]) > 0
            and len(splits_data["val"]) > 0
            and len(splits_data["test"]) > 0
        ):
            return splits_data

    # Collect all image files
    all_filepaths: List[str] = []
    all_labels: List[int] = []

    for cls_name in CLASS_NAMES:
        cls_dir = dataset_path / cls_name
        if not cls_dir.exists():
            raise FileNotFoundError(f"Required class directory not found: {cls_dir}")

        cidx = CLASS_TO_IDX[cls_name]
        files = sorted(list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.jpeg")))
        for f in files:
            all_filepaths.append(str(f.resolve()))
            all_labels.append(cidx)

    if not all_filepaths:
        raise ValueError(f"No valid image files found in {dataset_path}")

    filepaths_arr = np.array(all_filepaths)
    labels_arr = np.array(all_labels)

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=val_test_ratio, random_state=random_seed)
    train_idx, val_test_idx = next(sss1.split(filepaths_arr, labels_arr))

    # Second split: val vs test (proportional)
    val_rel_ratio = val_ratio / val_test_ratio
    val_test_labels = labels_arr[val_test_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - val_rel_ratio), random_state=random_seed)
    val_sub_idx, test_sub_idx = next(sss2.split(val_test_idx, val_test_labels))

    val_idx = val_test_idx[val_sub_idx]
    test_idx = val_test_idx[test_sub_idx]

    splits_data = {
        "train": [(filepaths_arr[i], int(labels_arr[i])) for i in train_idx],
        "val": [(filepaths_arr[i], int(labels_arr[i])) for i in val_idx],
        "test": [(filepaths_arr[i], int(labels_arr[i])) for i in test_idx],
    }

    # Save manifest
    splits_file.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "class_name", "class_idx", "split"])
        for split_name, items in splits_data.items():
            for fpath, cidx in items:
                writer.writerow([fpath, IDX_TO_CLASS[cidx], cidx, split_name])

    logger.info(f"Created new split manifest: {len(splits_data['train'])} train, {len(splits_data['val'])} val, {len(splits_data['test'])} test.")
    return splits_data


def get_data_loaders(
    dataset_dir: str | Path = "dataset 1",
    splits_csv_path: str | Path = "data/splits/split_manifest.csv",
    batch_size: int = 32,
    num_workers: int = 0,
    img_size: int = 224,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """Build train, val, and test DataLoaders."""
    splits = create_or_load_splits(
        dataset_dir=dataset_dir,
        splits_csv_path=splits_csv_path,
        random_seed=random_seed,
    )
    transforms_dict = get_transforms(img_size=img_size)

    train_ds = FaceMaskDataset(splits["train"], transform=transforms_dict["train"])
    val_ds = FaceMaskDataset(splits["val"], transform=transforms_dict["val"])
    test_ds = FaceMaskDataset(splits["test"], transform=transforms_dict["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    counts = {
        "train": len(train_ds),
        "val": len(val_ds),
        "test": len(test_ds),
    }

    return train_loader, val_loader, test_loader, counts


if __name__ == "__main__":
    print("Preparing dataset splits...")
    splits = create_or_load_splits(force_recreate=True)
    print(f"Train samples: {len(splits['train'])}")
    print(f"Val samples  : {len(splits['val'])}")
    print(f"Test samples : {len(splits['test'])}")
    print("Split manifest saved at data/splits/split_manifest.csv")
