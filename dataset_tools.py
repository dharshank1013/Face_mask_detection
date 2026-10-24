"""Dataset validation and analysis tools for Face Mask Detection.

Handles Dataset 1 (training image folders) and Dataset 2 (PASCAL VOC XML annotations).
"""

from __future__ import annotations

import os
from pathlib import Path
import random
from typing import Dict, List, Tuple, Any, Optional
import xml.etree.ElementTree as ET
from PIL import Image

try:
    from lxml import etree as lxml_etree
    HAS_LXML = True
except ImportError:
    HAS_LXML = False


VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

REQUIRED_DATASET1_CLASSES = [
    "mask_weared_incorrect",
    "with_mask",
    "without_mask",
]

DISPLAY_NAME_MAP = {
    "mask_weared_incorrect": "Incorrect Mask",
    "with_mask": "With Mask",
    "without_mask": "Without Mask",
}


def validate_dataset1(
    base_dir: str | Path,
    expected_classes: Optional[List[str]] = None,
    verify_images: bool = True,
    sample_verify_limit: Optional[int] = None
) -> Dict[str, Any]:
    """Validate Dataset 1 structure, image counts, and file integrity.

    Args:
        base_dir: Path to Dataset 1 folder.
        expected_classes: List of folder names expected.
        verify_images: If True, attempts opening image files to catch corruption.
        sample_verify_limit: Max number of images to deeply verify per class (None for all).

    Returns:
        Dictionary with validation results, warnings, counts, and corrupt files.
    """
    base_path = Path(base_dir)
    if expected_classes is None:
        expected_classes = REQUIRED_DATASET1_CLASSES

    results: Dict[str, Any] = {
        "valid": True,
        "base_path": str(base_path),
        "exists": base_path.exists(),
        "classes": {},
        "total_images": 0,
        "warnings": [],
        "errors": [],
        "corrupt_files": [],
        "class_balance_ratio": 1.0,
    }

    if not base_path.exists():
        results["valid"] = False
        results["errors"].append(f"Dataset 1 directory not found: {base_path}")
        return results

    counts: Dict[str, int] = {}

    for cls in expected_classes:
        cls_dir = base_path / cls
        if not cls_dir.exists() or not cls_dir.is_dir():
            results["valid"] = False
            results["errors"].append(f"Missing required class directory: {cls_dir}")
            counts[cls] = 0
            results["classes"][cls] = {"count": 0, "status": "missing"}
            continue

        # Enumerate valid image files
        image_files = [
            f for f in cls_dir.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS
        ]
        counts[cls] = len(image_files)

        corrupt_in_class = []
        if verify_images and image_files:
            to_test = image_files
            if sample_verify_limit is not None and len(to_test) > sample_verify_limit:
                to_test = random.sample(to_test, sample_verify_limit)

            for img_path in to_test:
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception as e:
                    corrupt_in_class.append((str(img_path), str(e)))

        results["corrupt_files"].extend(corrupt_in_class)
        results["classes"][cls] = {
            "count": len(image_files),
            "display_name": DISPLAY_NAME_MAP.get(cls, cls),
            "status": "ok" if len(image_files) > 0 else "empty",
            "corrupt_count": len(corrupt_in_class),
        }

    total = sum(counts.values())
    results["total_images"] = total

    if results["corrupt_files"]:
        results["valid"] = False
        results["errors"].append(f"Found {len(results['corrupt_files'])} corrupt image(s).")

    if counts:
        min_c = min(counts.values())
        max_c = max(counts.values())
        ratio = (min_c / max_c) if max_c > 0 else 0.0
        results["class_balance_ratio"] = ratio
        if ratio < 0.6 and total > 0:
            results["warnings"].append(
                f"Class imbalance detected: min class has {min_c} images, max has {max_c} (ratio: {ratio:.2f})."
            )

    return results


def parse_voc_annotation(xml_path: Path) -> Dict[str, Any]:
    """Parse a single Pascal VOC XML annotation file.

    Returns dict with image filename, image size, and list of objects with bboxes.
    """
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
    except Exception as e:
        return {"error": str(e), "objects": []}

    filename = root.findtext("filename", "")
    size_elem = root.find("size")
    width = int(size_elem.findtext("width", 0)) if size_elem is not None else 0
    height = int(size_elem.findtext("height", 0)) if size_elem is not None else 0

    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        bndbox = obj.find("bndbox")
        if bndbox is not None:
            try:
                xmin = int(float(bndbox.findtext("xmin", 0)))
                ymin = int(float(bndbox.findtext("ymin", 0)))
                xmax = int(float(bndbox.findtext("xmax", 0)))
                ymax = int(float(bndbox.findtext("ymax", 0)))
                objects.append({
                    "name": name,
                    "bbox": (xmin, ymin, xmax, ymax),
                    "width": max(0, xmax - xmin),
                    "height": max(0, ymax - ymin),
                })
            except (ValueError, TypeError):
                continue

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "objects": objects,
    }


def analyze_dataset2(base_dir: str | Path) -> Dict[str, Any]:
    """Inspect and analyze Dataset 2 annotations and images.

    Args:
        base_dir: Path to Dataset 2 directory.

    Returns:
        Dictionary with statistics on annotations, images, and object labels.
    """
    base_path = Path(base_dir)
    annotations_dir = base_path / "annotations"
    images_dir = base_path / "images"

    summary: Dict[str, Any] = {
        "exists": base_path.exists(),
        "has_annotations": annotations_dir.exists(),
        "has_images": images_dir.exists(),
        "total_xml_files": 0,
        "total_image_files": 0,
        "total_objects": 0,
        "object_counts": {},
        "matched_pairs": 0,
        "warnings": [],
    }

    if not base_path.exists():
        summary["warnings"].append(f"Dataset 2 directory not found: {base_path}")
        return summary

    if not annotations_dir.exists():
        summary["warnings"].append(f"Dataset 2 'annotations' subfolder missing: {annotations_dir}")
        return summary

    xml_files = list(annotations_dir.glob("*.xml"))
    summary["total_xml_files"] = len(xml_files)

    if images_dir.exists():
        img_files = [f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS]
        summary["total_image_files"] = len(img_files)
        img_basenames = {f.stem for f in img_files}
        xml_basenames = {f.stem for f in xml_files}
        summary["matched_pairs"] = len(img_basenames.intersection(xml_basenames))

    obj_counts: Dict[str, int] = {}
    total_objs = 0

    for xml_f in xml_files:
        parsed = parse_voc_annotation(xml_f)
        for obj in parsed.get("objects", []):
            label = obj["name"]
            obj_counts[label] = obj_counts.get(label, 0) + 1
            total_objs += 1

    summary["total_objects"] = total_objs
    summary["object_counts"] = obj_counts

    return summary


def get_sample_images_dataset1(base_dir: str | Path) -> Dict[str, Optional[Path]]:
    """Return a random image path for each class in Dataset 1 for UI preview."""
    base_path = Path(base_dir)
    samples: Dict[str, Optional[Path]] = {}

    for cls in REQUIRED_DATASET1_CLASSES:
        cls_dir = base_path / cls
        if cls_dir.exists() and cls_dir.is_dir():
            images = [
                f for f in cls_dir.iterdir()
                if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS
            ]
            samples[cls] = random.choice(images) if images else None
        else:
            samples[cls] = None

    return samples


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate and analyze Face Mask datasets.")
    parser.add_argument("--dataset1", default="dataset 1", help="Path to Dataset 1 folder")
    parser.add_argument("--dataset2", default="dataset 2", help="Path to Dataset 2 folder")
    parser.add_argument("--verify-all", action="store_true", help="Verify all images for corruption")
    args = parser.parse_args()

    print("=" * 65)
    print("   DATASET VALIDATION & SUMMARY REPORT")
    print("=" * 65)

    print("\n[1] Validating Dataset 1 (Classifier Training Source)...")
    d1_res = validate_dataset1(args.dataset1, verify_images=args.verify_all, sample_verify_limit=None if args.verify_all else 50)
    print(f"Directory: {d1_res['base_path']}")
    print(f"Status   : {'VALID' if d1_res['valid'] else 'INVALID'}")
    print(f"Total    : {d1_res['total_images']} images")
    for cls, info in d1_res["classes"].items():
        print(f"  - {info['display_name']} ({cls}): {info['count']} images (status: {info['status']})")

    if d1_res["warnings"]:
        print("\nWarnings:")
        for w in d1_res["warnings"]:
            print(f"  ! {w}")

    if d1_res["errors"]:
        print("\nErrors:")
        for e in d1_res["errors"]:
            print(f"  * {e}")

    print("\n[2] Analyzing Dataset 2 (Supplementary Detection Annotations)...")
    d2_res = analyze_dataset2(args.dataset2)
    print(f"Annotations XMLs : {d2_res['total_xml_files']}")
    print(f"Images           : {d2_res['total_image_files']}")
    print(f"Matched Pairs    : {d2_res['matched_pairs']}")
    print(f"Total Objects    : {d2_res['total_objects']}")
    print("Object labels breakdown:")
    for label, count in d2_res["object_counts"].items():
        print(f"  - {label}: {count} bounding boxes")

    print("\n" + "=" * 65)
