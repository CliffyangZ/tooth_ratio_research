"""Convert COCO bounding boxes into an Ultralytics YOLO detection dataset."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _natural_key(value: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def _read_coco(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read COCO annotations from {path}: {exc}") from exc

    for key in ("images", "annotations", "categories"):
        if not isinstance(data.get(key), list):
            raise ValueError(f"COCO field {key!r} must be a list")
    if not data["images"]:
        raise ValueError("COCO dataset contains no images")
    if not data["categories"]:
        raise ValueError("COCO dataset contains no categories")
    return data


def _unique_by_id(items: list[dict[str, Any]], kind: str) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for item in items:
        item_id = item.get("id")
        if not isinstance(item_id, int):
            raise ValueError(f"Every COCO {kind} must have an integer id")
        if item_id in indexed:
            raise ValueError(f"Duplicate COCO {kind} id: {item_id}")
        indexed[item_id] = item
    return indexed


def _split_images(
    images: list[dict[str, Any]], train_ratio: float, val_ratio: float, seed: int
) -> dict[str, list[dict[str, Any]]]:
    test_ratio = 1.0 - train_ratio - val_ratio
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    if test_ratio < -1e-9:
        raise ValueError("train_ratio + val_ratio cannot exceed 1")

    shuffled = sorted(images, key=lambda item: _natural_key(str(item["file_name"])))
    random.Random(seed).shuffle(shuffled)
    train_count = int(len(shuffled) * train_ratio)
    val_count = int(len(shuffled) * val_ratio)
    test_count = len(shuffled) - train_count - val_count
    if train_count == 0 or val_count == 0 or (test_ratio > 1e-9 and test_count == 0):
        raise ValueError("Dataset is too small for the requested train/val/test ratios")

    splits = {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
    }
    if test_count:
        splits["test"] = shuffled[train_count + val_count :]
    return splits


def _validate_image_dimensions(image_path: Path, width: int, height: int) -> None:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency is declared by this project
        raise RuntimeError("OpenCV is required for image dimension checks") from exc

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unable to decode image: {image_path}")
    actual_height, actual_width = image.shape[:2]
    if (actual_width, actual_height) != (width, height):
        raise ValueError(
            f"Image dimensions disagree with COCO for {image_path.name}: "
            f"JSON={width}x{height}, file={actual_width}x{actual_height}"
        )


def _yolo_line(
    annotation: dict[str, Any],
    image: dict[str, Any],
    category_to_class: dict[int, int],
) -> str:
    annotation_id = annotation.get("id", "<unknown>")
    category_id = annotation.get("category_id")
    if category_id not in category_to_class:
        raise ValueError(
            f"Annotation {annotation_id} references unknown category {category_id}"
        )

    bbox = annotation.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Annotation {annotation_id} must contain COCO bbox [x, y, w, h]")
    try:
        x, y, box_width, box_height = map(float, bbox)
        width, height = float(image["width"]), float(image["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid geometry in annotation {annotation_id}") from exc
    if not all(math.isfinite(value) for value in (x, y, box_width, box_height)):
        raise ValueError(f"Annotation {annotation_id} contains non-finite bbox values")
    if width <= 0 or height <= 0 or box_width <= 0 or box_height <= 0:
        raise ValueError(f"Annotation {annotation_id} contains non-positive dimensions")

    tolerance = 1e-6
    if (
        x < -tolerance
        or y < -tolerance
        or x + box_width > width + tolerance
        or y + box_height > height + tolerance
    ):
        raise ValueError(
            f"Annotation {annotation_id} bbox is outside image {image['file_name']}"
        )

    # Absorb harmless floating-point drift at an image edge.
    x1 = min(max(x, 0.0), width)
    y1 = min(max(y, 0.0), height)
    x2 = min(max(x + box_width, 0.0), width)
    y2 = min(max(y + box_height, 0.0), height)
    center_x = ((x1 + x2) / 2.0) / width
    center_y = ((y1 + y2) / 2.0) / height
    normalized_width = (x2 - x1) / width
    normalized_height = (y2 - y1) / height
    values = (center_x, center_y, normalized_width, normalized_height)
    if any(value < 0.0 or value > 1.0 for value in values):
        raise AssertionError("Normalized YOLO coordinates must be in [0, 1]")
    return f"{category_to_class[category_id]} " + " ".join(
        f"{value:.8f}" for value in values
    )


def _copy_image(source: Path, destination: Path, copy_mode: str) -> None:
    if copy_mode == "copy":
        shutil.copy2(source, destination)
    elif copy_mode == "hardlink":
        os.link(source, destination)
    elif copy_mode == "symlink":
        destination.symlink_to(source.resolve())
    else:
        raise ValueError(f"Unsupported copy mode: {copy_mode}")


def augment_medical_xray_intensity(image: np.ndarray, seed: int) -> np.ndarray:
    """Apply deterministic intensity-only augmentation without moving any pixel."""

    if image.dtype != np.uint8 or image.ndim not in {2, 3}:
        raise ValueError("Medical X-ray augmentation expects a 2D/3D uint8 image")
    rng = np.random.default_rng(seed)
    values = image.astype(np.float32) / 255.0

    # These narrow ranges simulate acquisition/display variation while retaining
    # anatomical geometry and avoiding synthetic local edges.
    contrast = float(rng.uniform(0.985, 1.015))
    brightness = float(rng.uniform(-0.002, 0.002))
    gamma = float(rng.uniform(0.985, 1.015))
    noise_sigma = float(rng.uniform(0.0005, 0.001))

    mean = float(values.mean())
    augmented = np.clip((values - mean) * contrast + mean, 0.0, 1.0)
    augmented = np.power(augmented, gamma)
    noise_shape = image.shape[:2] + ((1,) if image.ndim == 3 else ())
    noise = rng.normal(0.0, noise_sigma, size=noise_shape).astype(np.float32)
    augmented = np.clip(augmented + brightness + noise, 0.0, 1.0)
    return np.rint(augmented * 255.0).astype(np.uint8)


def _augmentation_seed(seed: int, file_name: str, variant: int) -> int:
    payload = f"{seed}:{file_name}:{variant}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _write_augmented_image(source: Path, destination: Path, seed: int) -> None:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency is declared by this project
        raise RuntimeError("OpenCV is required for medical image augmentation") from exc

    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unable to decode image for augmentation: {source}")
    augmented = augment_medical_xray_intensity(image, seed)
    if augmented.shape != image.shape:
        raise AssertionError("Intensity augmentation must preserve image dimensions")
    if not cv2.imwrite(str(destination), augmented):
        raise OSError(f"Unable to write augmented image: {destination}")


def prepare_yolo_dataset(
    coco_path: Path,
    images_dir: Path,
    output_dir: Path,
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    copy_mode: str = "copy",
    overwrite: bool = False,
    verify_images: bool = True,
    augmentations_per_train_image: int = 0,
) -> dict[str, Any]:
    """Build a YOLO dataset and return its reproducibility metadata."""

    coco_path = coco_path.resolve()
    images_dir = images_dir.resolve()
    output_dir = output_dir.resolve()
    if images_dir == output_dir or images_dir.is_relative_to(output_dir):
        raise ValueError("Output directory cannot contain the source image directory")
    if coco_path.is_relative_to(output_dir):
        raise ValueError("Output directory cannot contain the source COCO annotations")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_dir}; use overwrite=True")
    if augmentations_per_train_image < 0:
        raise ValueError("augmentations_per_train_image cannot be negative")

    coco = _read_coco(coco_path)
    images_by_id = _unique_by_id(coco["images"], "image")
    categories_by_id = _unique_by_id(coco["categories"], "category")
    _unique_by_id(coco["annotations"], "annotation")

    ordered_categories = sorted(categories_by_id.values(), key=lambda item: item["id"])
    category_to_class = {
        category["id"]: class_id for class_id, category in enumerate(ordered_categories)
    }
    class_names = [str(category.get("name", category["id"])) for category in ordered_categories]

    file_names: set[str] = set()
    label_names: set[str] = set()
    for image in images_by_id.values():
        file_name = image.get("file_name")
        if not isinstance(file_name, str) or not file_name or Path(file_name).name != file_name:
            raise ValueError("COCO image file_name must be a plain filename, not a path")
        if file_name in file_names:
            raise ValueError(f"Duplicate image file_name: {file_name}")
        file_names.add(file_name)
        label_name = str(Path(file_name).with_suffix(".txt"))
        if label_name in label_names:
            raise ValueError(f"Image names collide after conversion to labels: {label_name}")
        label_names.add(label_name)

        source = images_dir / file_name
        if not source.is_file():
            raise FileNotFoundError(f"COCO image does not exist: {source}")
        try:
            width, height = int(image["width"]), int(image["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid dimensions for COCO image {file_name}") from exc
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions for COCO image {file_name}")
        if verify_images:
            _validate_image_dimensions(source, width, height)

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        image_id = annotation.get("image_id")
        if image_id not in images_by_id:
            raise ValueError(
                f"Annotation {annotation.get('id')} references unknown image {image_id}"
            )
        # Validate all boxes before creating any output.
        _yolo_line(annotation, images_by_id[image_id], category_to_class)
        annotations_by_image[image_id].append(annotation)

    splits = _split_images(list(images_by_id.values()), train_ratio, val_ratio, seed)
    generated_names = {
        f"{Path(image['file_name']).stem}__aug{variant:02d}.png"
        for image in splits["train"]
        for variant in range(1, augmentations_per_train_image + 1)
    }
    collisions = generated_names & file_names
    if collisions:
        raise ValueError(
            "Generated augmentation names collide with source images: "
            + ", ".join(sorted(collisions))
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        split_metadata: dict[str, Any] = {}
        for split, split_images in splits.items():
            image_output = staging / "images" / split
            label_output = staging / "labels" / split
            image_output.mkdir(parents=True)
            label_output.mkdir(parents=True)
            annotation_count = 0
            augmented_image_count = 0
            class_counts: Counter[int] = Counter()
            for image in split_images:
                source = images_dir / image["file_name"]
                _copy_image(source, image_output / source.name, copy_mode)
                annotations = sorted(
                    annotations_by_image[image["id"]], key=lambda item: item["id"]
                )
                lines = [
                    _yolo_line(annotation, image, category_to_class)
                    for annotation in annotations
                ]
                label_text = "\n".join(lines) + ("\n" if lines else "")
                output_names = [source.name]
                if split == "train":
                    for variant in range(1, augmentations_per_train_image + 1):
                        augmented_name = f"{source.stem}__aug{variant:02d}.png"
                        _write_augmented_image(
                            source,
                            image_output / augmented_name,
                            _augmentation_seed(seed, source.name, variant),
                        )
                        output_names.append(augmented_name)
                        augmented_image_count += 1
                for output_name in output_names:
                    label_path = label_output / Path(output_name).with_suffix(".txt")
                    label_path.write_text(label_text, encoding="utf-8")
                    annotation_count += len(annotations)
                    class_counts.update(
                        category_to_class[annotation["category_id"]]
                        for annotation in annotations
                    )
            split_metadata[split] = {
                "source_images": len(split_images),
                "augmented_images": augmented_image_count,
                "images": len(split_images) + augmented_image_count,
                "annotations": annotation_count,
                "class_counts": {
                    class_names[class_id]: class_counts[class_id]
                    for class_id in range(len(class_names))
                },
                "files": [image["file_name"] for image in split_images],
            }

        yaml_lines = [
            f"path: {json.dumps(str(output_dir))}",
            "train: images/train",
            "val: images/val",
        ]
        if "test" in splits:
            yaml_lines.append("test: images/test")
        yaml_lines.extend(
            ["names:"]
            + [
                f"  {i}: {json.dumps(name)}"
                for i, name in enumerate(class_names)
            ]
        )
        (staging / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

        metadata = {
            "source": {
                "coco": str(coco_path),
                "images": str(images_dir),
                "coco_sha256": hashlib.sha256(coco_path.read_bytes()).hexdigest(),
            },
            "seed": seed,
            "ratios": {
                "train": train_ratio,
                "val": val_ratio,
                "test": max(0.0, 1.0 - train_ratio - val_ratio),
            },
            "copy_mode": copy_mode,
            "augmentations_per_train_image": augmentations_per_train_image,
            "augmentation": {
                "scope": "train only",
                "geometry": "unchanged",
                "operations": [
                    "global contrast (0.985-1.015)",
                    "global brightness (-0.002-0.002)",
                    "global gamma (0.985-1.015)",
                    "grayscale Gaussian noise sigma (0.0005-0.001)",
                ],
            },
            "classes": class_names,
            "splits": split_metadata,
        }
        (staging / "split.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return metadata
