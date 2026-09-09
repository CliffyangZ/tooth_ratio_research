"""Validation and packaging helpers for CVAT-compatible COCO polygons."""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path
from typing import Any

CVAT_COCO_ANNOTATION_PATH = "annotations/instances_default.json"


def central_tooth_review_reasons(
    *,
    sam_score: float,
    mask_fraction: float,
    touches_image_border: bool,
    outside_box_fraction: float,
    has_polygon: bool,
    score_threshold: float = 0.7,
    min_mask_fraction: float = 0.02,
    max_mask_fraction: float = 0.45,
    max_outside_box_fraction: float = 0.15,
) -> list[str]:
    """Return conservative QA flags; these do not turn pseudo-labels into truth."""
    reasons: list[str] = []
    if sam_score < score_threshold:
        reasons.append("low_sam_score")
    if mask_fraction < min_mask_fraction:
        reasons.append("mask_too_small")
    if mask_fraction > max_mask_fraction:
        reasons.append("mask_too_large")
    if touches_image_border:
        reasons.append("touches_image_border")
    if outside_box_fraction > max_outside_box_fraction:
        reasons.append("mostly_outside_prompt_box")
    if not has_polygon:
        reasons.append("no_valid_polygon")
    return reasons


def validate_coco_polygons(coco: dict[str, Any]) -> None:
    """Fail closed on malformed COCO instance polygons before CVAT import."""
    for key in ("images", "annotations", "categories"):
        if not isinstance(coco.get(key), list):
            raise ValueError(f"COCO field {key!r} must be a list")

    images = {image["id"]: image for image in coco["images"]}
    category_ids = {category["id"] for category in coco["categories"]}
    if len(images) != len(coco["images"]):
        raise ValueError("COCO image IDs must be unique")

    annotation_ids: set[int] = set()
    for annotation in coco["annotations"]:
        annotation_id = annotation["id"]
        if annotation_id in annotation_ids:
            raise ValueError("COCO annotation IDs must be unique")
        annotation_ids.add(annotation_id)

        if annotation["image_id"] not in images:
            raise ValueError(f"Annotation {annotation_id} references an unknown image")
        if annotation["category_id"] not in category_ids:
            raise ValueError(f"Annotation {annotation_id} references an unknown category")
        if annotation.get("iscrowd") != 0:
            raise ValueError("Polygon annotations must use iscrowd=0")

        image = images[annotation["image_id"]]
        width, height = image["width"], image["height"]
        polygons = annotation.get("segmentation")
        if not isinstance(polygons, list) or not polygons:
            raise ValueError(f"Annotation {annotation_id} has no polygons")
        for polygon in polygons:
            if len(polygon) % 2:
                raise ValueError("Each polygon must contain an even number of coordinates")
            if len(polygon) < 6:
                raise ValueError("Each polygon needs at least three coordinate pairs")
            if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in polygon):
                raise ValueError("Polygon coordinates must be finite numbers")
            xs, ys = polygon[0::2], polygon[1::2]
            if min(xs) < 0 or max(xs) >= width or min(ys) < 0 or max(ys) >= height:
                raise ValueError(f"Annotation {annotation_id} has coordinates outside its image")

        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            raise ValueError(f"Annotation {annotation_id} has an invalid bbox")
        if annotation.get("area", 0) <= 0:
            raise ValueError(f"Annotation {annotation_id} has a non-positive area")


def write_cvat_coco_package(coco: dict[str, Any], output_path: Path) -> None:
    """Validate and write an annotation-only COCO archive."""
    validate_coco_polygons(coco)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(coco, indent=2).encode("utf-8")
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(CVAT_COCO_ANNOTATION_PATH, payload)


def write_cvat_coco_dataset(
    coco: dict[str, Any],
    image_paths: list[Path],
    output_path: Path,
    subset: str = "default",
) -> None:
    """Write a complete CVAT-importable COCO dataset archive with images."""
    validate_coco_polygons(coco)
    paths_by_name = {Path(path).name: Path(path) for path in image_paths}
    if len(paths_by_name) != len(image_paths):
        raise ValueError("Input image filenames must be unique")

    coco_names = [image["file_name"] for image in coco["images"]]
    if len(set(coco_names)) != len(coco_names):
        raise ValueError("COCO image filenames must be unique")
    missing = sorted(set(coco_names) - set(paths_by_name))
    extra = sorted(set(paths_by_name) - set(coco_names))
    if missing or extra:
        raise ValueError(f"COCO/image filename mismatch: missing={missing}, extra={extra}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path = f"annotations/instances_{subset}.json"
    payload = json.dumps(coco, indent=2).encode("utf-8")
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(annotation_path, payload)
        for name in sorted(coco_names):
            archive.write(paths_by_name[name], f"images/{subset}/{name}")
