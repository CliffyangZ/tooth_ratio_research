"""Convert selected CVAT XML image annotations into a COCO 1.0 dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.segmentation.cvat_coco import validate_coco_polygons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, required=True, help="CVAT XML file")
    parser.add_argument("--images", type=Path, required=True, help="Source image directory")
    parser.add_argument("--output", type=Path, required=True, help="COCO dataset root")
    parser.add_argument(
        "--selected-ids",
        type=int,
        nargs="+",
        required=True,
        help="Numeric image stems to include, in dataset order",
    )
    return parser.parse_args()


def polygon_area(polygon: list[float]) -> float:
    points = list(zip(polygon[0::2], polygon[1::2]))
    return abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
        )
    ) / 2.0


def polygon_from_cvat(shape: ET.Element) -> list[list[float]]:
    polygon = [
        coordinate
        for point in shape.attrib["points"].split(";")
        for coordinate in map(float, point.split(","))
    ]
    if len(polygon) < 6 or len(polygon) % 2:
        raise ValueError("CVAT polygon must contain at least three coordinate pairs")
    return [polygon]


def polygons_from_cvat_mask(shape: ET.Element) -> list[list[float]]:
    width = int(shape.attrib["width"])
    height = int(shape.attrib["height"])
    left = int(shape.attrib["left"])
    top = int(shape.attrib["top"])
    runs = np.asarray(
        [int(value.strip()) for value in shape.attrib["rle"].split(",")],
        dtype=np.int64,
    )
    if np.any(runs < 0) or int(runs.sum()) != width * height:
        raise ValueError("CVAT mask RLE does not match its declared dimensions")

    # CVAT uses row-major alternating background/foreground runs, beginning
    # with background. COCO polygons use coordinates in the full image frame.
    values = np.arange(len(runs), dtype=np.uint8) % 2
    mask = np.repeat(values, runs).reshape(height, width)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is not None and any(parent != -1 for parent in hierarchy[0, :, 3]):
        raise ValueError("A CVAT mask with holes cannot be represented exactly as COCO polygons")

    polygons: list[list[float]] = []
    for contour in contours:
        if len(contour) < 3 or cv2.contourArea(contour) <= 0:
            continue
        coordinates = contour.reshape(-1, 2).astype(float)
        coordinates[:, 0] += left
        coordinates[:, 1] += top
        polygons.append(coordinates.flatten().tolist())
    if not polygons:
        raise ValueError("CVAT mask did not contain a valid foreground polygon")
    return polygons


def annotation_geometry(polygons: list[list[float]]) -> tuple[list[float], float]:
    xs = [x for polygon in polygons for x in polygon[0::2]]
    ys = [y for polygon in polygons for y in polygon[1::2]]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min], sum(
        polygon_area(polygon) for polygon in polygons
    )


def build_coco(xml_path: Path, selected_ids: list[int]) -> dict:
    root = ET.parse(xml_path).getroot()
    images_by_name = {image.attrib["name"]: image for image in root.findall("image")}
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Selected IDs must be unique")

    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    annotation_id = 1
    for selected_id in selected_ids:
        file_name = f"{selected_id}.png"
        if file_name not in images_by_name:
            raise ValueError(f"Selected image is absent from CVAT XML: {file_name}")
        image = images_by_name[file_name]
        width = int(image.attrib["width"])
        height = int(image.attrib["height"])
        coco_images.append(
            {
                "id": selected_id,
                "width": width,
                "height": height,
                "file_name": file_name,
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0,
            }
        )

        for shape in image:
            if shape.tag == "polygon":
                polygons = polygon_from_cvat(shape)
            elif shape.tag == "mask":
                polygons = polygons_from_cvat_mask(shape)
            else:
                continue
            if shape.attrib.get("label") != "tooth":
                raise ValueError(f"Unsupported label: {shape.attrib.get('label')!r}")
            bbox, area = annotation_geometry(polygons)
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": selected_id,
                    "category_id": 1,
                    "segmentation": polygons,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                    "attributes": {"occluded": shape.attrib.get("occluded") == "1"},
                }
            )
            annotation_id += 1

    dumped = root.findtext("meta/dumped", default="")
    coco = {
        "info": {
            "description": "Selected expert-reviewed tooth segmentations",
            "url": "",
            "version": "1.0",
            "year": int(dumped[:4]) if dumped[:4].isdigit() else 0,
            "contributor": "",
            "date_created": dumped,
        },
        "licenses": [{"id": 0, "name": "", "url": ""}],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": 1, "name": "tooth", "supercategory": "tooth"}],
    }
    validate_coco_polygons(coco)
    return coco


def write_dataset(coco: dict, source_images: Path, output: Path) -> None:
    annotations_dir = output / "annotations"
    images_dir = output / "images" / "default"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    expected_names = {image["file_name"] for image in coco["images"]}
    unexpected = sorted(
        path.name for path in images_dir.iterdir() if path.name not in expected_names
    )
    if unexpected:
        raise ValueError(f"Output image directory contains unexpected files: {unexpected}")

    for image in coco["images"]:
        source = source_images / image["file_name"]
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, images_dir / source.name)

    annotation_path = annotations_dir / "instances_default.json"
    annotation_path.write_text(
        json.dumps(coco, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    coco = build_coco(args.xml, args.selected_ids)
    write_dataset(coco, args.images, args.output)
    print(
        f"Wrote {len(coco['images'])} images and "
        f"{len(coco['annotations'])} annotations to {args.output}"
    )


if __name__ == "__main__":
    main()
