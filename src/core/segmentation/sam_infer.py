"""Batch SAM (ViT-B) inference over preprocessed dental X-rays, producing
pseudo-label masks and a COCO 1.0 annotation file importable into CVAT.

Prompting follows preprocess.md Step 3: a box prompt over the central tooth
(BOX_FRACTIONS, x1/y1/x2/y2 as fractions of width/height) plus a positive
point at the box center and a negative point near the box's inner edge.
Mask post-processing (Step 4) keeps the largest connected component and
applies a 5x5 morphological closing.

Every output mask is a pseudo-label (see preprocess.md Step 5) and must be
corrected by a dental expert in CVAT before use as training ground truth.

Usage:
    python -m src.core.segmentation.sam_infer \
        --input data/processed \
        --checkpoint src/models/weights/sam_vit_b_01ec64.pth \
        --model-type vit_b \
        --device auto \
        --output reports/models/sam_pseudo_labels
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch

from .cvat_coco import (
    central_tooth_review_reasons,
    validate_coco_polygons,
    write_cvat_coco_dataset,
    write_cvat_coco_package,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
BOX_FRACTIONS = (0.32, 0.05, 0.68, 0.95)
SAM_SCORE_REVIEW_THRESHOLD = 0.7


def _natural_key(path: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.stem)]


def discover_images(input_dir: Path) -> list[Path]:
    paths = [p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    paths.sort(key=_natural_key)
    return paths


def box_and_points(w: int, h: int, box_fractions: tuple[float, float, float, float]):
    x1 = int(box_fractions[0] * w)
    y1 = int(box_fractions[1] * h)
    x2 = int(box_fractions[2] * w)
    y2 = int(box_fractions[3] * h)
    box = np.array([x1, y1, x2, y2], dtype=np.float32)

    point_coords = np.array(
        [
            [(x1 + x2) / 2, (y1 + y2) / 2],
            [x1 + 0.03 * (x2 - x1), (y1 + y2) / 2],
        ],
        dtype=np.float32,
    )
    point_labels = np.array([1, 0], dtype=np.int32)
    return box, point_coords, point_labels


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 1:
        return mask.astype(bool)
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest_label


def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    clean = keep_largest_component(mask)
    kernel = np.ones((5, 5), np.uint8)
    clean = cv2.morphologyEx(clean.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return clean.astype(bool)


def mask_to_polygons(mask: np.ndarray) -> list[list[float]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        polygons.append(contour.reshape(-1).astype(float).tolist())
    return polygons


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda but CUDA is not available")
    return torch.device(requested)


def main() -> None:
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise SystemExit(
            "segment_anything is not installed. Run "
            "`uv sync --extra segmentation`."
        ) from exc

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("src/models/weights/sam_vit_b_01ec64.pth"),
    )
    parser.add_argument("--model-type", default="vit_b", choices=sorted(sam_model_registry.keys()))
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--output", type=Path, default=Path("reports/models/sam_pseudo_labels"))
    parser.add_argument(
        "--package-cvat-dataset",
        action="store_true",
        help="Also create a full CVAT COCO ZIP containing images (about the input size).",
    )
    parser.add_argument("--box-fractions", type=float, nargs=4, default=BOX_FRACTIONS, metavar=("X1", "Y1", "X2", "Y2"))
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"SAM checkpoint not found: {args.checkpoint}")

    paths = discover_images(args.input)
    if not paths:
        raise SystemExit(f"No images found under {args.input}")
    print(f"Found {len(paths)} images under {args.input}")

    device = resolve_device(args.device)
    print(f"Loading SAM {args.model_type} on {device}...")
    sam = sam_model_registry[args.model_type](checkpoint=str(args.checkpoint))
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)

    masks_dir = args.output / "masks"
    overlays_dir = args.output / "overlays"
    metadata_dir = args.output / "metadata"
    for d in (masks_dir, overlays_dir, metadata_dir):
        d.mkdir(parents=True, exist_ok=True)

    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    all_metadata: list[dict] = []
    next_annotation_id = 1

    for image_id, path in enumerate(paths, start=1):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        box, point_coords, point_labels = box_and_points(w, h, tuple(args.box_fractions))

        with torch.inference_mode():
            predictor.set_image(rgb)
            masks, scores, _ = predictor.predict(
                box=box,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )

        best_index = int(np.argmax(scores))
        sam_score = float(scores[best_index])
        final_mask = postprocess_mask(masks[best_index])
        mask_area = int(final_mask.sum())
        mask_fraction = mask_area / final_mask.size
        polygons = mask_to_polygons(final_mask)

        x1, y1, x2, y2 = box.astype(int)
        inside_box_area = int(final_mask[y1 : y2 + 1, x1 : x2 + 1].sum())
        outside_box_fraction = (
            (mask_area - inside_box_area) / mask_area if mask_area else 1.0
        )
        touches_image_border = bool(
            final_mask[0, :].any()
            or final_mask[-1, :].any()
            or final_mask[:, 0].any()
            or final_mask[:, -1].any()
        )
        review_reasons = central_tooth_review_reasons(
            sam_score=sam_score,
            mask_fraction=mask_fraction,
            touches_image_border=touches_image_border,
            outside_box_fraction=outside_box_fraction,
            has_polygon=bool(polygons),
            score_threshold=SAM_SCORE_REVIEW_THRESHOLD,
        )

        mask_path = masks_dir / f"{path.stem}.png"
        overlay_path = overlays_dir / f"{path.stem}.png"
        metadata_path = metadata_dir / f"{path.stem}.json"

        cv2.imwrite(str(mask_path), final_mask.astype(np.uint8) * 255)

        overlay = rgb.copy()
        color = np.array([255, 80, 40], dtype=np.uint8)
        overlay[final_mask] = (
            0.55 * overlay[final_mask] + 0.45 * color
        ).astype(np.uint8)
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        metadata = {
            "image": str(path),
            "target_scope": "single_central_tooth",
            "model": f"sam_{args.model_type}",
            "checkpoint": args.checkpoint.name,
            "device": str(device),
            "box_xyxy": box.astype(int).tolist(),
            "point_coords_xy": point_coords.round(1).tolist(),
            "point_labels": point_labels.tolist(),
            "sam_score": round(sam_score, 4),
            "mask_area_pixels": mask_area,
            "mask_fraction": round(mask_fraction, 4),
            "outside_box_fraction": round(outside_box_fraction, 4),
            "touches_image_border": touches_image_border,
            "status": "pseudo-label; requires expert correction",
            "flagged_for_review": bool(review_reasons),
            "review_reasons": review_reasons,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))
        all_metadata.append(metadata)

        coco_images.append(
            {"id": image_id, "file_name": path.name, "width": w, "height": h}
        )

        if polygons:
            ys, xs = np.where(final_mask)
            bbox = [
                float(xs.min()),
                float(ys.min()),
                float(xs.max() - xs.min()),
                float(ys.max() - ys.min()),
            ]
            coco_annotations.append(
                {
                    "id": next_annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": polygons,
                    "area": float(mask_area),
                    "bbox": bbox,
                    "iscrowd": 0,
                    "attributes": {
                        "sam_score": round(sam_score, 4),
                        "requires_expert_correction": True,
                    },
                }
            )
            next_annotation_id += 1

        print(f"[{image_id}/{len(paths)}] {path.name}: sam_score={sam_score:.3f} mask_fraction={mask_fraction:.3f}")

    coco = {
        "info": {
            "description": "SAM pseudo-labels for one central tooth per dental X-ray",
            "version": "1.0",
        },
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": 1, "name": "tooth", "supercategory": "tooth"}],
    }
    validate_coco_polygons(coco)

    coco_path = args.output / "coco_annotations.json"
    coco_path.write_text(json.dumps(coco, indent=2))
    cvat_zip_path = args.output / "cvat_coco_annotations.zip"
    write_cvat_coco_package(coco, cvat_zip_path)
    cvat_dataset_path = args.output / "cvat_coco_dataset.zip"
    if args.package_cvat_dataset:
        write_cvat_coco_dataset(coco, paths, cvat_dataset_path)

    all_metadata_path = args.output / "metadata.json"
    all_metadata_path.write_text(json.dumps(all_metadata, indent=2))

    scores = np.array([m["sam_score"] for m in all_metadata])
    flagged = sum(1 for m in all_metadata if m["flagged_for_review"])
    print()
    print(
        f"sam_score: mean={scores.mean():.3f} median={np.median(scores):.3f} "
        f"min={scores.min():.3f} max={scores.max():.3f}"
    )
    print(f"Flagged for expert review by score/geometry QA: {flagged}/{len(all_metadata)}")
    print(f"Wrote COCO 1.0 JSON to {coco_path}")
    print(f"Wrote CVAT COCO archive to {cvat_zip_path}")
    if args.package_cvat_dataset:
        print(f"Wrote full CVAT COCO dataset to {cvat_dataset_path}")
    print(f"Wrote per-image metadata to {metadata_dir} and {all_metadata_path}")


if __name__ == "__main__":
    main()
