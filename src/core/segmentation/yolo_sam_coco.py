"""Use YOLO tooth boxes as SAM prompts and export COCO 1.0 pseudo-labels."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .cvat_coco import (
    central_tooth_review_reasons,
    validate_coco_polygons,
    write_cvat_coco_dataset,
    write_cvat_coco_package,
)
from .sam_infer import (
    discover_images,
    mask_to_polygons,
    postprocess_mask,
    resolve_device,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_YOLO_MODEL = (
    PROJECT_ROOT
    / "reports/models/yolo11/tooth_detect_intensity_aug/weights/best.pt"
)
DEFAULT_SAM_CHECKPOINT = PROJECT_ROOT / "src/models/weights/sam_vit_b_01ec64.pth"
DEFAULT_INPUT = PROJECT_ROOT / "data/processed"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports/models/yolo_sam_pseudo_labels"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--yolo-model", type=Path, default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--sam-checkpoint", type=Path, default=DEFAULT_SAM_CHECKPOINT)
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--confidence", type=float, default=0.08)
    parser.add_argument("--review-confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--max-det", type=int, default=10)
    parser.add_argument("--sam-score-threshold", type=float, default=0.7)
    parser.add_argument(
        "--single-mask",
        action="store_true",
        help="Request one SAM mask instead of choosing the highest-scoring of three",
    )
    parser.add_argument("--limit", type=int, help="Process only the first N images")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--package-cvat-dataset",
        action="store_true",
        help="Also package all images into a large CVAT dataset ZIP",
    )
    return parser.parse_args()


def mask_geometry(mask: np.ndarray) -> tuple[list[list[float]], list[float], int]:
    """Return external polygons, tight COCO bbox, and pixel area for a mask."""

    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D array")
    binary = mask.astype(bool)
    area = int(binary.sum())
    polygons = mask_to_polygons(binary)
    if not polygons or area == 0:
        return [], [], area
    ys, xs = np.where(binary)
    bbox = [
        float(xs.min()),
        float(ys.min()),
        float(xs.max() - xs.min()),
        float(ys.max() - ys.min()),
    ]
    return polygons, bbox, area


def mask_prompt_geometry(mask: np.ndarray, box_xyxy: np.ndarray) -> tuple[float, bool]:
    """Measure mask spill outside its prompt box and whether it touches an image edge."""

    height, width = mask.shape
    x1, y1, x2, y2 = box_xyxy.astype(int)
    x1, x2 = max(0, x1), min(width - 1, x2)
    y1, y2 = max(0, y1), min(height - 1, y2)
    area = int(mask.sum())
    inside = int(mask[y1 : y2 + 1, x1 : x2 + 1].sum())
    outside_fraction = (area - inside) / area if area else 1.0
    touches_border = bool(
        mask[0, :].any()
        or mask[-1, :].any()
        or mask[:, 0].any()
        or mask[:, -1].any()
    )
    return outside_fraction, touches_border


def review_reasons(
    *,
    yolo_confidence: float,
    review_confidence: float,
    sam_score: float,
    mask_fraction: float,
    touches_image_border: bool,
    outside_box_fraction: float,
    has_polygon: bool,
    sam_score_threshold: float,
) -> list[str]:
    reasons = central_tooth_review_reasons(
        sam_score=sam_score,
        mask_fraction=mask_fraction,
        touches_image_border=touches_image_border,
        outside_box_fraction=outside_box_fraction,
        has_polygon=has_polygon,
        score_threshold=sam_score_threshold,
    )
    if yolo_confidence < review_confidence:
        reasons.insert(0, "low_yolo_confidence")
    return reasons


def _draw_overlay(
    rgb: np.ndarray,
    instances: list[dict[str, Any]],
    masks: list[np.ndarray],
) -> np.ndarray:
    overlay = rgb.copy()
    palette = np.array(
        [
            [255, 90, 60],
            [40, 190, 255],
            [100, 220, 100],
            [220, 100, 255],
            [255, 205, 60],
            [60, 230, 210],
        ],
        dtype=np.uint8,
    )
    for index, (instance, mask) in enumerate(zip(instances, masks)):
        color = palette[index % len(palette)]
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)
        x1, y1, x2, y2 = instance["yolo_box_xyxy"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color.tolist(), 2)
        label = f"{instance['annotation_id']} {instance['yolo_confidence']:.2f}"
        cv2.putText(
            overlay,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color.tolist(),
            2,
            cv2.LINE_AA,
        )
    return overlay


def _validate_args(args: argparse.Namespace) -> None:
    if not args.input.is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.input}")
    if not args.yolo_model.is_file():
        raise FileNotFoundError(f"YOLO model not found: {args.yolo_model}")
    if not args.sam_checkpoint.is_file():
        raise FileNotFoundError(f"SAM checkpoint not found: {args.sam_checkpoint}")
    if not 0.0 < args.confidence <= 1.0:
        raise ValueError("confidence must be in (0, 1]")
    if not args.confidence <= args.review_confidence <= 1.0:
        raise ValueError("review-confidence must be >= confidence and <= 1")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    output = args.output.resolve()
    input_dir = args.input.resolve()
    if input_dir == output or input_dir.is_relative_to(output):
        raise ValueError("Output directory cannot contain the input images")
    if args.yolo_model.resolve().is_relative_to(output):
        raise ValueError("Output directory cannot contain the YOLO checkpoint")
    if args.sam_checkpoint.resolve().is_relative_to(output):
        raise ValueError("Output directory cannot contain the SAM checkpoint")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}; pass --overwrite")

    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise SystemExit(
            "segment_anything is not installed. Run "
            "`uv sync --extra yolo --extra segmentation`."
        ) from exc
    from ultralytics import YOLO

    if args.sam_model_type not in sam_model_registry:
        raise ValueError(
            f"Unknown SAM model type {args.sam_model_type!r}; "
            f"available={sorted(sam_model_registry)}"
        )

    paths = discover_images(args.input.resolve())
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise ValueError(f"No images found under {args.input}")

    device = resolve_device(args.device)
    print(f"Loading YOLO from {args.yolo_model} on {device}...")
    detector = YOLO(str(args.yolo_model.resolve()))
    print(f"Loading SAM {args.sam_model_type} from {args.sam_checkpoint}...")
    sam = sam_model_registry[args.sam_model_type](checkpoint=str(args.sam_checkpoint.resolve()))
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    masks_dir = staging / "masks"
    overlays_dir = staging / "overlays"
    metadata_dir = staging / "metadata"
    for directory in (masks_dir, overlays_dir, metadata_dir):
        directory.mkdir(parents=True)

    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    image_metadata: list[dict[str, Any]] = []
    annotation_id = 1
    try:
        for image_id, path in enumerate(paths, start=1):
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Could not decode image: {path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            coco_images.append(
                {
                    "id": image_id,
                    "file_name": path.name,
                    "width": width,
                    "height": height,
                }
            )

            detection = detector.predict(
                source=rgb,
                imgsz=args.imgsz,
                conf=args.confidence,
                iou=args.iou,
                max_det=args.max_det,
                device=str(device),
                verbose=False,
            )[0]
            boxes = detection.boxes.xyxy.detach().cpu().numpy()
            confidences = detection.boxes.conf.detach().cpu().numpy()
            order = np.argsort((boxes[:, 0] + boxes[:, 2]) / 2.0) if len(boxes) else []

            instances: list[dict[str, Any]] = []
            final_masks: list[np.ndarray] = []
            if len(boxes):
                with torch.inference_mode():
                    predictor.set_image(rgb)
                    for detection_index in order:
                        box = boxes[detection_index].astype(np.float32)
                        yolo_confidence = float(confidences[detection_index])
                        candidate_masks, scores, _ = predictor.predict(
                            box=box,
                            multimask_output=not args.single_mask,
                        )
                        best_index = int(np.argmax(scores))
                        sam_score = float(scores[best_index])
                        final_mask = postprocess_mask(candidate_masks[best_index])
                        polygons, bbox, area = mask_geometry(final_mask)
                        outside_fraction, touches_border = mask_prompt_geometry(
                            final_mask, box
                        )
                        reasons = review_reasons(
                            yolo_confidence=yolo_confidence,
                            review_confidence=args.review_confidence,
                            sam_score=sam_score,
                            mask_fraction=area / final_mask.size,
                            touches_image_border=touches_border,
                            outside_box_fraction=outside_fraction,
                            has_polygon=bool(polygons),
                            sam_score_threshold=args.sam_score_threshold,
                        )
                        yolo_box = np.rint(box).astype(int).tolist()
                        instance = {
                            "annotation_id": annotation_id,
                            "yolo_box_xyxy": yolo_box,
                            "yolo_confidence": round(yolo_confidence, 6),
                            "sam_score": round(sam_score, 6),
                            "mask_area_pixels": area,
                            "mask_fraction": round(area / final_mask.size, 6),
                            "outside_box_fraction": round(outside_fraction, 6),
                            "touches_image_border": touches_border,
                            "flagged_for_review": bool(reasons),
                            "review_reasons": reasons,
                        }
                        instances.append(instance)
                        final_masks.append(final_mask)

                        if polygons:
                            coco_annotations.append(
                                {
                                    "id": annotation_id,
                                    "image_id": image_id,
                                    "category_id": 1,
                                    "segmentation": polygons,
                                    "area": float(area),
                                    "bbox": bbox,
                                    "iscrowd": 0,
                                    "attributes": {
                                        "yolo_confidence": round(yolo_confidence, 6),
                                        "sam_score": round(sam_score, 6),
                                        "yolo_box_xyxy": yolo_box,
                                        "requires_expert_correction": True,
                                        "flagged_for_review": bool(reasons),
                                        "review_reasons": reasons,
                                    },
                                }
                            )
                            instance_mask_dir = masks_dir / path.stem
                            instance_mask_dir.mkdir(exist_ok=True)
                            cv2.imwrite(
                                str(instance_mask_dir / f"{annotation_id}.png"),
                                final_mask.astype(np.uint8) * 255,
                            )
                        annotation_id += 1

            overlay = _draw_overlay(rgb, instances, final_masks)
            cv2.imwrite(
                str(overlays_dir / f"{path.stem}.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )
            per_image = {
                "image": str(path),
                "image_id": image_id,
                "detections": len(boxes),
                "valid_masks": sum(
                    1 for instance in instances if instance["mask_area_pixels"] > 0
                ),
                "instances": instances,
                "status": "pseudo-label; requires expert correction",
            }
            (metadata_dir / f"{path.stem}.json").write_text(
                json.dumps(per_image, indent=2), encoding="utf-8"
            )
            image_metadata.append(per_image)
            flagged = sum(i["flagged_for_review"] for i in instances)
            print(
                f"[{image_id}/{len(paths)}] {path.name}: "
                f"detections={len(boxes)} masks={len(instances)} flagged={flagged}"
            )

        coco = {
            "info": {
                "description": "YOLO11 box-prompted SAM tooth pseudo-labels",
                "version": "1.0",
                "models": {
                    "detector": args.yolo_model.name,
                    "segmenter": args.sam_checkpoint.name,
                },
                "thresholds": {
                    "yolo_confidence": args.confidence,
                    "yolo_iou": args.iou,
                    "sam_score_review": args.sam_score_threshold,
                },
                "status": "pseudo-labels; require dental expert correction",
            },
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": [{"id": 1, "name": "tooth", "supercategory": "tooth"}],
        }
        validate_coco_polygons(coco)
        (staging / "coco_annotations.json").write_text(
            json.dumps(coco, indent=2), encoding="utf-8"
        )
        write_cvat_coco_package(coco, staging / "cvat_coco_annotations.zip")
        if args.package_cvat_dataset:
            write_cvat_coco_dataset(
                coco, paths, staging / "cvat_coco_dataset.zip"
            )
        summary = {
            "images": len(coco_images),
            "annotations": len(coco_annotations),
            "images_without_detections": sum(
                item["detections"] == 0 for item in image_metadata
            ),
            "flagged_annotations": sum(
                instance["flagged_for_review"]
                for item in image_metadata
                for instance in item["instances"]
            ),
            "per_image": image_metadata,
        }
        (staging / "metadata.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        if output.exists():
            shutil.rmtree(output)
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"Wrote {len(coco_annotations)} COCO annotations for {len(paths)} images")
    print(f"COCO JSON: {output / 'coco_annotations.json'}")
    print(f"CVAT ZIP: {output / 'cvat_coco_annotations.zip'}")


if __name__ == "__main__":
    main()
