"""Prepare the tooth detection dataset and fine-tune Ultralytics YOLO11."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from src.models.yolo_dataset import prepare_yolo_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coco",
        type=Path,
        default=PROJECT_ROOT / "data/bounding_box/instances_default.json",
        help="COCO detection annotations",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed",
        help="Directory containing the COCO images",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "data/yolo_tooth",
        help="Generated Ultralytics dataset root",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy-mode", choices=("copy", "hardlink", "symlink"), default="copy"
    )
    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="Replace an existing generated dataset",
    )
    parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="Trust dimensions in COCO JSON instead of decoding every image",
    )
    parser.add_argument(
        "--augmentations-per-image",
        type=int,
        default=3,
        help="Intensity-only augmented copies per training image; use 0 to disable",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Convert and validate the data without importing Ultralytics",
    )

    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics device, e.g. 0, cpu, mps; omitted means automatic",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument(
        "--project",
        type=Path,
        default=PROJECT_ROOT / "reports/models/yolo11",
    )
    parser.add_argument("--name", default="tooth_detect")
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from a previous last.pt checkpoint",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Do not evaluate best.pt against the held-out test split",
    )
    return parser.parse_args()


def _print_dataset_summary(metadata: dict[str, Any], yaml_path: Path) -> None:
    print(f"YOLO dataset: {yaml_path}")
    for split, stats in metadata["splits"].items():
        print(
            f"  {split}: {stats['images']} images, "
            f"{stats['annotations']} annotations"
        )


def _validate_existing_dataset(args: argparse.Namespace, metadata: dict[str, Any]) -> None:
    expected = {
        "coco": str(args.coco.resolve()),
        "images": str(args.images_dir.resolve()),
        "coco_sha256": hashlib.sha256(args.coco.read_bytes()).hexdigest(),
    }
    source = metadata.get("source", {})
    ratios = metadata.get("ratios", {})
    stale = any(source.get(key) != value for key, value in expected.items())
    stale = stale or metadata.get("seed") != args.seed
    stale = stale or metadata.get("copy_mode") != args.copy_mode
    stale = stale or metadata.get("augmentations_per_train_image") != args.augmentations_per_image
    stale = stale or not math.isclose(
        float(ratios.get("train", -1)), args.train_ratio, abs_tol=1e-9
    )
    stale = stale or not math.isclose(
        float(ratios.get("val", -1)), args.val_ratio, abs_tol=1e-9
    )
    if stale:
        raise ValueError(
            "The generated dataset does not match the requested source or split "
            "settings; run again with --rebuild-dataset"
        )


def main() -> None:
    args = parse_args()
    dataset_yaml = args.dataset_dir.resolve() / "dataset.yaml"
    metadata_path = args.dataset_dir.resolve() / "split.json"
    if args.rebuild_dataset or not dataset_yaml.is_file():
        metadata = prepare_yolo_dataset(
            args.coco,
            args.images_dir,
            args.dataset_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            copy_mode=args.copy_mode,
            overwrite=args.rebuild_dataset,
            verify_images=not args.skip_image_check,
            augmentations_per_train_image=args.augmentations_per_image,
        )
    else:
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Existing dataset has no split metadata: {metadata_path}; "
                "run with --rebuild-dataset"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        _validate_existing_dataset(args, metadata)
    _print_dataset_summary(metadata, dataset_yaml)
    if args.prepare_only:
        return

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Run `uv sync --extra yolo`, then retry."
        ) from exc

    train_kwargs: dict[str, Any] = {
        "data": str(dataset_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "patience": args.patience,
        "seed": args.seed,
        "deterministic": True,
        "project": str(args.project.resolve()),
        "name": args.name,
        "plots": True,
        # No online augmentation may alter dental anatomy. Intensity-only
        # variants are generated offline with unchanged labels.
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    if args.resume:
        model = YOLO(str(args.resume.resolve()))
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        model.train(**train_kwargs)

    if args.skip_test or "test" not in metadata["splits"]:
        return
    best_path = Path(model.trainer.best)
    if not best_path.is_file():
        raise FileNotFoundError(f"Training completed but best checkpoint is absent: {best_path}")
    test_model = YOLO(str(best_path))
    val_kwargs: dict[str, Any] = {
        "data": str(dataset_yaml),
        "split": "test",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(args.project.resolve()),
        "name": f"{args.name}_test",
        "plots": True,
    }
    if args.device is not None:
        val_kwargs["device"] = args.device
    test_model.val(**val_kwargs)


if __name__ == "__main__":
    main()
