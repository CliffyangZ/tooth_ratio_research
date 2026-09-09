import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.models.yolo_dataset import (
    augment_medical_xray_intensity,
    prepare_yolo_dataset,
)


class YoloDatasetTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        images_dir = root / "images"
        images_dir.mkdir()
        images = []
        annotations = []
        for image_id in range(1, 5):
            file_name = f"{image_id}.png"
            (images_dir / file_name).write_bytes(b"image fixture")
            images.append(
                {"id": image_id, "file_name": file_name, "width": 100, "height": 200}
            )
            annotations.append(
                {
                    "id": image_id,
                    "image_id": image_id,
                    "category_id": 7,
                    "bbox": [10.0, 20.0, 30.0, 80.0],
                }
            )
        coco = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 7, "name": "tooth"}],
        }
        coco_path = root / "instances.json"
        coco_path.write_text(json.dumps(coco), encoding="utf-8")
        return coco_path, images_dir

    def test_converts_coco_boxes_and_creates_deterministic_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            coco_path, images_dir = self._fixture(root)
            output = root / "yolo"
            metadata = prepare_yolo_dataset(
                coco_path,
                images_dir,
                output,
                train_ratio=0.5,
                val_ratio=0.25,
                seed=11,
                verify_images=False,
            )

            self.assertEqual(
                {split: stats["images"] for split, stats in metadata["splits"].items()},
                {"train": 2, "val": 1, "test": 1},
            )
            self.assertEqual(
                sum(stats["annotations"] for stats in metadata["splits"].values()), 4
            )
            labels = list((output / "labels").glob("*/*.txt"))
            self.assertEqual(len(labels), 4)
            self.assertEqual(
                labels[0].read_text(),
                "0 0.25000000 0.30000000 0.30000000 0.40000000\n",
            )
            self.assertIn('path: "', (output / "dataset.yaml").read_text())
            self.assertTrue((output / "split.json").is_file())

    def test_rejects_out_of_bounds_bbox_before_writing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            coco_path, images_dir = self._fixture(root)
            coco = json.loads(coco_path.read_text())
            coco["annotations"][0]["bbox"] = [90.0, 20.0, 30.0, 80.0]
            coco_path.write_text(json.dumps(coco))
            output = root / "yolo"

            with self.assertRaisesRegex(ValueError, "outside image"):
                prepare_yolo_dataset(
                    coco_path, images_dir, output, verify_images=False
                )
            self.assertFalse(output.exists())

    def test_refuses_to_replace_existing_dataset_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            coco_path, images_dir = self._fixture(root)
            output = root / "yolo"
            output.mkdir()

            with self.assertRaises(FileExistsError):
                prepare_yolo_dataset(
                    coco_path, images_dir, output, verify_images=False
                )

    def test_medical_augmentation_preserves_geometry_and_grayscale(self) -> None:
        gradient = np.tile(np.arange(100, dtype=np.uint8), (200, 1))
        image = np.repeat(gradient[:, :, None], 3, axis=2)

        first = augment_medical_xray_intensity(image, seed=123)
        second = augment_medical_xray_intensity(image, seed=123)

        self.assertEqual(first.shape, image.shape)
        self.assertEqual(first.dtype, image.dtype)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[:, :, 0], first[:, :, 1])
        np.testing.assert_array_equal(first[:, :, 1], first[:, :, 2])
        self.assertFalse(np.array_equal(first, image))

    def test_augments_train_only_and_copies_labels_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            coco_path, images_dir = self._fixture(root)
            for image_path in images_dir.glob("*.png"):
                image = np.full((200, 100, 3), 128, dtype=np.uint8)
                self.assertTrue(cv2.imwrite(str(image_path), image))
            output = root / "yolo"

            metadata = prepare_yolo_dataset(
                coco_path,
                images_dir,
                output,
                train_ratio=0.5,
                val_ratio=0.25,
                seed=11,
                verify_images=True,
                augmentations_per_train_image=2,
            )

            self.assertEqual(metadata["splits"]["train"]["source_images"], 2)
            self.assertEqual(metadata["splits"]["train"]["augmented_images"], 4)
            self.assertEqual(metadata["splits"]["train"]["images"], 6)
            self.assertEqual(metadata["splits"]["val"]["augmented_images"], 0)
            self.assertEqual(metadata["splits"]["test"]["augmented_images"], 0)
            for augmented_label in (output / "labels/train").glob("*__aug*.txt"):
                original_label = augmented_label.with_name(
                    augmented_label.name.split("__aug", 1)[0] + ".txt"
                )
                self.assertEqual(augmented_label.read_bytes(), original_label.read_bytes())


if __name__ == "__main__":
    unittest.main()
