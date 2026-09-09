import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from src.core.segmentation.cvat_coco import (
    central_tooth_review_reasons,
    validate_coco_polygons,
    write_cvat_coco_dataset,
    write_cvat_coco_package,
)


class CvatCocoPackageTests(unittest.TestCase):
    def test_writes_cvat_coco_zip_with_instances_default_json(self) -> None:
        coco = {
            "images": [{"id": 1, "file_name": "1.png", "width": 100, "height": 200}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": [[10.0, 20.0, 40.0, 20.0, 40.0, 80.0]],
                    "area": 1800.0,
                    "bbox": [10.0, 20.0, 30.0, 60.0],
                    "iscrowd": 0,
                }
            ],
            "categories": [{"id": 1, "name": "tooth", "supercategory": "tooth"}],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "cvat_coco_annotations.zip"
            write_cvat_coco_package(coco, output)

            with zipfile.ZipFile(output) as archive:
                self.assertEqual(archive.namelist(), ["annotations/instances_default.json"])
                packaged = json.loads(
                    archive.read("annotations/instances_default.json")
                )

            self.assertEqual(packaged, coco)

    def test_writes_full_cvat_dataset_with_images(self) -> None:
        coco = {
            "images": [{"id": 1, "file_name": "1.png", "width": 100, "height": 200}],
            "annotations": [],
            "categories": [{"id": 1, "name": "tooth"}],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / "1.png"
            image.write_bytes(b"fake image bytes")
            output = root / "cvat_coco_dataset.zip"

            write_cvat_coco_dataset(coco, [image], output)

            with zipfile.ZipFile(output) as archive:
                self.assertEqual(
                    archive.namelist(),
                    ["annotations/instances_default.json", "images/default/1.png"],
                )
                self.assertEqual(
                    archive.read("images/default/1.png"), b"fake image bytes"
                )

    def test_rejects_polygon_with_odd_coordinate_count(self) -> None:
        coco = {
            "images": [{"id": 1, "file_name": "1.png", "width": 100, "height": 200}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": [[10.0, 20.0, 40.0, 20.0, 40.0]],
                    "area": 100.0,
                    "bbox": [10.0, 20.0, 30.0, 60.0],
                    "iscrowd": 0,
                }
            ],
            "categories": [{"id": 1, "name": "tooth"}],
        }

        with self.assertRaisesRegex(ValueError, "even number"):
            validate_coco_polygons(coco)

    def test_flags_geometrically_implausible_central_tooth_mask(self) -> None:
        reasons = central_tooth_review_reasons(
            sam_score=0.95,
            mask_fraction=0.55,
            touches_image_border=True,
            outside_box_fraction=0.25,
            has_polygon=True,
        )

        self.assertEqual(
            reasons,
            ["mask_too_large", "touches_image_border", "mostly_outside_prompt_box"],
        )


if __name__ == "__main__":
    unittest.main()
