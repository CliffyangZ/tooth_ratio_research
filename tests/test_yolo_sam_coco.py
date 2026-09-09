import unittest

import numpy as np

from src.core.segmentation.cvat_coco import validate_coco_polygons
from src.core.segmentation.yolo_sam_coco import (
    mask_geometry,
    mask_prompt_geometry,
    review_reasons,
)


class YoloSamCocoTests(unittest.TestCase):
    def test_mask_geometry_produces_valid_coco_annotation(self) -> None:
        mask = np.zeros((100, 80), dtype=bool)
        mask[20:70, 10:40] = True
        polygons, bbox, area = mask_geometry(mask)
        coco = {
            "images": [{"id": 1, "file_name": "1.png", "width": 80, "height": 100}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": polygons,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                }
            ],
            "categories": [{"id": 1, "name": "tooth"}],
        }

        validate_coco_polygons(coco)
        self.assertEqual(bbox, [10.0, 20.0, 29.0, 49.0])
        self.assertEqual(area, 1500)

    def test_prompt_geometry_detects_spill_and_image_border(self) -> None:
        mask = np.zeros((20, 20), dtype=bool)
        mask[0:10, 2:12] = True
        outside, touches = mask_prompt_geometry(
            mask, np.array([2, 0, 6, 9], dtype=np.float32)
        )

        self.assertAlmostEqual(outside, 0.5)
        self.assertTrue(touches)

    def test_review_reasons_include_detector_and_segmenter_quality(self) -> None:
        reasons = review_reasons(
            yolo_confidence=0.1,
            review_confidence=0.25,
            sam_score=0.6,
            mask_fraction=0.1,
            touches_image_border=False,
            outside_box_fraction=0.0,
            has_polygon=True,
            sam_score_threshold=0.7,
        )

        self.assertEqual(reasons, ["low_yolo_confidence", "low_sam_score"])


if __name__ == "__main__":
    unittest.main()

