"""Fine-tune SAM (ViT-B)'s mask decoder on expert-corrected tooth masks.

INFRASTRUCTURE ONLY -- do not run yet. This script consumes a COCO 1.0
instance-segmentation export from CVAT (Tasks -> Export annotations ->
COCO 1.0), i.e. the pseudo-labels produced by
`src/core/segmentation/sam_infer.py` *after* a dental expert has corrected
them in CVAT (preprocess.md Step 5). As of 2026-08-23 no such export exists
in this repo yet, so there is no ground truth to train against. Run this
only once a corrected COCO export is available.

Approach: freeze the image encoder and prompt encoder (their weights are
generic and not what needs adapting for tooth boundaries); train only the
mask decoder against the corrected masks, using a box prompt derived from
each mask's bounding box. This keeps training cheap (single GPU) and low
risk of catastrophic forgetting.

Usage (once a corrected COCO export exists):
    python src/models/finetune_sam.py \
        --coco-annotations <path to corrected CVAT COCO export>/annotations.json \
        --images-dir <path to the corresponding images> \
        --checkpoint ~/.cache/tooth_ratio/sam_vit_b_01ec64.pth \
        --output src/models/weights/sam_vit_b_tooth_finetuned.pth \
        --epochs 20 --lr 1e-5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader, Dataset


def polygons_to_mask(segmentation: list[list[float]], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in segmentation:
        points = np.array(polygon, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


class CocoMaskDataset(Dataset):
    """One sample per COCO annotation: image + corrected binary mask + box prompt."""

    def __init__(self, coco_path: Path, images_dir: Path, target_length: int):
        coco = json.loads(coco_path.read_text())
        self.images_dir = images_dir
        self.transform = ResizeLongestSide(target_length)

        images_by_id = {img["id"]: img for img in coco["images"]}
        self.samples = []
        for ann in coco["annotations"]:
            if not ann.get("segmentation"):
                continue
            image_info = images_by_id[ann["image_id"]]
            self.samples.append((image_info, ann))

        if not self.samples:
            raise ValueError(f"No annotated segmentations found in {coco_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        image_info, ann = self.samples[idx]
        image_path = self.images_dir / image_info["file_name"]
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not read image: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        gt_mask = polygons_to_mask(ann["segmentation"], h, w)
        ys, xs = np.where(gt_mask)
        box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)

        input_image = self.transform.apply_image(rgb)
        input_box = self.transform.apply_boxes(box[None, :], (h, w))[0]

        return {
            "image": torch.as_tensor(input_image).permute(2, 0, 1).contiguous(),
            "original_size": (h, w),
            "box": torch.as_tensor(input_box, dtype=torch.float32),
            "gt_mask": torch.as_tensor(gt_mask, dtype=torch.float32),
        }


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(-2, -1))
    union = probs.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return 1 - ((2 * intersection + eps) / (union + eps)).mean()


def mask_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred_bool = pred > 0.5
    target_bool = target > 0.5
    intersection = (pred_bool & target_bool).sum().item()
    union = (pred_bool | target_bool).sum().item()
    return intersection / (union + eps)


def forward_sample(sam, sample: dict, device: torch.device) -> torch.Tensor:
    """Returns full-resolution mask logits for one sample."""
    image = sam.preprocess(sample["image"].to(device).unsqueeze(0).float())
    with torch.no_grad():
        image_embedding = sam.image_encoder(image)
        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points=None,
            boxes=sample["box"].to(device).unsqueeze(0),
            masks=None,
        )

    low_res_masks, _ = sam.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    return sam.postprocess_masks(low_res_masks, sample["image"].shape[-2:], sample["original_size"])[0, 0]


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda but CUDA is not available")
    return torch.device(requested)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coco-annotations", type=Path, required=True, help="CVAT COCO 1.0 export with expert-corrected masks.")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("~/.cache/tooth_ratio/sam_vit_b_01ec64.pth").expanduser())
    parser.add_argument("--model-type", default="vit_b", choices=sorted(sam_model_registry.keys()))
    parser.add_argument("--output", type=Path, default=Path("src/models/weights/sam_vit_b_tooth_finetuned.pth"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    device = resolve_device(args.device)
    sam = sam_model_registry[args.model_type](checkpoint=str(args.checkpoint))
    sam.to(device=device)

    for param in sam.image_encoder.parameters():
        param.requires_grad_(False)
    for param in sam.prompt_encoder.parameters():
        param.requires_grad_(False)

    dataset = CocoMaskDataset(args.coco_annotations, args.images_dir, target_length=sam.image_encoder.img_size)
    n_val = max(1, int(len(dataset) * args.val_fraction))
    train_set, val_set = torch.utils.data.random_split(
        dataset, [len(dataset) - n_val, n_val], generator=torch.Generator().manual_seed(0)
    )
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    optimizer = torch.optim.Adam(sam.mask_decoder.parameters(), lr=args.lr)
    bce = torch.nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        sam.mask_decoder.train()
        epoch_loss = 0.0
        for batch in train_loader:
            sample = {k: v[0] if isinstance(v, list) else v for k, v in batch.items()}
            sample["original_size"] = (int(sample["original_size"][0]), int(sample["original_size"][1]))

            logits = forward_sample(sam, sample, device)
            target = sample["gt_mask"].to(device)

            loss = bce(logits, target) + dice_loss(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        sam.mask_decoder.eval()
        ious = []
        with torch.no_grad():
            for batch in val_loader:
                sample = {k: v[0] if isinstance(v, list) else v for k, v in batch.items()}
                sample["original_size"] = (int(sample["original_size"][0]), int(sample["original_size"][1]))
                logits = forward_sample(sam, sample, device)
                ious.append(mask_iou(torch.sigmoid(logits), sample["gt_mask"].to(device)))

        print(f"epoch {epoch}/{args.epochs}: train_loss={epoch_loss / len(train_loader):.4f} val_iou={np.mean(ious):.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sam.state_dict(), args.output)
    print(f"Saved fine-tuned checkpoint to {args.output}")


if __name__ == "__main__":
    main()
