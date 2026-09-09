# tooth_ratio_research

## YOLO11 tooth detection fine-tuning

The COCO bounding boxes in `data/bounding_box/instances_default.json` can be
validated, converted, split, and used to fine-tune YOLO11 with one command:

```bash
uv sync --extra yolo
uv run python -m src.models.finetune_yolo11 --prepare-only
uv run python -m src.models.finetune_yolo11 --device 0
```

The generated dataset is written to `data/yolo_tooth/`, while training runs and
`best.pt` are written below `reports/models/yolo11/`. See
[`docs/yolo11_finetune.md`](docs/yolo11_finetune.md) for configuration and
evaluation details.
