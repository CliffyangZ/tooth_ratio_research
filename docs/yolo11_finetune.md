# YOLO11 牙齒偵測微調

此流程使用 `data/bounding_box/instances_default.json` 的 COCO `[x, y, w, h]`
bounding boxes，以及 `data/processed/` 中的對應影像，微調 Ultralytics
YOLO11 detection model。

## 1. 安裝

```bash
uv sync --extra yolo
```

若要使用 NVIDIA GPU，請先依主機 CUDA/driver 版本安裝相容的 CUDA PyTorch；
`torch.cuda.is_available()` 必須回傳 `True`。Ultralytics 套件本身不會修復主機的
NVIDIA driver。

## 2. 轉換與驗證資料

```bash
uv run python -m src.models.finetune_yolo11 --prepare-only
```

預設使用 seed 42 做逐影像的 80/10/10 分割，目前 40 張影像會得到
32 train、4 validation、4 test。每張 train image 另產生 3 張只改變像素強度的版本，
因此訓練目錄共有 128 張影像；validation 與 test 不做 augmentation。輸出：

```text
data/yolo_tooth/
├── dataset.yaml
├── split.json
├── images/{train,val,test}/*
└── labels/{train,val,test}/*.txt
```

`split.json` 保存來源 JSON 的 SHA-256、seed、各 split 檔名與標註數，供實驗重現。
轉換器會檢查 ID、影像存在性、實際尺寸、category reference、bbox 有效性與邊界，
再將 bbox 正規化為 YOLO 的 `class x_center y_center width height`。

### 醫學影像 augmentation 原則

此流程不使用任何會改變解剖結構或位置的 augmentation。旋轉、平移、縮放、剪裁、
水平／垂直翻轉、shear、perspective、mosaic、mixup、cutmix 與 copy-paste 全部關閉。
只對 train split 做小幅全域亮度、全域對比、gamma 與灰階 Gaussian noise；影像尺寸、
像素位置與 bbox 標籤完全不變。強度範圍刻意限制在原值附近，實際資料產生版本的
SSIM 最低值須維持在 0.95 以上。原始 `data/processed` 檔案不會被覆寫。

可調整每張影像的增強數量，或設為 0 關閉：

```bash
uv run python -m src.models.finetune_yolo11 \
  --prepare-only --rebuild-dataset --augmentations-per-image 3
```

重新產生不同 split 時必須明確指定 `--rebuild-dataset`：

```bash
uv run python -m src.models.finetune_yolo11 \
  --prepare-only --rebuild-dataset --seed 123
```

目前 JSON 不含 patient/group ID，因此只能逐影像分割。若同一病人的近似影像可能同時
出現在資料集，正式效能報告前應改用 patient/group split，避免資料洩漏。

## 3. Fine-tune

單張 NVIDIA GPU：

```bash
uv run python -m src.models.finetune_yolo11 --device 0
```

CPU smoke test（只確認 pipeline 能跑，不代表有效訓練）：

```bash
uv run python -m src.models.finetune_yolo11 \
  --device cpu --epochs 1 --imgsz 640 --batch 2 --workers 0 --skip-test
```

預設參數為 `yolo11n.pt`、100 epochs、`imgsz=960`、batch 8、early stopping
patience 30。影像是灰階 X-ray，所有線上幾何與色彩 augmentation 均關閉，使用上一節
產生的 intensity-only train variants。訓練本身會執行 validation；完成後，
程式另載入 `best.pt` 在 test split 評估。要跳過 test 可加 `--skip-test`。

較大模型或自訂參數範例：

```bash
uv run python -m src.models.finetune_yolo11 \
  --model yolo11s.pt --epochs 150 --imgsz 1024 --batch 4 --device 0 \
  --name tooth_yolo11s
```

輸出位於 `reports/models/yolo11/<run name>/`，部署時使用
`weights/best.pt`，不要使用 `last.pt`。只有 40 張標註影像，test 指標變異會很大；
建議後續增加專家標註，或以 grouped k-fold cross-validation 比較模型。

## 4. 中斷後續訓

```bash
uv run python -m src.models.finetune_yolo11 \
  --resume reports/models/yolo11/tooth_detect/weights/last.pt
```

續訓會使用 checkpoint 內原本的訓練參數。

## 5. YOLO boxes → SAM masks → COCO 1.0

使用微調後 YOLO11 的 bounding boxes 作為 SAM ViT-B prompts，為
`data/processed/` 全部影像產生 instance segmentation pseudo-labels：

```bash
uv sync --extra yolo --extra segmentation
uv run python -m src.core.segmentation.yolo_sam_coco --device cuda
```

預設 detector confidence 為 0.08，因為本資料 test F1 curve 的最佳 threshold 約為
0.081；低於 0.25 的 detection 仍保留，但會標記 `low_yolo_confidence` 供人工優先檢查。
輸出位於 `reports/models/yolo_sam_pseudo_labels/`，包含：

- `coco_annotations.json`：COCO 1.0 instance segmentation pseudo-labels。
- `cvat_coco_annotations.zip`：可上傳至既有 CVAT task 的 annotation archive。
- `masks/<image>/<annotation_id>.png`：逐 instance 二值 mask。
- `overlays/<image>.png`：YOLO box 與 SAM mask 疊圖。
- `metadata.json`、`metadata/<image>.json`：YOLO/SAM scores 與 QA flags。

輸出仍是 pseudo-label，必須由牙科專家校正後才能當作 ground truth。
