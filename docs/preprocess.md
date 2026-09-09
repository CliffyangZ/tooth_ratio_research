# 資料前處理流程（Segmentation 任務）

本文件定義 `data/` 目錄下的牙科 PA (periapical) X-ray 影像，如何前處理後送入 SAM (Segment Anything) 進行牙齒分割標註，最終產出可用於 CRR (Crown-to-Root Ratio) landmark 標註的資料集。

## 1. 資料總覽

| 目錄 | 內容 | 數量 |
| - | - | - |
| `data/original/1~100`, `101~200`, `201~300` | 原始蒐集影像，未去重，尺寸不一（如 1200×825、1708×1197、552×793） | 300 張 |
| `data/raw_images/` | 去重後的代表影像，後續所有流程以此為輸入 | 282 張 |

`data/original` 與 `data/raw_images` 皆已加入 `.gitignore`，不進版控，只在本機保留。

## 2. Pipeline 總覽

```text
data/original/**            (300 張，含近重複)
    │  check_similarity.py  (SSIM pairwise, resize=256, grayscale)
    ▼
similarity_reports.json
    │  generate_raw_images.py (threshold=0.95, union-find 分群，取群內平均相似度最高者為代表)
    ▼
data/raw_images/            (282 張，去重代表影像)
    │  src/data/preprocess.py
    │  影像正規化 (grayscale → 品質量測 → 1–99 百分位正規化 → CLAHE → 複製為 3 通道 RGB)
    ▼
data/processed/             (282 張，SAM 輸入影像) + reports/data/quality_report.json
    │  src/core/segmentation/sam_infer.py
    │  Box/Point prompt 產生
    ▼
SAM (ViT-B) 推論              → mask + IoU score
    │  後處理：取最大連通元件 + morphological closing (5×5 kernel)
    ▼
reports/models/sam_pseudo_labels/coco_annotations.json (COCO 1.0)
    │  CVAT (deploy/cvat/tooth-sam)  → 匯入預標注 + 半自動標註，人工校正 pseudo-label
    ▼
最終 segmentation 標註（mask / landmark），用於 CRR 計算
    │  src/models/finetune_sam.py（待 CVAT 校正匯出後執行）
    ▼
微調後的 SAM ViT-B（src/models/weights/）
```

## 3. Step 1：去重（Deduplication）

近重複影像（同一顆牙齒多張近似拍攝）會讓資料集失衡並汙染 train/val split，因此先以結構相似度（SSIM）去重。

```bash
python src/data/check_similarity.py \
    --data-dir data/original \
    --output similarity_reports.json \
    --resize 256 \
    --workers 8

python src/data/generate_raw_images.py \
    --report similarity_reports.json \
    --output data/raw_images \
    --threshold 0.95
```

- SSIM 計算前先 grayscale 化並縮放到 256×256（僅用於相似度比較，不影響後續分割用的原圖）。
- threshold 0.95：目前資料集在此門檻下產生 282 群，其中 17 群為重複群（最大群集 3 張），共移除 18 張近重複影像。
- 每群僅保留「與群內其他成員平均相似度最高」的一張作為代表，寫入 `data/raw_images/`，去重紀錄見 `dedup_report.json`。

## 4. Step 2：影像正規化（送入 SAM 前）

由 `src/data/preprocess.py` 批次執行，對 `data/raw_images/` 中每張影像做以下轉換，輸出到 `data/processed/`（對應正規化公式亦見於 `deploy/cvat/tooth-sam/nuclio/model_handler.py::normalize_dental_xray`，該檔案為獨立 nuclio docker build context，未跨目錄 import，因此兩處各自實作同一公式）：

```bash
python src/data/preprocess.py \
    --input data/raw_images \
    --output data/processed \
    --quality-report reports/data/quality_report.json
```

1. **不做尺寸縮放**：保留原始解析度。原圖尺寸差異大，且 CRR 是 crown/root 兩段長度的比值（單位一致即可），縮放不影響比值本身，但會降低邊界/landmark 的像素精度，因此分割階段一律使用原圖尺寸；SAM 內部會自行將長邊縮放到 1024 做 embedding，不需要我們手動 resize。
2. **灰階化**：X-ray 本質為單通道影像，用 `cv2.IMREAD_GRAYSCALE` 取得灰階。
3. **品質量測**（見第 8 節）：在做任何增強前，於灰階原圖上計算 `laplacian_variance` / `mean_intensity` / `contrast_std`，寫入 `reports/data/quality_report.json`。
4. **1–99 百分位正規化**：以灰階影像的第 1、第 99 百分位為上下界做線性拉伸並 clip 到 `[0, 255]`，避免感光/曝光差異或金屬贗復物造成的極端亮/暗值壓縮對比。

    ```python
    low, high = np.percentile(gray, (1.0, 99.0))
    normalized = np.clip((gray - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
    ```

5. **CLAHE（局部對比強化，預設開啟）**：在百分位正規化後的灰階影像上套用 `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))`。

    > **決策紀錄（2026-08-23）**：此步驟原先僅在 `sigle_image.ipynb` 作視覺比較，正式 pipeline 因「局部對比增強可能在牙根尖端等低對比區域引入假邊界，干擾 SAM 的 mask 邊界判斷」而不採用，且要求重新評估前需先用同一張影像比較 CLAHE 前後的 mask IoU。此次由使用者明確決定直接將 CLAHE 設為 `src/data/preprocess.py` 的預設步驟，**未先執行該 IoU 比較**即調整。若後續 CVAT 校正發現根尖/CEJ 等低對比區域的 SAM pseudo-label 邊界品質明顯下降，應以 `--no-clahe` 重跑並比較，再決定是否恢復原決策。
6. **複製為 3 通道 RGB**：SAM 的 image encoder 需要 H×W×3 uint8 輸入，灰階影像直接 `np.repeat(..., 3, axis=2)`，存成 PNG（避免 jpg 再壓縮）。

## 5. Step 3：SAM 半自動標註 Prompt

以中央目標牙齒為對象，提供 SAM 一組 box prompt + point prompt：

- **Box prompt**：以影像寬高比例框選中央牙齒，預設 `BOX_FRACTIONS = (0.32, 0.05, 0.68, 0.95)`（x1, y1, x2, y2 皆為 0–1 比例），換算成像素座標。CVAT 互動標註時由標註者手動框選。
- **Point prompt**：box 中心給一個正點（前景），box 內側邊緣附近給一個負點（排除相鄰齒），`point_labels = [1, 0]`。
- 呼叫 `predictor.predict(box=..., point_coords=..., point_labels=..., multimask_output=True)`，取 `argmax(scores)` 的候選 mask。

## 6. Step 4：Mask 後處理

1. **保留最大連通元件**：`cv2.connectedComponentsWithStats`，去除 SAM 輸出中與目標牙齒不相連的雜訊區塊。
2. **Morphological closing**（5×5 kernel）：填補 mask 內部因牙本質/牙髓腔低對比造成的小孔洞，讓 mask 邊界連續。

## 6.5. Step 3+4 批次執行：`src/core/segmentation/sam_infer.py`

Step 3（prompt）與 Step 4（後處理）對 `data/processed/` 全部影像批次執行，並輸出可直接匯入 CVAT 的預標注：

```bash
python -m src.core.segmentation.sam_infer \
    --input data/processed \
    --checkpoint src/models/weights/sam_vit_b_01ec64.pth \
    --model-type vit_b \
    --device auto \
    --output reports/models/sam_pseudo_labels \
    --package-cvat-dataset
```

輸出至 `reports/models/sam_pseudo_labels/`：

- `masks/<name>.png`、`overlays/<name>.png`：逐張二值 mask 與疊圖，供人工抽查。
- `metadata/<name>.json` 與彙整後的 `metadata.json`：保存 `sam_score`、`mask_fraction`、`box_xyxy`、`point_coords_xy`、`status`，並以 `flagged_for_review` / `review_reasons` 記錄低分與幾何異常，供人工校正排序。
- `coco_annotations.json`：**COCO 1.0** instance segmentation 格式（`images` / `annotations`（polygon segmentation）/ `categories=[{"id":1,"name":"tooth"}]`），是匯入 CVAT 的檔案（見第 7 節）。
- `cvat_coco_annotations.zip`：annotation-only ZIP，內含 `annotations/instances_default.json`；已先建立影像 task 時，直接 Upload annotations 使用 `coco_annotations.json` 最單純。
- `cvat_coco_dataset.zip`：只有指定 `--package-cvat-dataset` 才建立，包含 `images/default/*` 與 `annotations/instances_default.json`，可作為完整 COCO dataset 匯入 CVAT。這會複製所有輸入影像，檔案大小約等於 `data/processed/`，不需要完整 archive 時請省略該旗標。
- `metadata.json` 的 `review_reasons` 不只檢查 SAM score，也會標記 mask 過大/過小、碰觸影像邊界、超出 prompt box 過多或無有效 polygon。這些只是 QA 篩選規則，不能取代牙科專家校正。

## 7. Step 5：CVAT 人工校正

SAM pseudo-label 僅為初稿，`sam_score` 與 `mask_fraction` 等 metadata 皆標記 `"status": "pseudo-label; requires expert correction"`，必須經牙科專家於 CVAT 校正後才可作為訓練標籤：

```bash
cd ~/Program/Medical-CV/cvat
CVAT_HOST=localhost docker compose up -d
```

1. 用 `data/processed/` 建立 CVAT task。
2. Task → Upload annotations → format `COCO 1.0`，選取 `reports/models/sam_pseudo_labels/coco_annotations.json`，取得逐張預標注（詳見 `docs/cvat_user_guide.md`）。
3. 標註者在已有預標注的基礎上做校正；若某顆牙齒的預標注品質太差（尤其 `flagged_for_review: true` 的影像），可改用 `deploy/cvat/tooth-sam/nuclio/`（`sam_vit_b_01ec64.pth`, ViT-B）部署的互動式 Nuclio function 重新框選/加點，取得新的分割結果（RLE 編碼，見 `model_handler.mask_to_cvat_rle`）——兩者互補：批次匯入省去逐張手動框選，互動式 function 處理批次結果不佳的個別牙齒。
4. 校正重點：牙冠/牙根交界（CEJ, cemento-enamel junction）、根尖、齒尖是否落在 mask 邊界內，避免相鄰齒重疊或牙周膜間隙被誤納入；並留意 CLAHE 是否在根尖等低對比區域引入假邊界（見第 4 節決策紀錄）。

## 8. 品質檢查

`src/data/preprocess.py` 已自動化此節指標，見第 4 節：

- `laplacian_variance`：影像模糊程度，過低代表對焦不佳。資料集內位於後 5 百分位者標記 `possible_blur`。
- `mean_intensity` / `contrast_std`：曝光異常（過曝/欠曝）或對比過低。`contrast_std` 位於後 5 百分位者標記 `low_contrast`。
- 結果寫入 `reports/data/quality_report.json`，可在跑 SAM 前先過濾被標記的影像。

`src/core/segmentation/sam_infer.py` 則自動標記 `sam_score`（predicted IoU）< 0.7 的影像為 `flagged_for_review`，代表應人工重新給 prompt 而非直接採用。

## 9. 模型微調（待資料齊備）：`src/models/finetune_sam.py`

批次匯入的預標注經 CVAT 專家校正、匯出（Task → Export annotations → `COCO 1.0`）後，可用該匯出檔微調 SAM 的 mask decoder：

```bash
python src/models/finetune_sam.py \
    --coco-annotations <CVAT 匯出的 COCO json> \
    --images-dir <對應影像目錄，通常就是 data/processed> \
    --checkpoint ~/.cache/tooth_ratio/sam_vit_b_01ec64.pth \
    --output src/models/weights/sam_vit_b_tooth_finetuned.pth \
    --epochs 20 --lr 1e-5
```

- 凍結 image encoder 與 prompt encoder，只訓練 mask decoder（訓練成本低、不易破壞既有的通用分割能力）。
- Loss = BCE + Dice，box prompt 取自每個校正後 mask 的 bounding box。
- 產出的 checkpoint 可直接替換 `deploy/cvat/tooth-sam/nuclio/function.yaml` 的 `MODEL_PATH` 供互動式標註使用。
- **截至 2026-08-23，repo 內尚無任何 CVAT 校正匯出的標註**，此腳本目前僅為基礎設施，需等第一批人工校正完成後才能實際執行。

## 10. 目前尚未定案（TBD）

- Train/val/test 切分策略：建議以「去重前的原始分群」為單位切分（而非用 `data/raw_images` 逐張隨機切），避免同一顆牙齒的近似影像同時落入 train 與 val 造成資料洩漏；比例暫定 80/10/10。
