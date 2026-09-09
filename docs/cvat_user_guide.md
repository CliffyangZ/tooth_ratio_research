# 日常啟動與停止
```
cd ~/Program/Medical-CV/cvat
```

```
CVAT_HOST=localhost docker compose up -d
```

```
docker compose down
```

```
docker compose ps
```

```
docker compose logs -f cvat_server
```

# 匯入 SAM 批次預標注（pseudo-label）

`src/core/segmentation/sam_infer.py` 會對 `data/processed/` 全部影像跑 SAM ViT-B，輸出
`reports/models/sam_pseudo_labels/coco_annotations.json`（COCO 1.0 格式）。用這個檔案可以
讓新建立的 CVAT task 一開始就帶有每張影像的牙齒 mask，標註者只需校正而不用逐張手動框選：

1. 啟動 CVAT（見上）後開啟 CVAT 網頁介面。
2. **Tasks → Create a new task**，建立名稱完全相同的 `tooth` label，Data 選擇 `data/processed/` 目錄下的全部影像（此為 CLAHE 增強後、SAM 實際使用的輸入影像，見 `preprocess.md` 第 4 節）。影像檔名與 COCO `images[].file_name` 必須一致。
3. Task 建立完成後，進入該 task → **Actions → Upload annotations**，Format 選 **COCO 1.0**，上傳
   `reports/models/sam_pseudo_labels/coco_annotations.json`。
   腳本也會輸出 annotation-only 的 `cvat_coco_annotations.zip`（內含 `annotations/instances_default.json`）。對已建立且已上傳影像的 task，優先使用前述 JSON。
   若推論時加上 `--package-cvat-dataset`，另會產生包含 `images/default/*` 與 `annotations/instances_default.json` 的完整 `cvat_coco_dataset.zip`，可直接以 COCO dataset 匯入；但目前 282 張 processed images 約 401 MB，會額外占用相近空間。
4. 匯入後每張影像會出現預先產生的 mask（多邊形），標註者逐張檢查並校正邊界（CEJ、根尖、齒尖，見
   `preprocess.md` 第 7 節）。
5. 若某張影像的預標注品質太差（可對照 `reports/models/sam_pseudo_labels/metadata.json` 裡
   `flagged_for_review: true`，即 `sam_score < 0.7` 的影像），刪掉該預標注後，改用下方的互動式
   Tooth SAM function 重新框選/加點取得新 mask——批次匯入與互動式標註互補，不是互相取代。
6. 校正完成後，**Actions → Export annotations**，Format 同樣選 **COCO 1.0**，匯出檔可作為
   `src/models/finetune_sam.py` 微調 mask decoder 的訓練資料（見 `preprocess.md` 第 9 節）。

## 互動式 Tooth SAM（個別牙齒重新框選）

`deploy/cvat/tooth-sam/nuclio/` 部署了一個互動式 SAM ViT-B 的 Nuclio serverless function
（`sam_vit_b_01ec64.pth`），CVAT 前端可對單一物件呼叫它取得即時分割結果（框選/加點後即時出 mask），
適合用來處理批次匯入結果不理想的個別牙齒，或全新加入的影像。
