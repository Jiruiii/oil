# 完整技術文件：資料前處理 → 模型輸入 → 模型架構 → 推論結果

> 本文件是 `README.md`（入口/總覽）的細節版，逐段對照實際程式碼（`preprocess/` +
> `main/`）寫成，目標是讓一個完全沒碰過這個 repo 的人，看完後能理解：
> 每一步在做什麼、輸入輸出格式是什麼、關鍵數字/公式是什麼、為什麼這樣設計。
> 所有敘述皆以「目前保留的唯一 yaml」
> （`main/experiments_CV_358clean_gt_expand_prithvi300m_dlv3_det.yaml`）
> 的實際設定為準；程式支援但本 yaml 未使用的分支只簡短帶過並註明「本 yaml 未使用」。

---

## 0. 全流程總覽

```
原始 8-band TIF（每場景一個資料夾，每 band 一個 GeoTIFF）
    │
    ├─ 1.2 build_vrt_ms6.py           → 每場景一個 .vrt（8-band 虛擬疊圖，不佔額外磁碟）
    ├─ 1.3 json_to_mask_tif.py /
    │      batch_event_gpkg_to_mask.py → 每場景一個 GT Mask TIF（0=Oil/1=BG/2=Others/255=未標註）
    ├─ 1.4 build_manifest.py          → manifest.csv（資料集帳本，偵測新增/異動場景）
    ├─ 1.5 resplit_stratified_v2.py   → scene_level 3-fold stratified split
    ├─ 1.6 generate_patch_coords_gt_expand.py → patch_level_gt_expand（GT-centric patch 座標）
    ├─ 1.7 generate_bg_patches.py     → patch_level_gt_expand_bg（背景 patch 座標）
    ├─ 1.8 preflight_check.py /
    │      check_dataset_completeness.py → 訓練前體檢（fail-fast）
    ▼
2. main_runner.py --config <yaml>
    │   SegmentationDataset：從 VRT 動態讀 patch → /10000 正規化 → pixel_mapping →
    │   Albumentations augment → mean/std Normalize
    ▼
3. DeepLabAdapter（model_family=prithvi_eo_v2_dlv3）
    │   Prithvi-EO-2.0-300M ViT encoder（6-band tap 最後層）→ torchvision DeepLabV3(ASPP) head
    │   → TverskyLoss(α=0.3, β=0.7) → AdamW + differential LR + Cosine schedule
    ▼
4. recon_gt_aware_module.py
    │   test_fold{N}.txt 的油汙周邊 patch → TTA 推論 → 貼回原場景座標 → per-scene / pooled IoU
    │   → predmask_codec.py 寫 pixel-exact JSON artifact
    ▼
5. tracking_module.log_run_json → restructure_excel.py → 彙整 Excel 總表
```

---

## 1. 資料前處理（`preprocess/`）

### 1.1 原始資料

- 來源：Sentinel-2 經 Sen2Like 調和後的 8-band 影像，每個場景一個資料夾，內含
  8 個單 band GeoTIFF：`*_B01.tif ... *_B12.tif`（B01/B02/B03/B04/B08/B8A/B11/B12）。
- 目錄結構有兩種：
  - 3-level：`MS6_sen2like/{NOAA_LOCATION}/{YEAR}/{SCENE_NAME}/`
  - 2-level：`MS6_sen2like/{EVENT_NAME}/{SCENE_NAME}/`
- 原生解析度不同：B01/B11/B12 = 60m/20m，B02–B08 = 10m，B8A = 20m。
- 這批原始 tif 本身 **521GB**，是後續所有 VRT 的實際像素來源（VRT 只是指標，見 1.2）。

### 1.2 建立 VRT — `preprocess/build_vrt_ms6.py`

- 對每個場景，以 **B02（10m, 10980×10980）** 作為參考解析度，手動組裝 VRT XML。
- Band 順序固定為（依波長升序）：`B01→B02→B03→B04→B08→B8A→B11→B12`。
- 建置前逐 band 驗證幾何契約（不符即整景失敗，不靜默）：
  - CRS 必須與參考 band 一致
  - bounds 誤差 `<1e-3`
  - transform 不得含旋轉項（`b==0, d==0`）
  - 縮放比例必須等比（`w/src_w == h/src_h`），容許 S2 的 6×/2×/1× 與 L8/L9 的 1.5×
  - `nodata` 必須是 `0` 或未設
- 各 band 依實際像素尺寸寫 `SrcRect`/`DstRect`，GDAL 用 `resampleAlg="nearest"` 自動 resample：
  - B01（1830×1830）→ upsample 6× 到 10980×10980
  - B02–B08（10980×10980）→ 1:1
  - B8A/B11/B12（5490×5490）→ upsample 2× 到 10980×10980
- **VRT 本身只是 XML 指標檔**（每景幾 KB，全部場景合計約 6.3MB），每個 band 的
  `<SourceFilename>` 指向 1.1 的原始單 band tif；讀取時 GDAL 才即時組合成 8-band 影像。
  **原始 521GB tif 不能刪**，刪了 VRT 就讀不到任何像素。

### 1.3 GT Mask 產生 — `preprocess/json_to_mask_tif.py`（+ `batch_event_gpkg_to_mask.py`）

- 輸入：LabelMe JSON（`shapes` 內每個 polygon 帶 `label` 欄：`oil`/`background`/`others`）。
- 用 `cv2.fillPoly` 把每個 polygon 依 label 填色到一張 `(H, W)` 的 `uint8` 陣列：

  | 填值 | 語意 | 來源 |
  |---|---|---|
  | 0 | Oil | 明確標註的油污 polygon |
  | 1 | Background | 明確標註的背景 polygon |
  | 2 | Others | 明確標註的「其他」polygon |
  | 255 | 未標註 | 沒有任何 polygon 覆蓋到的像素（初始值） |

- 輸出 GeoTIFF 的 CRS/geotransform 取自同場景的 B02 tif（與 VRT 座標系對齊），
  `uint8` 單 band、LZW 壓縮。
- `batch_event_gpkg_to_mask.py` 是「事件場景」（NOAA 以外、以 GPKG 提供標註的場景）的
  前置轉換：GPKG → LabelMe JSON → 再走同一套 mask 邏輯。

### 1.4 資料集帳本 — `preprocess/build_manifest.py`

- 掃全部場景，產出/更新 `manifest.csv`，作為「單一真相帳本」。
- 重跑時會 diff 出新增/異動的場景（依檔案 hash），供後續步驟判斷哪些場景要重新處理。

### 1.5 Scene-level 3-fold split — `preprocess/resplit_stratified_v2.py`

這是**目前主用**的 split 定義腳本（`v2`），流程：

1. 從舊 `5_fold/scene_level` union 出全部場景名單，扣掉 `dirty.txt`/`blur.txt`
   標記的髒場景（GT 油污不明顯，per-scene 空檔標記）。
2. 對剩下每個場景讀 mask，算 `oil_px = (mask == 0).sum()`。
3. 依 oil 像素數的 **tertile**（1/3、2/3 分位數）分 3 層：`small` / `medium` / `large`。
4. `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)` 依這個分層標籤切
   3 個 test fold，確保每個 fold 的 test set 都有均衡的三種難度場景。
5. 每個 fold 的 `train_val` pool 再依同樣的分層標籤、seeded 隨機抽 **20%** 當 val，
   其餘當 train（`VAL_RATIO=0.20`）。
6. 輸出到 `3_fold_stratified_v2/scene_level/{train,val,test}_fold{1,2,3}.txt`，
   同時輸出 `scene_metadata.csv`（scene/oil_px/oil_bin/sensor）與 `bin_edges.txt`
   供後續分析使用；跑完會印每個 fold × split 的 small/medium/large 場景數分佈表供 audit。

> 這是 **scene 級別** 的 split（一行一個場景名）。1.6 會把它轉成 **patch 級別** 的 split。

### 1.6 GT-centric patch 座標 — `preprocess/generate_patch_coords_gt_expand.py`

這是分支 B（GT_expand）的核心設計，取代舊版「全圖滑窗切 patch」策略：

1. 對每個場景的 mask，取 `mask == 0`（oil）做二值化，用
   `cv2.connectedComponentsWithStats(8-connectivity)` 找每一塊獨立的油污區域
   （connected component）。
2. 每個 component 取其 bounding box：
   - 若 `bbox <= 256×256`：直接取 **1 個** patch，中心對齊 bbox 中心。
   - 若 `bbox > 256×256`：用 **256×256 網格鋪磚**覆蓋整個 bbox，patch 中心依比例
     均勻分布（`n_rows/n_cols` 由 bbox 尺寸除 256 無條件進位決定）。
3. 所有 patch 座標 clip 到影像邊界內（避免超出範圍）。
4. 環境變數 `SPLIT_DIR`/`OUT_DIR` **必填**（無 default，故意設計成裸跑會直接報錯退出，
   避免誤用舊的 5-fold split 生出過期 patch 座標）；`N_FOLDS` 由 `SPLIT_DIR` 內
   `train_fold*.txt` 的數量自動偵測並驗證連續、完整（每 fold 都要有 train/val/test 三檔）。
5. 每個 unique 場景只算一次（`multiprocessing.Pool(8)` 平行處理），再依 1.5 的
   scene→fold/split 對應關係分派成最終的 patch-level txt。
6. 輸出：`patch_level_gt_expand/{train,val,test}_fold{N}.txt`，每行格式
   `{SCENE_NAME}_patch_x{COL_OFF}_y{ROW_OFF}`（這個檔名格式後面訓練/推論全程都靠這個
   pattern `^(.+)_patch_x(\d+)_y(\d+)$` 解析場景名與座標）。同時輸出 `stats.csv`
   記錄每個 component 的 bbox 尺寸/oil 像素數/切出幾個 patch/是否為鋪磚模式。

> 這批文件已複製一份到 `onboarding/data_split/patch_level_gt_expand/`（見 README）。

### 1.7 背景（無油）patch 座標 — `preprocess/generate_bg_patches.py`

GT_expand 策略只圍著油污取樣，天生缺乏「background 長什麼樣」的知識，這支腳本補強：

1. 對每個場景，在 mask 上以 `stride=512` 滑窗取 `256×256` 候選 window，需同時滿足：
   - 零 oil 像素（`(win==0).sum()==0`）
   - `(background | others)` 像素占比 ≥ 30%（`MIN_VALID_BG_FRAC`）
   - `ignore(255)` 占比 < 50%（`MAX_IGNORE_FRAC`）
2. 每場景用「場景名 blake2b 雜湊 + 固定 SEED(42)」建構 reproducible RNG，
   隨機抽 **`N_PER_SCENE=10`** 個候選（本 yaml 用的是這個最基本模式；程式碼另外支援
   P7 配額比例模式 / P8 擴池模式，本 yaml **未使用**，此處略過）。
3. 輸出 `patch_level_gt_expand_bg/bg_coords.tsv`（欄位：`scene, x, y`）+ `stats.csv`。
4. 訓練時透過 **`bg_scene_filter`**（同場景集合過濾，而非舊版的 `fold_only`）把
   這份 BG 座標池按「當前 fold 當前 split 出現過的場景」過濾，避免 val/test 場景的
   BG patch 洩漏進 train（`deeplab_adapter.py` 的 `_load_bg_coords`）。

### 1.8 訓練前體檢 — `preprocess/preflight_check.py` + `check_dataset_completeness.py`

- 收集 split 目錄裡**所有** fold × split txt + `bg_coords.tsv` 引用到的每個 unique 場景。
- 逐場景驗證：
  1. VRT 存在且可讀（實際讀一個 16×16 window）
  2. Mask 存在且可讀
  3. VRT 與 Mask 的 shape / transform / CRS 完全對齊
  4. Mask 像素值 ⊆ `{0, 1, 2, 255}`（沒有非法值）
  5. VRT 波段數 = 8
- 任一項不符 → 印出**全部**問題清單、`exit(1)`（fail-fast，不會讓訓練帶著壞資料默默跑下去）。
- `check_dataset_completeness.py` 做更廣的完整性掃描（JSON/Mask/COG 存在性），適合在
  1.2/1.3 跑完後先執行一次，確認沒有場景漏轉。

---

## 2. 資料送入模型（`main/deeplab_adapter.py` → `SegmentationDataset`）

### 2.1 一個 Dataset，三種樣本來源

`SegmentationDataset.__len__` = `len(pos_coords) + len(image_files) + len(bg_coords)`，
`__getitem__` 依 index 落在哪個區段分派：

| 來源 | 本 yaml 是否使用 | 說明 |
|---|---|---|
| `pos_coords` | ✅ 使用 | 從 `train/val/test_fold{N}.txt` 解析出的油污周邊 patch 座標，走 `_get_pos_vrt_item`（`read_images_from_vrt: true`） |
| `image_files` | ❌ 未使用 | 舊版 flat-dir（實體切好的 patch 圖片）相容模式，本 yaml 未設 `image_dir` |
| `bg_coords` | ✅ 使用 | 1.7 產生的背景座標，走 `_get_coord_vrt_item` |

### 2.2 影像讀取與正規化

`_get_pos_vrt_item` / `_get_coord_vrt_item` 共用的處理鏈：

1. `rasterio` 開 VRT（per-process cache，避免每次 `__getitem__` 都重開檔），用
   `Window(x, y, 256, 256)` 讀 8-band，`boundless=True, fill_value=0`（超出邊界補 0）。
2. **`/ 10000.0`**：Sentinel-2 反射率原始整數尺度轉回 0–1 附近的浮點尺度。
3. `nan_to_num`：NaN/±inf → 0；並把 `>100` 或 `<-100` 的異常 nodata 填值（例如
   float32 nodata 常見的 `9.97e36`）強制歸零。
4. `band_indices`/`spectral_index` 切片：本 yaml **兩者都未設**，8-band 原封不動交給
   模型（6-band 的選擇是在模型內部做的，見第 3 節）。
5. `(C, H, W) → (H, W, C)`，交給 Albumentations。
6. BG patch 額外規則：讀出的 patch 若 **NaN 比例 > 10%**，視為無效樣本回傳 `None`，
   由 `oil_collate_fn` 在 batch 組裝時濾掉（`bg_pool_cfg` 擴池模式下會改成 fail-fast，
   本 yaml 未設 `bg_pool_cfg`，維持「濾掉」的舊行為）。

### 2.3 Mask 讀取與 `pixel_mapping`

- 正樣本的 mask：從 `labels_dir/{scene}.tif` 用同一個 Window 讀，`fill_value=255`。
- **`pixel_mapping`（本 yaml 實際設定，務必注意）**：

  ```yaml
  pixel_mapping:
    0: 0     # Oil        → class 0（前景）
    1: 1     # Background → class 1（背景）
    2: 1     # Others     → class 1（背景）
    255: 1   # 未標註      → class 1（背景）
  ```

  ⚠️ **這裡把「未標註（255）」直接映射成背景 class 1，不是 ignore。**
  也就是說模型訓練時，畫面裡「沒被人標註過的區域」會被當成**真背景**來監督，
  而不是被排除在 loss 之外。這跟 `json_to_mask_tif.py` 檔頭註解裡舉例的
  「典型設定」（`{0:0, 1:1, 2:255, 255:255}`，把 others/未標註都設 ignore）**不同**，
  是本次champion 設定的實際選擇，理解評估數字時要記住這點。
- BG patch（`_get_coord_vrt_item`）的 mask 不查表，直接整張填 class 1（純背景監督）。

### 2.4 `oil_ignore_ring`（本 yaml 未啟用，但機制存在）

- 只在 **train** dataset 生效（`apply_oil_ignore_ring=True` 固定傳入，val/test 不受影響）。
- `dataset_cfg.get('oil_ignore_ring', 0)`：本 yaml 未設此欄位，預設 `0` = 停用。
- 若設為 `N > 0`：在 oil 區域外圈 `N` px 內，且原始 mask 值為 `255`（未標註）的像素，
  訓練時改標為 `255`（ignore，需搭配非本 yaml 的 pixel_mapping 才有意義）。
  動機是標註 polygon 常是保守框，邊緣沒框到的薄油若被 255→bg 映射教成背景，
  會系統性把模型推向 under-segmentation（漏檢油污邊緣）。

### 2.5 資料增強（Albumentations）

```python
# train
A.RandomResizedCrop(size=(256,256), scale=(0.9, 1.0), p=0.5)  # GT_expand patch 緊貼 bbox，收緊到 0.9 避免裁掉油污
A.HorizontalFlip(p=0.5)
A.VerticalFlip(p=0.5)
A.RandomRotate90(p=0.8)
A.GaussNoise(std_range=(0.005, 0.02), p=0.4)
A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3)
A.Normalize(mean=mean, std=std, max_pixel_value=1.0)
ToTensorV2()

# val / test
A.Resize(256, 256)
A.Normalize(mean=mean, std=std, max_pixel_value=1.0)
ToTensorV2()
```

- `mean`/`std` 直接從 yaml 讀（8 槽，見第 3.2 節的官方統計值），不用另外抽樣計算。
- Albumentations 2.0.x 的 `Compose` 有**獨立於 torch/numpy/random 的 RNG**，訓練時會
  顯式 `seed=` 建構、每個 DataLoader worker 再各自 `set_random_seed` 一次，確保跨 run 的
  augmentation 可重現。

### 2.6 DataLoader

- `batch_size=16`，`nominal_batch_size=16` → 無梯度累積（`accumulation_steps=1`）。
- `num_workers=8`，`persistent_workers=True`（讓 worker 內的 rasterio file handle
  cache 跨 epoch 保持暖機）。
- `drop_last=True`（train）/ `False`（val，避免丟資料影響 best checkpoint 選擇）。
- `collate_fn=oil_collate_fn`：過濾 `None`（NaN 過多的 BG patch）。
- Seed-aware：`loader_generator` 固定種子、`worker_init_fn` 對每個 worker 設定
  `numpy/random/Albumentations` 獨立 seed，訓練順序/增強皆可重現。
- 本 yaml 未設 `bg_pool_cfg.draws_freeze_csv`，所以不會啟用 `EpochRotatingBGSampler`
  （P8 的 BG 池輪替機制），單純 `shuffle=True`。

### 2.7 Loss — `TverskyLoss`（本 yaml 使用的損失函數）

```yaml
dataset:
  loss:
    type: 'tversky'
    alpha: 0.3
    beta: 0.7
```

```
p  = softmax(logits)[:, oil_class]         # (B,H,W) 模型預測「是 oil」的機率
vm = (target != ignore_index)              # 有效像素遮罩
g  = (target == oil_class)                 # GT 是 oil 的像素
TP = Σ p·g·vm
FP = Σ p·(1-g)·vm
FN = Σ (1-p)·g·vm
TI = (TP + smooth) / (TP + α·FP + β·FN + smooth)
loss = 1 - TI
```

- `α=0.3 < β=0.7` → **FN（漏檢）的懲罰比 FP（誤檢）重**，把訓練推向拉高 recall。
- `focal_gamma`（本 yaml 未設，預設 `1.0`）：若 >1 會變成 Focal Tversky
  `(1-TI)^γ`，本 yaml 走純 Tversky 路徑。
- 這是目前判定的**標竿 loss 設定**：相對 baseline 的 class-weighted Focal/CE，
  在大/中型油污上全面提升，tiny 油污無顯著傷害（詳見 README 第六節數字）。
- 备用路徑（本 yaml 未用）：`class_weights` + `FocalLoss(gamma=2.0)`，或無權重的純 `FocalLoss`。

---

## 3. 模型架構（`main/deeplab_adapter.py` + `prithvi_backbone.py` + `prithvi_upernet.py` + `prithvi_deeplab.py`）

### 3.1 `model_family` 路由

`DeepLabAdapter.__init__` 依 `architecture_cfg.model_family` 分派到 4 條建構路徑之一
（`torchvision_deeplabv3` / `smp_deeplabv3plus` / `prithvi_eo_v2` / `prithvi_eo_v2_dlv3`）。
本 yaml 設 `model_family: 'prithvi_eo_v2_dlv3'` → `_build_prithvi_eo_v2_dlv3()`。

### 3.2 共用 Encoder 建構 — `_prithvi_build_encoder()`

這段是 **`prithvi_eo_v2`（UPerNet head）與 `prithvi_eo_v2_dlv3`（DeepLabV3 head）
兩個 family 共用**的同一份程式碼（所以 `prithvi_upernet.py` 就算只用 dlv3 head 也是
硬依賴，因為它裡面的 `PrithviViTEncoder` 類別被兩邊 import）。建構前會做一連串
**寫死不可繞過**的驗證（違反即 `raise`，不會靜默修正）：

1. **Spec 一致性**：`in_channels` 必須是 8；yaml 的 `input_band_names` /
   `backbone_band_names` / `input_band_indices` 必須逐字等於官方 300M spec：
   ```python
   input_band_names     = ['B01','B02','B03','B04','B08','B8A','B11','B12']  # 8-band VRT 順序
   backbone_band_names  = ['B02','B03','B04','B8A','B11','B12']              # 實際餵進 ViT 的 6 band
   input_band_indices   = [1, 2, 3, 5, 6, 7]                                  # 對應的 0-based index
   ```
   即 **B01（索引0）與 B08（索引4）不會被送進 backbone**。
2. **mean/std 逐槽比對**：yaml 的 8 槽 mean/std，對應到上面 6 個實際使用的 band，
   必須與官方統計值（除以 10000 換算到同尺度）逐位元相符：
   ```
   mean_ref = [0.1087, 0.1342, 0.1433, 0.2734, 0.1958, 0.1363]  # B02,B03,B04,B8A,B11,B12
   std_ref  = [0.2248, 0.2179, 0.2178, 0.1850, 0.1242, 0.1049]
   ```
   未使用的 B01/B08 兩槽必須是 identity `(mean=0, std=1)`（yaml 裡確實如此，見第 5 節表格）。
3. **Checkpoint 溯源驗證**：`backbone_ckpt` 路徑必須存在，旁邊必須有
   `<ckpt>.manifest.json`（由 `main/fetch_prithvi_checkpoint.py` 下載時產生），
   manifest 的 `revision`/`sha256`/`size_bytes` 三者都要對上官方 pin 值，**再對整個
   checkpoint 檔案實際重算一次 SHA256** 二次確認（`faab2c8b...`，1,326,658,396 bytes）。
4. **Encoder 建構**：`PrithviViTEncoder(img_size=256, patch_size=(1,16,16), num_frames=1,
   in_chans=6, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4)`。
5. **C6 嚴格載入**：從 checkpoint 的 `encoder.*` 命名空間取權重載入 encoder；
   **只允許 `pos_embed` 缺席**（因為預訓練用 T=4 的網格，跟本專案 T=1/256px 不同形，
   改用本地重建的 sin-cos 位置編碼，等同官方 inference.py 的做法）；`decoder.*` 與
   `mask_token`（MAE 預訓練專用的解碼器權重）允許被丟棄；**其餘任何 key 缺失或
   shape 不符一律 `raise`**，不允許 `strict=False` 泛用容錯（避免把權重載錯而不自知）。

### 3.3 PrithviViT 內部結構（`prithvi_backbone.py`）

- **`PatchEmbed`**：`nn.Conv3d(in_chans=6, embed_dim=1024, kernel=patch_size=(1,16,16),
  stride=(1,16,16))`，把 `(B, 6, 1, 256, 256)` 的輸入直接 3D 卷積切成
  `16×16=256` 個 token，每個 token 對應原圖 16×16 px 的區域。
- **位置編碼**：`get_3d_sincos_pos_embed` 產生固定（非學習）的 3D sin-cos 位置編碼；
  本專案輸入尺寸（256px）與官方預訓練尺寸不同時，`interpolate_pos_encoding` 用
  bicubic 對 patch 網格做插值對齊。
- **Transformer 主體**：`cls_token` + 256 個 patch token 串接，過 **24 層**
  `timm.models.vision_transformer.Block`（標準 ViT block：LayerNorm → Multi-Head
  Self-Attention → LayerNorm → MLP，`num_heads=16, mlp_ratio=4`），最後 `norm` 正規化。
- **`forward_features`**：回傳每一層 block 的輸出（list），供上層依 `feature_index`
  取用；因為是分割任務、用完整影像（非 MAE 預訓練的隨機遮罩），走 `forward_features`
  而不是 `forward`（`forward` 才會做 `random_masking`，那是 MAE 自監督預訓練用的）。

### 3.4 DeepLabV3(ASPP) Head — `prithvi_deeplab.py: PrithviDeepLabV3Segmenter`

```yaml
feature_indices: [23]              # 單一 tap，必須 = depth-1（最後一層，含 terminal norm）
aspp_atrous_rates: [6, 12, 18]     # 現行值；程式預註冊只允許 {(6,12,18), (2,4,6)}，其餘 raise
```

1. `encoder.forward_features(x)` 取第 **23 層**（最後一層，`assert feature_index == depth-1`
   保證一定是經過 encoder `norm` 的 terminal 特徵，不能是中間層）。
2. `prepare_features_for_image_model`：去掉 `cls_token`，
   `rearrange("(t h w) e -> (t e) h w")` 把 256 個 token 還原成空間特徵圖
   `(B, 1024, 16, 16)`（T=1 時通道數就是 `embed_dim=1024`）。
3. `torchvision.models.segmentation.deeplabv3.DeepLabHead(in_channels=1024,
   num_classes=2, atrous_rates=(6,12,18))` — **整包沿用 torchvision 原生結構**
   （ASPP 五分支 + projection + `Dropout(0.5)`（torchvision 內建）+ 3×3 conv +
   1×1 classifier），本專案只參數化 `in_channels` 與 `atrous_rates`，不自己重組 ASPP。
4. `DeepLabHead` 輸出 `(B, 2, 16, 16)`，`bilinear` 上採樣回輸入解析度
   `(B, 2, 256, 256)`（`align_corners=False`，跟 torchvision 慣例一致）。
5. 無 aux classifier / aux loss（跟 torchvision 標準 DeepLabV3 的雙頭設計不同，
   這是本專案 decoder 消融實驗的協商定案之一）。

### 3.5 完整 forward 流程圖

```
輸入 patch：(B, 8, 256, 256)  ← 2.2 節正規化後的 8-band Sentinel-2 patch
   │
   │ select_bands（index_select 通道 1,2,3,5,6,7 = B02,B03,B04,B8A,B11,B12）
   ▼
(B, 6, 256, 256)
   │ unsqueeze(2)  ← 顯式建立時間維 T=1
   ▼
(B, 6, 1, 256, 256)
   │ PrithviViT.forward_features()
   │   PatchEmbed(Conv3d) → (B, 256, 1024) token
   │   + pos_embed（bicubic 插值對齊 256px 網格）
   │   + cls_token → (B, 257, 1024)
   │   → 24 × ViT Block → norm
   ▼
取第 23 層（最後層，含 terminal norm）→ 去 cls_token → reshape
   ▼
(B, 1024, 16, 16)  ← 16×16 = 256px / 16px(patch_size)
   │ DeepLabHead（ASPP rates=6,12,18 + projection + dropout0.5 + 3×3conv + 1×1classifier）
   ▼
(B, 2, 16, 16)
   │ bilinear upsample（align_corners=False）
   ▼
(B, 2, 256, 256) = logits  →  {'out': logits}
```

### 3.6 訓練超參數與最佳化流程（`deeplab_adapter.py: DeepLabAdapter.train()`）

| 項目 | 值 | 備註 |
|---|---|---|
| `epochs` | 300 | 搭配 `patience=25` 的 early stopping |
| `batch_size` / `nominal_batch_size` | 16 / 16 | 無梯度累積 |
| `lr0` | 5e-5 | base learning rate |
| `backbone_lr_mult` | 0.1 | encoder 用 `5e-5 × 0.1 = 5e-6`；DeepLabHead 整包用 `5e-5`（`differential LR`，避免破壞預訓練特徵） |
| optimizer | `AdamW` | `weight_decay=1e-3` |
| scheduler | `CosineAnnealingLR` | `T_max=epochs`（未設 `max_optimizer_steps`），`eta_min=lr0*0.01` |
| AMP | `use_amp: true` | `torch.autocast('cuda')` + `GradScaler` |
| grad clip | `max_norm=1.0` | `clip_grad_norm_` |
| `min_physical_batch_size` | 2 | Prithvi head 的 ASPPPooling 全域池化 + BatchNorm 在 `batch=1` 下無法訓練；不足的 microbatch 整批跳過（不推進 optimizer step 計數） |
| checkpoint 策略 | `best`（預設） | 依 `val_miou` 挑 `best.pt`；另存每 epoch 的 `last.pt`（含 optimizer/scheduler/scaler/RNG 狀態，可 resume） |

- IoU 計算口徑（train/val/test 全程一致）：`ignore(255)` 一律視為 `bg class 1` 再算，
  避免模型靠「畫一個框圈住 ignore 區域」灌水 IoU。
- Val：`drop_last=False`，不丟資料，確保 best checkpoint 選擇時用到完整驗證集。

---

## 4. 推論與重組（`main/recon_gt_aware_module.py`）

### 4.1 為什麼用 `gt_aware` 而不是舊版 sliding-window

`yaml.post_tests[0].reconstruction.mode: 'gt_aware'` 讓 `main_runner.py` 在每個 fold
訓練完後，派發到 `recon_gt_aware_module.run_reconstruction_gt_aware`（而不是
`reconstruct_module.run_reconstruction` 那套舊版全圖 sliding-window 重組）：

- **不會**把整張圖切滿密集 sliding window（含大量純海/雲 patch）丟給模型做 OOD 推論。
- 只在 `test_fold{N}.txt` 定義的「油污周邊 patch」座標做 inference（跟訓練時 GT-centric
  的取樣分布一致），貼回原場景座標，其餘區域維持「未測試」狀態。
- per-scene IoU 只在「實際被 inference 覆蓋」的像素範圍內計算，數字更代表 GT_expand
  這個設計實際能力，而非被大量必然正確的純背景區域稀釋/灌水。

### 4.2 推論流程

1. `_group_patches_by_scene(test_fold{N}.txt)` → `{scene: [(x, y), ...]}`。
2. 每個場景：找對應 VRT + mask；逐 patch 座標從 VRT 讀 window → 用**跟訓練相同**的
   正規化鏈（`/10000` → nan/異常值清理 → `(x-mean)/std`）→ 做 **TTA**：
   ```
   pred = argmax( softmax( (logits(原圖) + logits(水平翻轉還原) + logits(垂直翻轉還原)) / 3 ) )
   ```
   （`_tta_predict_prob`，三份 logits 平均後再 softmax）。
3. 每個 patch 的預測寫進整場景大小的 `pred_canvas`（初始全 `255`=未測試），
   同時記錄 `tested_mask`（patch 若有重疊，**後寫入者覆蓋前者**，last-write-wins）。

### 4.3 指標計算

- `_compute_scene_iou`：只在 `tested_mask` 範圍內算；GT 裡的 `255`（ignore）同樣視為
  `bg class 1`（跟訓練/驗證口徑一致，避免虛胖）。
- 同時輸出兩種統計，語意不同，引用時要分清楚：
  - **per-scene mean**：每個場景各自算一個 Oil IoU 再平均（受小場景影響較大）。
  - **pooled**：把所有場景的 intersection/union 累加起來算一個總 IoU（受大場景主導）。
  - `restructure_excel.py` 把 **pooled Oil IoU** 當作正式「決策指標」
    （`GATE = 0.362` 雙閘門判準）。

### 4.4 Pixel-exact 預測產物 — `main/predmask_codec.py`

除了 `per_scene_iou.csv`，`gt_aware_recon` 還會用 `predmask_codec.py` 寫一份
**逐 pixel 精確可還原**的 JSON 格式預測結果（`schema=oil.predmask/1`）：

- 預測 mask 用 RLE（run-length encoding，`labelrle` codec）壓縮成 base64 字串存進 JSON，
  機率圖（`prob_oil`）則另存 `.npz`（避免 JSON 過大）。
- 每個 scene 的 JSON 內含完整 **provenance**：git commit、checkpoint 路徑、
  當時的 `dataset`/`architecture_cfg`/`reconstruction` config 快照、confusion matrix、
  VRT/mask 檔案的 size+SHA256（小檔案才實際算 hash，避免對超大 NAS 檔案硬算）。
- 目的：任何人在未來拿到這份 JSON，都能**不重跑模型**精確還原出當時的預測 mask，
  也能追溯到底是用哪個 checkpoint / 哪份 config 跑出來的。
- 舊版的疊圖視覺化（預測黃色外框 + GT 青色外框疊在真彩色底圖上）預設**關閉**
  （因為有了 pixel-exact JSON，疊圖可以隨選重新產生），可用
  `main/visualize_prediction_artifact.py` 從 JSON 復原視覺化圖片。

### 4.5 結果彙整

1. 每個 fold 訓練完，`run_evaluation_job` 呼叫 `tracking_module.log_run_json` 寫一份
   `{fold_dir}/run_metrics.json`（`kind='patch'`，patch-level 指標）。
2. Reconstruction 跑完，同樣寫一份 `{fold_dir}/post_test_*/run_metrics.json`
   （`kind='recon'`，scene-level pooled/mean IoU）。
3. 每跑完一個 fold，`main_runner._refresh_restructured_excel` 就呼叫一次
   `main/restructure_excel.py`（全掃、幂等、原子替換），把 `RESULT_BASE` 下**所有**
   fold 的 `run_metrics.json` 彙整成一份 4-sheet Excel：
   - 🚦 決策（scene-recon pooled_oil_iou，`GATE=0.362`）
   - 🗺 Scene Recon
   - 🔬 Patch 診斷
   - ⚙ Run 設定
4. 這是**新版**資料流（2026-07 起）：舊版 wide-format Excel（`log_to_excel` /
   `LEGACY_XLSX`）已淘汰，只在「補一次性 migration」時才會被讀取。

---

## 5. yaml 關鍵欄位速查（`experiments_CV_358clean_gt_expand_prithvi300m_dlv3_det.yaml`）

| 欄位 | 值 | 對應本文件章節 |
|---|---|---|
| `architecture_cfg.model_family` | `prithvi_eo_v2_dlv3` | §3.1 |
| `architecture_cfg.backbone_ckpt` | Prithvi-EO-2.0-300M pin ckpt 路徑 | §3.2 |
| `architecture_cfg.input_band_indices` | `[1,2,3,5,6,7]` | §3.2 |
| `architecture_cfg.feature_indices` | `[23]` | §3.4 |
| `architecture_cfg.aspp_atrous_rates` | `[6,12,18]` | §3.4 |
| `architecture_cfg.backbone_lr_mult` | `0.1` | §3.6 |
| `architecture_cfg.min_physical_batch_size` | `2` | §3.6 |
| `cross_validation.split_dir` | 1.5 產出的 scene→patch fold split 目錄 | §1.6 |
| `dataset.vrt_dir` | 1.2 產出的 VRT 目錄 | §1.2 / §2.2 |
| `dataset.labels_dir` | 1.3 產出的 mask 目錄 | §1.3 / §2.3 |
| `dataset.bg_coord_txt` | 1.7 產出的 bg_coords.tsv | §1.7 / §2.6 |
| `dataset.pixel_mapping` | `{0:0, 1:1, 2:1, 255:1}` | §2.3（⚠ 255 不是 ignore） |
| `dataset.mean` / `dataset.std` | 8 槽，對應官方 Prithvi 統計值 | §3.2 |
| `dataset.loss` | `tversky, alpha=0.3, beta=0.7` | §2.7 |
| `epochs` / `patience` / `lr0` | `300 / 25 / 5e-5` | §3.6 |
| `post_tests[0].reconstruction.mode` | `gt_aware` | §4.1 |

---

## 6. 與 README.md 的關係

- `README.md`：入口文件，給「新人第一次打開這個資料夾」或「AI agent 需要快速定位
  該讀哪支程式」用，內容是總覽表 + 檔案職責對照表，不深入公式/演算法細節。
- 本文件（`PIPELINE_DETAIL.md`）：給需要**真正理解每一步在幹嘛**（要改程式、要重現
  結果、要向外解釋方法論）的人用，逐段對照原始碼寫成。
- 兩份文件的「目前保留的實驗設定」數字口徑一致（見 README 第六節），本文件不重複
  貼基準數字，只講「這個數字是怎麼被算出來的」（§4.3 的 pooled vs per-scene mean）。
