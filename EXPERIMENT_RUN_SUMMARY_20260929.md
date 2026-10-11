# GT-expand 3-fold 實驗重跑紀錄（2026-09-29）

## 本次重跑範圍

依本交付包 `README.md` 的 onboarding 路線，使用已調和的 8-band Sen2Like GeoTIFF
與已凍結的 3-fold patch 座標，重建訓練所需的 VRT、GT mask，完成三折訓練、
patch 測試與 GT-aware 場景重組。這是從交付資料重跑模型實驗，並非從衛星原始資料
重新執行 Sen2Like 調和或重新設計資料切分。

## 資料與前處理

- 輸入為 `MS6_sen2like/` 中 355 個場景的單波段 GeoTIFF：Sentinel-2 283 景、
  Landsat 8 51 景、Landsat 9 21 景；每景 8 個波段。
- 從這些波段建立 355 個 8-band VRT（`MS6_sen2like_vrt/`），並從 JSON/GPKG
  標註產生 355 個 GT mask（`mask/`）。VRT 是指向原始波段檔的虛擬疊圖，
  並未產生實體合併影像。
- 執行 `preflight_check.py`，檢查凍結名冊引用的場景、8 個波段、VRT/GT mask
  的尺寸、CRS、transform 與標籤值域。
- 執行 `verify_frozen_labels.py`：355 景的 oil 像素數與交付的凍結統計相符；
  3,510 個背景 patch 座標均不含 oil 像素。
- 沿用交付包 `data_split/` 的三折名冊：共 2,893 個油汙 patch 座標與
  3,510 個背景 patch 座標。訓練時從 VRT 即時讀取 256×256 window，沒有
  預先輸出實體 patch 圖片。

## 環境與訓練設定

- 主機：NVIDIA GeForce RTX 4090 24 GB，驅動 550.144.03。
- 使用本地虛擬環境與 `requirements.local-cu118.txt`。相較交付包的 CUDA 12.8
  PyTorch wheel，本機改用 CUDA 11.8 wheel；其餘套件版本依原清單。
- 模型：Prithvi-EO-2.0-300M encoder + DeepLabV3 ASPP head；Tversky loss
  `alpha=0.3, beta=0.7`。
- `seed=42`、3-fold CV、每折最多 300 epochs、`patience=25` early stopping、
  `batch_size=16`、`lr0=5e-5`、AMP。設定檔為 `main/local_gt_expand.yaml`。
- 透過 systemd 使用者服務執行 `run_local_3fold.sh`；2026-09-29 17:08 開始，
  21:13 正常結束（exit code 0），總歷時約 4 小時 6 分。

## 三折結果

下表的正式比較指標是 **GT-aware 場景重組的 pooled Oil IoU**；patch Oil IoU
是另一個評估口徑，不應直接與場景指標混比。

| Fold | 實際 epochs | 訓練分鐘 | 測試場景 | Patch Oil IoU | 場景重組 pooled Oil IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 37 | 50.29 | 119 | 0.4542 | 0.4637 |
| 2 | 66 | 74.46 | 118 | 0.4825 | 0.4634 |
| 3 | 100 | 108.98 | 118 | 0.3907 | 0.3726 |
| 三折平均 | — | — | — | 0.4425 | **0.4332** |

三折都因連續 25 輪驗證指標未改善而提前停止，共訓練 203 epochs。交付包
`README.md` 記載同設定前人逐折 pooled Oil IoU 平均約 0.44；本次 0.4332
與之相差約 0.0068。交付包沒有前人完整逐折紀錄，故無法作嚴格的逐折比對。

## 輸出與重現線索

- 訓練日誌：`training_3fold.log`。
- 每折結果：`result-seg/CV_358clean_gt_expand_prithvi300m_dlv3_det/`
  底下的 `20260929-*_local_gt_expand_run_fold{1,2,3}/`，含 `results.csv`、
  `weights/best.pt`、patch `run_metrics.json`、重組 `run_metrics.json`、
  `per_scene_iou.csv` 與預測 artifact。
- 四頁彙整表：
  `result_excel/result_CV_358clean_gt_expand_prithvi300m_dlv3_det_log_restructured.xlsx`。
- 本機操作步驟：`LOCAL_REPRODUCTION.md`。
- 訓練成功後，自動彙整 Excel 曾因呼叫不存在的系統 `python` 而失敗；已修正
  `main/main_runner.py` 改用目前的 Python 解譯器，並從已完成的 JSON 結果補產生
  Excel。此修正沒有重跑或更動訓練結果。

## 未重新執行的上游步驟

- 沒有從 Sentinel-2／Landsat 的原始衛星資料重新執行 Sen2Like 調和；輸入的
  `MS6_sen2like/` 已包含處理好的 8 個波段。
- 沒有重建 manifest、scene-level split、油汙與背景 patch 座標；使用交付包
  已凍結的 `data_split/`。`README.md` 明示 onboarding 版保證可重現的起點
  是此名冊，更早的切分流程依賴未交付資料。
