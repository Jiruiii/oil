# 油汙偵測專案 — Onboarding 版（GT_expand 分支 B）

> 本目錄是從正式研究 repo `OIL_PROJECT_MutiBand_GT_expand` 分出來的**精簡版**，
> 只保留「能跑出整個 pipeline」所需的程式碼與目前最強的已釋出架構設定。
> 不含研究過程中的分析腳本、個人筆記、單元測試、實驗記錄與已否決的架構分支——
> 這些留在正式 repo，此處只給新人一條可以照著跑一遍的主線。

---

## 一、專案一句話

用 8-band Sentinel-2（Sen2Like 調和）衛星影像訓練語意分割模型，對海面油汙做逐像素
偵測。目前最強設定 = **Prithvi-EO-2.0-300M（ViT）encoder + DeepLabV3(ASPP) head +
Tversky loss**，用 **GT-centric patch 策略**（每個 patch 圍繞真實油汙 bbox 取樣，
另混入部分純背景 patch）訓練，3-fold cross-validation，每折訓練完自動把 patch 級
預測拼回全場景大圖算 IoU。

這是研究專案的**分支 B**（GT-centric patch，估監督式分割效能上限），fork 自
sliding-window 版本的分支 A（論文主軸，整圖偵測）。兩者 IoU 分母不同、**不可直接互比**。

---

## 二、環境設置

實驗直接跑在 host 的 conda 環境（無獨立訓練容器）：

```bash
# Python 3.11, CUDA 12.8（需支援 RTX 50 系 / Blackwell sm_120 的話 torch 必須 >= 2.7）
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

`requirements.txt` 已從實際在跑的環境版本鎖定（rasterio/albumentations/geopandas 等）。

**Docker 版本（可選，適合連同環境一起交接給別人）**

`Docker/Dockerfile` 是乾淨版訓練環境（不含任何個人 git 身份 / SSH /
Obsidian vault 掛載；有裝 Claude Code 與 Codex CLI 供新人開發用，但不含你
個人的登入狀態/agent 記憶），base image 跟上面 pip 安裝用的是同一個 cu128
版本。這個 cu128 wheel 是多架構編譯，從 Ampere（sm_86，例如 RTX 3090）到
Blackwell（sm_120，RTX 50 系）都涵蓋，換到別張卡不需要改 Dockerfile，只要
對方 host 的 NVIDIA driver 支援 CUDA 12.8（Linux 需 driver >= 570.26，這是
CUDA 12.8 GA 對應的門檻，用 `nvidia-smi` 查版本；550.x 只夠 CUDA 12.4，Blackwell
sm_120 也是從 570 系列驅動才開始支援）。

對方拿到這個資料夾後，第一次設定分兩步：

```bash
# 1. host 端一次性設定：讓 Docker 容器能存取 GPU（需要 sudo）
#    這支只在「host 主機」上跑，不是在容器裡跑
bash Docker/setup_host_gpu.sh

# 2. build image（要在本目錄根目錄下跑，因為 Dockerfile 裡的 COPY 要抓到根目錄的 requirements.txt）
docker build -f Docker/Dockerfile -t oil-detect:latest .

# 3a. 跑容器，把程式碼和資料集分別掛進去（一次性、關掉終端機容器就沒了）
docker run --rm -it --gpus all --shm-size=32g \
    -v "$PWD:/workspace" \
    -v "/path/to/oil_dataset:/data" \
    oil-detect:latest

# 3b. 或用 Docker/run.sh：背景常駐 container，關終端機/斷線不影響訓練，
#     之後重跑同一個指令會直接 exec 進同一個 container（不會重建）
DATA_DIR=/path/to/oil_dataset ./Docker/run.sh
```

`Docker/setup_host_gpu.sh` 只需要在每台新機器上跑一次（裝
`nvidia-container-toolkit` 並接上 docker runtime）；之後同一台機器上 build/run
不同版本的 image 都不用重跑。

`Docker/run.sh` 是乾淨版的常駐啟動腳本，只掛程式碼目錄和 `DATA_DIR` 兩個路徑，
不含任何個人 SSH / agent 狀態（跟正式 repo 裡那份給你自己開發機用的 `run.sh`
是兩回事，那份不適合交接給別人）。

---

## 三、資料前置（大型資料不含在本目錄，需自行準備 / 指到既有資料）

本目錄含**程式碼 + 已凍結的 3-fold split 名冊**（`data_split/`，內容跟正式
repo 逐檔一致，已核對過：三折共用同一組 2,893 個油汙 patch、background patch
3,510 筆／351 scenes，train/val/test 的 patch 與 scene 交集皆為 0）。
不含的是原始影像、GT 標註、或訓練好的權重（皆為大型二進位/資料檔，不適合
放進程式碼分支）。實際訓練需要：

| 資料                     | 用途                                  | 典型路徑（正式 repo 現況，需按你自己的環境調整）                                     |
| ------------------------ | ------------------------------------- | ------------------------------------------------------------------------------------ |
| 8-band Sen2Like TIF      | 原始影像（每場景 8 個 band TIF）      | `.../full_band/MS6_sen2like/`                                                      |
| VRT 虛擬多波段檔         | 訓練時實際讀取的來源（見下方步驟 A）  | `.../full_band/MS6_sen2like_vrt/`                                                  |
| GT Mask TIF              | 逐場景 0=Oil/1=BG/2=Others/255=未標註 | `.../full_band/mask/`                                                              |
| fold split 名冊 / bg 座標 | 訓練/驗證/測試場景與背景 patch 分配   | **已包含**：`data_split/patch_level_gt_expand/`、`data_split/patch_level_gt_expand_bg/` |
| Prithvi-EO-2.0-300M 權重 | ViT encoder 預訓權重                  | 用`main/fetch_prithvi_checkpoint.py` 下載，會存到 `<repo_root>/pretrained_ckpt/` |

> `DATA_DIR` 底下要放的目錄結構、數量、大小，跟一個容易踩到的地雷（VRT 是
> XML 指標檔，不能單獨複製過去就用），細節都寫在 **`DATA_MANIFEST.md`**，
> 開始準備資料前建議先看那份。

> **`main/experiments_CV_358clean_gt_expand_prithvi300m_dlv3_det.yaml`
> 裡的所有絕對路徑，都是原始 repo 在這台機器上的路徑**（`results_base_dir` /
> `excel_log_path` / `backbone_ckpt` / `split_dir` / `labels_dir` / `bg_coord_txt` /
> `vrt_dir` / `mask_dir`）——**這份是研究記錄，不要直接改著跑**。新人請改用
> **`main/experiments.example.yaml`**：複製一份成 `main/local_gt_expand.yaml`，
> 只需要填 `<DATA_DIR>` 相關的幾個欄位（範本裡有標 `<FILL ME>`），
> `split_dir`/`bg_coord_txt` 已經指向本目錄內建的 `data_split/`，不用自己重切；
> `run` 預設是 `false`，路徑確認沒問題再手動改成 `true`。
>
> 下載完 Prithvi checkpoint 後記得回頭把 yaml 的 `backbone_ckpt` 指到新下載的
> 檔案位置——`fetch_prithvi_checkpoint.py` 不會自動幫你改 yaml
> （`experiments.example.yaml` 預設值已假設你用它的預設輸出路徑）。

> **前處理腳本（`preprocess/` 下除了 `preflight_check.py`）目前是「參考用」，
> 不保證能直接重跑**：這些腳本路徑多半寫死在正式機器上，且部分依賴正式 repo
> 裡沒有交付的舊版名冊/腳本。本 onboarding 目錄實際保證能跑通的路徑是「從已
> 凍結的 `data_split/` 直接開始訓練」，不是「從原始 JSON/GPKG 重建整條
> pipeline」。如果你真的需要重建 split/patch 座標，請回頭參考正式 repo。

---

## 四、Pipeline 總覽

```
原始 8-band TIF（每場景）
    │
    ├─ preprocess/build_vrt_ms6.py            → 每場景一個 .vrt（省磁碟，避免實體疊圖）
    │
    ├─ preprocess/json_to_mask_tif.py         → LabelMe JSON → 全場景 GT Mask TIF
    │  preprocess/batch_event_gpkg_to_mask.py → （事件場景）GPKG → JSON → Mask TIF
    │
    ├─ preprocess/build_manifest.py           → 全資料集 manifest.csv（單一真相帳本，
    │                                             重跑會 diff 出新增/異動場景）
    │
    ├─ preprocess/resplit_stratified_v2.py    → 依油汙面積 tertile 分層抽樣，
    │                                             產出 3-fold scene-level split
    │
    ├─ preprocess/generate_patch_coords_gt_expand.py
    │       → GT-centric patch 座標：每個油汙 connected component 取 bbox 中心
    │         256×256 patch；bbox 過大則用 256-grid 鋪磚覆蓋整個 bbox
    │
    ├─ preprocess/generate_bg_patches.py
    │       → 同場景油汙範圍外滑窗取樣純背景 patch（混入訓練集，避免模型只看得到油）
    │
    ├─ preprocess/preflight_check.py          → 訓練前體檢：VRT/Mask 存在性、
    │  preprocess/check_dataset_completeness.py  shape/CRS 對齊、mask 值域合法性
    │
    ▼
main/fetch_prithvi_checkpoint.py             → 下載 Prithvi-EO-2.0-300M 權重（pin
                                                 revision + SHA256，程式不隱式連網）
    ▼
main/main_runner.py --config <experiment.yaml>
    │   （讀 YAML → 每個實驗 → cross_validation.enabled 時跑 3-fold）
    │
    ├─ for fold in 1..3:
    │     training_module.train_model()        → deeplab_adapter.DeepLabAdapter
    │                                              建模/訓練/存 best.pt
    │         ├─ model_family: prithvi_eo_v2_dlv3
    │         │     prithvi_backbone.py  → 共用底層 ViT 模組
    │         │     prithvi_upernet.py   → PrithviViTEncoder（兩個 family 共用的
    │         │                             encoder 建構＋權重嚴格載入）
    │         │     prithvi_deeplab.py   → PrithviDeepLabV3Segmenter（ASPP head）
    │         │
    │     recon_gt_aware_module.run_reconstruction_gt_aware()
    │                                    → patch 預測拼回原場景座標，算 pooled_oil_iou
    │         （用到 predmask_codec.py 做 pixel-exact 預測遮罩編碼）
    │     tracking_module.log_to_excel() / log_run_json()
    │                                    → 寫入 per-fold run_metrics.json
    │
    ▼
main/restructure_excel.py --result-base <result_dir> --dst <xlsx>
    → 把各 fold 的 run_metrics.json 彙整成一份 Excel 總表
```

`reconstruct_module.py` 是 `main_runner.py` 一律會 import 的舊版整圖重組模組
（非 GT-aware 模式時使用）；本 yaml 走 `mode: 'gt_aware'`，實際重組用的是
`recon_gt_aware_module.py`。`evaluation_module.py` 是 patch-level 評估（CV 流程中
目前不主動觸發，僅 `mode: 'test'` 且未設 `post_tests` 時才會走到）。
`backfill_patch_macro_metrics.py` / `visualize_prediction_artifact.py` 是維運工具：
前者補算既有 patch 的巨集指標，後者可從 `recon_gt_aware_module` 產出的 pixel-exact
JSON 預測還原成圖片。

---

## 五、檔案職責對照表

### `preprocess/`

| 檔案                                   | 職責                                                   |
| -------------------------------------- | ------------------------------------------------------ |
| `build_vrt_ms6.py`                   | 8-band TIF → VRT 虛擬多波段檔（參考波段固定 B02/10m） |
| `json_to_mask_tif.py`                | LabelMe JSON → GT Mask TIF                            |
| `batch_event_gpkg_to_mask.py`        | 事件場景：GPKG → JSON → Mask TIF                     |
| `build_manifest.py`                  | 產出/更新資料集 manifest.csv，偵測 mask 變動、缺檔     |
| `resplit_stratified_v2.py`           | 目前主用的 3-fold stratified split 定義腳本            |
| `generate_patch_coords_gt_expand.py` | GT-centric patch 座標產生（分支 B 核心）               |
| `generate_bg_patches.py`             | 純背景 patch 座標產生（混入訓練集）                    |
| `preflight_check.py`                 | 訓練前資料體檢                                         |
| `check_dataset_completeness.py`      | 資料集完整性掃描（JSON/Mask/COG）                      |

### `main/`

| 檔案                                                                      | 職責                                                           |
| ------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `main_runner.py`                                                        | 入口，讀 YAML、跑 CV 實驗排程、seed 控制                       |
| `training_module.py`                                                    | 建模＋訓練迴圈調度                                             |
| `deeplab_adapter.py`                                                    | 模型定義（含 model_family 路由）、Dataset、train()/val()       |
| `prithvi_backbone.py` / `prithvi_upernet.py` / `prithvi_deeplab.py` | Prithvi ViT encoder（共用）＋ DeepLabV3(ASPP) head             |
| `recon_gt_aware_module.py`                                              | GT-aware 全場景重組評估（目前主用）                            |
| `reconstruct_module.py`                                                 | 舊版整圖 sliding-window 重組（`main_runner.py` 固定 import） |
| `predmask_codec.py`                                                     | Pixel-exact 預測遮罩編碼／解碼                                 |
| `evaluation_module.py`                                                  | Patch-level 評估（CV 流程中非主要路徑）                        |
| `tracking_module.py`                                                    | 結果寫入 Excel／JSON                                           |
| `utils.py`                                                              | 產生暫時 data.yaml、計算圖片數量等小工具                       |
| `restructure_excel.py`                                                  | 彙整各 fold`run_metrics.json` 成總表 Excel                   |
| `backfill_patch_macro_metrics.py`                                       | 補算既有 patch 的巨集指標                                      |
| `visualize_prediction_artifact.py`                                      | 從 pixel-exact JSON 預測還原成可視化圖片                       |
| `fetch_prithvi_checkpoint.py`                                           | 下載並驗證 Prithvi-EO-2.0-300M 權重                            |
| `prithvi_vendor_manifest.json`                                          | 上述權重的 pin revision / SHA256 紀錄                          |
| `experiments_CV_358clean_gt_expand_prithvi300m_dlv3_det.yaml`           | 目前最強已釋出架構的實驗設定（研究記錄，不要直接改著跑）       |
| `experiments.example.yaml`                                              | **新人請從這份開始**：路徑改成佔位符的 portable 範本           |

---

## 六、目前保留的實驗設定說明

正式 repo 底下有數十個 yaml（各種架構/loss/資料消融實驗），本目錄**只保留一份**：

**`experiments_CV_358clean_gt_expand_prithvi300m_dlv3_det.yaml`**

- Backbone：Prithvi-EO-2.0-300M（ViT，凍結權重嚴格比對載入，`backbone_lr_mult=0.1`）
- Head：torchvision DeepLabV3 ASPP head，`aspp_atrous_rates=[6,12,18]`（這是目前判定
  為現行值的設定；曾實驗過 `[2,4,6]` 但未證實有效益，故不採用）
- Loss：Tversky（`alpha=0.3, beta=0.7`）——相對 baseline Focal/CE 全面提升（大油汙、
  中型油汙皆顯著變好，tiny 油汙無顯著傷害），目前是標竿 loss 設定
- 輸入：8-band，實際餵入 backbone 6 band（`input_band_indices=[1,2,3,5,6,7]`，即
  B02/B03/B04/B8A/B11/B12；B01/B08 兩槽在 backbone 端不使用）
- 3-fold cross validation，`epochs=300, patience=25, batch_size=16, lr0=5e-5`
- 訓練完每折自動觸發 `mode: 'gt_aware'` reconstruction

**已知現行基準數字**（此設定 + Tversky，2026-09 決定性重跑後）：逐折 pooled
Oil IoU 平均 ≈ **0.44**（不同統計口徑——逐折平均 vs 全語料 pooled——會有落差，
正式引用請以正式 repo 當時最新的實驗記錄為準，本目錄不含實驗歷史）。
