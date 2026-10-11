# 本機重跑：GT-expand 3-fold 實驗

本流程使用交付資料中的 355 個場景與 `data_split/` 已凍結的 3-fold 名冊，重跑
`main/experiments.example.yaml` 所代表的 Prithvi-EO-2.0-300M + DeepLabV3 設定。
依 `README.md`，這是 onboarding 版本保證可重現的起點；舊版 scene split 產生流程
依賴未交付的資料，因此不重新切分 fold。

## 本機環境差異

- GPU：NVIDIA GeForce RTX 4090，24 GB；主機驅動 550.144.03。
- 原始 `requirements.txt` 的 PyTorch CUDA 12.8 wheel 需要較新的驅動。
- `requirements.local-cu118.txt` 只把 PyTorch 三個 wheel 改為 CUDA 11.8，
  其餘套件版本與原始清單相同。這是**環境差異**，不能宣稱與原文中的 CUDA 12.8
  執行環境逐位元相同；報告結果時須註明。

## 執行順序

從本專案根目錄執行：

```bash
cd /home/ray950623/oil_dataset/OIL_PROJECT_MutiBand_GT_expand

# 1. 環境（已建立 .venv 時可略過 venv 這行）
python -m venv .venv
.venv/bin/python -m pip install -r requirements.local-cu118.txt

# 2. 從 8 個波段與 JSON/GPKG 標註產生 VRT 和 GT mask；可續跑
.venv/bin/python preprocess/prepare_local_data.py \
  --data-dir /home/ray950623/oil_dataset

# 3. 檢查凍結名冊引用的全部場景、8 band、mask 值域及空間對齊
.venv/bin/python preprocess/preflight_check.py \
  --split-dir data_split/patch_level_gt_expand \
  --bg-tsv data_split/patch_level_gt_expand_bg/bg_coords.tsv \
  --vrt-dir /home/ray950623/oil_dataset/MS6_sen2like_vrt \
  --mask-dir /home/ray950623/oil_dataset/mask

# 4. 核對重新產生的 mask 與凍結統計及 3,510 個背景 patch 座標
.venv/bin/python preprocess/verify_frozen_labels.py \
  --mask-dir /home/ray950623/oil_dataset/mask \
  --stats data_split/patch_level_gt_expand/stats.csv \
  --bg-tsv data_split/patch_level_gt_expand_bg/bg_coords.tsv

# 5. 下載並驗證 pin 住的 Prithvi 300M checkpoint
.venv/bin/python main/fetch_prithvi_checkpoint.py

# 6. 完整 3-fold 訓練、patch 測試和 GT-aware scene reconstruction
NO_ALBUMENTATIONS_UPDATE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  .venv/bin/python main/main_runner.py --config main/local_gt_expand.yaml
```

訓練設定在 `main/local_gt_expand.yaml`。結果會寫入 `result-seg/`，彙整表會寫入
`result_excel/`。三折均完成後，檢查每折的 `run_metrics.json`、GT-aware
`per_scene_iou.csv`、預測 artifact，以及彙整 Excel 的 pooled Oil IoU。

`main_runner.py` 每次啟動會建立新時間戳結果目錄；中斷後直接重跑會產生新實驗，
不能把新舊 fold 的結果當作同一次完整 3-fold 比較。

本機長時間執行使用 `oil-gt-expand-3fold-20260929.service` 使用者 systemd 服務，
實際命令在 `run_local_3fold.sh`，輸出在 `training_3fold.log`。查看狀態：

```bash
systemctl --user status oil-gt-expand-3fold-20260929.service
tail -f training_3fold.log
```

本機已啟用 `loginctl enable-linger ray950623`，`Linger=yes`；
使用者服務在 SSH 工作階段結束後仍會繼續執行。主機重新開機會中斷這個暫時性服務。

事件場景的 mask 使用 GPKG 原始座標產生；現成 JSON 的座標已四捨五入到小數一位，
直接用 JSON 會讓 33 個場景的油汙像素數與凍結統計略有差異。
`2022年祕魯La_Pampilla煉油廠洩油事件_20220202_S2_18LTM` 的 GPKG 另含四個
未納入凍結統計的油汙 polygon，準備腳本會以 JSON 辨識保留的 polygon，
再用對應 GPKG 的完整精度座標產生 mask。
