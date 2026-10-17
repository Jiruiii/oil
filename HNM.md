# 多波段 OOF HNM 第一版使用方式

更新說明（2026-10-16）：本文件與下列指令只對應已接入的 EfficientNetV2-S＋OOF
第一版，不是學長提供的分割誤報＋光譜過濾 HNM_SI。HNM_SI 尚未移植，接入方案見
[HNM_SI_INTEGRATION.md](HNM_SI_INTEGRATION.md)，原始碼快照見
[references/hnm_si_original](references/hnm_si_original/README.md)。兩種方法尚無正式 HNM 實驗結果。

本實作在已凍結的三折實驗中，只篩選各折 **train** 的背景座標。
原有 GT-centric 名冊與 val/test 背景不變。沒有啟用 `hnm.enabled` 的設定沿用原流程。

## 方法與論文的關係

- 使用八通道 `tf_efficientnetv2_s` 二元分類器，直接讀取 VRT 的 256×256 window。
  反射率除以 10000，再使用設定檔的逐波段 mean/std。
- 每個 outer train pool 內做五折 StratifiedGroupKFold。預設按場景分組；
  可提供 `scene\tgroup` TSV 綁定事件。若事件跨越既有 outer split，明確中止，
  不會為了事件分組偷偷改掉基準三折。
- 分類標籤由原始 mask 的 `0=Oil` 判定。GT-centric 名冊中的小圖不一定含油污，
  這些固定背景仍參與 miner 訓練，但不納入新增背景候選的抽樣配額。
- 背景 EL2N = `sqrt(2) * p_oil`。每筆候選只由未訓練過其群組的分類器評分。
- 各 miner 訓練固定 epochs，不用 OOF held-out fold 挑 checkpoint、epochs 或做早停。
  若未來需要早停，必須再從 miner 的訓練部分切出獨立的內部驗證組。
- 另用全部 **outer train** 資料訓練一個共用 embedding 模型。它不負責 OOF 難度分數。
- 先選高 EL2N 候選，再從剩餘候選以 cosine facility-location 選具代表性的樣本。
  零名額回傳空集合，不會出現參考 SAR 程式的 `k=0` 全選問題。
- 目前沒有對齊的陸地標註，所以不套用 land/coast/ocean 配額與 SAR 亮度過濾。
  這是多波段適配版，不宣稱完整重現截圖以外的論文方法。

每折共訓練 5 個 OOF miner 加 1 個 embedding miner，三折合計 18 個模型。
預設 70 epochs、batch 8；這不是已調優的多波段超參數。
預設 `pretrained: true` 可能下載 timm 權重，timm 會將輸入權重適配到八波段。
`pretrained: false` 可離線測試，但屬不同實驗設定。

## 操作

以下指令在 repo 根目錄執行。HNM config 中的相對路徑以 **config 檔所在目錄** 解析；
既有分割訓練 YAML 的相對路徑則仍以 repo 工作目錄解析。

1. 檢查 `preprocess/hnm/config.example.yaml` 的資料與輸出路徑。
   範本假設 repo 與 `MS6_sen2like_vrt/`、`mask/` 同在資料集根目錄中。
   如要更改，可複製為 `preprocess/hnm/config.local.yaml`。
2. 先檢查三折分組、波段／mask 對齊與分類標籤（不訓練、不寫產物）：

   ```bash
   .venv/bin/python -m preprocess.hnm.pipeline \
     --config preprocess/hnm/config.example.yaml --stage plan
   ```

3. 驗證所有 train 候選不含油污，且至少 30% 為標註背景、未標註比例低於 50%：

   ```bash
   .venv/bin/python -m preprocess.hnm.pipeline \
     --config preprocess/hnm/config.example.yaml --stage validate
   ```

4. 確認 GPU 可用後正式訓練 miner 與抽樣：

   ```bash
   NO_ALBUMENTATIONS_UPDATE=1 .venv/bin/python -m preprocess.hnm.pipeline \
     --config preprocess/hnm/config.example.yaml --stage mine
   ```

   可加 `--fold 1` 只跑一折。`device: cuda` 時找不到 GPU 會明確失敗；
   CPU 測試需明確設定 `device: cpu`。不會因找不到 GPU 靜默啟動長時間 CPU 訓練。

5. 複製 `main/hnm.example.yaml` 為 `main/local_hnm.yaml`，填入 `<DATA_DIR>`，
   核對 `hnm.artifacts_dir` 指向完成的輸出根目錄，將實驗 `run` 改為 `true`：

   ```bash
   NO_ALBUMENTATIONS_UPDATE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
     .venv/bin/python main/main_runner.py --config main/local_hnm.yaml
   ```

## 配額與公平比較

預設 `keep_fraction: 0.5` 是保留 train 背景候選池的一半，
`hard_fraction: 0.5` 是將其中一半名額給最困難的候選，剩餘名額給多樣性抽樣。
這是可執行的起始設定，**不是論文確定的比例，也不是負／正樣本 1.5 倍**。
目前既有背景池並不大；若全部保留，HNM 就沒有改變資料組成。

`candidate_bg_tsv` 可指向更大的候選池，`baseline_bg_tsv` 則保留原始名冊，
用來維持 val/test 的背景集合。候選池的非 train 場景不參與本折 mining。
不得把未標註區域一律當作乾淨背景。

HNM 減少樣本數也會改變每 epoch 步數。正式比較應另設相同背景數量的隨機抽樣對照，
並記錄總 optimizer steps、seed、候選池、抽樣數量及 GPU 環境。

## 產物與重跑

每個 `hnm_runs/oof_v1/fold_N/` 包含：

- `plan.json`：設定、分組、來源名冊 SHA256 與數量。
- `miner_oof_*.pt`、`miner_embeddings.pt`：分類器權重。
- `scores.tsv`、`embeddings.npy`：同序候選分數與共用特徵。
- `selected_train_bg.tsv`：選出的 train 背景。
- `bg_coords.tsv`：選出的 train 背景，加上原樣保留的非 train 背景。
- `training_history.json`：固定 epochs 的訓練 loss/accuracy。
- `manifest.json`：最後才寫入的完成標記與產物雜湊。

分割訓練會核對 fold、三份 split、原始背景清單、產物雜湊、資料路徑、patch size，
並再次確認非 train 背景沒有改變。HNM 與既有 rotating BG sampler 不可同時啟用。
不對大型影像逐檔做內容雜湊；請把原始影像及 mask 視為不可變資料，變更後另跑 HNM。

目前不支援 epoch 中途接續。已有 fold 輸出時不會覆寫或盲目沿用舊 checkpoint；
中斷後使用新的 `output_dir` 重跑該折。各完整 fold 可透過 `--fold` 分開執行。
不要把 CPU smoke-test 結果當成正式實驗產物。

## 開發驗證

```bash
NO_ALBUMENTATIONS_UPDATE=1 .venv/bin/python -m unittest discover -s tests -v
```

測試涵蓋群組隔離、零配額、重複座標、真實八通道 window 讀取、GT 小圖實際標籤、
小模型完整訓練／評分／抽樣／訓練設定接線，以及資料修改與不完整產物的拒絕。

本機初次驗證：三折的 mask 分類及背景候選有效性檢查均通過；原名冊未修改。

| Outer fold | 實際含油 GT 小圖 | 固定 GT 背景小圖 | 背景候選 | 預設選取數 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1168 | 250 | 1860 | 930 |
| 2 | 1349 | 389 | 1860 | 930 |
| 3 | 1163 | 273 | 1870 | 935 |

另已驗證未載入預訓練權重的八通道 EfficientNetV2-S 可前向／反向運算，
共用 embedding 維度為 1280。這些是程式與資料檢查，尚不是正式 HNM 效能結果。
正式 miner 與後續分割訓練尚未執行。
