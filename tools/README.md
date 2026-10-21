# tools

專案目錄外原本散落、之後仍會用到的腳本。只收程式，不收報告、簡報與其產出的 csv/xlsx。

| 檔案 | 用途 | 來源 |
| --- | --- | --- |
| `audit_background_inventory.py` | 盤點凍結名冊中各折、各場景的背景小圖數量與標註組成，輸出 csv／json | 原為 `../reports/2026-10-15/background_inventory_20261014/audit_inventory.py`，內容原樣複製 |

注意：腳本內的 `ROOT`、`OUT` 是寫死的本機絕對路徑（`/home/ray950623/oil_dataset`），換機器要先改。
| `run_hnm_si_full.sh` | 依序跑 HNM_SI 三折 mine、select（hnm_si arm）與匯出圖片；用法 `nohup tools/run_hnm_si_full.sh > hnm_runs/hnm_si_v1/run.log 2>&1 &`（2026-10-17 實際跑法，三折探勘約 2.4 小時） | 本專案 |
| `analyze_baseline_fp.py` | 計畫步驟 1：基準偽陽性診斷（GT-aware 重組輸出＋各折 test 背景小圖推論），分類 FP 落點、已選背景含陸地／雲比例、SR 指數飽和；輸出到 `hnm_runs/diagnostics/`。用法 `.venv/bin/python tools/analyze_baseline_fp.py --config preprocess/hnm_si/config.example.yaml`（三折約 25 分鐘，GPU 只用於背景小圖推論） | 本專案 |
| `make_hnm_arm_configs.py` | 計畫步驟 2：由 `main/local_gt_expand.yaml` 產生 `main/local_hnm_{random,hnm,hnm_si}.yaml`，並檢查除結果路徑、名稱與 `hnm` 區塊外逐行相同；`--seed` 供步驟 4 | 本專案 |
| `run_hnm_step2.sh` | 計畫步驟 2：依序訓練三個 arm（各三折，約 12 小時 GPU）；用法 `nohup tools/run_hnm_step2.sh > hnm_runs/step2_training.log 2>&1 &`，可只給部分 arm 名稱 | 本專案 |
| `compare_hnm_arms.py` | 計畫步驟 2：讀各組 `analyze_baseline_fp.py` 輸出（基準＋各 arm 用 `--run-dir` 分析），算逐折與三折平均配對差及場景 bootstrap 信賴區間；輸出 `hnm_runs/diagnostics_step2/comparison.md` | 本專案 |
