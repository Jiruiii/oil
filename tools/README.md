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
| `run_hnm_oof.sh` | E oof／F oof_si：三折 OOF 分類器評分 → 選取 → 產生設定 → 訓練 | 本專案 |
| `make_ring_config.py` | 產生 `main/local_ring<N>.yaml`：基準加 `oil_ignore_ring: N`（只影響訓練）並開啟重組的背景誤報與 Boundary IoU 評估；檢查只差允許的行 | 本專案 |
| `rescore_arms.sh` | 用 `analyze_baseline_fp.py` 重算 A～F 六組（含 Boundary IoU），輸出 `hnm_runs/diagnostics_v2/` | 本專案 |
| `threshold_sweep.py` | 重新推論 GT-aware 重組與 test 背景小圖，存 P(oil) 直方圖供門檻掃描 | 本專案 |
| `compare_threshold_sweep.py` | 同 recall（或同背景警報率）下比較各組，場景配對 bootstrap | 本專案 |
| `run_threshold_sweep.sh` | 七組三折跑 `threshold_sweep.py` 後接比較，輸出 `hnm_runs/threshold_sweep/` | 本專案 |
| `oil_component_prob.py` | 每個 GT 油汙區塊的 P(oil) 直方圖與局部機率圖（推論同門檻掃描） | 本專案 |
| `analyze_lost_oil.py` | 以 A 為參考：HNM 整塊漏掉／變小的油、依屬性分層 recall、同地點硬負樣本曝露、漏掉區塊圖集 | 本專案 |
| `run_lost_oil.sh` | A～D 三折跑 `oil_component_prob.py` 後接分析，輸出 `hnm_runs/lost_oil/` | 本專案 |
| `show_hard_negative_tiles.py` | 某組選進訓練的背景小圖（依基準誤報數排序）真彩圖，用來和漏掉的油對照 | 本專案 |
| `feature_space.py` | 在基準 A 的特徵（head 倒數第二層、encoder）中，比較 HNM 漏掉／保住的油與硬負樣本、訓練油、隨機背景的距離（kNN、AUC、PCA 圖） | 本專案 |
| `filter_hard_negatives.py` | G hnm_knn／G0 hnm_swap：在 A 的 encoder 特徵中，把 C 已選硬負樣本裡離訓練油太近的（10 鄰居中訓練油 ≥ 30%）換成同場景的其他候選；G0 同場景同張數隨機替換。`--stage simulate／extract／select` | 本專案 |
| `run_knn_filter.sh` | 抽候選池向量 → 產生 G、G0 名冊與設定 → 兩組各訓練三折，輸出 `hnm_runs/knn_filter/` | 本專案 |
| `run_knn_eval.sh` | G、G0 跑門檻掃描、油汙區塊 P(oil)、診斷，再與 A～F、R 既有輸出比較，輸出 `hnm_runs/knn_filter/{threshold_sweep,lost_oil,diagnostics}/` | 本專案 |
| `run_seed43_eval.sh` | C、G、G0 的 seed 43 重跑做同樣評估，與 seed 42 的 A、B 比較，輸出 `hnm_runs/seed43/` | 本專案 |
