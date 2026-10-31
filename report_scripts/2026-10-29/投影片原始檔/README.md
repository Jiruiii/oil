# 10/29 簡報原始檔

程式在這裡；圖片（`img/`）與輸出的 pptx 在 repo 外的 `../reports/2026-10-29/`。

- `build.js`：產生 `reports/2026-10-29/HNM週進度報告_20261029.pptx`（pptxgenjs 3.12）。
  `NODE_PATH=<node_modules> APPLY_THEME=<pptx skill 的 apply_theme.js> node build.js <輸出.pptx>`；圖片讀 `reports/2026-10-29/投影片原始檔/img/`。
- 第 2 頁「實驗組別」（10/28 晚上加，使用者移到第 2 頁）沒有圖；10/29 改成七列表格（每欄一個可能不同的設定，相同值合併），取代原本的卡片與分隔線。
- `example.py`：A 與 R 在同一位置的預測對照圖（備用第 18 頁），在 repo 根目錄用 `.venv/bin/python` 執行：
  `report_scripts/2026-10-29/投影片原始檔/example.py <fold> <scene> <輸出前綴> <視窗邊長> [pct]`；本次用 fold 3 `..._20210617_S2_15RVM`、fold 2 `..._20180627_S2_16RCT`，邊長 240。
  預設固定拉伸（反射率 ÷ 0.15）。15RVM 海面反射率約 0.17、油約 0.30，會整片截成白色，所以 10/29 改加 `pct`（依視窗 2–98% 拉伸）重新出圖；預測與像素數不變。
- `lookalike_figs.py`：第 11、13、14 頁的圖（門檻掃描、漏掉的油與硬負樣本、特徵 PCA），在 repo 根目錄用 `.venv/bin/python` 執行，輸出到 `reports/2026-10-29/投影片原始檔/img/`。
- 數字來源：`hnm_runs/diagnostics_v2/comparison.md`、`paired_differences.csv`。
  look-alike 頁：`hnm_runs/threshold_sweep/matched_comparison.md`、`hnm_runs/lost_oil/summary.md`、`hnm_runs/feature_space/summary.md`。
  第 15 頁（G／G0 初步結果）：`hnm_runs/knn_filter/{threshold_sweep/matched_comparison.md,lost_oil/summary.md,diagnostics/comparison.md}`。
