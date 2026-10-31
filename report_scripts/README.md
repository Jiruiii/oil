# 報告用程式

每週報告的分析與出圖程式，依報告日期分資料夾。程式放在這裡進版控；
輸出（圖、csv、pptx）仍寫到 repo 外的 `../reports/<日期>/` 同名資料夾，不進 git。

都在 repo 根目錄用 `.venv/bin/python` 執行，例如：

```bash
.venv/bin/python report_scripts/2026-10-22/analysis/contrast_recall.py --help
```

| 資料夾 | 程式 | 內容 | 輸出 |
| --- | --- | --- | --- |
| `2026-10-15/background_inventory_20261014/` | `audit_inventory.py` | 背景小圖數量盤點（每折、每場景） | `../reports/2026-10-15/background_inventory_20261014/` |
| `2026-10-22/analysis/` | `contrast_recall.py` | 依 B8A 對比（亮／暗／弱對比）分層的 recall；`tools/analyze_lost_oil.py` 沿用同一定義 | `../reports/2026-10-22/analysis/contrast_recall*/` |
| | `boundary_tolerance.py` | 排除 GT 邊界兩側 TOL px 後的 Oil IoU，以及漏掉的油位在哪裡 | `../reports/2026-10-22/analysis/boundary_tolerance/` |
| | `build_review.py` | 可疑標註的抽樣圖與判讀表（10/22 試做，標註後來確定以 NOAA 為準） | `../reports/2026-10-22/analysis/review/` |
| | `label_check.py`、`label_overview.py` | 標註檢查用的裁切圖與全景縮圖 | `../reports/2026-10-22/analysis/label_check/` |
| | `oof_composition.py`、`sample_arm_sheets.py` | E、F 背景小圖的陸地／雲／霧耀光組成；各組背景抽樣看圖 | `../reports/2026-10-22/analysis/hnm_vs_literature/` |
| `2026-10-29/投影片原始檔/` | `build.js`、`example.py`、`lookalike_figs.py` | 10/29 簡報與圖，見該資料夾的 README | `../reports/2026-10-29/` |
