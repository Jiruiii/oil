# tools

專案目錄外原本散落、之後仍會用到的腳本。只收程式，不收報告、簡報與其產出的 csv/xlsx。

| 檔案 | 用途 | 來源 |
| --- | --- | --- |
| `audit_background_inventory.py` | 盤點凍結名冊中各折、各場景的背景小圖數量與標註組成，輸出 csv／json | 原為 `../reports/2026-10-15/background_inventory_20261014/audit_inventory.py`，內容原樣複製 |

注意：腳本內的 `ROOT`、`OUT` 是寫死的本機絕對路徑（`/home/ray950623/oil_dataset`），換機器要先改。
| `run_hnm_si_full.sh` | 依序跑 HNM_SI 三折 mine、select（hnm_si arm）與匯出圖片；用法 `nohup tools/run_hnm_si_full.sh > hnm_runs/hnm_si_v1/run.log 2>&1 &`（2026-10-17 實際跑法，三折探勘約 2.4 小時） | 本專案 |
