# 學長 HNM_SI 原始碼快照

收錄日期：2026-10-16。

來源為使用者提供的 `/home/ray950623/oil_dataset/OIL_PROJECT_MutiBand_thesis`。
本目錄保留來源的相對目錄結構與檔案原始內容，供方法溯源與移植比對；
本目錄 README 與 `SHA256SUMS` 是本專案新增的說明與檢查檔。

## 收錄範圍

- `preprocess/HNM_SI/README.md`：原始方法說明。
- `preprocess/HNM_SI/run_HNM_SI.py`：四步 driver。
- `preprocess/HNM_SI/step1_mine_fp.py`：分割誤報探勘。
- `preprocess/HNM_SI/step2_si_distance.py`：油污光譜參考與馬氏距離。
- `preprocess/HNM_SI/step3_si_filter.py`：光譜距離分位數過濾。
- `preprocess/HNM_SI/step4_assemble.py`：抽樣與名冊組合。
- `preprocess/HNM_SI/analysis_top_q_validation/gt_oil_d_si_validation.py`：原始門檻驗證腳本。
- `main/experiments_CV_MS5_deeplab_HNM_probe.yaml`：原始探勘模型訓練設定。

未收錄整包中的其他模型、資料名冊、結果圖表、分析 CSV、日誌與 Python 快取。
完整來源約 145 MB，仍留在 repo 外的原位置；本快照不是完整專案備份。

## 使用方式與已知差異

這是參考快照，不是可直接在本專案執行的入口。保留了來源機器的絕對路徑、
11 波段模型與五折假設；未修改原始 YAML 中的 `run: true`。
未來適配程式應另置於 `preprocess/hnm_si/`，保留本快照供比較。

原 README 第三個 SI 名稱 `SR_B12_B3B8A` 與程式不同；程式實際計算
`(B1 + B2) / (B3 + B8A)`。原 Step 4 實際將 HN 追加到整份原名冊，
而非只合併油污 patch；原 Step 2 的共變異矩陣由多折候選場景合併建立。
上述差異在快照中保留，適配處理見 [接入方案](../../HNM_SI_INTEGRATION.md)。

可在本目錄執行 `sha256sum -c SHA256SUMS` 檢查八個來源檔案是否保持原樣。
