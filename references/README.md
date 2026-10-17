# HNM 參考來源

本目錄保存兩個不同來源的原始碼快照，均不是本專案的執行入口。

## 學長提供的多光譜 HNM_SI

[hnm_si_original](hnm_si_original/README.md) 保存 `OIL_PROJECT_MutiBand_thesis` 的
HNM_SI 四步流程、driver、光譜門檻驗證腳本與 probe 訓練設定，逐檔保持原樣並附 SHA256。
來源、收錄範圍與已知差異見該目錄 README；移植方案見
[HNM_SI_INTEGRATION.md](../HNM_SI_INTEGRATION.md)。目前尚未接入。

## SAR OOF 第一版的參考來源

`pipeline_step6_HNM_MaskSplit_NoLand_CoastOcean_CV.py` 是使用者提供的原始 SAR
參考程式之原樣快照，不是本專案的執行入口。原檔仍保留在資料集根目錄。
本專案尚未取得完整論文書目；使用者提供的截圖為第 3.5.3 節「EL2N 困難度評分」，
描述 EfficientNetV2-S、五折分組 OOF、背景 EL2N = sqrt(2) × p_oil，以及共用 embedding 模型。
不要將未知的論文作者、標題或引用編號補成確定的書目資訊。

已確認的原程式限制：

- 依賴 PNG/JPG、0/90/255 SAR 遮罩與未提供的外層 CV helper。
- 預設關閉 inner OOF，inner K 設為 3，與截圖的 K=5 不同。
- `greedy_facility_location(k=0)` 回傳全部索引，會破壞 LAND_ALLOC=0 的意圖。
- 多處重複設定；部分註解與實際程式不同。

多波段適配實作位於 `preprocess/hnm/`，不執行此參考檔。
使用方式、保留的設計與實驗差異記載於根目錄 `HNM.md`。
