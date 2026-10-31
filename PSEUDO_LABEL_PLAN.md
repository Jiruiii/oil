# MariNeXt 偽標籤計畫

建立日期：2026-10-31。10/29 報告後的方向調整：HNM 先暫停，改用外部 SOTA 模型 MariNeXt 產生偽標籤，
重建訓練資料集。決策與目前結論見 [RESEARCH_GOAL.md](RESEARCH_GOAL.md)，執行紀錄見 [RUN_LOG.md](RUN_LOG.md)。

## 1. 目標

1. 用 MariNeXt 對本專案的場景做偽標籤。
2. 只保留「原 GT 油汙 ∪ MariNeXt 判為油」的範圍作為新訓練資料集，**標籤仍用原 GT**。
   重新訓練分割模型（Prithvi-EO-2.0-300M＋DeepLabV3，其他設定不變）。
   預期：模型在 MariNeXt 也會判成油、但 GT 不是油的地方（look-alike 候選）多看到背景樣本，減少對雜訊與 look-alike 的誤判。

## 2. MariNeXt 與本專案資料的差異

| 項目 | MariNeXt（MADOS） | 本專案 |
| --- | --- | --- |
| 感測器 | Sentinel-2 | Sentinel-2 283 景、Landsat 8 51 景、Landsat 9 21 景（共 355） |
| 波段 | 11 個：B1～B8A、B11、B12 | 8 個：B01、B02、B03、B04、B08、B8A、B11、B12（Sen2Like 調和）；模型用其中 6 個 |
| 缺少 | — | **B05、B06、B07（紅邊）**；Landsat 沒有對應波段 |
| 類別 | 15 類，含 Oil Spill、Marine Water、Turbid Water、Sediment-Laden Water、Foam、Waves & Wakes、Dense Sargassum、Natural Organic Material、Ship 等 | 油汙、手標背景、船、未標註 |
| 小圖 | 240×240 | 256×256 |

- 來源：[程式與權重（MIT）](https://github.com/gkakogeorgiou/mados)、[資料集](https://zenodo.org/records/10664073)；
  Kikaki et al., ISPRS J. Photogramm. Remote Sens., 2024。
- 待確認：MADOS 影像的處理層級與大氣校正方式、反射率尺度、波段解析度（重採樣到 10 m 的方式），
  以及程式是否提供全圖推論（repo 只寫了在 MADOS 上評估的指令）。

## 3. 步驟

| # | 步驟 | 完成的判斷 |
| --- | --- | --- |
| 0 | 取得 MariNeXt 程式與 5 組權重，在 MADOS test 上重現論文數字 | 與論文表格一致 |
| 1 | 讀論文與程式，確認輸入前處理（第 2 節的待確認項） | 寫下輸入規格 |
| 2 | 283 個 S2 場景補 B05～B07：依場景日期與 tile 下載原始產品，處理成與 MADOS 相同的輸入 | 每景 11 波段，與既有 8 波段在幾何上對齊 |
| 3 | 全圖推論，存每像素類別與 Oil Spill 機率 | 283 景完成 |
| 4 | **偽標籤品質檢查**（建資料集前） | 見第 4 節 |
| 5 | 建新資料集：油汙小圖照舊；另以「MariNeXt 判為油、GT 非油」的區塊為中心切小圖，標籤用原 GT | 名冊、數量統計 |
| 6 | 訓練：新資料集 vs A；同張數隨機背景對照；各兩個 seed（42、43） | 三折完成 |
| 7 | 評估：與目前相同的指標（第 5 節） | 配對 CI、對照判讀表 |

Landsat 72 景：MariNeXt 無法直接使用，先不做偽標籤，訓練時照舊使用原 GT 與原背景小圖。

## 4. 偽標籤品質檢查

在建資料集之前先做，結果不好就回頭檢查前處理，不直接用：

- 在有 GT 的 S2 場景上，MariNeXt Oil Spill 對 GT 油汙的 recall／precision，依場景、亮暗油、弱對比分層。
- 在 test 背景小圖（無油）上，MariNeXt 判成油的小圖比例，與 A 的 4.8% 比較。
- MariNeXt 判為油、GT 非油的區塊：數量、面積分布、離 GT 油汙的距離（油汙旁邊一圈的比例），並抽樣看圖。
- 附帶用途：MariNeXt 的其他類別（濁水、泥沙水、浪花、海藻等）可用來替本專案模型的誤報粗分 look-alike 類型
  （RESEARCH_GOAL §5 的待確認項），取代目前未經驗證的光譜規則。

## 5. 評估

沿用目前的口徑，以題目（look-alike 不誤判成油）為主：

- 主要：同 recall 下 test 背景小圖的警報率。
- 次要：Oil IoU、含背景 Oil IoU、弱對比油的 recall（HNM 的代價出現在這裡）、
  油汙邊界誤報（8 px 內＋與油汙相連）、Boundary IoU。
- 不確定性：場景配對 bootstrap 2,000 次；每組至少兩個 seed。
  10/29 seed 43 顯示同一份資料換 seed，Oil IoU 可差 0.04，單一 seed 的差異不下結論。

## 6. 待和學姊確認

1. 「只保留聯集」是指訓練小圖的取樣範圍，還是聯集以外的像素也不計分？
2. 原本每場景隨機 10 張的背景小圖要不要保留？
3. 「MariNeXt 判為油、GT 非油」的區塊作用接近由外部模型挑出的硬負樣本。
   除了 A，是否也要以 B（同池隨機）的方式做同張數對照，以區分「多了背景」與「挑選」的效果？
4. Landsat 場景的處理方式。

## 7. 判讀規則（訓練前補齊）

步驟 4 完成、資料集定案後、開始訓練前寫定，格式同 RUN_LOG 的判讀表。
