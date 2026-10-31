# 研究目標與決策紀錄

更新日期：2026-10-31。只記「為什麼做、怎麼判斷成功、已定案的決策、目前結論」；
怎麼跑見 [README.md](README.md)，每次執行的細節見 [RUN_LOG.md](RUN_LOG.md)。
新的決策或結論出現時更新本檔，舊的不刪，改標日期。

## 1. 題目

**讓油汙分割模型不把 look-alike 誤判成油。**（老師給的題目；2026-10-28 確認）

- 模型：Prithvi-EO-2.0-300M＋DeepLabV3，Sentinel-2／Landsat（Sen2Like 調和）八波段影像，
  逐像素判斷油汙／背景；凍結三折、GT-centric 小圖訓練。細節見 [PIPELINE_DETAIL.md](PIPELINE_DETAIL.md)。
- look-alike：外觀像油、實際不是油的現象，例如雲與雲影、霧與耀光、陸地與海岸、暗色海面、
  細長暗線、濁水。資料**沒有** look-alike 的標註，只有油汙（0）、手標背景（1）、船（2）與未標註（255）。
- 主要手段：HNM（困難負樣本挖掘）——挑出容易被誤判成油的背景放進訓練，讓模型學會分辨。
  HNM 只改訓練用的背景，分割模型不變。

## 2. 核心問題與評估方式

**核心問題：在漏掉一樣多油的條件下，模型在無油背景上的誤報有沒有變少？**

- 判斷油的門檻（P(oil) ≥ 0.5）一調高，誤報和抓到的油會一起變少；所以只看預設門檻的誤報
  分不出「學會分辨 look-alike」和「只是變保守」。主要評估是**同 recall 下比背景誤報**
  （門檻掃描：`tools/threshold_sweep.py`、`tools/compare_threshold_sweep.py`）。
- 背景誤報在 test 背景小圖上量（每折 1,170 張、不含油、以手標背景為主）：
  每百萬像素誤報、有任何油像素的小圖比例（look-alike 警報率）。
- **原主指標**（GT-aware 重組 pooled Oil IoU，只評油汙周邊小圖）保留為次要指標，
  確認找油能力沒有變差。它量不到題目：主指標的誤報 88% 在油汙邊界，look-alike 類約 5%。
- 輔助指標：含背景 Oil IoU、Boundary IoU（d＝8 px）。
- 不確定性：多數組單一 seed（C、G、G0 另有 seed 43）；逐折配對、場景 bootstrap 2,000 次。

## 3. 已定案的決策

| 日期 | 決策 | 來源 |
| --- | --- | --- |
| 10/19 | HNM 的對照組是同池隨機背景 B，不是原基準 A（兩池幾乎不重疊） | [HNM_EXPERIMENT_PLAN.md](HNM_EXPERIMENT_PLAN.md) |
| 10/22 | 評估要納入背景誤報（學姊） | 10/22 報告後討論 |
| 10/28 | 標註是 NOAA 專家標註，以標註為準，不考慮標錯；油旁未標註就是非油 | 學姊／組內討論 |
| 10/28 | `oil_ignore_ring`（訓練時忽略油外圈）不採用：油旁多畫，precision 與 Boundary IoU 顯著變差 | RUN_LOG 2026-10-27 |
| 10/28 | 題目是 look-alike 誤判；油汙邊界改善降為副線 | 使用者確認 |
| 10/31 | HNM 暫停；改用 MariNeXt 對資料集做偽標籤，只保留原 GT ∪ MariNeXt 判為油的範圍重建訓練資料集（標籤仍用原 GT） | 學姊 10/29 報告後回饋；[PSEUDO_LABEL_PLAN.md](PSEUDO_LABEL_PLAN.md) |
| 10/31 | 油汙外圍誤報另以「油汙邊界誤報（8 px 內＋與油汙相連）」呈現；Boundary IoU 只看內側邊界帶，不單獨用來看這個問題 | 學姊 10/29 報告後回饋 |
| 10/28 | 簡報與報告的組名改為顯示判斷者：B 隨機（同池）、C HNM-A、D HNM-A+SI、E HNM-EffNet、F HNM-EffNet+SI；字母與 `hnm_runs/` 目錄名不變 | 使用者確認（避免老師以為只有 C、D 是 HNM） |

## 4. 目前結論（除註明外為單一 seed）

- 原主指標：C～F 四種 HNM 都輸給隨機背景 B（−0.026～−0.053，CI 不含 0），主因 recall 下降。
- 背景誤報：HNM_SI（D）讓 test 背景小圖誤報約減半、有誤報的小圖 4.7% → 1.5%。
- 含背景 Oil IoU：只有 C−B 變得不顯著；背景誤報只占含背景聯集 2～5%，排序不變。
- **門檻掃描（10/28）：HNM 不是單純變保守。** 模型輸出幾乎是二值，降門檻只多 0.03～0.13 recall。
  在各組曲線重疊的範圍內，C hnm 同 recall 下的背景警報率比 A、B 低（三折平均 −0.023、−0.027，
  CI 不含 0 但屬條件式）；D 也較低，但 fold 2、3 有一部分油任何門檻都找不回來。B 與 A 無差異。
- **被推掉的油（10/28）：** HNM 自己的代價（C−B）集中在**弱對比、細（6–15 px）、乾淨暗水面上的淡油**，
  亮油、暗油、大片厚油沒有差異（D 另外還丟了亮油與大片油）。被整塊漏掉的區塊 C 給的最高 P(oil) 中位數 0.03，
  且以場景為單位整片消失；C、D 漏掉的高度重疊，是系統性效果。同地點硬負樣本數量解釋不了（Spearman 0.03），
  但 HNM 選進的背景小圖（NOAA 標為背景的細暗線、淡色帶、條紋）外觀與漏掉的淡油非常像。
- **特徵空間（10/28）：** 在基準 A 的 encoder 特徵中，被 HNM 漏掉的油最近的參考鄰居有 30～50% 是硬負樣本（保住的油約 0），
  AUC 0.78（C）／0.83（D），比對比、大小等屬性更能預測誰會被漏掉；這些油離訓練油遠、離背景近，是「非典型」的油。
- **SI 過濾的額外效果（10/29 整理，D−C、F−E）：** 目前設定（三個光譜指數排除最像油的 5%）沒有額外好處。
  test 背景小圖誤報 D−C −566／M、F−E −432／M，CI 皆含 0；同 recall 警報率 D 在 fold 2、3 達不到目標，F−E 在 fold 2、3 反而較高。
  代價是 recall：D−C −0.033、F−E −0.045（CI 不含 0），D−C Oil IoU −0.021。先前「D 讓背景誤報減半」是相對 B，主要來自 HNM 本身。
  只測過一種 SI 設定、單一 seed，不代表光譜指數本身沒有資訊。
- **kNN 過濾硬負樣本（10/29，G／G0）：** 把 C 的硬負樣本中，A 的 encoder 特徵離訓練油太近的 22～33% 換成同場景其他候選（G）。
  - **look-alike 分辨力：** 同 recall 背景警報率與 C 相同（2.54% vs 2.52%），仍低於 A、B。
  - **C 整塊漏掉的淡油：** G 抓回約一半（recall 0.03 → 0.50）。隨機替換同樣張數的 G0 只到 0.08（G−G0 +0.35，CI 不含 0）。
  - **主指標：** Oil IoU 與 C 相同，仍低於 B；弱對比油 G−C 的 CI 含 0。
  - **對照組：** G0 本身比 C 全面變差（Oil IoU −0.047），換一批硬負樣本造成的變動很大，要補 seed 才能確認。
  - **seed 43（10/29 補跑）：** 「G 抓回一半」「G−G0」「G0 比 C 差」都沒有重現。C、G0 換 seed 後 Oil IoU 各差約 0.04、方向相反
    （C 0.421 → 0.384、G0 0.374 → 0.410；G 0.422 → 0.415），G0−C 的正負號翻過來。
    C 整塊漏掉的區塊兩個 seed 只重疊 30 塊；在 seed 42 定義的 73 塊上，seed 43 的 C／G／G0 recall 都在 0.29～0.31。
    kNN 替換的效果兩個 seed 下**沒有確認**；組間 0.04 左右的 Oil IoU 差異在訓練變動範圍內。
  - **兩個 seed 一致的：** HNM 各組 test 背景小圖誤報都比 B 少（0.5 門檻有誤報的小圖 1.4～2.2% vs 4.7%）；
    代價集中在弱對比、細的淡油（C−B 兩次都為負）；G 的 Oil IoU 低於 B、同 recall 背景警報率低於 B（CI 皆不含 0）。
    A、B 只有 seed 42。
- **目前的主要問題：** HNM 挖到的 look-alike 與淡油在模型特徵中重疊，HNM 把決策邊界推過重疊區，真油一起被推到背景。
  依特徵排除重疊區的硬負樣本（G）在 seed 42 救回一部分淡油，但 seed 43 沒有重現；主指標兩個 seed 都低於 B。
  補 seed 後看到訓練變動（同一份背景名冊換 seed，Oil IoU 差 0.04）與目前各組的差異同量級，
  單一 seed 的組間比較（含 C～F 的排序）都要保留。
- **下一步（10/31 更新）：** HNM 暫停（B seed 43、混合背景、hard positive 都先不做），改做 MariNeXt 偽標籤，
  步驟見 [PSEUDO_LABEL_PLAN.md](PSEUDO_LABEL_PLAN.md)。先處理波段差異（MariNeXt 用 S2 11 波段，本專案缺 B05～B07，Landsat 無法直接用），
  建資料集前先量偽標籤在本專案資料上的品質；新組別每組至少兩個 seed。

## 5. 待確認

- look-alike 測試集：依類型（陸地、雲、霧／耀光、暗色海面…）分開算誤報率。目前類別是
  `tools/analyze_baseline_fp.py` 的光譜粗估規則，未經驗證；候選做法是公開海岸線資料、
  標準雲偵測、MariNeXt 的其他類別（濁水、泥沙水、浪花、海藻等），加上抽樣請學長姊看圖確認。
- 偽標籤資料集的細節（PSEUDO_LABEL_PLAN §6）：「只保留聯集」的範圍、原隨機背景小圖是否保留、是否加同張數隨機對照、Landsat 怎麼處理。
- 若 HNM 只是變保守：改用其他方法（例如先找疑似油區塊、再分類油／look-alike 的兩階段做法）。

## 6. 文件地圖

| 文件 | 內容 |
| --- | --- |
| [README.md](README.md) | 環境、資料、訓練與評估怎麼跑 |
| [PIPELINE_DETAIL.md](PIPELINE_DETAIL.md) | 前處理 → 模型 → 推論的完整技術說明 |
| [DATA_MANIFEST.md](DATA_MANIFEST.md)、[LOCAL_REPRODUCTION.md](LOCAL_REPRODUCTION.md) | 資料交付格式、本機重跑步驟 |
| [EXPERIMENT_RUN_SUMMARY_20260929.md](EXPERIMENT_RUN_SUMMARY_20260929.md) | 基準 A 三折的重跑紀錄 |
| [HNM_EXPERIMENT_PLAN.md](HNM_EXPERIMENT_PLAN.md) | HNM 步驟 1～3 的實驗設計與決策閘門 |
| [PSEUDO_LABEL_PLAN.md](PSEUDO_LABEL_PLAN.md) | MariNeXt 偽標籤與重建訓練資料集的步驟（10/31 起） |
| [HNM_SI_INTEGRATION.md](HNM_SI_INTEGRATION.md) | 學長 HNM_SI 的適配與 `preprocess/hnm_si/` 用法（C、D 組） |
| [HNM.md](HNM.md) | `preprocess/hnm/` EfficientNet OOF 第一版的用法（E、F 組改用 `preprocess/hnm_si/oof.py`） |
| [RUN_LOG.md](RUN_LOG.md) | 每次執行的指令、輸出、結果（新的在上） |
| [tools/README.md](tools/README.md) | 分析與比較工具一覽 |
