# HNM 後續實驗規劃

更新日期：2026-10-19。

目前狀態：HNM_SI 的三折探勘與 `hnm_si` 選取已完成，選出的背景圖已匯出給學姊檢視
（`hnm_runs/hnm_si_v1/hnm_si_selected_images/`）。**尚未用這些背景訓練任何分割模型，
因此不宣稱 HNM 有改善。** 本文件規劃接下來的驗證，依「先便宜、有訊號再加碼」排序。

**2026-10-19 進度：** 步驟 1 診斷已完成（結果見下方「步驟 1 結果」與 [RUN_LOG.md](RUN_LOG.md)）。
步驟 2 已完成（10/19 13:11–10/20 00:29，9 個 run），分析見 [RUN_LOG.md](RUN_LOG.md) 2026-10-20：
單一 seed 下，主指標三折平均 A 0.4332、B random 0.4469、C hnm 0.4205、D hnm_si 0.3993；
C−B −0.026、D−B −0.048（場景 bootstrap CI 皆不含 0），主因是 recall 下降。
HNM 讓 test 背景小圖誤報少了約一半，但主指標不評這部分。**依決策閘門屬「C、D 劣於 B」，
不進步驟 3、4；也不宣稱 HNM 有改善。**

相關文件：[HNM_SI_INTEGRATION.md](HNM_SI_INTEGRATION.md)（HNM_SI 實作與用法）、
[HNM.md](HNM.md)（EfficientNet OOF 第一版）、[MODEL_HNM_OVERVIEW.md](MODEL_HNM_OVERVIEW.md)。
每次實際執行的指令、輸出位置與前後差別記在 [RUN_LOG.md](RUN_LOG.md)。

## 0. 起點與已知數字

| 項目 | 數值 | 來源 |
| --- | --- | --- |
| 基準三折 pooled Oil IoU | 0.4637 / 0.4634 / 0.3726，平均 **0.4332** | `EXPERIMENT_RUN_SUMMARY_20260929.md` |
| 基準每折訓練時間 | 約 50 / 74 / 109 分鐘（RTX 4090） | 同上 |
| 每折訓練背景數（基準） | 1860 / 1860 / 1870 | `data_split/patch_level_gt_expand_bg/` |
| HNM_SI 每折選出背景數 | 1860 / 1860 / 1870（與基準相同） | `hnm_runs/hnm_si_v1/arms/hnm_si/fold_N/manifest.json` |
| 每折探勘窗數／有 FP | 272,201／12,521、265,839／8,885、267,716／9,453 | 探勘日誌 |

**折間差異（0.37 到 0.46）比預期的 HNM 效果大**，加上 early stopping 帶來的 seed 雜訊，
所有比較都必須逐折配對並附不確定性，不能只比三折平均。

## 1. 核心問題

1. 基準的偽陽性是不是 HNM 挑得到的那一類？（步驟 1）
2. 用 HNM 背景取代原背景，Oil IoU／誤報／漏報有沒有變化？（步驟 2）
3. 變化來自「更難的背景」，還是「換了一批背景」？（步驟 2 的 random 對照）
4. 光譜指數過濾是否有額外貢獻？（步驟 2 的 hnm 對 hnm_si）

## 2. 步驟 1：基準偽陽性診斷（不訓練、不佔 GPU 太久）

**目的：** 決定步驟 2 值不值得花約 12 小時 GPU。HNM 只能壓低它挑得到的誤報類型。

**資料：** 基準三折的 GT-aware 重組輸出
`result-seg/CV_358clean_gt_expand_prithvi300m_dlv3_det/<run>/post_test_*/gt_aware_recon/`，
內含 `per_scene_iou.csv` 與逐場景 pixel-exact 預測 `prediction_artifacts/*.pred.json`
（以 `main/predmask_codec.py` 解碼，`main/visualize_prediction_artifact.py` 可還原圖片）。

**做法：** 對三折 test 場景，統計「預測為油、GT 為背景（mask==1）」的像素落在哪類區域，
並分別算像素數與佔全部 FP 的比例：

| 類別 | 判定方式（先做簡單版，不引入新模型） |
| --- | --- |
| 油汙邊界附近 | 距 GT 油汙 ≤ N 像素（N 先取 8，並掃 4／16 看敏感度） |
| 陸地／海岸 | 需要陸地依據；沒有現成陸地遮罩，先以「近紅外／SWIR 亮度＋MNDWI 門檻」粗判，結果標註為粗估 |
| 雲／霧 | 高亮度且各波段平坦（可見光到 SWIR 反射率都高）的粗判 |
| 海面反光／船／尾跡 | 小面積高亮連通區，粗判；無法判定者歸「其他」 |
| 其他乾淨海面 | 以上皆非 |

另外輸出：
- 各折 FP 像素總量、占 test 背景像素比例。
- 依感測器（S2／L8／L9）與依油汙大小（tiny／中／大，沿用既有分層）拆分。
- 每類前 20 個最大 FP 區域的截圖，供人工核對粗判是否合理。

**同時量化已選背景：** 對 `hnm_si_selected_images` 的 5,590 張，估計含陸地、含雲霧的比例
（同一組粗判規則）。學姊檢視圖片時大概率會問，且這個比例也決定是否需要步驟 3 的「排除雲／陸地」版。

**判讀與決策：**

| 結果 | 決策 |
| --- | --- |
| FP 主要在雲、霧、反光、陸地等 HNM 候選類型 | 進入步驟 2 |
| FP 主要貼著油汙邊界，或大多在「其他乾淨海面」 | HNM 預期效益低；先與學姊討論標註邊界或其他改法，再決定是否仍做步驟 2 |
| 粗判無法分辨 | 以人工抽樣 100 個 FP 區域確認後再決定 |

**產出：** `reports/` 外不另放；腳本放 `tools/`（例如 `tools/analyze_baseline_fp.py`），
結果 csv 與圖放 `hnm_runs/diagnostics/`（已被 `.gitignore` 排除）。

### 步驟 1 結果（2026-10-19）

腳本 `tools/analyze_baseline_fp.py`，輸出 `hnm_runs/diagnostics/`，細節與逐折數字見 [RUN_LOG.md](RUN_LOG.md)。
重算三折 pooled Oil IoU 與基準完全一致（0.4637／0.4634／0.3726）。

- 主指標（GT-aware 重組）的 FP：99% 落在未標註（255）像素，只有 1% 在 mask==1（HNM 的探勘定義）；
  87.6% 是邊界類（距油汙 ≤ 8 px 55.6%，連著油汙延伸 32.0%），HNM 類（陸地／雲／霧反光／小亮點）只占 5.4%，
  其他海面 7.0%。
- 上限：把重組中所有非邊界 FP 全部消掉，三折平均 Oil IoU 只從 0.4332 到 0.4450（+0.012）；
  只消 mask==1 的 FP 則到 0.4341（+0.0009）。fold 2、3 漏報多於誤報。
- HNM 類誤報出現在主指標不評的 test 背景小圖：每折 3.6–6.6% 的小圖有 FP，71% 為暗色水域等其他海面，
  29% 為霧反光／雲／海岸。
- 選出的背景含陸地／雲的比例低於基準（陸地 7% 對 18%，雲 11% 對 23%）；SR 指數飽和只有 1–4%。

**依上表判讀：** 屬於「FP 主要貼著油汙邊界」，HNM 對現行主指標的預期效益低。
若仍要做步驟 2，建議同時報告 test 背景小圖的 FP 率（HNM 唯一能明顯發揮的地方），
否則主指標幾乎看不到差異；另外可能更值得投入的是標註邊界與漏報（recall）。

## 3. 步驟 2：四組同條件對照（主實驗）

**設計：** 固定下列條件，只換 train 背景來源。

- 固定：GT-centric 三折名冊、val／test 背景（逐筆不變）、每折背景數量、seed 42、
  優化器設定（`lr0=5e-5`、Tversky α=0.3 β=0.7、batch 16、AMP、patience 25）、評估口徑
  （GT-aware 場景重組 pooled Oil IoU）。
- 變動：train 背景名冊。

| 組別 | 名冊 | 取得方式 | 狀態 |
| --- | --- | --- | --- |
| A. baseline | 凍結原名冊 | 既有 | 已有結果，0.4332 |
| B. random | 同一合格窗池隨機抽 | `select --arm random` | 待產生 |
| C. hnm | 探勘模型 FP 窗（無 SI 過濾） | `select --arm hnm` | 待產生 |
| D. hnm_si | FP＋SI 過濾 | 已完成 | 名冊已有 |

**關鍵：比較 C 對 B、D 對 C。** 不要只拿 D 去比 A。HNM 背景的取樣方式與基準本來就不同
（滑窗 stride 192、需 mask==1 ≥ 50%、未標註 ≤ 30%），沒有 B 就分不清進步來自 HNM 還是背景池改變。

**產生 B、C 的名冊（不需重新探勘）：**

```bash
CFG=preprocess/hnm_si/config.example.yaml
for arm in random hnm; do
  for f in 1 2 3; do
    .venv/bin/python -m preprocess.hnm_si.pipeline --config $CFG --stage select --arm $arm --fold $f
  done
done
```

**訓練設定：** 複製 `main/hnm.example.yaml` 為 `main/local_hnm_<arm>.yaml`，
`hnm.artifacts_dir` 分別指向 `hnm_runs/hnm_si_v1/arms/{random,hnm,hnm_si}`，
`experiment_name` 各自不同，`results_base_dir` 與 `excel_log_path` 分開。
基準名冊與 `local_gt_expand.yaml` 不動。三個 yaml 除了上述欄位外必須逐行相同。

**規模：** B、C、D 各三折，共 9 個 run，約 12 小時 GPU（沿用基準每折 50 到 110 分鐘）。
每個 run 記錄實際 epochs 與 `global_optimizer_step`。

> 備註：各 arm 的 train 樣本數相同，但 early stopping 會讓實際步數不同。
> 先沿用基準流程（結果可與 A 直接對照），並在報告中列出各 run 的實際步數；
> 若差異明顯，再補一組固定步數的對照。

### 評估指標

主要：逐折與三折平均的 **pooled Oil IoU**（同基準口徑）。必須同時看：

- precision、recall（避免只降低誤報卻漏掉更多真油）。
- test 背景區的 FP 像素總量與每百萬背景像素的 FP 數。
- 依油汙大小（tiny／中／大）分層的 IoU 與 recall。
- 依感測器（S2／L8／L9）分層的 IoU 與 FP。
- 逐場景 IoU，用於 bootstrap。

### 統計判讀

- 逐折配對差：`C − B`、`D − C`、`D − B`、`各組 − A`。
- 對 test 場景做 bootstrap（以場景為單位重抽，≥ 1000 次）得 95% 信賴區間。
- 單 seed 結果只視為訊號，不下結論。若步驟 2 出現明確方向，再補 seed 43、44（見步驟 4）。
- 不看 test 調整任何 HNM 參數；選參只用各折 train／val。

### 決策閘門

| 結果 | 下一步 |
| --- | --- |
| C、D 在多數折優於 B，且 FP 下降而 recall 未明顯下降 | 補 seed（步驟 4），再做步驟 3 消融 |
| C、D 與 B 無明顯差異 | 結論為「此資料集上 HNM 無明確效益」；回到步驟 1 的診斷找原因，不急著調參 |
| C、D 劣於 B | 檢查選到的背景是否含漏標油汙或雲／陸地過多；回頭檢視圖片與 FP 定義 |
| B 優於 A | 代表背景池本身有益，應單獨記錄，與 HNM 效果分開討論 |

## 4. 步驟 3：消融（僅在步驟 2 有正向訊號時做）

每項只改一個因素，其餘同步驟 2 的 D 組。

| 編號 | 變因 | 設定 | 要回答的問題 |
| --- | --- | --- | --- |
| E1 | SI 過濾強度 | `si.top_q` ＝ 0、0.05、0.2 | 過濾要多強才有用？ |
| E2 | 排序式選取 | 依 `n_fp` 或平均機率取最難，取代場景均衡隨機抽 | 「更難」是否比「均衡」重要？需新增選取模式 |
| E3 | 背景量 | 1×、1.5×、2× 基準背景數 | 進步來自更難還是更多？需用 `selection.target_count` 指定整數並確認候選足夠 |
| E4 | 雲／陸地排除 | 排除步驟 1 判為雲、陸地的候選 | 這些圖是有益還是雜訊？需先有可靠判定 |
| E5 | 混合 | 一半 hnm_si、一半 random | 保留多樣性是否更穩？ |

E2、E4、E5 目前程式尚未支援，要做時各需在 `preprocess/hnm_si/pipeline.py` 增加選取模式並補測試。

## 5. 步驟 4：補 seed 與最終比較

- 條件：步驟 2 有正向訊號。
- 做法：對 B、C、D（必要時加 A 重跑）補 seed 43、44，共 +18 個 run，約 24 小時。
- 最終全貌約 36 個 run、48 小時左右。
- 報告：三個 seed 的逐折平均與標準差、配對差的信賴區間；seed 之間不平均到看不出變異。

## 6. 另一條 HNM：EfficientNet OOF（選做）

`preprocess/hnm/` 的 OOF 版已實作但尚未正式探勘（需每折 5 個 OOF 分類器加 1 個 embedding 模型，三折共 18 個模型）。
若要把兩條方法一起比較，加為第 5 組「E. oof」；它與 HNM_SI 共用同一個匯出腳本
（`python -m preprocess.export_hnm_selection`）。建議放在步驟 2 有結果之後再決定，避免同時引入太多變因。

## 7. 風險與限制（寫報告時要誠實交代）

- **within-fold 探勘：** 探勘模型是各折基準 `best.pt`，該權重曾用該折 val 選模，
  且探勘掃的是同折 train 場景。這不是場景層級 OOF，不能宣稱探勘模型沒見過這些場景。
  嚴格 OOF 需另在 outer train 內按場景／事件分組訓練 probe，成本高，先不做。
- **背景標籤品質：** 被選中的小圖整張當背景訓練，但其中可能有未標註或 Others 區域、雲、陸地。
  遮罩 `mask==1` 才是確認背景，其餘不保證是海面。
- **SI 尺度：** 影像有 +1000 DN offset，光譜指數已扣除；扣除後海面反射率很小，
  `SR_B1B2_B3B8A` 常被 ±10 的 clip 截斷。若步驟 1 顯示該指數飽和比例高，要重新評估它的用處。
- **探勘參數是原法起點：** `top_q=0.05`、`mask1≥0.5` 等都不是為本資料集驗證過的最佳值。
- **候選池與基準背景不同：** 這正是需要 random 對照的原因。

## 8. 時程與產物

| 步驟 | 工作量（粗估） | 產物 |
| --- | --- | --- |
| 1 診斷 | 半天到一天（含人工核對） | 診斷腳本、各類 FP 比例、已選背景含雲／陸地比例 |
| 2 主實驗 | 產生名冊數分鐘；訓練約 12 小時 GPU；分析約半天 | 9 個 run 的結果表、配對差與信賴區間 |
| 3 消融 | 視 E1 到 E5 選幾項而定，每項約 4 到 6 小時 GPU | 消融表 |
| 4 補 seed | 約 24 小時 GPU | 最終比較表 |

所有程式放進 repo（`tools/`、`preprocess/hnm_si/`），資料與大型輸出留在被 `.gitignore`
排除的 `hnm_runs/`、`result-seg/`；每個階段完成後 commit 並在本文件更新狀態。

## 9. 待確認

- 步驟 2 是否沿用基準的 early stopping，還是另設固定步數？
- 要不要把 EfficientNet OOF 版納入第一輪比較？
- 步驟 1 的陸地／雲判定，是否接受先用光譜粗判，或你手上有可用的陸地遮罩？
- 步驟 2 顯示 HNM 降低背景誤報但犧牲 recall：要以單一 seed 結論收尾，還是補 seed／固定步數確認？
- 是否把 test 背景小圖 FP 率正式列為次要指標（實際應用會掃整景，這部分的誤報才會出現）？
- 依閘門，是否抽查 hnm／hnm_si 選中背景中 `n_fp` 最高者，確認有無漏標油汙？
