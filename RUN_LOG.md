# 執行紀錄

每次實際跑了什麼、用哪個指令、輸出在哪、跟前一次差在哪。新的放最上面。
規劃見 [HNM_EXPERIMENT_PLAN.md](HNM_EXPERIMENT_PLAN.md)；結果輸出大多在被 `.gitignore`
排除的 `hnm_runs/`、`result-seg/`，這份文件是唯一進版控的流水帳。

## 一覽

| 日期時間 | 做什麼 | 範圍 | 用 GPU | 輸出 | 狀態 |
| --- | --- | --- | --- | --- | --- |
| 10/20 00:35–01:00 | 步驟 2 分析：三組套用步驟 1 診斷＋配對比較 | 3 組 × 三折 | 小量 | `hnm_runs/diagnostics_step2/` | 完成 |
| 10/19 13:11–10/20 00:29 | 步驟 2 訓練：random → hnm → hnm_si，各三折 | 9 個 run | 是（11.3 小時） | `result-seg/CV_gt_expand_hnm_si_v1_<arm>/` | 完成 |
| 10/19 00:29–01:11 | 步驟 1 正式診斷（基準 FP 落點） | 三折 | 小量（背景小圖推論） | `hnm_runs/diagnostics/` | 完成 |
| 10/19 00:30 | 產生步驟 2 訓練 yaml 並檢查 | 3 個 arm | 否 | `main/local_hnm_{random,hnm,hnm_si}.yaml` | 完成 |
| 10/19 00:29 | 產生 random、hnm 背景名冊 | 三折 | 否 | `hnm_runs/hnm_si_v1/arms/{random,hnm}/` | 完成 |
| 10/19 00:1x | 步驟 1 煙霧測試（舊分類規則） | 只有 fold 1 | 小量 | 暫存目錄，**未保留、不採用** | 完成 |
| 10/17 16:38–19:12 | HNM_SI 三折探勘＋hnm_si 選取＋匯出圖片 | 三折 | 是（約 2.4 小時） | `hnm_runs/hnm_si_v1/` | 完成 |
| 09/29 17:08–21:13 | 基準三折訓練＋GT-aware 重組 | 三折 | 是（約 4.1 小時） | `result-seg/CV_358clean_gt_expand_prithvi300m_dlv3_det/` | 完成 |

步驟 2 已完成（見 2026-10-20）：單一 seed 下，HNM 兩組在主指標上**劣於**隨機背景對照，但 test 背景小圖的誤報明顯減少。

---

## 2026-10-20

### (1) 步驟 2 分析：四組配對比較 —— 完成

```bash
for arm in random hnm hnm_si; do   # 每組各折 run 目錄用 --run-dir 1=... 2=... 3=... 指定
  .venv/bin/python tools/analyze_baseline_fp.py --config preprocess/hnm_si/config.example.yaml \
    --skip-backgrounds --run-dir 1=<fold1 run> --run-dir 2=<fold2 run> --run-dir 3=<fold3 run> \
    --out hnm_runs/diagnostics_step2/$arm
done
.venv/bin/python tools/compare_hnm_arms.py --arm A=hnm_runs/diagnostics \
  --arm B=hnm_runs/diagnostics_step2/random --arm C=hnm_runs/diagnostics_step2/hnm \
  --arm D=hnm_runs/diagnostics_step2/hnm_si
```

- 四組都用步驟 1 同一支腳本評分（重組像素＋test 背景小圖推論），保證口徑一致；
  重算的各 run pooled Oil IoU 與訓練時 `run_metrics.json` 相同。
- 信賴區間：test 場景配對 bootstrap 2,000 次（兩組抽同一批場景）。**只反映 test 場景抽樣，
  不含訓練 seed 的變異**，單一 seed 下實際不確定性更大。
- 輸出：`hnm_runs/diagnostics_step2/comparison.md`（完整表）、`comparison_by_fold.csv`、
  `paired_differences.csv`、`strata_{sensor,size_bin}.csv`，各組明細在 `<arm>/`。

**各組結果（三折）：**

| 組 | 實際 epochs | Oil IoU（1／2／3） | 平均 | precision | recall | test 背景小圖 FP（每百萬像素） | 有 FP 的背景小圖 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| A 基準 | 37／66／100 | 0.4637／0.4634／0.3726 | 0.4332 | 0.653 | 0.566 | 2,785 | 4.8% |
| B random | 105／55／78 | 0.5189／0.4521／0.3698 | 0.4469 | 0.712 | 0.545 | 2,697 | 4.7% |
| C hnm | 42／50／100 | 0.4789／0.4253／0.3572 | 0.4205 | 0.672 | 0.534 | 1,747 | 1.8% |
| D hnm_si | 42／57／81 | 0.4866／0.4098／0.3014 | 0.3993 | 0.668 | 0.501 | 1,181 | 1.5% |

**關鍵配對差（三折平均，[95% CI]）：**

| 比較 | Oil IoU | recall | test 背景小圖 FP（每百萬像素） |
| --- | --- | --- | --- |
| C − B（HNM 對隨機） | −0.026 [−0.048, −0.006] | −0.011 [−0.038, +0.015] | −950 [−2,034, +225] |
| D − C（SI 過濾的額外效果） | −0.021 [−0.039, −0.002] | −0.033 [−0.059, −0.007] | −566 [−1,532, +412] |
| D − B | −0.048 [−0.072, −0.025] | −0.044 [−0.075, −0.016] | −1,516 [−2,388, −711] |
| B − A（換背景池本身） | +0.014 [−0.007, +0.035] | −0.022 [−0.048, +0.003] | −89 [−1,035, +750] |

**判讀：**
- **主指標上 HNM 劣於隨機對照，SI 過濾再更差。** C、D 在三折中都低於 B（D−B 三折皆為負，fold 2、3 的 CI 不含 0）。
  依計畫決策閘門，屬「C、D 劣於 B」：不進步驟 3 消融與步驟 4 補 seed。
- **退步主要來自 recall 下降。** D 的 recall 比 B 低 0.044，fold 3 低 0.086；precision 也沒有比 B 好。
  推測：更像油的困難背景讓模型在油汙周邊變得保守，漏掉更多真油。
- **HNM 確實壓低了它針對的誤報。** test 背景小圖的 FP：D 比 B 少約 56%（2,697 → 1,181 每百萬像素），
  有 FP 的小圖從 4.7% 降到 1.5%。但主指標不評這些小圖，所以這個好處在 Oil IoU 上看不到，
  只看到 recall 的代價。
- **B 對 A 沒有確定差異**（+0.014，CI 含 0）。fold 1 的 +0.055 伴隨 epoch 從 37 增到 105，
  更可能是 early stopping／訓練長度造成，不宜解讀為背景池較好。
- **限制：** 單一 seed；各 run 實際 epoch 差異大（fold 1：37／105／42／42），early stopping 依 val 決定。
  若要確認「HNM 劣於隨機」，需補 seed 或固定步數對照；若接受目前訊號，下一步依閘門回頭檢查選到的背景
  是否含漏標油汙。

## 2026-10-19

### (5) 步驟 2 訓練：三組背景對照 —— 完成（分析見 2026-10-20）

```bash
setsid nohup tools/run_hnm_step2.sh > hnm_runs/step2_training.log 2>&1 &
```

- **13:11:42 開始**，依序跑 random（B）→ hnm（C）→ hnm_si（D），每組三折，共 9 個 run，預估約 12 小時。
  基準（A）沿用 09/29 結果，不重跑。
- 步驟 1 顯示 HNM 對主指標的上限很小（見 (4)），仍決定先跑訓練取得實際數字。
- 設定為 (3) 產生的 `main/local_hnm_<arm>.yaml`：與基準只差 train 背景名冊、結果路徑與名稱；
  seed 42、優化器、early stopping（patience 25）、評估口徑皆相同。
- 開跑檢查：fold 1 log 顯示 `[HNM] Fold 1: 1860 training backgrounds; validation/test backgrounds unchanged`；
  每 epoch 204 步、第一個 epoch 5 分 22 秒，與基準一致（基準 204 步、5 分 24 秒）。
- 輸出：`result-seg/CV_gt_expand_hnm_si_v1_{random,hnm,hnm_si}/<時間戳>_local_hnm_<arm>_run_foldN/`，
  彙整表 `result_excel/result_CV_gt_expand_hnm_si_v1_<arm>_log.xlsx`，總 log `hnm_runs/step2_training.log`。
- 若中斷：`main_runner.py` 每次啟動會建新時間戳目錄，不能把新舊 fold 混成同一次比較；
  可只重跑未完成的 arm（`tools/run_hnm_step2.sh hnm_si`）。
- 跑完後要做：逐折配對比較 C−B、D−C、D−B、各組−A，場景 bootstrap 信賴區間；
  另以 `tools/analyze_baseline_fp.py` 的方式比較各組 test 背景小圖 FP 率（需擴充腳本以讀取新 run）。


### (4) 步驟 1 正式診斷：基準偽陽性落在哪裡 —— 完成

```bash
nohup .venv/bin/python tools/analyze_baseline_fp.py \
  --config preprocess/hnm_si/config.example.yaml --quiet \
  > hnm_runs/diagnostics/run.log 2>&1 &
```

- **目的：** 計畫步驟 1。判斷基準的誤報是不是 HNM 挑得到的類型，決定要不要花 12 小時跑步驟 2。
- **不訓練。** 只用 09/29 基準各折的 `best.pt` 與既有重組輸出。
- **內容（每折）：**
  1. 讀 GT-aware 重組預測（`*.pred.json`），也就是 pooled Oil IoU 實際評分的像素，把每個 FP 像素分類，
     並依 GT 標籤（1 確認背景／2 Others／255 未標註）拆開。主指標把 2、255 都當背景，所以都算 FP。
  2. 用 `best.pt` 推論該折 test 背景小圖（重組時的同款 TTA、FP32）。重組只評油汙周邊小圖，
     不評這些背景小圖，但它們才是 HNM 要壓的誤報類型。
  3. 估計四組訓練背景（基準、random、hnm、hnm_si）含陸地／雲／霧反光的比例。
  4. 統計 SR_B1B2_B3B8A 指數卡在 ±10 上限的比例（計畫第 7 節的風險）。
- **FP 分類（依序，先符合者優先）：** nodata → near_oil（距 GT 油汙 ≤ 8 px）→ oil_attached
  （更遠，但所在 FP 連通區碰到油汙）→ land_coast → cloud → haze_glint → bright_small → other_sea。
  陸地、雲、霧都是光譜粗判，沒有現成遮罩；每類輸出前 20 大區域截圖供人工核對。
- **輸出：** `hnm_runs/diagnostics/summary.md`（彙整表）、`scene_totals.csv`、`fp_pixels.csv`、
  `fp_regions.csv`、`near_oil_sweep.csv`、`background_composition.csv`、`crops/<來源>/<類別>/`。
- **正確性：** 由預測重算的三折 pooled Oil IoU = 0.4637／0.4634／0.3726，與 09/29 完全一致。
- **結果一：主指標的 FP 幾乎都是油汙邊界問題，不是 HNM 挑得到的類型。**

  | 折 | 重組 FP 像素 | 在 255 未標註上 | 在 mask==1 上 | 邊界類（≤8 px＋連著油汙） | HNM 類（陸地／雲／霧反光／小亮點） | 其他海面 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | 1 | 1,116,626 | 99.8% | 0.2% | 95.8% | 2.7% | 1.5% |
  | 2 | 1,010,483 | 99.1% | 0.9% | 82.6% | 11.3% | 6.1% |
  | 3 | 1,120,570 | 98.3% | 1.7% | 83.9% | 2.9% | 13.2% |
  | 合計 | 3,247,679 | 99.0% | 1.0% | 87.6% | 5.4% | 7.0% |

  距油汙 ≤4／8／16 px 的比例：fold 1 為 43／65／75%，fold 2 為 40／55／65%，fold 3 為 35／47／55%。
  連著油汙的大塊 FP 常沿小圖邊界呈方塊狀，代表模型在油汙周邊整片判成油；部分則像真的油膜延伸到未標註區。
- **結果二：HNM 對主指標的上限很小。** 假設把重組中所有非邊界類 FP 全部消掉，
  三折平均 Oil IoU 也只從 0.4332 升到 **0.4450（+0.012）**；只消掉 mask==1 上的 FP（HNM 的探勘定義）
  則只到 0.4341（+0.0009）。這個上限比折間差異（0.37 到 0.46）小得多。
  另外 fold 2、3 的漏報比誤報多：誤報占錯誤像素 53%／38%／33%，recall 0.65／0.58／0.47。
- **結果三：HNM 類誤報確實存在，但在主指標不評的 test 背景小圖上。** 每折 1,170 張 test 背景小圖中
  有 49／42／77 張（4.2／3.6／6.6%）有 FP，每百萬背景像素 2,238／3,456／2,661 個 FP。
  其中 71% 是其他海面（多為暗色水域類似物，也混有粗判抓不到的暗冬季場景），29% 是霧反光／雲／海岸。
- **結果四：HNM 選出的背景反而比基準少陸地與雲。** 含陸地（≥1%）的小圖：基準 17.9%、random 17.2%、
  hnm 7.0%、hnm_si 6.9%；含雲（≥5%）：基準 22.8%、random 23.2%、hnm 10.7%、hnm_si 10.8%。
  霧反光占半張以上的比例四組都在 46–50%，霧反光規則可能偏寬，這個數字僅供參考。
  計畫步驟 3 的 E4（排除雲／陸地）從這點看不急。
- **結果五：SR_B1B2_B3B8A 卡在 ±10 上限的比例只有 1–4%**（探勘 FP 區域、油汙參考、hnm_si 選中者皆是），
  計畫第 7 節擔心的飽和問題不成立。
- **粗判的限制：** 陸地、雲、霧都是光譜規則，不是遮罩；截圖抽查大致合理，但 other_sea 混有幾個很暗的
  冬季 19TDE 場景。正式引用前請看 `crops/` 人工核對。
- **與 (1) 煙霧測試的差別：** 三折（不只 fold 1）；用修正後的分類規則（見 (1)）；
  多了訓練背景組成；輸出到正式位置。

### (3) 產生步驟 2 訓練設定 —— 完成

```bash
.venv/bin/python tools/make_hnm_arm_configs.py
```

- 由 `main/local_gt_expand.yaml` 產生 `main/local_hnm_{random,hnm,hnm_si}.yaml`（被 `.gitignore`
  排除，可隨時用這個指令重產）。
- 與基準只差：`results_base_dir`、`excel_log_path`、`experiment_name`、post-test 名稱，
  加上指向 `hnm_runs/hnm_si_v1/arms/<arm>` 的 `hnm` 區塊。腳本會比對差異，多改一行就拒絕寫出。
- 已用訓練端檢查 `apply_hnm_to_fold` 試跑 3 arm × 3 折：全部通過，train 背景數
  1860／1860／1870，val／test 背景不變。
- 結果會寫到 `result-seg/CV_gt_expand_hnm_si_v1_<arm>/`。啟動指令（**尚未執行**）：
  `nohup tools/run_hnm_step2.sh > hnm_runs/step2_training.log 2>&1 &`

### (2) 產生 random、hnm 背景名冊 —— 完成

```bash
CFG=preprocess/hnm_si/config.example.yaml
for arm in random hnm; do for f in 1 2 3; do
  .venv/bin/python -m preprocess.hnm_si.pipeline --config $CFG --stage select --arm $arm --fold $f
done; done
```

- 沿用 10/17 的探勘結果，**沒有重新探勘**、不用 GPU，數秒完成。
- 三組名冊都來自同一個合格窗池，數量都等於基準（1860／1860／1870）：
  - `random`：合格窗中隨機抽（對照組 B）。
  - `hnm`：只取探勘模型有 FP 的窗，不做 SI 過濾（C）。
  - `hnm_si`：10/17 已產生，FP＋SI 過濾（D）。
- 輸出：`hnm_runs/hnm_si_v1/arms/{random,hnm}/fold_N/`。

### (1) 步驟 1 煙霧測試 —— 完成，結果不採用

```bash
.venv/bin/python tools/analyze_baseline_fp.py --config preprocess/hnm_si/config.example.yaml \
  --fold 1 --skip-backgrounds --out <暫存目錄>
```

- 只跑 fold 1，約 5 分鐘。目的是確認程式正確並校正分類規則。
- **正確性：** 由預測重算的 fold 1 pooled Oil IoU = 0.4637，與 09/29 基準紀錄一致。
- **看截圖後發現舊規則有兩個問題，因此 (4) 改了規則重跑：**
  - 雲的門檻太嚴（要求 NIR ≥ 0.08），霧、反光幾乎都被歸成 other_sea。
    改為：厚雲（藍 ≥ 0.15、NIR ≥ 0.12、SWIR ≥ 0.08）與霧／反光（NIR ≥ 0.03、SWIR ≥ 0.025；
    乾淨海水的 NIR、SWIR 接近 0）分開。
  - 沒有「連著油汙」這類：大塊 FP 碰到 GT 油汙、往外延伸到未標註區，被算成乾淨海面。
    新增 oil_attached；並新增連通區碰到陸地即算海岸（land_coast）。
- 舊規則下 fold 1 的初步觀察（僅供參考，以 (4) 為準）：
  - 重組評分範圍內的 FP，99.75% 在未標註（255）像素，只有 0.25% 在確認背景（mask==1），
    而 HNM 探勘只把 mask==1 當 FP。
  - 約 65% 的 FP 在油汙 8 px 內。
  - test 背景小圖上的 FP 多是暗色水域、霧、條紋，才是 HNM 的目標類型，但主指標不評這些小圖。
  - SR_B1B2_B3B8A 卡在上限的比例只有約 2%，比原先擔心的低。

---

## 2026-10-17 HNM_SI 三折探勘與 hnm_si 選取 —— 完成

```bash
nohup tools/run_hnm_si_full.sh > hnm_runs/hnm_si_v1/run.log 2>&1 &
```

- 設定 `preprocess/hnm_si/config.example.yaml`；探勘模型為 09/29 各折基準 `best.pt`。
- 先跑 preflight（16:27，`hnm_runs/hnm_si_v1/preflight/fold_1.json`）。
- **mine**（16:38–19:02）：只掃各折 train 場景，256×256、stride 192 的合格窗。

  | 折 | 場景數 | 合格窗 | 有 FP 的窗 | 耗時 |
  | --- | ---: | ---: | ---: | --- |
  | 1 | 188 | 272,201 | 12,521 | 約 48 分 |
  | 2 | 189 | 265,839 | 8,885 | 約 48 分 |
  | 3 | 189 | 267,716 | 9,453 | 約 48 分 |

- **select --arm hnm_si**（19:02）：FP 窗 → SI 過濾（排除最像油的 5%）→ 依場景均衡抽到基準數量。

  | 折 | 選出 | 涵蓋場景 | 排除：太像油 | 排除：FP 區 SI 像素不足 | 合格但超出配額 | SI 距離門檻 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | 1 | 1,860 | 185 | 568 | 1,171 | 8,922 | 0.147 |
  | 2 | 1,860 | 185 | 400 | 887 | 5,738 | 0.142 |
  | 3 | 1,870 | 186 | 421 | 1,039 | 6,123 | 0.146 |

- **export**（19:02–19:12）：選中的背景輸出成圖片，給學姊檢視：
  `hnm_runs/hnm_si_v1/hnm_si_selected_images/fold_N/`。
- 只產生了 hnm_si 一組名冊，沒有訓練。

## 2026-09-29 基準三折 —— 完成（參考起點）

- `run_local_3fold.sh` → `main/local_gt_expand.yaml`，詳見
  [EXPERIMENT_RUN_SUMMARY_20260929.md](EXPERIMENT_RUN_SUMMARY_20260929.md)。
- pooled Oil IoU：0.4637／0.4634／0.3726，平均 0.4332。之後所有 HNM 工作都以這三個 `best.pt`
  為探勘模型與比較基準。
