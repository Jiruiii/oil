# 執行紀錄

每次實際跑了什麼、用哪個指令、輸出在哪、跟前一次差在哪。新的放最上面。
規劃見 [HNM_EXPERIMENT_PLAN.md](HNM_EXPERIMENT_PLAN.md)；結果輸出大多在被 `.gitignore`
排除的 `hnm_runs/`、`result-seg/`，這份文件是唯一進版控的流水帳。

## 一覽

| 日期時間 | 做什麼 | 範圍 | 用 GPU | 輸出 | 狀態 |
| --- | --- | --- | --- | --- | --- |
| 10/31 | 方向調整：HNM 暫停，改做 MariNeXt 偽標籤（計畫，未執行） | — | 否 | [PSEUDO_LABEL_PLAN.md](PSEUDO_LABEL_PLAN.md) | 規劃 |
| 10/29 | C 整塊漏掉的油改以 B 為參考重算（回應報告準備時的問題） | 三折 | 否 | 終端輸出（讀 `hnm_runs/lost_oil/components.csv`） | 完成 |
| 10/29 19:45–21:15 | seed 43 評估：C、G、G0 門檻掃描、漏掉的油、診斷，與 seed 42 的 A、B 比較（`tools/run_seed43_eval.sh`） | 3 組 × 三折 | 是（90 分鐘） | `hnm_runs/seed43/` | 完成 |
| 10/29 10:06–19:44 | C、G、G0 補 seed 43：同一份背景名冊，只換訓練 seed | 3 組 × 三折 | 是（9.6 小時） | `result-seg/CV_gt_expand_hnm_si_v1_{hnm,hnm_knn,hnm_swap}_seed43/` | 完成 |
| 10/29 05:46–06:40 | G、G0 評估：門檻掃描、每個油汙區塊 P(oil)＋漏掉的油、診斷與配對比較 | 2 組 × 三折 | 是（54 分鐘） | `hnm_runs/knn_filter/{threshold_sweep,lost_oil,diagnostics}/`、`hnm_runs/diagnostics_v2/{G,G0}/` | 完成 |
| 10/28 23:17–10/29 05:44 | G hnm_knn／G0 hnm_swap：抽 C 候選池的 encoder 向量 → 替換硬負樣本 → 兩組各訓練三折 | 2 組 × 三折 | 是（6.5 小時） | `hnm_runs/knn_filter/`、`arms/{hnm_knn,hnm_swap}/`、`result-seg/CV_gt_expand_hnm_si_v1_{hnm_knn,hnm_swap}/` | 完成 |
| 10/28 23:00 | 硬負樣本 kNN 過濾離線模擬（用 (3) 已存特徵） | 三折 | 否 | 終端輸出 | 完成 |
| 10/28 20:59–21:38 | 特徵空間：在 A 的特徵中，HNM 漏掉的油是否靠近其硬負樣本 | 三折 | 是（39 分鐘） | `hnm_runs/feature_space/` | 完成 |
| 10/28 20:18–20:48 | HNM 漏掉的是哪種油：A～D 存每個 GT 油汙區塊的 P(oil)，依屬性分層與同地點硬負樣本比較 | 4 組 × 三折 | 是（30 分鐘） | `hnm_runs/lost_oil/` | 完成 |
| 10/28 15:05–15:57 | 門檻掃描：七組重新推論存 P(oil)，同 recall 下比背景誤報（HNM 是學會分辨 look-alike，還是只是變保守） | 7 組 × 三折 | 是（52 分鐘） | `hnm_runs/threshold_sweep/` | 完成 |
| 10/27 23:05–10/28 04:05 | 重算 A～F 六組＋R（加含背景 Oil IoU、Boundary IoU）與配對比較 | 7 組 × 三折 | 少量 | `hnm_runs/diagnostics_v2/` | 完成 |
| 10/27 23:01–10/28 03:20 | 基準＋oil_ignore_ring 8 px 重訓（R ring8），自動評估加背景誤報與 Boundary IoU | 三折 | 是（4.3 小時） | `result-seg/CV_gt_expand_ring8/` | 完成 |
| 10/21 19:47–10/22 08:49 | OOF 兩組：EfficientNet 場景 OOF 評分 → 選取 → 分割訓練（E oof、F oof_si） | 三折 | 是（13 小時） | `hnm_runs/hnm_si_v1/oof/`、`arms/{oof,oof_si}/`、`result-seg/CV_gt_expand_hnm_si_v1_{oof,oof_si}/` | 完成 |
| 10/20 00:35–01:00 | 步驟 2 分析：三組套用步驟 1 診斷＋配對比較 | 3 組 × 三折 | 小量 | `hnm_runs/diagnostics_step2/` | 完成 |
| 10/19 13:11–10/20 00:29 | 步驟 2 訓練：random → hnm → hnm_si，各三折 | 9 個 run | 是（11.3 小時） | `result-seg/CV_gt_expand_hnm_si_v1_<arm>/` | 完成 |
| 10/19 00:29–01:11 | 步驟 1 正式診斷（基準 FP 落點） | 三折 | 小量（背景小圖推論） | `hnm_runs/diagnostics/` | 完成 |
| 10/19 00:30 | 產生步驟 2 訓練 yaml 並檢查 | 3 個 arm | 否 | `main/local_hnm_{random,hnm,hnm_si}.yaml` | 完成 |
| 10/19 00:29 | 產生 random、hnm 背景名冊 | 三折 | 否 | `hnm_runs/hnm_si_v1/arms/{random,hnm}/` | 完成 |
| 10/19 00:1x | 步驟 1 煙霧測試（舊分類規則） | 只有 fold 1 | 小量 | 暫存目錄，**未保留、不採用** | 完成 |
| 10/17 16:38–19:12 | HNM_SI 三折探勘＋hnm_si 選取＋匯出圖片 | 三折 | 是（約 2.4 小時） | `hnm_runs/hnm_si_v1/` | 完成 |
| 09/29 17:08–21:13 | 基準三折訓練＋GT-aware 重組 | 三折 | 是（約 4.1 小時） | `result-seg/CV_358clean_gt_expand_prithvi300m_dlv3_det/` | 完成 |

步驟 2 已完成（見 2026-10-20）：單一 seed 下，HNM 兩組在主指標上**劣於**隨機背景對照，但 test 背景小圖的誤報明顯減少。

10/29 seed 43（見 2026-10-29 (4)）：C、G0 換 seed 後 Oil IoU 各差約 0.04、方向相反，seed 42 的 G−G0 與「G 抓回 C 漏掉的一半」沒有重現；HNM 降低背景誤報、代價在弱對比淡油、Oil IoU 低於 B 三點兩個 seed 一致。

10/27 結果（見 2026-10-27）：納入背景誤報後 C−B 不再顯著，其餘排序不變；R ring8 的 recall 上升、precision 與 Boundary IoU 明顯下降，Oil IoU 與 A 無顯著差異（點估計 −0.020）；標註以 NOAA 專家為準，判定變差、不採用。

---

## 2026-10-31

### (1) 方向調整：HNM 暫停，改做 MariNeXt 偽標籤 —— 規劃

- 10/29 報告後的回饋：HNM 先暫停，改用 MariNeXt 對資料集做偽標籤，只保留原 GT ∪ MariNeXt 判為油的範圍重建訓練資料集（標籤仍用原 GT）。
- 原本待決定的 B seed 43、C／G seed 44、混合背景、hard positive 都先不做；HNM 的輸出（`hnm_runs/`、`result-seg/`）保留。
- 已知問題：MariNeXt（MADOS）用 Sentinel-2 11 波段（B1～B8A、B11、B12），本專案只有 8 波段，缺 B05～B07；Landsat 72 景無法直接使用。
- 步驟、品質檢查與待確認事項見 [PSEUDO_LABEL_PLAN.md](PSEUDO_LABEL_PLAN.md)。報告問題的回覆在 `reports/2026-10-31/學姊問題回覆_20261031.md`。

## 2026-10-29

### (4) C、G、G0 seed 43 結果 —— 完成

訓練 10/29 10:06–19:44（G0 → G → C），評估 19:45–21:15：

```bash
setsid nohup tools/run_seed43_eval.sh > hnm_runs/seed43_eval.log 2>&1 &
```

- `tools/run_seed43_eval.sh`：步驟與 `tools/run_knn_eval.sh` 相同，輸出放 `hnm_runs/seed43/`；
  比較時 C、G、G0 是 seed 43，A、B 沿用 seed 42（A、B 沒有補 seed）。
- 「seed 42 的 C 漏掉的 73 塊」在 seed 43 的 recall 另外算（讀兩份 `components.csv`，同一套區塊；場景配對 bootstrap 2,000 次，三折平均），沒有寫成工具。
- 訓練 epochs（早停）：C 47／50／34、G 60／53／39、G0 50／61／63。seed 43 的 C fold 3 只跑 34 epochs，與 seed 42 的 G0 fold 3 相同的情況。

**兩個 seed 並列（重組影像、三折平均，0.5 門檻）**

| 組 | Oil IoU s42 | Oil IoU s43 | recall s42 | recall s43 | precision s42 | precision s43 |
| --- | --- | --- | --- | --- | --- | --- |
| B（只有 s42） | 0.447 | — | 0.545 | — | 0.712 | — |
| C | 0.421 | 0.384 | 0.534 | 0.480 | 0.672 | 0.656 |
| G | 0.422 | 0.415 | 0.548 | 0.518 | 0.648 | 0.668 |
| G0 | 0.374 | 0.410 | 0.469 | 0.510 | 0.655 | 0.675 |

| 配對（Oil IoU） | seed 42 | seed 43 |
| --- | --- | --- |
| G−C | +0.001 [−0.016, +0.019] | +0.031 [+0.012, +0.052] |
| G−G0 | +0.048 [+0.025, +0.071] | +0.005 [−0.023, +0.032] |
| G0−C | −0.047 [−0.073, −0.023] | +0.026 [−0.005, +0.059] |
| G−B | −0.025 [−0.047, −0.003] | −0.032 [−0.054, −0.008] |
| C−B | −0.026 [−0.048, −0.006] | −0.063 [−0.091, −0.037] |

C 與 G0 在兩個 seed 間各差約 0.04，方向相反；G0−C 的正負號翻過來。G 兩個 seed 只差 0.007。

**look-alike（test 背景小圖）**

| 組 | 有誤報的小圖（0.5 門檻）s42 | s43 | 同 recall 警報率 s42 | s43 |
| --- | --- | --- | --- | --- |
| A | 4.79% | — | 4.79% | — |
| B | 4.67% | — | 5.19% | — |
| C | 1.77% | 2.17% | 2.52% | 達不到（fold 1、2） |
| G | 2.17% | 2.17% | 2.54% | 3.52% |
| G0 | 1.37% | 1.68% | 達不到 | 達不到（fold 1、3） |

- seed 43：G−B −0.017 [−0.031, −0.008]、G−A −0.013 [−0.025, −0.005]（條件式）。G 在兩個 seed 的三折都達得到 A 的 recall；C、G0 至少有一個 seed 達不到。
- test 背景小圖誤報（每百萬像素）seed 43：C−B −1,583 [−2,556, −718]、G−B −1,592 [−2,484, −828]、G−C −9 [−508, +449]。

**C 整塊漏掉的區塊**

- 兩個 seed 的 C 整塊漏掉的區塊只部分重疊：seed 42 有 73 塊、seed 43 有 64 塊，兩次都漏掉的只有 30 塊。
- 在 seed 42 定義的 73 塊上，seed 43 的 C／G／G0 recall 是 0.29／0.31／0.31。G−G0 +0.007 [−0.108, +0.250]，G−C +0.021 [−0.039, +0.160]。seed 42 的 G 0.48、G0 0.13 沒有重現。
- 在 seed 43 自己定義的 64 塊上，G 0.37、G0 0.21。G−G0 +0.157 [−0.073, +0.326]（CI 含 0），G0−C +0.208（這一層對 C 的偏差大小，見 (1) 的注意）。
- 兩次都漏掉的 30 塊（占油像素 0.6%）：G 0.12／0.12、G0 0.07／0.06（s42／s43）。
- seed 43 的 C 漏掉的區塊特性與 seed 42 相同：中位面積 839、寬度 6.9、對比 0、C 最高 P(oil) 中位數 0.008。

**弱對比油（recall 差）**

| 配對 | seed 42 | seed 43 |
| --- | --- | --- |
| C−B | −0.054 [−0.107, −0.015] | −0.095 [−0.162, −0.039] |
| G−C | +0.028 [−0.014, +0.066] | +0.057 [+0.005, +0.094] |
| G−G0 | +0.064 [+0.036, +0.113] | +0.037 [−0.007, +0.071] |

**對照判讀表：第四列（三組在兩個 seed 間差異都很大）。**

- **兩個 seed 一致的部分：**
  - HNM 各組在 test 背景小圖上的誤報都比 B 少：0.5 門檻有誤報的小圖 1.4～2.2% vs 4.7%。
  - G 在同 recall 下的背景警報率低於 B，CI 都不含 0。
  - G 的 Oil IoU 低於 B，CI 都不含 0。
  - HNM 的代價集中在弱對比油（C−B 都為負），C 漏掉的油都是小、細、對比 ≈ 0 的區塊。
  - G 的 Oil IoU 點估計兩次都不低於 C、G0。
- **不一致的部分：**
  - G−G0、G0−C 的大小與正負號；
  - C 整塊漏掉的是哪些區塊；
  - seed 42 看到的「G 抓回 C 漏掉的一半、G0 只有 0.08」。

**解讀：**

- C、G0 換一個訓練 seed，Oil IoU 就差 0.04，和 seed 42 用來判斷 G 效果的組間差異同一個量級。
- seed 42 的 G−G0 大，主要是 G0 那一次剛好變差（fold 2、3）；C 漏掉的區塊抓回一半，也有一大部分是 C 那一次剛好漏掉那些區塊。依 kNN 替換硬負樣本的效果，兩個 seed 下沒有確認。
- 兩個 seed 都成立的是：
  - HNM 挖到的背景確實讓 test 背景小圖誤報減少；
  - 代價落在弱對比、細的淡油上；
  - 至少到目前的設定為止，Oil IoU 仍低於 B。
- G 兩次結果最接近、兩次都達得到 A 的 recall，但兩個 seed 不足以說 G 比 C 穩定。
- 限制：每組兩個 seed；A、B 只有 seed 42，所以 seed 43 的 G−B、C−B 混了 seed 差異；同 recall 的 CI 是條件式；G0 的替換沒有重抽。

**下一步（待決定）：**

- 判讀表第四列建議評估是否所有主要組都補 seed。最優先的是 B seed 43（約 3 小時），讓 HNM−B 能在同 seed 比較。
- 若要確認 G 與 C 的差異，C、G 至少要再一個 seed（seed 44）。
- 下一個方法（混合背景、hard positive）也要以多 seed 評估。

### (3) C 整塊漏掉的油：改用 B 當參考 —— 完成

10/28 (2) 的「lost」以 A 為參考（A recall ≥ 0.5、C < 0.1）。B 才是 HNM 的對照組，所以改用 B 當參考重算一次，確認結論不受參考組影響。
不重新推論，直接讀 `hnm_runs/lost_oil/components.csv`（區塊 recall＝tp／area，門檻 0.5）。

| 定義 | 區塊數 | 中位面積 | 中位寬度 | 中位對比 | C 最高 P(oil) 中位數 |
| --- | --- | --- | --- | --- | --- |
| A 抓到、C 漏掉（原定義） | 73 | 545 | 6.7 | ≈ 0 | 0.03 |
| B 抓到、C 漏掉 | 58 | 419 | 6.4 | +0.016 | 0.029 |
| B、C 都抓到 | 777 | 1,583 | 9.2 | +0.073 | — |

- 原本 73 塊中，B 的分類：12 塊 B 也整塊漏掉、45 塊變小、16 塊相同或變大；B 在這 73 塊的像素 recall 0.62。
- 兩種定義重疊 48 塊。以 B 為參考時特性相同（小、細、對比接近 0、C 給的機率接近 0），結論不變。
- 限制：單一 seed；沒有做 CI。

### (2) C、G、G0 補 seed 43 —— 設計（開跑前寫定）

- **為什麼做：** (1) 的 G0 比 C 全面變差（Oil IoU −0.047），G−G0 有一部分來自 G0 自己變差；
  單一 seed 下分不出 G 的改善與 G0 的變差是依特徵挑選的效果，還是訓練本身的變動。依 10/28 (5) 判讀表的下一步補 seed。
- **做法：** C hnm、G hnm_knn、G0 hnm_swap 三組，背景名冊不變（沿用 seed 42 時產生的名冊），只把訓練 seed 改為 43；
  `make_hnm_arm_configs.py --seed 43` 產生 `main/local_hnm_{hnm,hnm_knn,hnm_swap}_seed43.yaml`，
  與 seed 42 的設定只差結果路徑、名稱與 `seed`。順序 G0 → G → C（G0 的疑問最大，先跑）。
- **這次只檢驗訓練變動，不檢驗選取變動：** G0 刪哪幾張、補哪幾張的隨機性沒有重抽。
- **評估：** 與 (1) 相同（門檻掃描、漏掉的油、診斷），seed 43 各組彼此比較，並與 seed 42 並列。

**判讀規則（開跑前寫定）**

| seed 43 結果 | 解讀 | 下一步 |
| --- | --- | --- |
| G0 回到 C 的水準，G 在 C 漏掉區塊與弱對比 recall 仍高於 C、同 recall 警報率仍與 C 相近 | seed 42 的 G0 是訓練變動；依特徵替換的效果重現 | G 當新的 HNM 版本，兩個 seed 合併報告 |
| G0 仍明顯比 C 差，G 的改善也重現 | 隨機替換本身有害（補進的窗不如 C 原本的），G−G0 不是乾淨的對照，以 G−C 為準 | 檢查 G0 補進的窗；考慮重抽 G0 的替換 |
| G 相對 C 的改善不重現（方向相反或接近 0） | seed 42 的 G−C 落在訓練變動範圍內 | 回到混合背景或 hard positive |
| 三組在兩個 seed 間差異都很大 | 單一 seed 的組間差異不可靠，前面各組比較都要保留 | 評估是否所有主要組都補 seed |

### (1) G hnm_knn／G0 hnm_swap 結果 —— 完成

訓練：10/28 23:46–10/29 05:44（每組三折約 3 小時）。評估：

```bash
setsid nohup tools/run_knn_eval.sh > hnm_runs/knn_eval.log 2>&1 &
```

- `tools/run_knn_eval.sh`：G、G0 各跑 `threshold_sweep.py`、`oil_component_prob.py`、`analyze_baseline_fp.py`，
  再與 A～F、R 既有的輸出一起比較；輸出放 `hnm_runs/knn_filter/`，舊組別的輸出不覆蓋。
- `compare_threshold_sweep.py`、`analyze_lost_oil.py`、`compare_hnm_arms.py` 加入 G−C、G−G0、G−B、G0−C、G−A、G0−A；
  `analyze_lost_oil.py` 另加「C 整塊漏掉的區塊」分層（`lost_by_C`）。A～F 既有數字不變（matched_differences 與 10/28 相同）。
  全部 71 個測試通過。
- **注意：** 「C 整塊漏掉的區塊」是用 C 自己的結果定義的，G−C 在這一層一定偏向 G（換一個模型多少會抓回一些）；
  G0−C 在這一層的 +0.11 可當作這個偏差的大小。這一層以 G−G0 為準。

**訓練時自動評估（patch 層級 test，三折平均，參考用）**

| 組 | Oil IoU | Recall | Precision |
| --- | --- | --- | --- |
| B 隨機（同池） | 0.458 | 0.583 | 0.681 |
| C HNM-A | 0.429 | 0.563 | 0.650 |
| G HNM-A+kNN | 0.434 | 0.584 | 0.628 |
| G0 隨機替換 | 0.386 | 0.499 | 0.633 |

**結果（三折平均，95% CI 為場景配對 bootstrap，單一 seed）**

1. **同 recall 下的背景警報率：G 與 C 相同，仍低於 A、B。**

   | 組 | fold 1 | fold 2 | fold 3 | 平均 |
   | --- | --- | --- | --- | --- |
   | A | 4.19% | 3.59% | 6.58% | 4.79% |
   | B | 5.21% | 4.36% | 5.98% | 5.19% |
   | C | 2.39% | 3.18% | 1.99% | 2.52% |
   | G | 1.71% | 2.08% | 3.85% | 2.54% |
   | G0 | 1.62% | 達不到 | 達不到 | 達不到 |

   G−C +0.000 [−0.008, +0.011]；G−A −0.022 [−0.032, −0.013]；G−B −0.026 [−0.038, −0.016]。
   G 三折都達得到 A 的 recall（CI 用到的抽樣比例 G−A 66%，C−A 52%），仍屬條件式。
   G0 在 fold 2、3 達不到 A 的 recall，有一部分油任何門檻都找不回來（與 D 相同的情況）。

2. **C 整塊漏掉的 73 個區塊（占油像素 1.7%）：G 抓回約一半。**

   | 組 | A | B | C | D | G | G0 |
   | --- | --- | --- | --- | --- | --- | --- |
   | recall | 0.894 | 0.621 | 0.029 | 0.276 | 0.502 | 0.079 |

   G−G0 +0.346 [+0.106, +0.556]（三折 +0.20／+0.19／+0.65）；G−C +0.458 [+0.232, +0.688]；G0−C +0.112 [+0.025, +0.268]；
   G−B −0.161 [−0.422, +0.133]。G 自己整塊漏掉的區塊 34 個（C 73、G0 91、B 30），同樣是淡、細、對比 ≈ 0 的油。

3. **弱對比油：G 介於 C 與 B 之間，與 C 的差異 CI 含 0。**
   recall A 0.530／B 0.528／C 0.474／G 0.502／G0 0.438。
   G−C +0.028 [−0.014, +0.066]、G−G0 +0.064 [+0.036, +0.113]、G−B −0.026 [−0.086, +0.021]。
   亮油 G−C +0.037 [−0.005, +0.076]、G−B +0.048 [+0.018, +0.096]；暗油各組無差異。

4. **整體 recall 與主指標（0.5 門檻）：**
   - 相對 A 的 recall 差：B −0.023、C −0.036、G −0.019、G0 −0.106。
   - Oil IoU：G−C +0.001 [−0.016, +0.019]、G−B −0.025 [−0.047, −0.003]、G−A −0.012 [−0.032, +0.009]。
   - Precision：G−C −0.023 [−0.046, −0.001]；重組影像中背景像素的誤報 G−C +3,441／M [+486, +7,222]，
     test 背景小圖的誤報 G−C +163／M [−862, +1,223]（無差異）。G 在油的周圍多畫了一些，look-alike 小圖上沒有增加。

   - 補充（10/29 報告準備時整理，數字取自 `hnm_runs/knn_filter/diagnostics/comparison.md`，重組影像、三折平均）：

     | 組 | recall | precision | 油汙周邊誤報（每折像素） | Oil IoU |
     | --- | --- | --- | --- | --- |
     | B | 0.545 | 0.712 | 679,880 | 0.447 |
     | C | 0.534 | 0.672 | 840,914 | 0.421 |
     | G | 0.548 | 0.648 | 959,281 | 0.422 |

     C 整塊漏掉的區塊只占油像素 1.7%，G 抓回一半約等於整體 recall +0.008。G 的 recall 已與 B 相當，
     Oil IoU 仍低於 B 主要來自 precision（油汙周邊多畫）；G 對 C 則是 recall 增加與 precision 下降互相抵銷。

5. **G0 比 C 全面變差，不只是在 C 漏掉的區塊：**
   - Oil IoU G0−C −0.047 [−0.073, −0.023]；
   - 「C 沒漏掉的區塊」recall −0.067 [−0.105, −0.040]，主要在 fold 2、3（fold 1 G0 與 C 接近）。
   - G0 fold 3 只訓練 34 epochs 就早停（C 100、G 55），該折 Oil IoU 0.277；fold 2 為 55 epochs（C 50）。
   - G0 的硬負樣本「訓練油鄰居比例」與 C 相近（0.141～0.200 vs 0.154～0.216），用這個分數解釋不了它為什麼變差。
   - 所以 G−G0 有一部分來自 G0 自己變差。fold 1 G0 正常，該折 G−G0 在 C 漏掉的區塊仍為 +0.20。

**對照判讀表：** 最接近第一列，但不完全符合。

- **符合：**
  - C 漏掉的區塊 G 抓回約一半，G−G0 CI 不含 0；
  - 同 recall 背景警報率與 C 相同，仍低於 A、B。
- **不符合：**
  - 弱對比油 G−C 的 CI 含 0；
  - 主指標 Oil IoU 與 C 相同、仍低於 B；
  - 對照組 G0 本身變差，第四列「C 的選取不穩定」的疑慮沒有排除：換掉兩到三成硬負樣本，結果可以差到 Oil IoU 0.05。

**解讀：** 把 A 特徵中離訓練油太近的硬負樣本換掉，C 對 look-alike 的分辨力沒有損失，C 整塊漏掉的淡油約一半回來了。
這支持 (3) 的看法：重疊區的硬負樣本是一部分淡油被推掉的原因。但整體 Oil IoU 沒有變好，precision 略降；
而且隨機替換（G0）的結果差異很大，在只有一個 seed 的情況下，無法確定 G 與 C、G0 的差異有多少來自依特徵挑選、
有多少來自訓練本身的變動。

**下一步：** 依判讀表補 seed。至少 C、G、G0 各加一個 seed，才能看 G0 的變差與 G 的改善是否重現。

## 2026-10-28

### (5) G hnm_knn／G0 hnm_swap：依 A 的 encoder kNN 替換硬負樣本 —— 設計（開跑前寫定）

- **為什麼做：** (3) 是相關性：被 C 漏掉的淡油在 A 的特徵中與硬負樣本混在一起。這裡直接改訓練資料驗證：
  把 C 的硬負樣本中「離訓練油太近」的換掉，看淡油的 recall 是否回來、look-alike 的誤報是否仍低。
- **過濾分數（只用訓練資料）：** 每折以 A（C 探勘用的同一個 `best.pt`）單次前向取 encoder 特徵；
  硬負樣本向量＝該小圖上 A 判成油的確認背景像素（P ≥ 0.5 且 mask＝1），與 (3) 相同。
  z-score 用 (3) 的參考集合（訓練油、random、C、D 的硬負樣本）。
  分數＝在「訓練油＋C 已選的硬負樣本」（扣掉自己）中，最近 10 個鄰居裡訓練油的比例。
  **τ＝0.3：分數 ≥ 0.3 的視為離油太近。** 不做 TTA 時沒有誤報像素、算不出向量的候選不刪、也不拿來補。
- **G hnm_knn：** 從 C 已選的 1,860／1,860／1,870 張中刪掉分數 ≥ τ 者；每刪一張，從同一場景、C 候選池中
  沒被選且分數 < τ 的窗補一張（隨機，seed 固定）；同場景不夠時，依原本的場景輪流規則從其他場景補。
  所以 G 與 C 只差被換掉的那些小圖，數量相同。
- **G0 hnm_swap（對照）：** 在相同場景刪掉相同張數，但刪哪幾張隨機；補的窗也來自同場景，從 C 候選池中
  沒被選的窗隨機抽（不看分數）。G0 每個場景的張數與 G 完全相同，只差「換掉哪些」是否依特徵決定。
- **其餘設定：** 訓練設定與 C 相同（`make_hnm_arm_configs.py` 檢查只差結果路徑與 `hnm` 區塊）；seed 42；三折。
- **評估：** 門檻掃描（`threshold_sweep.py`）、漏掉的油（`analyze_lost_oil.py`）、診斷與配對比較（`compare_hnm_arms.py`）。
  主要比較 G−C 與 G−G0；G−B 參考。
- **離線可行性檢查（(4)）：** 用 (3) 已存的特徵模擬 C 已選的硬負樣本，τ＝0.3 刪掉 22～33%；
  test 中 C 漏掉的油，其 10 個最近鄰居裡的硬負樣本有 43～52% 會被刪；漏掉的油的 kNN 硬負比例平均
  0.23～0.47 → 0.12～0.33。**這項檢查用到了 test 的 lost 標記，τ 是看過它之後選的**（τ＝0.5 只刪到四成，太弱），
  所以 τ 不是完全獨立於 test 決定的；過濾規則本身只用訓練資料。

**判讀規則（開跑前寫定）**

| 結果 | 解讀 | 下一步 |
| --- | --- | --- |
| G 的弱對比／C 漏掉區塊 recall 高於 C 與 G0，同 recall 背景警報率仍低於 A、B | 重疊區的硬負樣本是淡油被推掉的原因；換掉它們保住淡油、不失去分辨力 | G 當新的 HNM 版本；補 seed |
| G 的 recall 回來，但同 recall 背景警報率回到 B 的水準 | 被換掉的正是有用的 look-alike；A 的特徵分不開兩者 | 改加重淡油（hard positive）或兩階段 |
| G 與 G0 無差異（CI 含 0） | A 特徵中的重疊不是原因（或單一 seed 下看不出） | 回到混合背景（硬負＋隨機） |
| G0 也和 G 一樣改善 | 換一批硬負樣本本身就有影響，不是依特徵挑的效果 | 檢查 C 的選取是否不穩定；補 seed |

**執行（10/28 23:17 開跑）**

```bash
setsid nohup tools/run_knn_filter.sh > hnm_runs/knn_filter.log 2>&1 &
```

- `tools/filter_hard_negatives.py`：`--stage extract` 對 C 候選池（status 為 selected 或 not_selected_quota，
  每折約 12,500 窗）抽 A 的 encoder 向量；`--stage select` 算分數、產生兩組名冊與 manifest
  （`candidates_scored.tsv` 多一欄 `knn_train_share`，manifest 記 τ、k、特徵檔雜湊）。
  select 會先檢查重抽的 C 已選小圖向量與 (3) 存的相同，不同就中止。
- 接著 `make_hnm_arm_configs.py --arm hnm_knn --arm hnm_swap`、`run_hnm_step2.sh hnm_knn hnm_swap`。
- 測試 `tests/test_filter_hard_negatives.py`；全部 71 個測試通過。
- **抽特徵與選取（23:17–23:46）：** 每折約 11 分鐘（HDD 讀取為瓶頸，GPU 使用率約 8%）。

  | 折 | 候選池 | 有向量 | C 已選 | 換掉（G＝G0） | 從其他場景補 | 保留 C 的張數 | G∩G0 | 訓練油鄰居比例 C／G／G0 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
  | 1 | 12,521 | 12,485 | 1,860 | 403（21.7%） | 11 | 1,457 | 1,264 | 0.154／0.040／0.141 |
  | 2 | 8,885 | 8,847 | 1,860 | 605（32.5%） | 136 | 1,255 | 1,038 | 0.216／0.054／0.200 |
  | 3 | 9,453 | 9,415 | 1,870 | 500（26.7%） | 68 | 1,370 | 1,167 | 0.179／0.037／0.173 |

  - 重抽的 C 已選小圖向量與 (3) 相對差 ≤ 5.4×10⁻⁸；換掉比例與 (4) 模擬一致（21.7／32.7／26.9%）。
  - fold 2 候選池較小，有 136 張要從其他場景補，G 的場景分布與 C 略有不同；G0 每場景張數與 G 完全相同，所以 G−G0 不受影響。
  - 每組名冊皆通過 `_check_selection`（訓練場景、不與 GT 小圖重複、標註檢查）。23:46 開始訓練。

### (4) 硬負樣本 kNN 過濾的離線模擬 —— 完成

```bash
.venv/bin/python tools/filter_hard_negatives.py --stage simulate
```

- 用 (3) 存下的 `hnm_runs/feature_space/fold_*.npz`，不跑 GPU。結果見 (5) 的「離線可行性檢查」：

  | τ | 硬負樣本被刪比例 | C 漏掉的油的鄰近硬負樣本被刪比例 |
  | --- | --- | --- |
  | 0.5 | 13～20% | 36～43% |
  | 0.3 | 22～33% | 43～52% |
  | 0.1 | 45～58% | 65～79% |

- 漏掉的油旁邊的硬負樣本，訓練油鄰居比例（中位數 0.1～0.3）高於整體（0～0.1），過濾會優先刪到它們，但選擇性中等。

### (3) 特徵空間：漏掉的油是否靠近硬負樣本 —— 完成

```bash
.venv/bin/python tools/feature_space.py --out hnm_runs/feature_space \
    --components hnm_runs/lost_oil/components.csv --run-dir 1=<A fold 1> --run-dir 2=... --run-dir 3=...
```

- **為什麼做：** (2) 的看圖結果是「HNM 選進的背景小圖外觀像淡油」，但同地點曝露解釋不了。要確認這不只是肉眼印象：
  在挖硬負樣本所用的模型（基準 A）自己的特徵裡，被漏掉的油是不是真的跟硬負樣本靠在一起。
- **做法：** 用 A 單次前向（不做 TTA）取兩層特徵：DeepLabHead 倒數第二層（256 維，分類器對它是線性的）與 Prithvi encoder 輸出（1024 維），
  都是每 256 px 小圖 16×16 格。每個區域的向量＝它覆蓋格子的像素加權平均。每折四組參考向量：
  訓練油（每張訓練小圖的 GT 油像素）、C／D 的硬負樣本（每張選進的背景小圖上 A 判成油的像素）、隨機背景（random 組小圖的 mask＝1 像素）；
  加上 test 的 1,140 個油汙區塊。z-score 後用 cosine；每個區塊算與各組最近 5 個的平均相似度，
  以及「訓練油＋硬負樣本」中最近 10 個鄰居裡硬負樣本的比例。比較 (2) 的 lost 與 kept（A 與該組都 ≥ 0.5 recall），AUC 以場景 bootstrap 1,000 次。
- **檢查：** 硬負樣本小圖不做 TTA 仍有誤報的比例 1,852～1,870／1,860～1,870（與探勘一致）；測試 `tests/test_feature_space.py`；全部 68 個測試通過。
- **時間：** 20:59–21:38（每折約 13 分鐘）。輸出 `hnm_runs/feature_space/`（summary.md、auc.csv、similarities.csv、pca_pen.png、pca_enc.png）。

**結果（AUC＝漏掉的區塊分數高於保住的區塊的機率；0.5＝分不開）**

| 特徵 | 分數 | C：AUC [95% CI] | D：AUC [95% CI] |
| --- | --- | --- | --- |
| encoder | 最近 10 鄰居中硬負樣本比例 | **0.78 [0.71, 0.85]**（中位數 0.3 vs 0） | **0.83 [0.74, 0.92]**（0.5 vs 0） |
| encoder | 與訓練油相似度 | 0.22 [0.14, 0.31]（漏掉的離訓練油較遠） | 0.21 [0.12, 0.31] |
| encoder | 與隨機背景相似度 | 0.71 [0.61, 0.79] | 0.73 [0.62, 0.82] |
| 倒數第二層 | 最近 10 鄰居中硬負樣本比例 | 0.65 [0.56, 0.72] | 0.64 [0.54, 0.73] |
| 屬性對照 | 弱對比（−\|contrast\|） | 0.59 [0.49, 0.69] | 0.60 [0.51, 0.69] |
| 屬性對照 | 小（−log 面積） | 0.66 [0.56, 0.76] | 0.63 [0.49, 0.74] |

1. **在 A 的 encoder 特徵裡，被 HNM 漏掉的油確實和硬負樣本混在一起：** 它們最近的參考鄰居有 30～50% 是硬負樣本，保住的油幾乎是 0。
   這個分數比對比、大小等影像屬性更能預測哪些油會被漏掉。
2. **被漏掉的是「非典型」的油：** 離訓練油較遠、離一般背景較近，落在油與背景之間、硬負樣本最密的那一帶（pca_enc.png）。
3. 倒數第二層的差異較小（0.65）：A 的分類頭本來就把硬負樣本放在油那一側（它們就是 A 的誤報），所以這一層較難看出差別。

**解讀：** 與 (2) 的看圖一致——HNM 挖到的 look-alike 和淡油在模型看到的特徵中重疊；把這些 look-alike 標成背景訓練，
重疊區裡的真油也一起被推到背景。單一 seed、相關性分析（不是因果驗證）；特徵取自 A，不是 C／D 自己。

### (2) HNM 漏掉的是哪種油 —— 完成

```bash
setsid nohup tools/run_lost_oil.sh > hnm_runs/lost_oil.log 2>&1 &     # A～D 推論＋分析
.venv/bin/python tools/show_hard_negative_tiles.py --arm-dir hnm_runs/hnm_si_v1/arms/hnm \
    --site 2:34SGH --site 2:16RCT --site 3:15RVM --out hnm_runs/lost_oil/hard_negative_tiles.png
```

- **為什麼做：** 門檻掃描顯示 HNM 有一部分真油「任何門檻都找不回來」。要知道被推掉的是哪種油，才知道下一步怎麼保住它。
- **做法：** `tools/oil_component_prob.py` 用門檻掃描同一套推論（重組的小圖、TTA、拼接順序），
  對每個 GT 油汙區塊（8 連通、≥ 20 px，三折共 1,140 個；與 10/22 contrast_recall 的區塊清單完全相同）存 P(oil) 直方圖與局部機率圖。
  `tools/analyze_lost_oil.py` 以 A 為參考：
  - 區塊分類：lost＝A recall ≥ 0.5 且該組 < 0.1；shrunk＝兩邊都有抓到但該組較少；unrecoverable＝lost 且 P(oil) ≥ 0.05 仍 < 0.1。
  - 依大小、厚度（2×面積／周長）、對比（B8A，同 10/22）、耀光代理（油外圈 B8A）、感測器、是否靠岸分層算 recall，場景配對 bootstrap。
  - 「同地點曝露」：該組選進訓練、且基準探針有誤報的背景小圖中，與 test 場景同一 MGRS 圖塊的數量。
- **檢查：** A fold 1 的區塊數與油像素（352、2,779,923）與 10/22 components.csv 相同；recall 0.6499 與門檻掃描的 A 目標一致；
  對比分層占比（26.7%／47.9%／25.4%）與 10/22 相同。測試 `tests/test_lost_oil.py`。
- **時間：** 推論 20:18–20:48（四組各約 7.5 分鐘）。輸出 `hnm_runs/lost_oil/`（summary.md、components.csv、gallery.png、hard_negative_tiles.png）。

**結果（recall 為三折像素合併，單一 seed）**

| 組 | 相對 A 的 recall 差 | 整塊漏掉（lost） | 變小（shrunk） | 其他區塊變大 | lost 區塊數 | 救不回來的區塊 |
| --- | --- | --- | --- | --- | --- | --- |
| B random | −0.023 | −0.001 | −0.052 | +0.030 | 30 | 23 |
| C hnm | −0.036 | −0.015 | −0.056 | +0.034 | 73 | 53 |
| D hnm_si | −0.073 | −0.012 | −0.088 | +0.027 | 63 | 49 |

1. **C 漏掉的油主要是「淡、細、在乾淨暗水面上」的油。** 被 C 整塊漏掉的 73 個區塊：中位數 545 px（找到的 1,544）、
   厚 6.7 px（9.0）、對比 ≈ 0（+0.07）、外圈 NIR 0.021（0.042）。C 給這些區塊的最高 P(oil) 中位數只有 0.03，所以調門檻救不回來。
2. **HNM 自己造成的代價（C−B，扣掉換背景池的影響）集中在弱對比油：** weak −0.054 [−0.107, −0.015]、厚度 6–15 px −0.066 [−0.095, −0.041]；
   亮油、暗油、大片厚油與 B 沒有差異。B−A 自己的損失則在暗油與小區塊（換背景池的效果，非 HNM）。
3. **D 另外還丟了亮油與大片厚油**（D−B bright −0.072 [−0.126, −0.011]、厚度 ≥ 15 px −0.040、外圈 NIR < 0.01 −0.117），主要在 fold 2、3。
4. **以場景為單位整片消失：** 20 個場景 A 幾乎全抓、C 掉到 0 附近；6 個場景占 C 整塊漏掉像素的 73%，多為 NOAA 墨西哥灣。
   C 與 D 漏掉的高度重疊（C 漏掉像素的 59% D 也漏掉；B 只有 4%），是 HNM 的系統性效果，不是隨機。
5. **「同地點曝露」不能解釋：** 場景的同圖塊硬負樣本數與 C−A recall 差無關（Spearman 0.03，p = 0.62）；C 漏掉像素的 72% 在曝露最少的一組。
   但看圖（hard_negative_tiles.png）：HNM 選進的背景小圖有細暗線、淡色帶、條紋——NOAA 標為背景的 look-alike，
   外觀與被漏掉的淡油非常像。比較像是模型學到「這種外觀＝背景」，而不是記住地點。

**解讀：** HNM 讓 look-alike 的 P(oil) 降低，代價落在與 look-alike 外觀相近的淡油（弱對比、細、乾淨水面）；
在影像外觀上兩者很難分開，HNM 把決策邊界推過了淡油那一區。要保住它們，候選方向：混合背景（硬負＋隨機）降低 HNM 的推力、
同時加重淡油（hard positive）讓兩邊平衡、或先在模型特徵空間確認淡油與硬負樣本是否真的重疊。

### (1) 門檻掃描：同 recall 下比較背景誤報 —— 完成

```bash
setsid nohup tools/run_threshold_sweep.sh > hnm_runs/threshold_sweep.log 2>&1 &
```

- **為什麼做：** 10/22 學姊懷疑 HNM 輸是指標造成的；10/27 加了含背景 Oil IoU，結論不變。
  但 HNM 降低背景誤報的同時也降低 recall，兩者都可能只是「比較不敢畫油」（等同把門檻調高）。
  題目是 look-alike，要回答的是：**在漏掉一樣多油的條件下，HNM 的背景誤報是否較少**。
  - 較少 → HNM 真的學會分辨 look-alike；重疊 → 只是變保守。
- **做法：**
  - `tools/threshold_sweep.py`：用各 run 的 `best.pt`，照 `main/recon_gt_aware_module.py` 的切塊、正規化、
    TTA、拼接順序重新推論 GT-aware 重組與 test 背景小圖，把 P(oil) 存成直方圖（400 格，0.5 是格線，
    `P ≥ 0.5` 即 argmax）。每組每折一個 `hnm_runs/threshold_sweep/<組>/fold_<折>.npz`，不需存整張機率圖。
  - `tools/compare_threshold_sweep.py`：門檻 0～1 掃描，算 recall、precision、背景小圖每百萬像素誤報、
    「有任何油像素的背景小圖比例」（look-alike 警報率）。以 A 在 0.5 的 recall 為目標，每組各自選門檻達到同一 recall，
    比較背景誤報；反過來也比同警報率下的 recall。CI 用同樣的場景配對 bootstrap，每次抽樣重新選門檻。
- **驗證（fold 1 A）：** 0.5 門檻下 TP 1,806,758／FN 973,213／FP 1,116,629，背景誤報 171,638；
  既有結果為 TP 1,806,757／FN 973,214／FP 1,116,626／171,638。差 1～3 像素，來自 P 恰為 0.5 的像素，可忽略。
  全部七組三折與 `diagnostics_v2` 比對，0.5 門檻下 TP／FP／背景誤報每折差 ≤ 43 像素。
- **結果**（`hnm_runs/threshold_sweep/matched_comparison.md`、`sweep_recall_vs_bg_tiles.png`）：
  - **模型幾乎是二值輸出。** 門檻從 0.5 降到 0.0025，recall 只多 0.03～0.13；每個模型都有 25～50% 的
    GT 油像素 P(oil) < 0.0025。所以「同 recall」只在各組曲線重疊的範圍內比得了；達不到的標「unreachable」
    （只有把所有像素都判成油才達得到）。
  - **C hnm 的曲線在重疊範圍內都在 A、B 下方：同 recall、背景警報較少。** 以 A 在 0.5 的 recall 為目標，
    有任何油像素的背景小圖比例：fold 1 C 2.4%（A 4.2%、B 5.2%），fold 2 3.2%（3.6%、4.4%），
    fold 3 2.0%（6.6%、6.0%）；三折平均 C−A −0.023 [−0.033, −0.011]、C−B −0.027 [−0.040, −0.017]。
    CI 只用 C 達得到目標的抽樣（約一半），屬條件式，偏樂觀。要求警報至少 50 或 200 像素時結論相同。
  - **D hnm_si 同樣在 A、B 下方，但 fold 2、3 達不到 A 的 recall**（最高 0.577／0.399 vs A 0.580／0.469）：
    有一部分油被確定地判成背景，任何門檻都救不回來。F 在 fold 1 也達不到；E 在 fold 1 明顯較差。
  - B 與 A 無差異（0.004 [−0.005, +0.015]）：換一批隨機背景不改變分辨能力。R 與 A 相近。
- **解讀：** HNM 不是單純「變保守」（調門檻做不出它的效果）：在可比的範圍內，它確實讓 look-alike 的 P(oil)
  更低。代價是一部分真油也被推到背景那一側且無法用門檻找回，這就是主指標上 recall 下降的來源。
  下一個問題是：被 HNM 推掉的是哪種油（是否是外觀像 look-alike 的暗油、薄油）。單一 seed。

## 2026-10-27

10/22 報告後，學姊要求實作簡報「待討論」中的兩項：（1）主指標納入背景誤報；
（3）處理油汙邊界與漏報（訓練時油外圈未標註不計分的 `oil_ignore_ring`）。
參考文獻：Cheng et al., *Boundary IoU: Improving Object-Centric Image Segmentation Evaluation*,
CVPR 2021（arXiv:2103.16562）。

### (2) 重算 A～F 六組，加入新指標 —— 完成

```bash
setsid nohup tools/rescore_arms.sh > hnm_runs/diagnostics_v2.log 2>&1 &
.venv/bin/python tools/compare_hnm_arms.py --out hnm_runs/diagnostics_v2 \
  --arm A=hnm_runs/diagnostics_v2/A --arm B=hnm_runs/diagnostics_v2/B ... --arm R=<ring8 診斷>
```

- 同一支 `tools/analyze_baseline_fp.py` 重算，原有數字不變，多了 Boundary IoU 的計數欄位；
  輸出到新資料夾 `hnm_runs/diagnostics_v2/`，舊的 `diagnostics/`、`diagnostics_step2/` 保留。
- `tools/compare_hnm_arms.py` 新增兩個指標與配對差：
  - **含背景 Oil IoU**＝TP／(TP＋FN＋油汙周邊誤報＋test 背景小圖誤報)。背景小圖為凍結名冊中
    屬於該折 test 場景者（每折 1,170 張），以場景為單位和重組像素合併後做 bootstrap。
  - **Boundary IoU**（d＝8 px）：GT 與預測各取「油汙內、距輪廓 ≤ 8 px」的邊界帶，
    交集／聯集，三折各自 pooled。未推論的區域視為未知，不當成輪廓。
    論文以影像對角線 2% 為 d；256 小圖對角線 362 px，2% 約 7 px，取 8 px 與診斷的 near_oil 一致。

### (1) R ring8：基準加上 oil_ignore_ring 8 px 重訓 —— 完成

```bash
.venv/bin/python tools/make_ring_config.py --ring 8
NO_ALBUMENTATIONS_UPDATE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 setsid nohup \
  .venv/bin/python main/main_runner.py --config main/local_ring8.yaml > hnm_runs/ring8_training.log 2>&1 &
```

- **做什麼：** 訓練時把 GT 油汙外圈 8 px 內、原本未標註（255）的像素改為不計分（loss 忽略），
  手標背景與遠處未標註仍當背景。val／test 評分口徑完全不變，所以可直接和 A 比。
  動機：診斷顯示主指標誤報 55.6% 在油汙 8 px 內，且油旁未標註的誤判率是確認背景的 3.8 倍，
  部分像是沒框到的油；把它們當背景教，可能讓模型系統性少畫邊緣。
- **設定：** `main/local_ring8.yaml` 由腳本產生，和基準只差結果路徑、名稱、
  `dataset.oil_ignore_ring: 8`，以及重組的兩個新評估開關。訓練背景用 A 的名冊。
- **開跑前檢查：** 同 40 張 fold 1 訓練小圖，油像素數不變（652,486），ring 8 多出 216,189 個不計分像素。
- **自動評估新增（`main/recon_gt_aware_module.py` 選用開關，未開時輸出與舊版完全相同）：**
  - `bg_tiles: true` → 推論 test 背景小圖，`run_metrics.json` 多出
    `recon_gtaware_pooled_oil_iou_with_bg`、`recon_gtaware_bg_fp_per_M` 等；明細 `gt_aware_recon/bg_tiles.csv`。
  - `boundary_iou_px: 8` → 多出 `recon_gtaware_pooled_boundary_iou`。
  - 兩者與診斷腳本共用 `main/metrics_extra.py`。以 fold 1 基準實測，背景小圖結果與步驟 1 診斷完全相同
    （1,170 張、171,638 誤報像素、49 張有誤報）。
- **比較：** R 對 A（只差 ring）為主要比較，R 對 B 為參考；指標為 Oil IoU、含背景 Oil IoU、
  Boundary IoU、precision／recall、背景誤報。單一 seed。

### (3) 結果（10/28 04:05）

R 的診斷（`hnm_runs/diagnostics_v2/R/`，`--skip-backgrounds`）後重跑比較：

```bash
.venv/bin/python tools/compare_hnm_arms.py --out hnm_runs/diagnostics_v2 \
  --arm A=hnm_runs/diagnostics_v2/A ... --arm F=hnm_runs/diagnostics_v2/F --arm R=hnm_runs/diagnostics_v2/R
```

- **兩條路徑一致：** R 的訓練流程自動評估（`run_metrics.json`）與診斷腳本的 Oil IoU、含背景 Oil IoU、
  Boundary IoU 三折完全相同；背景小圖誤報像素差 ≤ 19 px／折（GPU 非決定性，< 0.01%）。
- **六組加新指標**（三折平均；配對差 [95% CI]）：

  | 組 | Oil IoU | 含背景 Oil IoU | Boundary IoU |
  | --- | ---: | ---: | ---: |
  | A | 0.4332 | 0.4145 | 0.2929 |
  | B | 0.4469 | 0.4255 | 0.3349 |
  | C | 0.4205 | 0.4069 | 0.2861 |
  | D | 0.3993 | 0.3893 | 0.2885 |
  | E | 0.4015 | 0.3865 | 0.2452 |
  | F | 0.3935 | 0.3822 | 0.2715 |
  | R | 0.4129 | 0.3905 | 0.1585 |

  - 含背景 Oil IoU 對 B：C −0.019 [−0.041, +0.005]（原指標 −0.026 顯著 → 不再顯著）、
    D −0.036 [−0.060, −0.012]、E −0.039 [−0.061, −0.017]、F −0.043 [−0.069, −0.018]。
  - 背景小圖誤報占含背景聯集：A 4.3%、B 4.4%、C 2.8%、D 1.9%、R 5.1%，所以新指標只小幅改變排序。
  - Boundary IoU：B 比 A 高 +0.042 [+0.032, +0.053]；C～F 都低於 B。
- **R ring8 對 A**：Oil IoU −0.020 [−0.051, +0.003]；含背景 −0.024 [−0.055, −0.000]；
  Boundary IoU −0.134 [−0.157, −0.115]；precision −0.089 [−0.121, −0.058]；recall +0.039 [+0.011, +0.070]。
  油汙 8 px 內誤報占比 55.6% → 62.4%（每折約 60 萬 → 106 萬像素）；背景小圖誤報 2,785 → 3,813／百萬（+1,028 [−114, +2,194]）。
  實際 epochs 92／82／32（A 37／66／100）。
- **解讀（10/28 與使用者確認：標註為 NOAA 專家標註，以標註為準、不考慮標錯）：**
  模型在油旁多畫一圈，多畫的就是誤報；precision、Boundary IoU 顯著變差（含背景 Oil IoU 的 CI 上界剛好碰 0），
  對應先定判讀規則的「R−A 顯著變差」：油旁未標註區是有用的邊界監督，**不採用 oil_ignore_ring**。
  後續方向改為加強邊界監督（例如油外圈加權）或處理推論端的方塊假影。

## 2026-10-21

### (1) 新增兩組：EfficientNet OOF（E）與 OOF＋光譜指數（F）—— 完成

```bash
setsid nohup tools/run_hnm_oof.sh > hnm_runs/oof_training.log 2>&1 &
```

- **為什麼加這兩組：** C、D 的「難」是由基準分割模型判斷，而它看過這些 train 場景（within-fold）。
  E、F 改由**沒看過該場景**的分類器判斷（場景層級 OOF），補上這個缺口。
- **設計（10/21 與使用者確認）：** 和 B、C、D 用**同一個候選池**（10/17 探勘的合格窗，每折約 27 萬）、
  **同樣數量**（1860／1860／1870）、同樣依場景平均抽。唯一變的是由誰判斷難不難，所以 B 可直接當對照。
  - **E oof：** 每折把 train 場景分成 5 組，各訓練一個 EfficientNetV2-S（ImageNet 預訓練、70 epochs，
    設定同 `preprocess/hnm`）。每個候選窗只由沒看過它所在場景的那個分類器評分，P(oil) ≥ 0.5 算「難」，
    再依場景平均抽。（對應 C 的「分割模型有誤判」）
  - **F oof_si：** E 的難窗再做光譜過濾：對整張小圖的確認背景（mask==1）有效像素取三個光譜指數中位數，
    與 hnm_si 同一份油汙參考、同一個共變異矩陣算馬氏距離，排除最像油的 5%。（對應 D）
- **與原 OOF 版（`preprocess/hnm`）的差異：** 原版候選池是 A 的背景、保留一半並用 facility-location 挑；
  這次為了能和 B、C、D 比，改成上述做法，也不訓練 embedding 模型。
- **為了速度的調整：** 影像是 LZW 條帶格式、放在 HDD，隨機讀一個 256 小窗要解壓 256 整列。因此：
  - 分類器的負樣本改為每折固定一批：每個場景隨機挑幾整列合格窗，共約 15,000 張，讀進記憶體快取（約 16 GB）；
    每 epoch 從中抽 5 倍正樣本數（原版是每 epoch 從整個池子抽）。
  - 評分整個池子時按列條帶循序讀，每折約 21 分鐘。
- **fold 1 實際數字：** 正樣本 1,168、GT 負樣本 250、池子負樣本 15,016；評分 272,201 窗。
  每個分類器約 23 分鐘，預估每折約 2.3 小時、三折約 7 小時，之後兩組分割訓練約 7.5 小時。
- **流程（`tools/run_hnm_oof.sh`）：** 三折 `--stage oof` → `select --arm oof／oof_si` →
  `make_hnm_arm_configs.py --arm oof --arm oof_si` → `run_hnm_step2.sh oof oof_si`。
  若某折 P(oil) ≥ 0.5 的窗不足 1860 張，選取會依原規則中止，需另行決定。
- **結果（10/22 08:49 完成；分析在 `reports/2026-10-22/analysis/`）：** 三折 pooled Oil IoU
  E 0.4253／0.4379／0.3412（平均 0.4015）、F 0.4412／0.4077／0.3316（0.3935）；
  對 B 的三折平均差 E −0.046 [−0.068, −0.024]、F −0.054 [−0.081, −0.029]，皆不含 0。
  換成沒看過場景的分類器挑背景，仍輸給隨機背景；背景誤報只小幅減少且不顯著。
- **程式：** `preprocess/hnm_si/oof.py`（新）、`pipeline.py` 新增 `--stage oof` 與兩個 arm、
  `preprocess/hnm` 的資料集與訓練函式加了可替換讀取來源的介面；測試 `tests/test_hnm_oof.py`，全部 54 個測試通過。

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
