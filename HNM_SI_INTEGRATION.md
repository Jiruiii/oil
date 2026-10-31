# 學長 HNM_SI 接入評估與實作計畫

更新日期：2026-10-16。

目標是將學長 `OIL_PROJECT_MutiBand_thesis/preprocess/HNM_SI` 的分割誤報探勘與
光譜指數過濾，適配到目前 Sentinel-2／Landsat、八波段 VRT、Prithvi＋DeepLabV3
及凍結三折實驗。**適配程式已實作於 `preprocess/hnm_si/`（見第 5 節），並以單元測試與
真實資料小量煙霧測試驗證。2026-10-17 已完成正式三折探勘，10/19 起用於分割訓練（C hnm、D hnm_si），
結果見 [RUN_LOG.md](RUN_LOG.md)。**

現有 `preprocess/hnm/` 是 EfficientNetV2-S＋OOF＋共用 embedding 第一版，
使用方式仍見 [HNM.md](HNM.md)。它與本計畫是兩個獨立方法。

## 1. 學長原始流程

原始說明與程式見 [HNM_SI 快照](references/hnm_si_original/README.md)。

| 步驟 | 原程式行為 | 本專案適配方向 |
| --- | --- | --- |
| Step 1 | 載入各折 11 通道 ResNet50＋DeepLabV3，掃描 train 與 val 找 FP | 載入各折既有 Prithvi＋DeepLabV3，只掃該折 train |
| Step 2 | 取得每景油污 SI 中位數、候選 FP 區域 SI 中位數，以 Ledoit-Wolf 共變異矩陣計算馬氏距離 | 改為八波段映射；每折獨立建立 train 油污統計 |
| Step 3 | 預設排除距離最小的 5%，即最像油污的候選 | 保留可設定的過濾機制；門檻只能由 train 資料決定 |
| Step 4 | 依場景抽樣，排除候選數超過平均＋3σ 的場景，將 HN 追加到原 train／val 名冊 | 固定 train 背景數量、去重，輸出背景 TSV 與 manifest；保留既有 val／test |

Step 1 原預設為 256×256、stride 192、整個 patch 不含 `mask==0`、
`mask==1` 至少 50%、`mask==255` 至多 30%、nodata 至多 30%，至少 1 個 FP 像素。
Step 2 再要求至少 100 個有效 FP 像素及可用的場景油污參考，因此「Step 1 有候選」
不代表一定能進入最終 HN 池。這些是原始參數，不是已驗證適合本資料集的設定。

## 2. 本機已確認的條件

- VRT 共 355 景：S2 283、L8 51、L9 21。三類各抽讀一景，可讀到八通道非零資料；
  這是抽查，尚未完成全資料集 SI 品質或物理尺度驗證。
- 八波段順序為 `B01, B02, B03, B04, B08, B8A, B11, B12`。
  Prithvi 只選 `B02, B03, B04, B8A, B11, B12`；SI 可以另外讀取原始 B01。
  設定中 B01／B08 的 mean=0、std=1 是未使用通道的 identity 設定，不能據此認定影像為零。
- 既有三折 baseline 權重位於下列目錄的 `weights/best.pt`，已確認檔案存在，
  尚未為本次 HNM_SI 驗證載入與推論：

  ```text
  result-seg/CV_358clean_gt_expand_prithvi300m_dlv3_det/
    20260929-170815_local_gt_expand_run_fold1/
    20260929-180401_local_gt_expand_run_fold2/
    20260929-192132_local_gt_expand_run_fold3/
  ```

- [main/hnm_config.py](main/hnm_config.py) 已提供每折背景替換與來源一致性檢查，
  可作為輸出接口；不是只放一份 TSV 就能接入，還需要完成標記與正確的 manifest。

## 3. 需要適配的程式與資料規則

### 模型載入與影像讀取

沿用 [DeepLabAdapter](main/deeplab_adapter.py) 的模型建構與 checkpoint 檢查，
使用對應折的架構設定及 baseline 權重，不直接使用原本 11 通道 `build_model()`。
探勘模型的影像處理要匹配 baseline：八通道 VRT window、除以 10000、
非有限值與異常 nodata 清理、原訓練 mean/std，以及油污類別 index 0。
不可在探勘時任意重新估計模型正規化。

重用 [preprocess/hnm/data.py](preprocess/hnm/data.py) 的場景索引、名冊解析與
影像／mask 對齊檢查。候選需檢查完整 window 邊界、有效像素、原始 mask，
並排除與既有 GT-centric 小圖完全重複的座標。

原 Step 1 把整景合格 patch 影像先放進記憶體，Step 2 則一次讀完整多波段場景；
本機抽讀場景為 10980×10980，移植應改為分批 window 推論與分塊 SI 統計，避免累積整景影像。
FP 區域資料可快取，減少 Step 2 重複模型推論。

### 光譜指數與感測器

原程式實際使用下列三個指數；索引是目前八通道的 **0-based** 索引：

| 指數 | 公式 | 使用索引 |
| --- | --- | --- |
| MNDWI | `(B03 - B11) / (B03 + B11)` | 2、6 |
| NDI_B1_B11 | `(B01 - B11) / (B01 + B11)` | 0、6 |
| SR_B1B2_B3B8A | `(B01 + B02) / (B03 + B8A)` | 0、1、2、5 |

實作應按波段名稱映射並處理接近零的分母、各必要波段 nodata 及非有限值。
原 README 將第三個指數誤名為 `SR_B12_B3B8A`，移植依實際公式命名。

模型正規化與 SI 物理尺度是兩件事：SI 使用已確認縮放及 offset 的反射率，
不能使用經 mean/std 標準化的模型輸入。須先核對 Sen2Like 產物的波段定義與尺度，
並在各折 train 內分 S2／L8／L9 檢查指數分布及篩選保留率。
若決定分感測器估計共變異矩陣或配額，需記錄為額外適配，且不得用 test 選擇方案。

### 三折隔離與方法命名

原 Step 2 先合併所有 fold 的候選場景，再估計一份共變異矩陣。交叉驗證中，其他折
的 train／val 可能包含本折 test 場景，因此移植時必須改為逐折計算，避免測試標籤
經光譜統計影響本折背景選擇。場景油污參考、油污像素池、共變異矩陣及分位數門檻
都只能來自本折 train；本折 val／test 不參與探勘或 SI 統計。

既有 baseline 的 best checkpoint 曾用 val 選模，這是原訓練流程的一部分。
用它挖同折 train 屬於 **within-fold 探勘**，不能宣稱模型未見過這些訓練場景，
也不能以「大多 patch 位置沒看過」當成場景 OOF。若後續需要嚴格 OOF，應另外
在 outer train 內訓練按場景／事件分組的分割 probe，再對 held-out 群組評分。

### 標籤、配額與輸出

FP 必須在有效影像與已確認背景內計算。原程式以 `mask==1` 作為 FP 背景，
現有 OOF 候選檢查則把 `1/2` 計入標註區域，兩者不可默默視為相同。
第一版 SI 適配建議保留 `mask==1` 的 FP 定義，另行統計 `Others=2` 與未標註比例；
`255` 不可計為已確認 FP，`Others` 也不可直接當成海洋。
目前背景 loader 會產生整張背景答案，因此候選品質檢查必須在輸出前完成。
缺乏有效場景油污參考者，原法會排除；需記錄排除數量，不應靜默給距離零值。

原 `ratio=0.5` 是新增 HN 數量／油污 patch 數，原 Step 4 仍保留原名冊中的背景。
現有 `keep_fraction=0.5` 則是背景候選池保留率。GT-centric 名冊也不保證每張含油，
如需計算油污 patch 基數，必須依 mask 判定，不能直接使用整份名冊長度。

建議擴大合格候選池，再選回固定數量的 train 背景；若候選不足需明確報告，
不能無紀錄地減量、重複抽樣或換入未檢查背景。原先 3σ 排除整景的規則需在本資料
檢視其影響後再決定，不視為必備 HNM 規則。

每折產物建議包含：

- 候選及篩選紀錄：場景、座標、感測器、FP 數、未標註／nodata 比例、SI 距離與排除原因。
- 該折油污參考、共變異矩陣與實際分位數門檻。
- `selected_train_bg.tsv`：選中的 train 背景，欄位 `scene/x/y`。
- `bg_coords.tsv`：選中的 train 背景＋原樣保留的非 train 背景。
- `manifest.json`：符合現有接口的 fold、來源名冊雜湊、設定、產物雜湊與完成狀態；
  另記錄 HNM_SI 方法名稱、probe 權重／設定來源與感測器處理政策。

原 GT-centric train／val／test 名冊不改寫。原始碼放在 `references/`；
適配實作建議新增 `preprocess/hnm_si/`，沿用現有背景接口並擴充必要的來源檢查。

## 4. 實作與驗證順序

1. 建立設定與 preflight：路徑、三折場景、權重與架構、波段映射、尺度、mask 對齊。
2. 先完成小量 train window 的模型載入、FP 探勘、候選紀錄與背景輸出，驗證現有接口可讀取。
3. 加入逐折 train-only SI 統計與過濾，驗證波段映射、零分母、無油污參考及空候選情況。
4. 驗證無跨折統計污染、無重複座標、候選不足會明確處理、val／test 背景逐筆不變；
   檢查產物被修改或尚未完成時，訓練接口會拒絕使用。
5. 正式比較固定數量隨機背景、HNM、HNM＋SI。對照從相同的合格背景搜尋池出發，
   記錄每種選擇規則；固定 GT-centric 名冊、背景數量、seed 與 optimizer steps，
   先完成候選數量盤點再決定共同可行的配額。

評估沿用固定 test 與 GT-aware 場景重組口徑，並同時觀察誤報、precision、recall、
Oil IoU，避免只降低誤報卻漏掉更多真油污。SI 的 5% 門檻是原法起點，
不是本資料集已確認的最佳值；正式結果產生前不宣稱改善。

## 5. 實作狀態與用法（2026-10-17）

程式位於 `preprocess/hnm_si/`，與 `preprocess/hnm/`（EfficientNetV2-S＋OOF）互不相依。

| 檔案 | 內容 |
| --- | --- |
| `indices.py` | 波段名稱映射、三個光譜指數、模型輸入處理（與分割載入器一致）、有效像素規則 |
| `mining.py` | 單景掃描：合格 window、探勘模型 FP、FP 區域 SI 中位數、場景油污參考 |
| `probe.py` | 由各折基準 run 的 `args.yaml`＋`weights/best.pt` 嚴格載入探勘模型 |
| `stats.py` | Ledoit-Wolf 共變異、馬氏距離、分位數門檻、可選 3σ、場景均衡抽樣 |
| `pipeline.py` | CLI：`preflight`、`mine`、`select` 三階段與 manifest |
| `config.example.yaml` | 設定範本（路徑相對於該檔） |

### 操作

```bash
CFG=preprocess/hnm_si/config.example.yaml
# 1. 資料與模型檢查、候選窗數量盤點、各感測器反射率／SI 尺度抽查（不寫探勘產物）
.venv/bin/python -m preprocess.hnm_si.pipeline --config $CFG --stage preflight
# 2. 逐折探勘（可 --fold N；--max-scenes N 只做煙霧測試，結果為 partial）
.venv/bin/python -m preprocess.hnm_si.pipeline --config $CFG --stage mine
# 3. 套用過濾並輸出背景名冊；--arm 為 hnm_si／hnm／random
.venv/bin/python -m preprocess.hnm_si.pipeline --config $CFG --stage select --arm hnm_si
```

`mine` 每景寫一組檔案，中斷後重跑會跳過已完成場景；若設定、名冊或權重與原先不同則拒絕混用。
`select` 不重跑模型，可針對同一份探勘結果輸出三種對照：`hnm_si`（FP＋SI 過濾）、
`hnm`（只用 FP）、`random`（同一批合格窗內的隨機對照），因此三者的搜尋池相同。
`top_q` 與 `min_oil_pool` 屬於選取階段設定，修改後不需重新探勘。

輸出在 `hnm_runs/hnm_si_v1/`：探勘結果在 `mining/fold_N/`，各 arm 的最終產物在
`arms/<arm>/fold_N/`（`selected_train_bg.tsv`、`bg_coords.tsv`、`candidates_scored.tsv`、
`si_reference.json`、`selection_report.json`、最後寫入的 `manifest.json`）。
訓練時把實驗 YAML 的 `hnm.artifacts_dir` 指向 `hnm_runs/hnm_si_v1/arms/<arm>`。
`main/hnm_config.py` 除原有檢查外，對 HNM_SI 產物另核對候選紀錄雜湊、選取結果與紀錄一致，
並拒絕 `partial` 狀態。

### 與原法不同、需知道的適配

- **反射率 offset（本機資料實測）：** 三種感測器的海面像素最小值都停在 DN 1001，表示交付影像有
  +1000 的加性 offset。光譜指數使用 `(DN − 1000) / 10000`（`si.reflectance_offset`）；
  分割模型輸入維持 `DN / 10000`，與基準訓練一致。未扣 offset 會讓三個指數整體偏移。
- **FP 區域極低反射率：** 扣 offset 後海面像素反射率很小，`SR_B1B2_B3B8A` 分母小，
  數值常被 ±10 的 clip 截斷（煙霧測試中 FP 區域中位數常停在 10）。原程式有同一個 clip，
  目前保留；正式探勘後請檢視該指數的飽和比例再決定是否調整。
- **無效像素不補零：** 原程式把 NaN／無窮大補 0；本實作標為無效並排除，不會產生假的指數值。
- **逐折 train-only：** 只掃本折 train 場景；油污參考、共變異矩陣與分位數門檻都只來自這些場景。
- **探勘不改基準名冊：** 候選排除與 GT-centric train 小圖座標完全重複者；選取數量預設等於
  該折基準 train 背景數（`selection.target_count: baseline`），候選不足時預設中止並寫出報告，
  `allow_shortfall: true` 才會接受較少數量並記錄在 manifest。
- **感測器：** 共變異矩陣為全感測器合併一份（與原法一致）；`selection_report.json` 依 S2／L8／L9
  列出各狀態計數，可檢視保留率是否偏向某個感測器。
- **探勘範圍：** 只掃 train（原法掃 train＋val）；val 背景維持基準名冊。這是 within-fold 探勘，
  `best.pt` 曾以該折 val 選模，不能宣稱是場景層級 OOF。

### 驗證現況

- `tests/test_hnm_si.py`：指數公式與無效像素、模型輸入與載入器一致、窗口規則、統計函式、
  三個 arm 的端到端流程、訓練接口接受與拒絕（竄改、partial、候選不足、名冊變更）。
- 真實資料煙霧測試（fold 1 前 3 個 train 場景、真實 Prithvi＋DeepLabV3 探勘模型）：
  權重嚴格載入成功，約每景 12 秒、3,968 個窗，三個 arm 皆可輸出。完整 fold 1 約 27.5 萬個
  合格窗，粗估探勘 1 小時內，三折合計數小時（GPU）。
- 完整三折探勘已於 2026-10-17 完成（每折約 27 萬合格窗、9 千～1.25 萬個有 FP），配額足夠，見 [RUN_LOG.md](RUN_LOG.md)。

### 匯出被選中的圖片（給人檢視）

`select` 完成後，可把實際餵入訓練的背景小圖輸出成資料夾：

```bash
.venv/bin/python -m preprocess.export_hnm_selection \
  --fold-dir hnm_runs/hnm_si_v1/arms/hnm_si/fold_1
```

預設寫到該折目錄下的 `preview/`（可用 `--out` 指定，`--limit N` 只匯出前 N 張，
`--no-individual` 只出總覽圖）。內容：

- `images/`：每張一個 PNG，左為真彩色（B04/B03/B02，反射率扣 offset 後固定拉伸），
  右為同一張疊上遮罩（洋紅＝未標註 255、黃＝Others 2）。被選中的小圖整張都會當背景訓練，
  所以染色區域並非確認的海面。
- `sheets/`：每頁 5×4 的總覽，標題含編號、日期＿感測器＿tile、座標、`n_fp`、`d_si`。
- `index.csv`：完整場景名、座標、遮罩組成與各項分數；`README.txt`：來源與雜湊。

同一支腳本也能讀 EfficientNet OOF 版的 `hnm_runs/oof_v1/fold_N`（分數欄位為 `el2n`）。
只匯出 `status: complete` 且名冊雜湊與 manifest 相符的產物。

