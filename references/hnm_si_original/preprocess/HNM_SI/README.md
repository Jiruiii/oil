# HNM_SI — Hard Negative Mining with Spectral-Index Filtering

四步 pipeline，產出最終訓練用的 train/val patch split：

```
step1_mine_fp.py        → cfhnm_within_mined_foldN.txt          (FP candidates per fold)
        ↓
step2_si_distance.py    → si_distance_per_candidate.csv         (Ledoit-Wolf 馬氏 d_si)
                          mahalanobis_matrices.npz
                          scene_oil_fingerprint.csv
        ↓
step3_si_filter.py      → cfhnm_within_si_filtered_foldN.txt    (top-5% cutoff)
        ↓
step4_assemble.py       → train_fold{1..5}.txt / val_fold{1..5}.txt    (final split)
                          cfhnm_within_si_metadata.csv
```

---

## 設計理由（為何選這三個 SI 指數）

FP contrast 分析（`docs/SPECTRAL_ANALYSIS_LOG.md`）：

| 對比 | MNDWI | NDI_B1_B11 | SR_B12_B3B8A | FAI | OSI |
|---|---|---|---|---|---|
| TP vs Global BG | **+0.69 (medium)** | **+0.65 (medium)** | **+0.60 (medium)** | negligible | -0.34 (small) |
| FP vs TP | -0.30 (small) | -0.29 (small) | -0.31 (small) | negligible | +0.25 (small, 反向) |
| FP vs Local BG | negligible | negligible | negligible | negligible | negligible |

- **選用三指數**：MNDWI / NDI_B1_B11 / SR_B12_B3B8A — TP vs BG 為 medium effect，能定義 oil signature
- **FP 介於 TP 與 BG**：FP-vs-TP small effect → SI 上 FP 是「弱油」、TP 是「強油」
- **FAI / OSI 不用**：FAI 全 negligible；OSI 反向（FP > TP），物理直覺不合

---

## 三個 SI 公式

11-band 升序索引：`B1=443 B2=492 B3=560 B4=665 B5=704 B6=740 B7=783 B8=833 B8A=865 B11=1614 B12=2202`

```
MNDWI         = (B3 - B11) / (B3 + B11)
NDI_B1_B11    = (B1 - B11) / (B1 + B11)
SR_B12_B3B8A  = (B1 + B2)  / (B3 + B8A)
```

---

## Step 1 — `step1_mine_fp.py`：挖 FP candidates（per fold）

對每個 fold N (1..5)：

1. Load fold N's DeepLab baseline `best.pt` + 該 fold 算的 mean/std
2. Sliding-window inference over fold N 的 **train + val** scenes
3. Patch-level filter（FP px count ≥ 1、cloud ratio 上限等）
4. 通過條件的 patch 寫入 `cfhnm_within_mined_foldN.txt`，每行：
   ```
   <scene>_patch_x<x>_y<y>\t<role>\t<fp_px>\t<mask255_px>\t<mean_prob>\t<max_prob>
   ```

**Within-fold 設計**：不用 fold N 的 test scenes（避免 leakage）。Model 對 train 看過 ~5% patches，
對 val 完全未訓練 → mining 仍 honest，因為 patch positions 95%+ 未見過。

---

## Step 2 — `step2_si_distance.py`：算 Ledoit-Wolf 馬氏 d_si

**核心數學**：

1. **Per-scene oil fingerprint** `f_oil(s) ∈ R^3`：scene 內所有 mask==0 oil pixels 的
   `[MNDWI, NDI_B1_B11, SR_B12_B3B8A]` median 向量
2. **Per-candidate FP-region p_si** `∈ R^3`：candidate patch 跑 HNM probe → 找 FP region
   (`pred_oil_prob > 0.5 ∩ mask == 1`)，**FP region ≥ 100 px 才算**（少於 100 → 跳過該 patch）
3. **Oil-pixel pool Σ via Ledoit-Wolf**：從每 valid scene 抽 1000 油污像素 → pool，
   `Σ_LW = LedoitWolf().fit(oil_pool).covariance_`（sample cov + optimal shrinkage）
4. **Mahalanobis distance**：
   ```
   d_si(c, s) = sqrt( (p_si - f_oil)^T Σ_LW⁻¹ (p_si - f_oil) )
   ```

**v4 改動 vs v3**：
- 距離從 standardized Euclidean 升 Mahalanobis（含 off-diagonal）
- Σ 估計用 Ledoit-Wolf optimal shrinkage（textbook 標準，無需 hyperparameter 辯護）
- 存 `mahalanobis_matrices.npz` 供 appendix

**輸出**：
- `si_distance_per_candidate.csv` — 每 candidate 的 d_si + 3 個 SI 原始值
- `scene_oil_fingerprint.csv` — per-scene f_oil
- `mahalanobis_matrices.npz` — Σ_sample / Σ_LW / Σ_LW⁻¹ / α / condition number
- `normalization_stats.csv` — per-dim σ + inter-dim correlation
- `dist_histogram_per_fold.png`
- `decision_summary.txt`

---

## Step 3 — `step3_si_filter.py`：套 SI cutoff

讀 `si_distance_per_candidate.csv`，per-fold quantile 套兩道閾值：

| Cutoff | 預設 | 作用 |
|---|---|---|
| `--top_q 0.05` | 開（5%） | 剔除距離最小（最像油）→ 防漏標油汙染 HN pool |
| `--bot_q 0`    | **關**   | 剔除距離最大（最不像油）— ⚠ FP-LocalBG ≈ 0 顯示多數 FP 是空間紋理錯，SI 上跟 BG 同質；砍掉會失去 spatial-pattern 訊號 |

第一刀其實已經發生在 Step 2：`fingerprint_valid=False`（n_fp < 100 或 scene oil fingerprint 不可算）
全被剔除。Step 3 的 top-5% 是第二刀。

**輸出**：
- `cfhnm_within_si_filtered_foldN.txt`（每行：patch_id, role, n_fp_px, n_mask255_px, mean_prob, max_prob, d_si）
- `cfhnm_within_si_filter_summary.csv`

---

## Step 4 — `step4_assemble.py`：組 final split

從 SI-filtered pool 做 stratified sampling（per-scene 均勻 + 3σ outlier scene 剔除），
照 `--ratio` (HN to oil) 採樣 → 跟 oil patch list 合併寫成 `train_foldN.txt` / `val_foldN.txt`。

**預設 ratio=0.5** 對齊 rand050 baseline 量級，做 apples-to-apples ablation。

**輸出**（在 `--out_split_dir`）：
- `train_fold{1..5}.txt` / `val_fold{1..5}.txt`
- `cfhnm_within_si_metadata.csv`（哪些 oil patch、哪些 HN patch、來自哪 scene、SI 值多少）

---

## Probe model paths

`step1_mine_fp.py` 跟 `step2_si_distance.py` 內部 hardcoded 的 `FOLD_BEST_PT` dict 指向 fold-N
DeepLab baseline 的 `best.pt`。換環境要改這 5 條路徑（在 step1 / step2 各一份，要對齊）。
