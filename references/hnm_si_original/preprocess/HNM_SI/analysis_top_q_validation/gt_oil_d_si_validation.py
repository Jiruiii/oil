"""
GT oil patches 的 d_si 分佈驗證 — 為 step3 `top_q=0.05` 提供 GT 支撐證據。

流程:
  1. 讀 v4 output 的 Σ_LW^-1 + scene f_oil + FP candidate d_si
  2. 對 list_oil.txt 內的 703 個 GT oil patches，逐 patch 算 p_si（median SI over GT oil pixels）
     → d_si = sqrt((p_si - f_oil)^T Σ_LW^-1 (p_si - f_oil))
  3. 疊圖 GT oil d_si vs FP candidate d_si；求 GT Q95 / Q99 對應 FP 分佈的哪個 quantile

輸出:
  gt_oil_d_si.csv
  overlay_histogram_global.png
  overlay_histogram_per_fold.png
  quantile_mapping.txt

Usage:
  conda run -n yolo11 python gt_oil_d_si_validation.py
"""
from collections import OrderedDict
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

# ===== CONFIG =====
V4_DIR = Path('/mnt/backups/sun/OIL_PROJECT_MutiBand_0422_VRT_training/result-seg/cfhnm_within_si_analysis_v4_full')
STACKED_TIF_DIR = Path('/home2/sun/OIL_PROJECT_dataset/full_band/processed/stack_tif')
MASK_DIR = Path('/home2/sun/OIL_PROJECT_dataset/full_band/mask')
LIST_OIL_TXT = Path('/mnt/backups/sun/OIL_PROJECT_dataset/full_band/data_split_MS5/5_fold/patch_level_GB1.0/list_oil.txt')
OUT_DIR = Path('/mnt/backups/sun/OIL_PROJECT_MutiBand_thesis/preprocess/HNM_SI/analysis_top_q_validation')

MIN_OIL_PIXELS = 100      # 對齊 step2 MIN_FP_PIXELS_FOR_SI
PATCH_SIZE = 256
EPS = 1e-8
MASK_CACHE_MAX = 8

# 11-band 升序 index (跟 step2 一致)
B1, B2, B3 = 0, 1, 2
B8A = 8
B11 = 9


def compute_si_per_pixel(arr: np.ndarray) -> np.ndarray:
    """arr: (11, H, W) float32 → (3, H, W) [MNDWI, NDI_B1_B11, SR_B1B2_B3B8A]."""
    mndwi = (arr[B3] - arr[B11]) / (arr[B3] + arr[B11] + EPS)
    ndi_b1_b11 = (arr[B1] - arr[B11]) / (arr[B1] + arr[B11] + EPS)
    sr = (arr[B1] + arr[B2]) / (arr[B3] + arr[B8A] + EPS)
    si = np.stack([mndwi, ndi_b1_b11, sr], axis=0).astype(np.float32)
    si = np.nan_to_num(si, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(si, -10.0, 10.0)


def valid_pixel_mask(arr: np.ndarray) -> np.ndarray:
    return np.abs(arr).sum(axis=0) > 1e-5


COORD_RE = re.compile(r'^(.+)_patch_x(\d+)_y(\d+)$')


def parse_patch_id(pid: str):
    m = COORD_RE.match(Path(pid).stem)
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load v4 outputs (reuse Σ, f_oil, FP d_si) ----
    print(f'[1/5] Load v4 artifacts from {V4_DIR}')
    npz = np.load(V4_DIR / 'mahalanobis_matrices.npz')
    cov_inv = npz['cov_inv']            # (3,3)
    print(f'  cov_inv shape={cov_inv.shape}, ledoit_wolf_alpha={float(npz["ledoit_wolf_alpha"]):.4f}')

    fp_df_all = pd.read_csv(V4_DIR / 'scene_oil_fingerprint.csv')
    fp_valid = fp_df_all[fp_df_all['oil_fingerprint_valid']].reset_index(drop=True)
    fp_lookup = {r['scene']: (r['oil_mndwi_med'], r['oil_ndi_med'], r['oil_sr_med'])
                 for _, r in fp_valid.iterrows()}
    print(f'  scene fingerprints: {len(fp_lookup)} valid scenes')

    fp_cand = pd.read_csv(V4_DIR / 'si_distance_per_candidate.csv')
    fp_valid_mask = fp_cand['fingerprint_valid'] == True
    fp_cand_valid = fp_cand[fp_valid_mask & fp_cand['d_si'].notna()].reset_index(drop=True)
    print(f'  FP candidates with valid d_si: {len(fp_cand_valid):,}')

    # ---- Load GT oil patch list ----
    print(f'\n[2/5] Load GT oil patches from {LIST_OIL_TXT}')
    lines = [l.strip() for l in LIST_OIL_TXT.read_text().splitlines() if l.strip()]
    patches = []
    for l in lines:
        info = parse_patch_id(l)
        if info is None:
            continue
        scene, x, y = info
        patches.append({'patch_id': Path(l).stem, 'scene': scene, 'x': x, 'y': y})
    print(f'  {len(patches)} patches parsed')

    # Filter to patches whose scene has valid fingerprint
    patches = [p for p in patches if p['scene'] in fp_lookup]
    print(f'  {len(patches)} patches with scene fingerprint available')

    # ---- Group by scene + per-scene mask cache (LRU) ----
    from collections import defaultdict
    by_scene = defaultdict(list)
    for p in patches:
        by_scene[p['scene']].append(p)
    print(f'  spanning {len(by_scene)} unique scenes')

    # ---- Compute d_si per GT oil patch ----
    print(f'\n[3/5] Compute p_si + d_si per patch (median over GT oil pixels)')
    rows = []
    n_skipped_low_oil = 0
    n_skipped_read = 0

    for scene, plist in tqdm(sorted(by_scene.items()), desc='  scenes'):
        tif_path = STACKED_TIF_DIR / f'{scene}.tif'
        mask_path = MASK_DIR / f'{scene}.tif'
        if not (tif_path.exists() and mask_path.exists()):
            n_skipped_read += len(plist)
            continue
        try:
            with rasterio.open(str(mask_path)) as msrc:
                full_mask = msrc.read(1)
            src = rasterio.open(str(tif_path))
        except Exception:
            n_skipped_read += len(plist)
            continue

        f_oil = np.array(fp_lookup[scene], dtype=np.float32)  # (3,)
        H, W = full_mask.shape

        try:
            for p in plist:
                x, y = p['x'], p['y']
                try:
                    arr = src.read(window=Window(x, y, PATCH_SIZE, PATCH_SIZE),
                                   boundless=True, fill_value=0).astype(np.float32)
                except Exception:
                    n_skipped_read += 1
                    continue
                np.nan_to_num(arr, nan=0.0, copy=False)
                valid = valid_pixel_mask(arr)

                # Crop mask window (bounds-safe, 預設 bg=1 padding)
                y0, y1 = y, min(y + PATCH_SIZE, H)
                x0, x1 = x, min(x + PATCH_SIZE, W)
                mask_patch = np.full((PATCH_SIZE, PATCH_SIZE), 1, dtype=full_mask.dtype)
                mask_patch[:y1 - y0, :x1 - x0] = full_mask[y0:y1, x0:x1]

                oil_mask = (mask_patch == 0) & valid
                n_oil = int(oil_mask.sum())
                if n_oil < MIN_OIL_PIXELS:
                    n_skipped_low_oil += 1
                    continue

                si = compute_si_per_pixel(arr)  # (3, H, W)
                p_si = np.array([np.median(si[0][oil_mask]),
                                 np.median(si[1][oil_mask]),
                                 np.median(si[2][oil_mask])], dtype=np.float32)

                diff = p_si - f_oil
                d_sq = float(diff @ cov_inv @ diff)
                d_si = float(np.sqrt(max(d_sq, 0.0)))

                rows.append({
                    'patch_id': p['patch_id'],
                    'scene': scene, 'x': x, 'y': y,
                    'n_oil_px': n_oil,
                    'p_mndwi': float(p_si[0]),
                    'p_ndi':   float(p_si[1]),
                    'p_sr':    float(p_si[2]),
                    'f_mndwi': float(f_oil[0]),
                    'f_ndi':   float(f_oil[1]),
                    'f_sr':    float(f_oil[2]),
                    'd_si': d_si,
                })
        finally:
            try: src.close()
            except Exception: pass

    gt_df = pd.DataFrame(rows)
    gt_df.to_csv(OUT_DIR / 'gt_oil_d_si.csv', index=False)
    print(f'\n  computed {len(gt_df):,} GT oil d_si  '
          f'(skipped: low_oil={n_skipped_low_oil}, read_error={n_skipped_read})')

    # ---- Quantile mapping ----
    print(f'\n[4/5] Quantile mapping')
    gt_d = gt_df['d_si'].values
    fp_d = fp_cand_valid['d_si'].values

    def q_of_value_in(arr, v):
        """value → quantile 位置 (0..1) in arr."""
        return float((arr < v).mean())

    lines_out = []
    lines_out.append('=== GT-oil-supported top_q validation ===')
    lines_out.append(f'GT oil patches: n={len(gt_d):,}')
    lines_out.append(f'FP candidates:  n={len(fp_d):,}')
    lines_out.append('')

    lines_out.append('--- GT oil d_si quantiles ---')
    for q in [0.50, 0.75, 0.90, 0.95, 0.99]:
        v = float(np.quantile(gt_d, q))
        lines_out.append(f'  Q{int(q*100):02d}: d={v:.4f}   → in FP dist ≈ Q{q_of_value_in(fp_d, v)*100:5.2f}%')
    lines_out.append('')

    lines_out.append('--- FP candidate d_si quantiles ---')
    for q in [0.01, 0.02, 0.05, 0.10, 0.25, 0.50]:
        v = float(np.quantile(fp_d, q))
        lines_out.append(f'  Q{int(q*100):02d}: d={v:.4f}   → in GT dist ≈ Q{q_of_value_in(gt_d, v)*100:5.2f}%')
    lines_out.append('')

    # per-fold
    lines_out.append('--- Per-fold FP quantile of GT Q95 / Q99 ---')
    gt_q95 = float(np.quantile(gt_d, 0.95))
    gt_q99 = float(np.quantile(gt_d, 0.99))
    lines_out.append(f'GT Q95={gt_q95:.4f}  GT Q99={gt_q99:.4f}')
    for fold in sorted(fp_cand_valid['fold'].unique()):
        sub = fp_cand_valid[fp_cand_valid['fold'] == fold]['d_si'].values
        q95_pos = q_of_value_in(sub, gt_q95)
        q99_pos = q_of_value_in(sub, gt_q99)
        lines_out.append(f'  fold {fold} (n={len(sub):,}): '
                         f'GT Q95 → FP Q{q95_pos*100:5.2f}%, '
                         f'GT Q99 → FP Q{q99_pos*100:5.2f}%')
    lines_out.append('')

    # Verdict
    fp_pct_at_gt_q95 = q_of_value_in(fp_d, gt_q95) * 100
    lines_out.append(f'--- Verdict ---')
    lines_out.append(f'Current top_q = 5.00%')
    lines_out.append(f'GT oil Q95 falls at FP percentile ≈ {fp_pct_at_gt_q95:.2f}%')
    if fp_pct_at_gt_q95 < 3:
        lines_out.append(f'→ 5% cutoff is more aggressive than GT evidence supports (may over-cut)')
    elif fp_pct_at_gt_q95 < 8:
        lines_out.append(f'→ 5% cutoff is well-supported by GT oil coverage')
    else:
        lines_out.append(f'→ 5% cutoff is conservative (could raise to ~{fp_pct_at_gt_q95:.1f}% for tighter coverage)')

    txt = '\n'.join(lines_out)
    (OUT_DIR / 'quantile_mapping.txt').write_text(txt)
    print('\n' + txt)

    # ---- Plots ----
    print(f'\n[5/5] Plots')
    # Global overlay (density) — full range + zoomed inset
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, xlim, title_suffix in [
        (axes[0], None, '(full range)'),
        (axes[1], (0, max(gt_d.max() * 1.1, np.quantile(fp_d, 0.75) * 1.5)),
         '(zoomed: 0 → GT max × 1.1)'),
    ]:
        if xlim is None:
            bins = np.linspace(0, max(gt_d.max(), fp_d.max()) * 1.02, 80)
        else:
            bins = np.linspace(0, xlim[1], 60)
        ax.hist(fp_d, bins=bins, alpha=0.55, color='steelblue',
                density=True, label=f'FP candidates (n={len(fp_d):,})', edgecolor='none')
        ax.hist(gt_d, bins=bins, alpha=0.65, color='seagreen',
                density=True, label=f'GT oil patches (n={len(gt_d):,})', edgecolor='none')
        ax.axvline(np.quantile(fp_d, 0.05), color='red', linestyle='--', linewidth=1.5,
                   label=f'FP Q05 = {np.quantile(fp_d, 0.05):.3f} (current 5% cutoff)')
        ax.axvline(gt_q95, color='orange', linestyle='--', linewidth=1.5,
                   label=f'GT Q95 = {gt_q95:.3f}')
        ax.axvline(gt_q99, color='darkorange', linestyle=':', linewidth=1.5,
                   label=f'GT Q99 = {gt_q99:.3f}')
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.set_xlabel('Mahalanobis SI distance d_si')
        ax.set_ylabel('density')
        ax.set_title(f'GT oil vs FP candidate {title_suffix}')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'overlay_histogram_global.png', dpi=140, bbox_inches='tight')
    plt.close()

    # Per-fold
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, fold in enumerate(sorted(fp_cand_valid['fold'].unique())):
        ax = axes[i]
        sub = fp_cand_valid[fp_cand_valid['fold'] == fold]['d_si'].values
        binf = np.linspace(0, max(gt_d.max(), sub.max()) * 1.02, 50)
        ax.hist(sub, bins=binf, alpha=0.55, color='steelblue', density=True,
                label=f'FP fold{fold} (n={len(sub):,})')
        ax.hist(gt_d, bins=binf, alpha=0.65, color='seagreen', density=True,
                label=f'GT oil (n={len(gt_d):,})')
        ax.axvline(np.quantile(sub, 0.05), color='red', linestyle='--',
                   label=f'FP Q05={np.quantile(sub, 0.05):.2f}')
        ax.axvline(gt_q95, color='orange', linestyle='--',
                   label=f'GT Q95={gt_q95:.2f}')
        ax.set_title(f'fold {fold}')
        ax.set_xlabel('d_si')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    # last axis: global overlay
    ax = axes[5]
    binf = np.linspace(0, max(gt_d.max(), fp_d.max()) * 1.02, 50)
    ax.hist(fp_d, bins=binf, alpha=0.55, color='steelblue', density=True,
            label=f'FP all (n={len(fp_d):,})')
    ax.hist(gt_d, bins=binf, alpha=0.65, color='seagreen', density=True,
            label=f'GT oil (n={len(gt_d):,})')
    ax.axvline(np.quantile(fp_d, 0.05), color='red', linestyle='--',
               label=f'FP Q05={np.quantile(fp_d, 0.05):.2f}')
    ax.axvline(gt_q95, color='orange', linestyle='--',
               label=f'GT Q95={gt_q95:.2f}')
    ax.set_title('ALL folds')
    ax.set_xlabel('d_si')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'overlay_histogram_per_fold.png', dpi=130, bbox_inches='tight')
    plt.close()

    print(f'\n[Done] outputs at {OUT_DIR}')


if __name__ == '__main__':
    main()
