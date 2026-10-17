"""
HNM_SI Step 2 — SI 距離分布分析（v4: FP-region + LW-Mahalanobis）。

對 step1_mine_fp.py 輸出的 candidates，計算：

  1. Per-scene oil fingerprint  f_oil(s) ∈ R^3:
     scene 內所有 mask==0 pixels (ground truth oil) 的
     [MNDWI, NDI_B1_B11, SR_B1B2_B3B8A] **median** 向量

  2. Per-candidate FP-region p_si:
     a. 跑 HNM probe (FOLD_BEST_PT[fold]) → pred_oil_prob
     b. FP mask = (pred_oil_prob > 0.5) & (label_mask == 1) & valid
     c. 跳過 FP region < MIN_FP_PIXELS_FOR_SI (預設 100)
     d. p_si(c) = median over FP region pixels (3-dim)

  3. Oil-pixel pool Σ via Ledoit-Wolf:
     從每 valid scene 抽 OIL_POOL_SAMPLES_PER_SCENE (預設 1000) 油污像素 → pool
     Σ_LW = LedoitWolf().fit(oil_pool).covariance_  (sample cov + optimal shrinkage)
     物理意義：油污 SI 的整體共變異結構（含 off-diagonal）

  4. Mahalanobis distance:
     d_si(c, s) = sqrt( (p_si - f_oil)^T Σ_LW^-1 (p_si - f_oil) )

  5. 過濾：剔除 d_si 最小的 5%（光譜太像油 → 疑似漏標，不適合當 HN）

v4 改動 vs v3:
  - 距離從 standardized Euclidean 升 Mahalanobis（含 off-diagonal）
  - Σ 估計用 Ledoit-Wolf optimal shrinkage（自動選 λ，textbook 標準）
  - 不需 hyperparameter 辯護
  - 存 mahalanobis_matrices.npz 供 paper appendix

v3 改動 vs v2:
  - σ 來自油污像素 pool
  - mask_cache LRU 限制（防 OOM）
  - load_state_dict 檢查 missing/unexpected（防 silent fail）
  - cudnn.deterministic=True（reproducibility）

輸出:
  si_distance_per_candidate.csv
  scene_oil_fingerprint.csv
  normalization_stats.csv        (per-dim σ + inter-dim correlation)
  dist_histogram_per_fold.png
  dist_histogram_global.png
  decision_summary.txt

Usage:
  conda run -n yolo11 python step2_si_distance.py \
      --step1_dir <step1_mine_fp.py output dir> \
      --out_dir   <SI analysis output dir> \
      --folds 1 2 3 4 5
"""
from collections import OrderedDict
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from rasterio.windows import Window
from sklearn.covariance import LedoitWolf
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm

EPS = 1e-8
PATCH_SIZE = 256

STACKED_TIF_DIR = Path('/home2/sun/OIL_PROJECT_dataset/full_band/processed/stack_tif')
MASK_DIR = Path('/home2/sun/OIL_PROJECT_dataset/full_band/mask')
PATCH_LEVEL_DIR = Path('/mnt/backups/sun/OIL_PROJECT_dataset/full_band/data_split_MS5/5_fold/patch_level_GB1.0')

# HNM baseline probe per fold (與 step1_mine_fp.py 對齊)
FOLD_BEST_PT = {
    1: '/mnt/backups/sun/OIL_PROJECT_MutiBand_0422_VRT_training/result-seg/CV_MS5/20260512-231439_MS5_CV_deeplabv3+_GB1.0_baseline_fold1/weights/best.pt',
    2: '/mnt/backups/sun/OIL_PROJECT_MutiBand_0422_VRT_training/result-seg/CV_MS5/20260513-173838_MS5_CV_deeplabv3+_GB1.0_baseline_fold2/weights/best.pt',
    3: '/mnt/backups/sun/OIL_PROJECT_MutiBand_0422_VRT_training/result-seg/CV_MS5/20260514-013955_MS5_CV_deeplabv3+_GB1.0_baseline_fold3/weights/best.pt',
    4: '/mnt/backups/sun/OIL_PROJECT_MutiBand_0422_VRT_training/result-seg/CV_MS5_rerun24/20260516-171513_MS5_CV_deeplabv3+_GB1.0_baseline_rerun_fold4/weights/best.pt',
    5: '/mnt/backups/sun/OIL_PROJECT_MutiBand_0422_VRT_training/result-seg/CV_MS5/20260514-034340_MS5_CV_deeplabv3+_GB1.0_baseline_fold5/weights/best.pt',
}

MIN_FP_PIXELS_FOR_SI = 100  # min FP pixels to compute stable median SI (對齊 fingerprint 100)
PROB_THRESHOLD = 0.5        # FP threshold (same as HNM mining)
OIL_POOL_SAMPLES_PER_SCENE = 1000  # oil pixels sampled per valid scene for σ estimation
MASK_CACHE_MAX = 8          # LRU bound on full-scene mask cache (防 OOM)
OIL_POOL_RNG_SEED = 42      # reproducibility for oil pixel sampling

# 11-band 升序：B1(443) B2(492) B3(560) B4(665) B5(704) B6(740) B7(783)
#               B8(833) B8A(865) B11(1614) B12(2202)
B1, B2, B3, B4 = 0, 1, 2, 3
B8, B8A = 7, 8
B11, B12 = 9, 10


def compute_si_per_pixel(arr: np.ndarray) -> np.ndarray:
    """arr: (11, H, W) float32. Return (3, H, W) SI maps: MNDWI, NDI_B1_B11, SR_B1B2_B3B8A."""
    mndwi = (arr[B3] - arr[B11]) / (arr[B3] + arr[B11] + EPS)
    ndi_b1_b11 = (arr[B1] - arr[B11]) / (arr[B1] + arr[B11] + EPS)
    sr_b1b2_b3b8a = (arr[B1] + arr[B2]) / (arr[B3] + arr[B8A] + EPS)
    si = np.stack([mndwi, ndi_b1_b11, sr_b1b2_b3b8a], axis=0).astype(np.float32)
    si = np.nan_to_num(si, nan=0.0, posinf=0.0, neginf=0.0)
    si = np.clip(si, -10.0, 10.0)
    return si


def valid_pixel_mask(arr: np.ndarray) -> np.ndarray:
    """arr: (11, H, W). 排 nodata pixels (sum-abs-bands < 1e-5)."""
    return np.abs(arr).sum(axis=0) > 1e-5


# ===== Probe model loading (跟 step1_mine_fp.py 同) =====
def build_model(in_channels=11, num_classes=2, device='cuda'):
    model = deeplabv3_resnet50(weights=None)
    if in_channels != 3:
        old = model.backbone.conv1
        model.backbone.conv1 = nn.Conv2d(in_channels, old.out_channels,
                                         kernel_size=old.kernel_size,
                                         stride=old.stride, padding=old.padding,
                                         bias=old.bias is not None)
    model.classifier[4] = nn.Conv2d(model.classifier[4].in_channels, num_classes, kernel_size=1)
    if model.aux_classifier:
        model.aux_classifier[4] = nn.Conv2d(model.aux_classifier[4].in_channels, num_classes, kernel_size=1)
    return model.to(device)


def load_mean_std(fold: int):
    """從 fold N's train patches 抽樣 500 patches 算 mean/std（11 band）。"""
    train_txt = PATCH_LEVEL_DIR / f'train_fold{fold}.txt'
    coord_re = re.compile(r'^(.+)_patch_x(\d+)_y(\d+)$')
    lines = [l.strip() for l in train_txt.read_text().splitlines() if l.strip()]
    coords = []
    for l in lines:
        stem = Path(l).stem
        m = coord_re.match(stem)
        if m:
            coords.append({'scene': m.group(1), 'x': int(m.group(2)), 'y': int(m.group(3))})
    rng = np.random.default_rng(42)
    sample_n = min(500, len(coords))
    idx = rng.choice(len(coords), size=sample_n, replace=False)
    samples = [coords[i] for i in idx]
    sum_px = np.zeros(11, dtype=np.float64)
    sum_sq = np.zeros(11, dtype=np.float64)
    n_pix = 0
    for c in tqdm(samples, desc=f'  [mean/std fold{fold}]', leave=False):
        tif = STACKED_TIF_DIR / f"{c['scene']}.tif"
        try:
            with rasterio.open(str(tif)) as src:
                p = src.read(window=Window(c['x'], c['y'], 256, 256),
                             boundless=True, fill_value=0).astype(np.float32)
            p = np.nan_to_num(p, nan=0.0)
            v = p.reshape(11, -1)
            sum_px += v.sum(axis=1)
            sum_sq += (v ** 2).sum(axis=1)
            n_pix += v.shape[1]
        except Exception:
            pass
    mean = (sum_px / n_pix)
    var = (sum_sq / n_pix) - mean ** 2
    var = np.clip(var, 1e-12, None)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


# ===== Per-scene oil fingerprint + oil pixel sample =====
def compute_scene_oil_fingerprint(scene: str, oil_pool_rng: np.random.Generator):
    """讀 scene 整張 stacked TIF + mask，從 mask==0 pixels 算 SI median (= f_oil)
    並抽 OIL_POOL_SAMPLES_PER_SCENE 個 oil pixels 的 SI 向量回傳（for σ estimation pool）。

    回傳 (info_dict, oil_pixel_si: (N, 3) array or None)
    """
    tif_path = STACKED_TIF_DIR / f'{scene}.tif'
    mask_path = MASK_DIR / f'{scene}.tif'
    if not (tif_path.exists() and mask_path.exists()):
        return None, None
    try:
        with rasterio.open(str(mask_path)) as msrc:
            mask = msrc.read(1)
        oil_mask = (mask == 0)
        n_oil = int(oil_mask.sum())
        if n_oil < 100:
            return ({'scene': scene, 'n_oil_px': n_oil, 'oil_fingerprint_valid': False,
                     'oil_mndwi_med': np.nan, 'oil_ndi_med': np.nan, 'oil_sr_med': np.nan},
                    None)
        with rasterio.open(str(tif_path)) as src:
            arr = src.read().astype(np.float32)  # (11, H, W)
        np.nan_to_num(arr, nan=0.0, copy=False)
        valid = valid_pixel_mask(arr) & oil_mask
        n_valid = int(valid.sum())
        if n_valid < 100:
            return ({'scene': scene, 'n_oil_px': n_oil, 'oil_fingerprint_valid': False,
                     'oil_mndwi_med': np.nan, 'oil_ndi_med': np.nan, 'oil_sr_med': np.nan},
                    None)
        si = compute_si_per_pixel(arr)  # (3, H, W)
        # f_oil: median per dim over all valid oil pixels
        mndwi_med = float(np.median(si[0][valid]))
        ndi_med = float(np.median(si[1][valid]))
        sr_med = float(np.median(si[2][valid]))
        # Random sample of oil pixel SI vectors for pool σ estimation
        ys, xs = np.where(valid)
        n_sample = min(OIL_POOL_SAMPLES_PER_SCENE, len(ys))
        sample_idx = oil_pool_rng.choice(len(ys), size=n_sample, replace=False)
        oil_pixel_si = si[:, ys[sample_idx], xs[sample_idx]].T.astype(np.float32)  # (n_sample, 3)
        return ({'scene': scene, 'n_oil_px': n_oil, 'n_valid_oil_px': n_valid,
                 'oil_fingerprint_valid': True,
                 'oil_mndwi_med': mndwi_med, 'oil_ndi_med': ndi_med, 'oil_sr_med': sr_med,
                 'n_oil_pool_sample': int(n_sample)},
                oil_pixel_si)
    except Exception as e:
        return ({'scene': scene, 'n_oil_px': 0, 'oil_fingerprint_valid': False,
                 'oil_mndwi_med': np.nan, 'oil_ndi_med': np.nan, 'oil_sr_med': np.nan,
                 'error': str(e)},
                None)


# ===== Per-candidate FP-region SI median (NEW in v2) =====
def compute_patch_fp_si_median(scene: str, x: int, y: int,
                                src_cache: dict, mask_cache: dict,
                                model, mean_arr, std_arr, device):
    """讀 patch + mask, 跑 probe → FP mask → SI median over FP region.
    Returns (mndwi_med, ndi_med, sr_med, n_fp_px) or None."""
    # Cache TIF handle
    if scene not in src_cache:
        tif_path = STACKED_TIF_DIR / f'{scene}.tif'
        if not tif_path.exists():
            return None
        src_cache[scene] = rasterio.open(str(tif_path))
    # LRU mask cache (bound: MASK_CACHE_MAX 防 OOM)
    if scene in mask_cache:
        mask_cache.move_to_end(scene)  # mark recently used
        full_mask = mask_cache[scene]
    else:
        mask_path = MASK_DIR / f'{scene}.tif'
        if not mask_path.exists():
            full_mask = None
        else:
            with rasterio.open(str(mask_path)) as msrc:
                full_mask = msrc.read(1)
        mask_cache[scene] = full_mask
        if len(mask_cache) > MASK_CACHE_MAX:
            mask_cache.popitem(last=False)  # evict LRU
    if full_mask is None:
        return None

    src = src_cache[scene]
    try:
        win = Window(x, y, PATCH_SIZE, PATCH_SIZE)
        arr = src.read(window=win, boundless=True, fill_value=0).astype(np.float32)
    except Exception:
        return None
    np.nan_to_num(arr, nan=0.0, copy=False)
    valid = valid_pixel_mask(arr)
    if valid.sum() < 100:
        return None

    # Crop mask patch (bounds-safe)
    H, W = full_mask.shape
    y0, y1 = y, min(y + PATCH_SIZE, H)
    x0, x1 = x, min(x + PATCH_SIZE, W)
    mask_patch = np.full((PATCH_SIZE, PATCH_SIZE), 1, dtype=full_mask.dtype)  # default bg
    src_h, src_w = y1 - y0, x1 - x0
    if src_h > 0 and src_w > 0:
        mask_patch[:src_h, :src_w] = full_mask[y0:y1, x0:x1]

    # Run probe → oil probability
    p_norm = (arr - mean_arr[:, None, None]) / std_arr[:, None, None]
    t = torch.from_numpy(p_norm).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(t)['out']
        probs = F.softmax(logits, dim=1)[0, 0].cpu().numpy()

    # FP mask = pred_oil AND mask==1 AND valid
    fp_mask = (probs > PROB_THRESHOLD) & (mask_patch == 1) & valid
    n_fp = int(fp_mask.sum())
    if n_fp < MIN_FP_PIXELS_FOR_SI:
        return None

    # SI median over FP region
    si = compute_si_per_pixel(arr)  # (3, H, W)
    mndwi_med = float(np.median(si[0][fp_mask]))
    ndi_med = float(np.median(si[1][fp_mask]))
    sr_med = float(np.median(si[2][fp_mask]))
    return mndwi_med, ndi_med, sr_med, n_fp


COORD_RE = re.compile(r'^(.+)_patch_x(\d+)_y(\d+)$')


def load_candidates(step1_dir: Path, fold: int) -> pd.DataFrame:
    txt = step1_dir / f'cfhnm_within_mined_fold{fold}.txt'
    rows = []
    for line in txt.read_text().splitlines():
        parts = line.strip().split('\t')
        if len(parts) < 6:
            continue
        m = COORD_RE.match(parts[0])
        if not m:
            continue
        rows.append({
            'fold': fold, 'patch_id': parts[0],
            'scene': m.group(1), 'x': int(m.group(2)), 'y': int(m.group(3)),
            'role': parts[1],
            'n_fp_px': int(parts[2]), 'n_mask255_px': int(parts[3]),
            'mean_oil_prob': float(parts[4]), 'max_oil_prob': float(parts[5]),
        })
    return pd.DataFrame(rows)


def detect_modality(dist_arr: np.ndarray) -> dict:
    """簡易判斷單峰/雙峰 — 算 histogram 局部極大值數量。"""
    if len(dist_arr) < 50:
        return {'shape': 'too_few_samples', 'n_peaks': 0, 'suggested_cutoff_top': 5.0,
                'suggested_cutoff_bottom': None}
    hist, edges = np.histogram(dist_arr, bins=40)
    s = np.convolve(hist, np.ones(5) / 5, mode='same')
    peaks = []
    for i in range(1, len(s) - 1):
        if s[i] > s[i - 1] and s[i] > s[i + 1] and s[i] > s.max() * 0.15:
            peaks.append((i, s[i], (edges[i] + edges[i + 1]) / 2))
    if len(peaks) >= 2:
        first_peak_idx, second_peak_idx = peaks[0][0], peaks[1][0]
        valley_idx = first_peak_idx + np.argmin(s[first_peak_idx:second_peak_idx + 1])
        valley_value = (edges[valley_idx] + edges[valley_idx + 1]) / 2
        return {'shape': 'bimodal', 'n_peaks': len(peaks), 'valley': valley_value,
                'suggested_cutoff_top': 5.0,
                'suggested_cutoff_bottom_at_valley': valley_value}
    return {'shape': 'unimodal', 'n_peaks': len(peaks),
            'suggested_cutoff_top': 5.0,
            'suggested_cutoff_bottom': 'disabled (unimodal)'}


def _apply_path_overrides(args):
    """CLI 傳入的路徑覆蓋 module 全域(不給則保留檔頭預設值,獨立執行仍可跑)。"""
    global MASK_DIR, STACKED_TIF_DIR, PATCH_LEVEL_DIR, FOLD_BEST_PT
    if args.mask_dir:        MASK_DIR = Path(args.mask_dir)
    if args.stacked_tif_dir: STACKED_TIF_DIR = Path(args.stacked_tif_dir)
    if args.patch_level_dir: PATCH_LEVEL_DIR = Path(args.patch_level_dir)
    if args.fold_best_pt:
        for e in args.fold_best_pt:
            k, v = e.split('=', 1)
            FOLD_BEST_PT[int(k)] = v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--step1_dir', required=True, type=Path)
    ap.add_argument('--out_dir', required=True, type=Path)
    ap.add_argument('--folds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument('--device', default='cuda')
    # 路徑覆蓋(driver 集中傳入;不給則用檔頭預設)
    ap.add_argument('--mask_dir')
    ap.add_argument('--stacked_tif_dir')
    ap.add_argument('--patch_level_dir')
    ap.add_argument('--fold_best_pt', nargs='+',
                    help='權重覆蓋,格式 N=/path/to/best.pt(可多個)')
    args = ap.parse_args()
    _apply_path_overrides(args)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility: cuDNN deterministic for probe inference
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 1. Load all candidates
    print(f'[1/5] Load candidates from {args.step1_dir}')
    dfs = []
    for fold in args.folds:
        d = load_candidates(args.step1_dir, fold)
        print(f'  fold {fold}: {len(d):,} candidates')
        dfs.append(d)
    cand_df = pd.concat(dfs, ignore_index=True)
    print(f'  total: {len(cand_df):,}')

    # 2. Per-scene oil fingerprint + oil pixel SI samples (for σ pool)
    print('\n[2/5] Compute per-scene oil fingerprint + sample oil-pixel SI pool')
    unique_scenes = sorted(cand_df['scene'].unique().tolist())
    print(f'  {len(unique_scenes)} unique scenes, sampling {OIL_POOL_SAMPLES_PER_SCENE} oil px/scene')
    oil_pool_rng = np.random.default_rng(OIL_POOL_RNG_SEED)
    fp_records = []
    oil_pixel_si_chunks = []
    for scene in tqdm(unique_scenes, desc='  fingerprints'):
        info, oil_si = compute_scene_oil_fingerprint(scene, oil_pool_rng)
        if info:
            fp_records.append(info)
        if oil_si is not None and len(oil_si) > 0:
            oil_pixel_si_chunks.append(oil_si)
    fp_df = pd.DataFrame(fp_records)
    fp_df.to_csv(args.out_dir / 'scene_oil_fingerprint.csv', index=False)
    n_valid = int(fp_df['oil_fingerprint_valid'].sum())
    print(f'  valid fingerprints: {n_valid}/{len(fp_df)} (剔 scenes with < 100 oil pixels)')

    oil_pool = np.vstack(oil_pixel_si_chunks) if oil_pixel_si_chunks else np.zeros((0, 3))
    print(f'  oil pixel SI pool: {len(oil_pool):,} pixels')

    fp_lookup = {r['scene']: r for r in fp_records if r['oil_fingerprint_valid']}

    # 3. Per-candidate FP-region SI median
    print('\n[3/5] Compute per-candidate FP-region SI median (probe + threshold + median)')
    cand_df['patch_mndwi_med'] = np.nan
    cand_df['patch_ndi_med'] = np.nan
    cand_df['patch_sr_med'] = np.nan
    cand_df['n_fp_px_si'] = 0
    cand_df['fingerprint_valid'] = False

    device = args.device
    fold_groups = cand_df.groupby('fold').indices

    for fold in args.folds:
        if fold not in fold_groups:
            continue
        row_idxs = fold_groups[fold]
        print(f'\n  --- Fold {fold}: {len(row_idxs):,} candidates ---')

        # Verify ckpt exists
        ckpt_path = Path(FOLD_BEST_PT[fold])
        if not ckpt_path.exists():
            raise FileNotFoundError(f'Probe checkpoint missing for fold {fold}: {ckpt_path}')
        print(f'  loading probe {ckpt_path}')
        model = build_model(device=device)
        state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
        # Check load (防 silent fail)
        missing, unexpected = model.load_state_dict(state, strict=False)
        n_loaded = sum(1 for k in state.keys() if k not in unexpected)
        print(f'  state_dict: loaded {n_loaded}/{len(state)} keys '
              f'(missing={len(missing)}, unexpected={len(unexpected)})')
        if n_loaded < 0.5 * len(state):
            raise RuntimeError(f'Probe load failure for fold {fold}: only {n_loaded}/{len(state)} keys loaded')
        model.eval()
        mean_arr, std_arr = load_mean_std(fold)

        src_cache: dict = {}
        mask_cache: OrderedDict = OrderedDict()
        try:
            for idx in tqdm(row_idxs, desc=f'  fold{fold} patches'):
                row = cand_df.iloc[idx]
                scene = row['scene']
                result = compute_patch_fp_si_median(
                    scene, int(row['x']), int(row['y']),
                    src_cache, mask_cache, model, mean_arr, std_arr, device,
                )
                if result is None:
                    continue
                mndwi_med, ndi_med, sr_med, n_fp = result
                cand_df.at[idx, 'patch_mndwi_med'] = mndwi_med
                cand_df.at[idx, 'patch_ndi_med'] = ndi_med
                cand_df.at[idx, 'patch_sr_med'] = sr_med
                cand_df.at[idx, 'n_fp_px_si'] = n_fp
                if scene in fp_lookup:
                    cand_df.at[idx, 'fingerprint_valid'] = True
        finally:
            for src in src_cache.values():
                try: src.close()
                except: pass
            mask_cache.clear()
        del model
        torch.cuda.empty_cache()

    # 4. Oil-pool Σ via Ledoit-Wolf (optimal shrinkage, textbook standard)
    print('\n[4/5] Oil-pool Σ via Ledoit-Wolf optimal shrinkage')
    if len(oil_pool) < 1000:
        raise RuntimeError(f'Oil pool too small ({len(oil_pool)}). Aborting Σ estimation.')

    # Reference stats (印 + 存，paper appendix 用)
    oil_pool_median = np.median(oil_pool, axis=0)
    oil_pool_mean = oil_pool.mean(axis=0)
    sigma_std = oil_pool.std(axis=0, ddof=1)
    corr = np.corrcoef(oil_pool.T)
    max_abs_corr = float(max(abs(corr[0,1]), abs(corr[0,2]), abs(corr[1,2])))
    print(f'  oil pool: n={len(oil_pool):,}, mean={oil_pool_mean.round(4)}, '
          f'median={oil_pool_median.round(4)}')
    print(f'  oil-pool σ (std): MNDWI={sigma_std[0]:.4f}, NDI={sigma_std[1]:.4f}, '
          f'SR={sigma_std[2]:.4f}')
    print(f'  oil-pool inter-dim correlation (Pearson):')
    print(f'    MNDWI-NDI: {corr[0,1]:+.3f}')
    print(f'    MNDWI-SR : {corr[0,2]:+.3f}')
    print(f'    NDI  -SR : {corr[1,2]:+.3f}')
    print(f'    max|corr| = {max_abs_corr:.3f}')

    # Ledoit-Wolf shrinkage estimator: Σ_LW = (1-α)·Σ_sample + α·μI
    lw = LedoitWolf(store_precision=True).fit(oil_pool)
    cov_lw = lw.covariance_
    cov_inv = lw.precision_  # already computed
    alpha_lw = float(lw.shrinkage_)
    cov_sample = np.cov(oil_pool.T, ddof=1)
    cond_sample = float(np.linalg.cond(cov_sample))
    cond_lw = float(np.linalg.cond(cov_lw))
    print(f'  Ledoit-Wolf α (shrinkage): {alpha_lw:.4f}')
    print(f'  cond(Σ_sample) = {cond_sample:.2e} → cond(Σ_LW) = {cond_lw:.2e}')
    print(f'  Σ_LW (3×3):')
    for r in cov_lw:
        print(f'    [{r[0]:+.4f}, {r[1]:+.4f}, {r[2]:+.4f}]')

    pd.DataFrame({
        'dim': ['MNDWI', 'NDI_B1_B11', 'SR_B1B2_B3B8A'],
        'oil_pool_mean':      oil_pool_mean,
        'oil_pool_median':    oil_pool_median,
        'oil_pool_std':       sigma_std,
        'n_oil_pool_px':      [len(oil_pool)] * 3,
        'corr_with_MNDWI':    corr[0],
        'corr_with_NDI':      corr[1],
        'corr_with_SR':       corr[2],
        'cov_sample_diag':    np.diag(cov_sample),
        'cov_lw_diag':        np.diag(cov_lw),
        'cov_inv_diag':       np.diag(cov_inv),
    }).to_csv(args.out_dir / 'normalization_stats.csv', index=False)
    np.savez(args.out_dir / 'mahalanobis_matrices.npz',
             oil_pool_median=oil_pool_median, oil_pool_mean=oil_pool_mean,
             oil_pool_std=sigma_std, corr=corr,
             cov_sample=cov_sample, cov_lw=cov_lw, cov_inv=cov_inv,
             ledoit_wolf_alpha=alpha_lw,
             cond_sample=cond_sample, cond_lw=cond_lw)

    # 5. Mahalanobis d_si = sqrt( (p-f)^T Σ_LW^-1 (p-f) )
    print('\n[5/5] Compute Mahalanobis d_si = sqrt((p-f)^T Σ_LW^-1 (p-f))')
    valid_mask = cand_df['fingerprint_valid'].values
    p_si_mat = cand_df[['patch_mndwi_med', 'patch_ndi_med', 'patch_sr_med']].values
    # Build f_oil per row via scene lookup
    f_oil_mat = np.full_like(p_si_mat, np.nan)
    for i, (scene, ok) in enumerate(zip(cand_df['scene'].values, valid_mask)):
        if ok and scene in fp_lookup:
            fp = fp_lookup[scene]
            f_oil_mat[i] = [fp['oil_mndwi_med'], fp['oil_ndi_med'], fp['oil_sr_med']]
    # Mahalanobis: d² = (p-f)^T Σ_inv (p-f) per row
    diff_mat = p_si_mat - f_oil_mat
    d_si_sq = np.einsum('ni,ij,nj->n', diff_mat, cov_inv, diff_mat)
    # 數值噪音 → 微負 (NaN exception) clip
    d_si_sq = np.where(np.isnan(d_si_sq), np.nan, np.maximum(d_si_sq, 0.0))
    d_si_arr = np.sqrt(d_si_sq)
    cand_df['d_si'] = d_si_arr
    # Also keep f_oil columns for transparency / debug
    cand_df['oil_mndwi_med'] = f_oil_mat[:, 0]
    cand_df['oil_ndi_med']   = f_oil_mat[:, 1]
    cand_df['oil_sr_med']    = f_oil_mat[:, 2]

    cand_df.to_csv(args.out_dir / 'si_distance_per_candidate.csv', index=False)
    n_dist = int(cand_df['fingerprint_valid'].sum())
    print(f'  candidates with valid d_si: {n_dist}/{len(cand_df)}')

    # --- Plots + summary ---
    valid_df = cand_df[cand_df['fingerprint_valid']].copy()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, fold in enumerate(args.folds):
        ax = axes[i]
        sub = valid_df[valid_df['fold'] == fold]['d_si']
        if len(sub):
            ax.hist(sub, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
            ax.axvline(sub.quantile(0.05), color='red', linestyle='--',
                       label=f'5%={sub.quantile(0.05):.3f}')
            ax.axvline(sub.quantile(0.50), color='orange', linestyle='--',
                       label=f'50%={sub.quantile(0.50):.3f}')
            ax.set_title(f'fold {fold} (n={len(sub):,})')
            ax.set_xlabel('SI distance (normalized) to scene oil fingerprint')
            ax.set_ylabel('count')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
    ax = axes[len(args.folds)] if len(args.folds) < 6 else axes[5]
    ax.hist(valid_df['d_si'], bins=50, alpha=0.7, color='darkgreen', edgecolor='black')
    ax.axvline(valid_df['d_si'].quantile(0.05), color='red', linestyle='--',
               label=f'5%={valid_df["d_si"].quantile(0.05):.3f}')
    ax.axvline(valid_df['d_si'].quantile(0.50), color='orange', linestyle='--',
               label=f'50%={valid_df["d_si"].quantile(0.50):.3f}')
    ax.set_title(f'ALL folds (n={len(valid_df):,})')
    ax.set_xlabel('SI distance (normalized)')
    ax.set_ylabel('count')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_dir / 'dist_histogram_per_fold.png', dpi=130, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(valid_df['d_si'], bins=60, alpha=0.8, color='darkgreen', edgecolor='black')
    plt.axvline(valid_df['d_si'].quantile(0.05), color='red', linestyle='--', linewidth=2,
                label=f'5% safety cutoff = {valid_df["d_si"].quantile(0.05):.3f}')
    plt.axvline(valid_df['d_si'].quantile(0.50), color='orange', linestyle='--', linewidth=2,
                label=f'50% too-easy cutoff = {valid_df["d_si"].quantile(0.50):.3f}')
    plt.xlabel('SI distance (normalized) to scene oil fingerprint')
    plt.ylabel('candidate count')
    plt.title(f'v4: Mahalanobis d_si (LW-shrunk oil-pool Σ) — '
              f'{len(valid_df):,} valid candidates × {len(args.folds)} folds')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_dir / 'dist_histogram_global.png', dpi=130, bbox_inches='tight')
    plt.close()

    summary_lines = []
    summary_lines.append(f'=== SI Distance Analysis Summary (v4: LW-Mahalanobis on oil pool) ===')
    summary_lines.append(f'Total candidates loaded: {len(cand_df):,}')
    summary_lines.append(f'Valid d_si (FP region >= {MIN_FP_PIXELS_FOR_SI} px + fingerprint OK): {len(valid_df):,}')
    summary_lines.append(f'')
    summary_lines.append(f'--- Oil-pool Σ via Ledoit-Wolf ---')
    summary_lines.append(f'  oil pool size: {len(oil_pool):,} pixels '
                         f'({OIL_POOL_SAMPLES_PER_SCENE}/scene × {n_valid} valid scenes)')
    summary_lines.append(f'  oil pool mean:   MNDWI={oil_pool_mean[0]:.4f}, '
                         f'NDI={oil_pool_mean[1]:.4f}, SR={oil_pool_mean[2]:.4f}')
    summary_lines.append(f'  oil pool median: MNDWI={oil_pool_median[0]:.4f}, '
                         f'NDI={oil_pool_median[1]:.4f}, SR={oil_pool_median[2]:.4f}')
    summary_lines.append(f'  oil pool std:    MNDWI={sigma_std[0]:.4f}, '
                         f'NDI={sigma_std[1]:.4f}, SR={sigma_std[2]:.4f}')
    summary_lines.append(f'  inter-dim correlation (Pearson):')
    summary_lines.append(f'    MNDWI-NDI: {corr[0,1]:+.3f}')
    summary_lines.append(f'    MNDWI-SR : {corr[0,2]:+.3f}')
    summary_lines.append(f'    NDI  -SR : {corr[1,2]:+.3f}')
    summary_lines.append(f'    max|corr| = {max_abs_corr:.3f}')
    summary_lines.append(f'  Ledoit-Wolf α (shrinkage to scaled identity): {alpha_lw:.4f}')
    summary_lines.append(f'  cond(Σ_sample) = {cond_sample:.2e} → cond(Σ_LW) = {cond_lw:.2e}')
    summary_lines.append(f'')
    summary_lines.append(f'--- Global distribution (normalized d_si) ---')
    for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        v = valid_df['d_si'].quantile(q)
        summary_lines.append(f'  Q{int(q*100):02d}: {v:.4f}')
    summary_lines.append(f'  mean: {valid_df["d_si"].mean():.4f}, std: {valid_df["d_si"].std():.4f}')
    summary_lines.append(f'')
    summary_lines.append(f'--- Modality detection ---')
    modality = detect_modality(valid_df['d_si'].values)
    for k, v in modality.items():
        summary_lines.append(f'  {k}: {v}')
    summary_lines.append(f'')
    summary_lines.append(f'--- Per-fold quantiles ---')
    for fold in args.folds:
        sub = valid_df[valid_df['fold'] == fold]['d_si']
        if len(sub):
            summary_lines.append(f'  fold {fold} (n={len(sub):,}): '
                                 f'Q05={sub.quantile(0.05):.3f}, '
                                 f'Q50={sub.quantile(0.50):.3f}, '
                                 f'Q95={sub.quantile(0.95):.3f}')
    summary_lines.append(f'')
    summary_lines.append(f'Next: step3_si_filter.py with --top_cutoff 0.05')

    summary_text = '\n'.join(summary_lines)
    (args.out_dir / 'decision_summary.txt').write_text(summary_text)
    print('\n' + summary_text)
    print(f'\n[Done] Outputs at: {args.out_dir}')


if __name__ == '__main__':
    main()
