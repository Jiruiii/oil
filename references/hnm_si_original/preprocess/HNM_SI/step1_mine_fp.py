"""
HNM_SI Step 1 — Mine FP candidates per fold (within-fold variant).

對每個 fold N (1..5):
  1. Load fold N's baseline best.pt + mean/std
  2. 對 fold N 的 TRAIN scenes 跑 sliding window inference → train FP pool
  3. 對 fold N 的 VAL scenes   跑 sliding window inference → val FP pool
  4. Patch-level filter（FP px count、cloud ratio 等）
  5. 通過所有條件的 patch → 記錄為 HN candidate, 標 role=train/val

Within-fold 設計：
   - 不用 fold N 的 test scenes (避免 leakage)
   - Mine fold N 自己的 train + val (model 對 train 看過 ~5% patches,
     對 val 完全未訓練; mining 仍 honest 因為 patch positions 95%+ 未見過)

輸出 (per fold N):
  cfhnm_within_mined_foldN.txt   每行: <scene>_patch_x<x>_y<y>  role  fp_px  mask255_px  mean_prob  max_prob
  cfhnm_within_mined_foldN_summary.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm

# ===== Paths =====
MASK_DIR = Path('/home2/sun/OIL_PROJECT_dataset/full_band/mask')
STACKED_TIF_DIR = Path('/home2/sun/OIL_PROJECT_dataset/full_band/processed/stack_tif')
SCENE_LEVEL_DIR = Path('/mnt/backups/sun/OIL_PROJECT_dataset/full_band/data_split_MS5/5_fold/scene_level')
PATCH_LEVEL_DIR = Path('/mnt/backups/sun/OIL_PROJECT_dataset/full_band/data_split_MS5/5_fold/patch_level_GB1.0')

FOLD_BEST_PT = {
    1: '/mnt/backups/sun/OIL_PROJECT_MutiBand_0422_VRT_training/result-seg/CV_MS5/20260512-231439_MS5_CV_deeplabv3+_GB1.0_baseline_fold1/weights/best.pt',
    2: '/mnt/backups/sun/OIL_PROJECT_MutiBand_0422_VRT_training/result-seg/CV_MS5/20260513-173838_MS5_CV_deeplabv3+_GB1.0_baseline_fold2/weights/best.pt',
    3: '/mnt/backups/sun/OIL_PROJECT_MutiBand_0422_VRT_training/result-seg/CV_MS5/20260514-013955_MS5_CV_deeplabv3+_GB1.0_baseline_fold3/weights/best.pt',
    4: '/mnt/backups/sun/OIL_PROJECT_MutiBand_0422_VRT_training/result-seg/CV_MS5_rerun24/20260516-171513_MS5_CV_deeplabv3+_GB1.0_baseline_rerun_fold4/weights/best.pt',
    5: '/mnt/backups/sun/OIL_PROJECT_MutiBand_0422_VRT_training/result-seg/CV_MS5/20260514-034340_MS5_CV_deeplabv3+_GB1.0_baseline_fold5/weights/best.pt',
}

PATCH_SIZE = 256
STRIDE = 192
MIN_MASK1_RATIO = 0.50
MAX_MASK255_RATIO = 0.30
MAX_NODATA_RATIO = 0.30
MIN_FP_PIXELS = 1


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
    import re
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
                p = src.read(window=rasterio.windows.Window(c['x'], c['y'], 256, 256),
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


def read_scenes(fold: int, role: str) -> list[str]:
    """role ∈ {'train', 'val'}"""
    txt = SCENE_LEVEL_DIR / f'{role}_fold{fold}.txt'
    scenes = []
    for line in txt.read_text().splitlines():
        line = line.strip()
        if line:
            scenes.append(line.split('/')[-1])
    return scenes


def mine_scene(model, mean, std, device, scene: str):
    tif = STACKED_TIF_DIR / f'{scene}.tif'
    mask_path = MASK_DIR / f'{scene}.tif'
    if not (tif.exists() and mask_path.exists()):
        return [], {'n_total_patches': 0, 'n_eligible': 0, 'n_with_fp': 0}

    with rasterio.open(str(mask_path)) as src:
        mask = src.read(1)
    H, W = mask.shape
    ys = list(dict.fromkeys(list(range(0, H - PATCH_SIZE, STRIDE)) + [max(H - PATCH_SIZE, 0)]))
    xs = list(dict.fromkeys(list(range(0, W - PATCH_SIZE, STRIDE)) + [max(W - PATCH_SIZE, 0)]))
    positions = [(y, x) for y in ys for x in xs]

    mean_arr = mean[:, None, None]
    std_arr = np.where(std < 1e-8, 1.0, std)[:, None, None]

    eligible = []
    src = rasterio.open(str(tif))
    try:
        for y, x in positions:
            m_patch = mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            if m_patch.shape != (PATCH_SIZE, PATCH_SIZE):
                continue
            patch_n = m_patch.size
            n_mask0 = int((m_patch == 0).sum())
            n_mask1 = int((m_patch == 1).sum())
            n_mask255 = int((m_patch == 255).sum())
            if n_mask0 > 0:
                continue
            if n_mask1 / patch_n < MIN_MASK1_RATIO:
                continue
            if n_mask255 / patch_n > MAX_MASK255_RATIO:
                continue
            p = src.read(window=rasterio.windows.Window(x, y, PATCH_SIZE, PATCH_SIZE)).astype(np.float32)
            np.nan_to_num(p, nan=0.0, copy=False)
            valid_pix = (np.abs(p).sum(axis=0) > 1e-5).sum()
            nodata_ratio = 1.0 - valid_pix / patch_n
            if nodata_ratio > MAX_NODATA_RATIO:
                continue
            eligible.append((y, x, p, m_patch))
    finally:
        src.close()

    if not eligible:
        return [], {'n_total_patches': len(positions), 'n_eligible': 0, 'n_with_fp': 0}

    candidates = []
    BATCH = 16
    with torch.no_grad():
        for b in range(0, len(eligible), BATCH):
            batch = eligible[b:b + BATCH]
            tensors = [torch.from_numpy((p - mean_arr) / std_arr) for (_, _, p, _) in batch]
            t = torch.stack(tensors).to(device)
            logits = model(t)['out']
            probs = F.softmax(logits, dim=1)[:, 0].cpu().numpy()
            for i, (y, x, _, m_patch) in enumerate(batch):
                pred_oil = (probs[i] > 0.5)
                fp = pred_oil & (m_patch == 1)
                n_fp = int(fp.sum())
                if n_fp >= MIN_FP_PIXELS:
                    candidates.append({
                        'scene': scene, 'x': x, 'y': y,
                        'n_fp_px': n_fp,
                        'n_mask255_px': int((m_patch == 255).sum()),
                        'mean_oil_prob': float(probs[i].mean()),
                        'max_oil_prob': float(probs[i].max()),
                    })
    summary = {'n_total_patches': len(positions), 'n_eligible': len(eligible), 'n_with_fp': len(candidates)}
    return candidates, summary


def _apply_path_overrides(args):
    """CLI 傳入的路徑覆蓋 module 全域(不給則保留檔頭預設值,獨立執行仍可跑)。"""
    global MASK_DIR, STACKED_TIF_DIR, SCENE_LEVEL_DIR, PATCH_LEVEL_DIR, FOLD_BEST_PT
    if args.mask_dir:        MASK_DIR = Path(args.mask_dir)
    if args.stacked_tif_dir: STACKED_TIF_DIR = Path(args.stacked_tif_dir)
    if args.scene_level_dir: SCENE_LEVEL_DIR = Path(args.scene_level_dir)
    if args.patch_level_dir: PATCH_LEVEL_DIR = Path(args.patch_level_dir)
    if args.fold_best_pt:
        for e in args.fold_best_pt:
            k, v = e.split('=', 1)
            FOLD_BEST_PT[int(k)] = v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--folds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument('--device', default='cuda')
    # 路徑覆蓋(driver 集中傳入;不給則用檔頭預設)
    parser.add_argument('--mask_dir')
    parser.add_argument('--stacked_tif_dir')
    parser.add_argument('--scene_level_dir')
    parser.add_argument('--patch_level_dir')
    parser.add_argument('--fold_best_pt', nargs='+',
                        help='權重覆蓋,格式 N=/path/to/best.pt(可多個)')
    args = parser.parse_args()
    _apply_path_overrides(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grand_summary = []

    for fold in args.folds:
        print(f'\n{"="*70}\n  Within-fold Mining: fold {fold}\n{"="*70}')

        train_scenes = read_scenes(fold, 'train')
        val_scenes = read_scenes(fold, 'val')
        print(f'  train scenes: {len(train_scenes)}, val scenes: {len(val_scenes)}')

        model = build_model(device=args.device)
        state = torch.load(FOLD_BEST_PT[fold], map_location=args.device, weights_only=True)
        model.load_state_dict(state, strict=False)
        model.eval()
        mean, std = load_mean_std(fold)
        np.save(out_dir / f'mean_std_fold{fold}.npy', {'mean': mean.tolist(), 'std': std.tolist()})

        all_cands = []
        per_scene = []
        for role, scenes in [('train', train_scenes), ('val', val_scenes)]:
            print(f'  [{role}] {len(scenes)} scenes')
            for scene in tqdm(scenes, desc=f'  fold{fold} {role}'):
                cands, summ = mine_scene(model, mean, std, args.device, scene)
                for c in cands:
                    c['role'] = role
                all_cands.extend(cands)
                summ['scene'] = scene
                summ['role'] = role
                summ['fold'] = fold
                per_scene.append(summ)

        # Save
        pool_txt = out_dir / f'cfhnm_within_mined_fold{fold}.txt'
        with open(pool_txt, 'w') as f:
            for c in all_cands:
                f.write(f"{c['scene']}_patch_x{c['x']}_y{c['y']}\t{c['role']}\t"
                        f"{c['n_fp_px']}\t{c['n_mask255_px']}\t"
                        f"{c['mean_oil_prob']:.4f}\t{c['max_oil_prob']:.4f}\n")
        pd.DataFrame(per_scene).to_csv(out_dir / f'cfhnm_within_mined_fold{fold}_summary.csv', index=False)

        n_train = sum(1 for c in all_cands if c['role'] == 'train')
        n_val = sum(1 for c in all_cands if c['role'] == 'val')
        print(f'\n  fold{fold} 完成: {len(all_cands):,} candidates ({n_train:,} train + {n_val:,} val)')
        grand_summary.append({
            'fold': fold,
            'n_train_scenes': len(train_scenes),
            'n_val_scenes': len(val_scenes),
            'candidates_train': n_train,
            'candidates_val': n_val,
            'total_candidates': len(all_cands),
        })

        del model
        torch.cuda.empty_cache()

    pd.DataFrame(grand_summary).to_csv(out_dir / 'grand_summary.csv', index=False)
    print('\n\n=== Grand Summary ===')
    print(pd.DataFrame(grand_summary).to_string(index=False))


if __name__ == '__main__':
    main()
