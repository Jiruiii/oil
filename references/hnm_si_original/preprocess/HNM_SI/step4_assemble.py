"""
HNM_SI Step 4 — 從 SI-filtered candidates 採樣 + 組 split dir。

從 step3_si_filter.py 的 filtered pool 做 stratified sampling
（per-scene 均勻 + 3σ outlier 剔除），輸出最終可訓練的 train/val fold txt。

Usage:
  conda run -n yolo11 python step4_assemble.py \\
      --filtered_dir <step3_si_filter.py output dir> \\
      --out_split_dir <new split dir> \\
      [--ratio 0.5]  # HN to oil ratio (預設 0.5，對應 rand050 量級)
"""
import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SCENE_LEVEL = Path('/mnt/backups/sun/OIL_PROJECT_dataset/full_band/data_split_MS5/5_fold/scene_level')
PATCH_LEVEL_SRC = Path('/mnt/backups/sun/OIL_PROJECT_dataset/full_band/data_split_MS5/5_fold/patch_level_GB1.0')
LIST_OIL = PATCH_LEVEL_SRC / 'list_oil.txt'
SEED = 42
COORD_RE = re.compile(r'^(.+)_patch_x(\d+)_y(\d+)$')


def load_filtered(filtered_dir: Path, fold: int) -> pd.DataFrame:
    """讀 step3 輸出，欄位 patch_id, role, n_fp_px, n_mask255_px, mean_oil_prob, max_oil_prob, d_si"""
    txt = filtered_dir / f'cfhnm_within_si_filtered_fold{fold}.txt'
    rows = []
    for line in txt.read_text().splitlines():
        parts = line.strip().split('\t')
        if len(parts) < 7:
            continue
        m = COORD_RE.match(parts[0])
        if not m:
            continue
        rows.append({
            'patch_id': parts[0],
            'scene': m.group(1), 'x': int(m.group(2)), 'y': int(m.group(3)),
            'role': parts[1],
            'n_fp_px': int(parts[2]), 'n_mask255_px': int(parts[3]),
            'mean_oil_prob': float(parts[4]), 'max_oil_prob': float(parts[5]),
            'd_si': float(parts[6]),
        })
    return pd.DataFrame(rows)


def detect_3sigma_outliers(df: pd.DataFrame, role: str) -> set:
    """3σ outlier scenes per role（依 candidate count）"""
    sub = df[df['role'] == role]
    counts = sub.groupby('scene').size().reset_index(name='n')
    if len(counts) < 3:
        return set()
    mu, sigma = counts['n'].mean(), counts['n'].std()
    thresh = mu + 3 * sigma
    outliers = set(counts.loc[counts['n'] > thresh, 'scene'].tolist())
    print(f'    [3σ rule, role={role}] μ={mu:.1f} σ={sigma:.1f} threshold={thresh:.1f}, outliers={sorted(outliers)}')
    return outliers


def count_oil(fold: int, role: str, list_oil_set: set) -> int:
    txt = PATCH_LEVEL_SRC / f'{role}_fold{fold}.txt'
    return sum(1 for l in txt.read_text().splitlines()
               if Path(l.strip()).stem in list_oil_set)


def stratified_sample(pool: pd.DataFrame, target_n: int, seed: int) -> pd.DataFrame:
    if len(pool) <= target_n:
        return pool.copy()
    rng = np.random.default_rng(seed)
    scenes = pool['scene'].unique()
    per_scene_q = max(1, target_n // len(scenes))
    picks = []
    for s in scenes:
        sub_idx = pool.index[pool['scene'] == s].values
        take = min(per_scene_q, len(sub_idx))
        idx = rng.choice(sub_idx, size=take, replace=False)
        picks.extend(idx)
    if len(picks) < target_n:
        remaining = pool.index.difference(picks)
        need = target_n - len(picks)
        if len(remaining) > 0:
            extra = rng.choice(remaining.values, size=min(need, len(remaining)), replace=False)
            picks.extend(extra)
    if len(picks) > target_n:
        picks = list(rng.choice(picks, size=target_n, replace=False))
    return pool.loc[picks].copy().reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--filtered_dir', required=True, type=Path)
    ap.add_argument('--out_split_dir', required=True, type=Path)
    ap.add_argument('--ratio', type=float, default=0.5)
    # 路徑覆蓋(driver 集中傳入;不給則用檔頭預設)
    ap.add_argument('--patch_level_src')
    ap.add_argument('--scene_level')
    ap.add_argument('--list_oil')
    args = ap.parse_args()

    global PATCH_LEVEL_SRC, LIST_OIL, SCENE_LEVEL
    if args.patch_level_src:
        PATCH_LEVEL_SRC = Path(args.patch_level_src)
        LIST_OIL = PATCH_LEVEL_SRC / 'list_oil.txt'
    if args.scene_level:
        SCENE_LEVEL = Path(args.scene_level)
    if args.list_oil:
        LIST_OIL = Path(args.list_oil)

    args.out_split_dir.mkdir(parents=True, exist_ok=True)
    list_oil_set = set(Path(l.strip()).stem for l in LIST_OIL.read_text().splitlines() if l.strip())

    rng = np.random.default_rng(SEED)
    summary, metadata = [], []

    for fold in [1, 2, 3, 4, 5]:
        print(f'\n=== Fold {fold} ===')
        df = load_filtered(args.filtered_dir, fold)
        n_train_cands = len(df[df['role'] == 'train'])
        n_val_cands = len(df[df['role'] == 'val'])
        print(f'  filtered candidates: train={n_train_cands}, val={n_val_cands}')

        train_outliers = detect_3sigma_outliers(df, 'train')
        val_outliers = detect_3sigma_outliers(df, 'val')

        n_before = len(df)
        df = df[~((df['role'] == 'train') & df['scene'].isin(train_outliers))]
        df = df[~((df['role'] == 'val') & df['scene'].isin(val_outliers))]
        n_after = len(df)
        print(f'  after 3σ filter: {n_after} (removed {n_before - n_after})')

        n_oil_train = count_oil(fold, 'train', list_oil_set)
        n_oil_val = count_oil(fold, 'val', list_oil_set)
        target_train = int(n_oil_train * args.ratio)
        target_val = int(n_oil_val * args.ratio)
        print(f'  target HN: train={target_train}, val={target_val}')

        train_pool = df[df['role'] == 'train'].reset_index(drop=True)
        val_pool = df[df['role'] == 'val'].reset_index(drop=True)
        train_hn = stratified_sample(train_pool, target_train, SEED + fold)
        val_hn = stratified_sample(val_pool, target_val, SEED + fold + 100)
        print(f'  sampled: train={len(train_hn)}, val={len(val_hn)}')

        original_train = [l.strip() for l in (PATCH_LEVEL_SRC / f'train_fold{fold}.txt').read_text().splitlines() if l.strip()]
        original_val = [l.strip() for l in (PATCH_LEVEL_SRC / f'val_fold{fold}.txt').read_text().splitlines() if l.strip()]

        def to_line(r):
            return f"{r['scene']}_patch_x{int(r['x'])}_y{int(r['y'])}"
        train_lines = [to_line(r) for _, r in train_hn.iterrows()]
        val_lines = [to_line(r) for _, r in val_hn.iterrows()]

        new_train = original_train + train_lines
        new_val = original_val + val_lines
        rng.shuffle(new_train)
        rng.shuffle(new_val)
        (args.out_split_dir / f'train_fold{fold}.txt').write_text('\n'.join(new_train) + '\n')
        (args.out_split_dir / f'val_fold{fold}.txt').write_text('\n'.join(new_val) + '\n')
        shutil.copy(PATCH_LEVEL_SRC / f'test_fold{fold}.txt', args.out_split_dir / f'test_fold{fold}.txt')

        for _, r in train_hn.iterrows():
            metadata.append({'target_fold': fold, 'role': 'train', 'patch': to_line(r),
                             'source_scene': r['scene'], 'n_fp_px': int(r['n_fp_px']),
                             'd_si': float(r['d_si']), 'mean_oil_prob': float(r['mean_oil_prob'])})
        for _, r in val_hn.iterrows():
            metadata.append({'target_fold': fold, 'role': 'val', 'patch': to_line(r),
                             'source_scene': r['scene'], 'n_fp_px': int(r['n_fp_px']),
                             'd_si': float(r['d_si']), 'mean_oil_prob': float(r['mean_oil_prob'])})

        summary.append({
            'fold': fold,
            'filtered_train': n_train_cands, 'filtered_val': n_val_cands,
            'train_outliers': len(train_outliers), 'val_outliers': len(val_outliers),
            'after_3sigma': n_after,
            'oil_train': n_oil_train, 'oil_val': n_oil_val,
            'target_train_hn': target_train, 'target_val_hn': target_val,
            'sampled_train_hn': len(train_hn), 'sampled_val_hn': len(val_hn),
            'orig_train_size': len(original_train), 'orig_val_size': len(original_val),
            'new_train_size': len(new_train), 'new_val_size': len(new_val),
        })

    for f in ['list_oil.txt', 'list_bg_candidates.txt', 'list_others.txt']:
        src = PATCH_LEVEL_SRC / f
        if src.exists():
            shutil.copy(src, args.out_split_dir / f)

    pd.DataFrame(summary).to_csv(args.out_split_dir / 'assembly_summary.csv', index=False)
    pd.DataFrame(metadata).to_csv(args.out_split_dir / 'cfhnm_within_si_metadata.csv', index=False)
    print('\n=== Final Summary ===')
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == '__main__':
    main()
