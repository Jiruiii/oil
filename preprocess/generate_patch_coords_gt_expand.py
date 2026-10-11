"""
GT-centric patch coord generator (GT-expand strategy).

對每個場景的 GT mask:
  1. cv2.connectedComponentsWithStats 找 connected components
  2. 每個 component 取 bbox 中心當 patch 中心
  3. 若 bbox <= PATCH_SIZE → 1 個 patch（中心對齊 component）
     若 bbox >  PATCH_SIZE → 用 PATCH_SIZE-grid 覆蓋 bbox，中心對齊 grid cells
  4. 所有 patch 座標 clip 到影像邊界內

輸入:
  - scene_level/{train,val,test}_fold{N}.txt   每行 {LOCATION}/{SCENE_NAME}
  - MASK_DIR/{SCENE_NAME}.tif                  mask 編碼: 0=oil, 1=annot bg, 2=other, 255=ignore

輸出 (與舊版 patch_level_GB1.0 同格式，可被既有 loader 沿用):
  - patch_level_gt_expand/{train,val,test}_fold{N}.txt
      每行 {SCENE_NAME}_patch_x{COL_OFF}_y{ROW_OFF}
  - patch_level_gt_expand/stats.csv
      scene_name, comp_id, bbox_h, bbox_w, n_oil, n_patches, tiled, error

效能設計：
  - 每個 unique scene 只處理一次（5-fold × 3 split = 15 個 txt 共享 ~358 unique scenes）
  - multiprocessing.Pool 並行 scene-level 計算
"""

import csv
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import rasterio
from tqdm import tqdm

# ===== 設定 =====
import os as _os
# SPLIT_DIR/OUT_DIR 必填:舊版預設指向 5_fold,裸跑會靜默用過期 split 重生 patch 座標(footgun),故改為必給
if not _os.environ.get('SPLIT_DIR') or not _os.environ.get('OUT_DIR'):
    sys.exit('[ERROR] 必須明確指定 SPLIT_DIR 與 OUT_DIR 環境變數(現行主 split = '
             '/mnt/backup/oil_dataset/new/full_band/data_split/3_fold_stratified_v2/...)')
SCENE_SPLIT_DIR = Path(_os.environ['SPLIT_DIR'])
MASK_DIR        = Path('/home/alanyh/oil_dataset/new/full_band/mask')
OUT_DIR         = Path(_os.environ['OUT_DIR'])
# 髒資料以 per-scene 空檔標記（dataset 慣例）：{scene_dir}/dirty.txt 或 blur.txt 存在即排除
DIRTY_SCAN_ROOT = Path(_os.environ.get('DIRTY_SCAN_ROOT',
    '/home/alanyh/oil_dataset/new/full_band/MS6_sen2like'))
DIRTY_MARKERS   = ('dirty.txt', 'blur.txt')
PATCH_SIZE      = 256
# [F8] N_FOLDS 由 split 目錄自動偵測並驗證完整 fold set（舊版預設 5,3-fold 下會靜默
# 忽略不存在輸入並寫出空 fold4/fold5）。若另給 N_FOLDS 環境變數,必須與偵測值一致。
import re as _re
_folds = sorted(int(_m.group(1)) for _p in SCENE_SPLIT_DIR.glob('train_fold*.txt')
                if (_m := _re.fullmatch(r'train_fold(\d+)\.txt', _p.name)))
if not _folds:
    sys.exit(f'[ERROR][F8] {SCENE_SPLIT_DIR} 內找不到 train_fold*.txt')
if _folds != list(range(1, len(_folds) + 1)):
    sys.exit(f'[ERROR][F8] fold 編號不連續: {_folds}')
_missing = [f'{_s}_fold{_n}.txt' for _n in _folds for _s in ('train', 'val', 'test')
            if not (SCENE_SPLIT_DIR / f'{_s}_fold{_n}.txt').exists()]
if _missing:
    sys.exit(f'[ERROR][F8] split 目錄缺檔: {_missing}')
N_FOLDS = len(_folds)
if _os.environ.get('N_FOLDS') and int(_os.environ['N_FOLDS']) != N_FOLDS:
    sys.exit(f"[ERROR][F8] 環境變數 N_FOLDS={_os.environ['N_FOLDS']} 與偵測到的 {N_FOLDS} 不符")
NUM_WORKERS     = 8
# ================


def _load_dirty():
    """掃 DIRTY_SCAN_ROOT 下所有 per-scene marker（dirty.txt / blur.txt）。
    空檔即旗標，回傳 {scene_name}（= marker 所在資料夾名）。"""
    d = set()
    if not DIRTY_SCAN_ROOT.exists():
        return d
    for marker in DIRTY_MARKERS:
        for p in DIRTY_SCAN_ROOT.rglob(marker):
            if p.parent == DIRTY_SCAN_ROOT:   # 排除頂層舊 central 檔（若還在）
                continue
            d.add(p.parent.name)
    return d


def collect_unique_scenes(split_dir: Path, n_folds: int) -> dict[str, list[tuple[int, str]]]:
    """掃 5 fold × 3 split 所有 txt，回傳 {scene_name: [(fold, split), ...]}."""
    assign: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for fold in range(1, n_folds + 1):
        for split in ('train', 'val', 'test'):
            txt = split_dir / f'{split}_fold{fold}.txt'
            if not txt.exists():
                continue
            for line in txt.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                scene = line.split('/', 1)[-1]
                assign[scene].append((fold, split))
    return assign


def patches_for_component(rmin, cmin, rmax, cmax, H, W):
    """回傳 ([(row_off, col_off), ...], tiled_bool). bbox inclusive."""
    bbox_h = rmax - rmin + 1
    bbox_w = cmax - cmin + 1

    def clip(off, dim_size):
        return max(0, min(off, dim_size - PATCH_SIZE))

    if bbox_h <= PATCH_SIZE and bbox_w <= PATCH_SIZE:
        cy = (rmin + rmax) // 2
        cx = (cmin + cmax) // 2
        return [(clip(cy - PATCH_SIZE // 2, H), clip(cx - PATCH_SIZE // 2, W))], False

    n_rows = max(1, -(-bbox_h // PATCH_SIZE))
    n_cols = max(1, -(-bbox_w // PATCH_SIZE))
    coords = []
    for i in range(n_rows):
        if n_rows == 1:
            center_r = (rmin + rmax) // 2
        else:
            t = i / (n_rows - 1)
            center_r = int(round(rmin + PATCH_SIZE / 2 + t * (bbox_h - PATCH_SIZE)))
        for j in range(n_cols):
            if n_cols == 1:
                center_c = (cmin + cmax) // 2
            else:
                t = j / (n_cols - 1)
                center_c = int(round(cmin + PATCH_SIZE / 2 + t * (bbox_w - PATCH_SIZE)))
            coords.append((clip(center_r - PATCH_SIZE // 2, H),
                           clip(center_c - PATCH_SIZE // 2, W)))
    return list(dict.fromkeys(coords)), True


def process_scene(scene_name: str):
    """回傳 (scene_name, patches, stats)."""
    mask_path = MASK_DIR / f'{scene_name}.tif'
    if not mask_path.exists():
        return scene_name, [], [{'scene_name': scene_name, 'error': 'mask_not_found'}]

    with rasterio.open(mask_path) as src:
        raw = src.read(1)
    mask = (raw == 0).astype(np.uint8)
    H, W = mask.shape

    if mask.sum() == 0:
        return scene_name, [], [{'scene_name': scene_name, 'error': 'no_oil_pixels'}]

    n_labels, _, cc_stats, _ = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
    patches = []
    stats = []
    for comp_id in range(1, n_labels):
        x, y, w, h, area = cc_stats[comp_id]
        rmin, rmax = int(y), int(y + h - 1)
        cmin, cmax = int(x), int(x + w - 1)
        coords, tiled = patches_for_component(rmin, cmin, rmax, cmax, H, W)
        patches.extend(coords)
        stats.append({
            'scene_name': scene_name,
            'comp_id': comp_id,
            'bbox_h': int(h),
            'bbox_w': int(w),
            'n_oil': int(area),
            'n_patches': len(coords),
            'tiled': tiled,
        })
    patches = list(dict.fromkeys(patches))
    return scene_name, patches, stats


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--sanity':
        sanity_check()
        return

    dirty = _load_dirty()
    assign = collect_unique_scenes(SCENE_SPLIT_DIR, N_FOLDS)
    # [dirty filter] 從 assign 移除髒場景
    dirty_hit = [s for s in list(assign) if s in dirty]
    for s in dirty_hit:
        assign.pop(s)
    scenes = sorted(assign)
    print(f'split_dir: {SCENE_SPLIT_DIR}')
    print(f'out_dir:   {OUT_DIR}')
    print(f'n_folds:   {N_FOLDS}')
    print(f'dirty:     {len(dirty)} listed, {len(dirty_hit)} hit → 已跳過')
    print(f'unique scenes to process: {len(scenes)}')
    print(f'workers: {NUM_WORKERS}\n')

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    scene_patches: dict[str, list[tuple[int, int]]] = {}
    all_stats = []

    with Pool(NUM_WORKERS) as pool:
        for scene_name, patches, stats in tqdm(
                pool.imap_unordered(process_scene, scenes), total=len(scenes), desc='scenes'):
            scene_patches[scene_name] = patches
            all_stats.extend(stats)

    # 按 fold/split 分派
    bucket: dict[tuple[int, str], list[str]] = defaultdict(list)
    for scene, assignments in assign.items():
        patches = scene_patches.get(scene, [])
        entries = [f'{scene}_patch_x{c}_y{r}' for (r, c) in patches]
        for fold, split in assignments:
            bucket[(fold, split)].extend(entries)

    print('\n=== 輸出 ===')
    for fold in range(1, N_FOLDS + 1):
        for split in ('train', 'val', 'test'):
            entries = bucket.get((fold, split), [])
            out_txt = OUT_DIR / f'{split}_fold{fold}.txt'
            out_txt.write_text('\n'.join(entries) + '\n')
            print(f'  {out_txt.name}: {len(entries)} patches')

    # stats csv
    fieldnames = ['scene_name', 'comp_id', 'bbox_h', 'bbox_w',
                  'n_oil', 'n_patches', 'tiled', 'error']
    with (OUT_DIR / 'stats.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for s in all_stats:
            w.writerow(s)
    print(f'\n stats → {OUT_DIR / "stats.csv"}')

    # 彙總印一下
    n_err = sum(1 for s in all_stats if s.get('error'))
    n_comp = sum(1 for s in all_stats if not s.get('error'))
    n_patch_all = sum(s.get('n_patches', 0) for s in all_stats)
    n_tiled = sum(1 for s in all_stats if s.get('tiled'))
    print(f'\n=== summary ===')
    print(f'  scenes errored:     {n_err}')
    print(f'  total components:   {n_comp}')
    print(f'  total patches:      {n_patch_all} (unique per scene, before fold dup)')
    print(f'  tiled components:   {n_tiled}')


def sanity_check():
    scene_txt = SCENE_SPLIT_DIR / 'train_fold1.txt'
    line0 = next(l for l in scene_txt.read_text().splitlines() if l.strip())
    scene = line0.split('/', 1)[-1]
    print(f'first scene: {scene}\n')
    _, patches, stats = process_scene(scene)
    print(f'  components: {len(stats)}')
    print(f'  patches:    {len(patches)}')
    for s in stats:
        print(f'    comp{s["comp_id"]}: bbox={s["bbox_h"]}x{s["bbox_w"]} '
              f'n_oil={s["n_oil"]} → {s["n_patches"]} patch(es) tiled={s["tiled"]}')


if __name__ == '__main__':
    main()
