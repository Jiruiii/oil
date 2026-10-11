"""Stratified 3-fold split v2.

策略：
  1. 從 dirty.txt 排除髒場景（GT 油汙不明顯的）
  2. 對剩下場景讀 mask，算 oil pixel count
  3. 依 oil 面積 tertile（小/中/大）分 3 層
  4. StratifiedKFold(n_splits=3) → 每 fold test 都有均衡的三種難度
  5. 每 fold 的 train+val 從其餘場景抽（80/20，也分層）
  6. 輸出到 3_fold_stratified_v2/scene_level/

Audit：跑完印每 fold 的 stratum × count 表。
"""
import re, csv, random
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import rasterio
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

# ---------- 路徑 ----------
SRC_5FOLD       = Path('/mnt/backup/oil_dataset/new/full_band/data_split/5_fold/scene_level')
# 髒資料以 per-scene 空檔標記（dataset 慣例）：{scene_dir}/dirty.txt 或 blur.txt 存在即排除
DIRTY_SCAN_ROOT = Path('/home/alanyh/oil_dataset/new/full_band/MS6_sen2like')
DIRTY_MARKERS   = ('dirty.txt', 'blur.txt')
MASK_DIR  = Path('/home/alanyh/oil_dataset/new/full_band/mask')
OUT_BASE  = Path('/mnt/backup/oil_dataset/new/full_band/data_split/3_fold_stratified_v2')
OUT_SCENE = OUT_BASE / 'scene_level'

SEED      = 42
N_FOLDS   = 3
VAL_RATIO = 0.20     # 從 train+val pool 分出 20% 當 val
N_WORKERS = 8


# ---------- 工具 ----------
def load_dirty():
    """掃 DIRTY_SCAN_ROOT 下所有 per-scene marker（dirty.txt / blur.txt）→ {scene_name}。"""
    d = set()
    if not DIRTY_SCAN_ROOT.exists():
        return d
    for marker in DIRTY_MARKERS:
        for p in DIRTY_SCAN_ROOT.rglob(marker):
            if p.parent == DIRTY_SCAN_ROOT:
                continue
            d.add(p.parent.name)
    return d


def collect_all_scenes():
    """從舊 scene_level 集合 union 出所有場景（去重）。"""
    scenes = set()
    for f in SRC_5FOLD.glob('*.txt'):
        for l in f.read_text().splitlines():
            l = l.strip()
            if l:
                scenes.add(l.split('/')[-1])
    return sorted(scenes)


def compute_oil_px(scene_name):
    mp = MASK_DIR / f'{scene_name}.tif'
    if not mp.exists():
        return scene_name, None
    try:
        with rasterio.open(mp) as src:
            m = src.read(1)
        return scene_name, int((m == 0).sum())
    except Exception:
        return scene_name, None


def parse_sensor(scene_name):
    """從場景檔名 parse S2 / L8 / L9 (or 其他)."""
    m = re.search(r'_(S2|L8|L9)_', scene_name)
    return m.group(1) if m else 'OTH'


def bin_oil_size(oil_px, bin_edges):
    """回傳 bin index: 0 小 / 1 中 / 2 大."""
    if oil_px is None or oil_px <= 0:
        return 0
    if oil_px < bin_edges[0]:  return 0
    if oil_px < bin_edges[1]:  return 1
    return 2


BIN_NAMES = ['small', 'medium', 'large']


# ---------- 主流程 ----------
def main():
    dirty = load_dirty()
    print(f'[dirty] {len(dirty)} 髒場景將被排除')

    scenes = collect_all_scenes()
    scenes = [s for s in scenes if s not in dirty]
    print(f'[scenes] 從 5-fold union 拿到 {len(scenes)} 個（已去 dirty）')

    # 讀 oil pixel count
    print(f'[audit] 讀 mask 算 oil pixel count（{N_WORKERS} workers）...')
    oil_counts = {}
    with Pool(N_WORKERS) as pool:
        for s, oil in tqdm(pool.imap_unordered(compute_oil_px, scenes), total=len(scenes)):
            if oil is not None:
                oil_counts[s] = oil
    print(f'  成功讀到 {len(oil_counts)} 張')

    # 剔除 mask 讀不到的
    scenes = [s for s in scenes if s in oil_counts]

    # 計算 tertile 分箱 edge
    oils = np.array([oil_counts[s] for s in scenes])
    e1, e2 = np.quantile(oils, [1/3, 2/3])
    bin_edges = (float(e1), float(e2))
    print(f'\n[bin] oil pixel tertile edges: {int(e1):,} / {int(e2):,}')
    print(f'      small: <{int(e1):,}  medium: {int(e1):,}~{int(e2):,}  large: >{int(e2):,}')

    # 每個場景的 stratum label + metadata
    scene_meta = []
    for s in scenes:
        px = oil_counts[s]
        b = bin_oil_size(px, bin_edges)
        sens = parse_sensor(s)
        scene_meta.append({'scene': s, 'oil_px': px, 'oil_bin': b, 'oil_bin_name': BIN_NAMES[b], 'sensor': sens})

    # print stratum × 感測器分佈
    print(f'\n[分佈] stratum × 感測器 場景數:')
    print(f'{"":<10} {"S2":>5} {"L8":>5} {"L9":>5} {"OTH":>5} {"合計":>6}')
    for b, name in enumerate(BIN_NAMES):
        s2 = sum(1 for m in scene_meta if m['oil_bin'] == b and m['sensor'] == 'S2')
        l8 = sum(1 for m in scene_meta if m['oil_bin'] == b and m['sensor'] == 'L8')
        l9 = sum(1 for m in scene_meta if m['oil_bin'] == b and m['sensor'] == 'L9')
        oth = sum(1 for m in scene_meta if m['oil_bin'] == b and m['sensor'] == 'OTH')
        print(f'  {name:<10} {s2:>5} {l8:>5} {l9:>5} {oth:>5} {s2+l8+l9+oth:>6}')

    # ---------- Stratified 3-fold ----------
    labels = np.array([m['oil_bin'] for m in scene_meta])
    scenes_arr = np.array([m['scene'] for m in scene_meta])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_splits = {}
    for fi, (train_val_idx, test_idx) in enumerate(skf.split(scenes_arr, labels), start=1):
        test_scenes = scenes_arr[test_idx].tolist()

        # train_val_idx 再分 train/val（也分層）
        tv_labels = labels[train_val_idx]
        # sklearn 沒有直接的 StratifiedShuffle，用 seeded 分層抽樣
        rng = random.Random(SEED + fi)
        by_bin = {0: [], 1: [], 2: []}
        for i in train_val_idx:
            by_bin[labels[i]].append(i)
        val_idx, train_idx = [], []
        for b, idxs in by_bin.items():
            rng.shuffle(idxs)
            n_val = int(round(len(idxs) * VAL_RATIO))
            val_idx.extend(idxs[:n_val])
            train_idx.extend(idxs[n_val:])
        train_scenes = scenes_arr[train_idx].tolist()
        val_scenes   = scenes_arr[val_idx].tolist()

        fold_splits[fi] = {
            'train': sorted(train_scenes),
            'val':   sorted(val_scenes),
            'test':  sorted(test_scenes),
        }

    # ---------- 輸出 + audit ----------
    OUT_SCENE.mkdir(parents=True, exist_ok=True)

    print(f'\n[輸出] 到 {OUT_SCENE}')
    for fi, s in fold_splits.items():
        for split in ('train', 'val', 'test'):
            p = OUT_SCENE / f'{split}_fold{fi}.txt'
            p.write_text('\n'.join(s[split]) + '\n')
        # audit
        for split in ('train', 'val', 'test'):
            names = s[split]
            small = sum(1 for x in names if bin_oil_size(oil_counts[x], bin_edges) == 0)
            medi  = sum(1 for x in names if bin_oil_size(oil_counts[x], bin_edges) == 1)
            large = sum(1 for x in names if bin_oil_size(oil_counts[x], bin_edges) == 2)
            print(f'  fold{fi} {split:5}: 小={small}  中={medi}  大={large}  合計={len(names)}')

    # 也寫 metadata csv 給後續分析用
    meta_csv = OUT_BASE / 'scene_metadata.csv'
    with open(meta_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['scene', 'oil_px', 'oil_bin', 'oil_bin_name', 'sensor'])
        w.writeheader(); w.writerows(scene_meta)
    print(f'\n[metadata] 寫 {meta_csv} ({len(scene_meta)} 場景)')

    # 也寫 bin_edges 給後續使用
    (OUT_BASE / 'bin_edges.txt').write_text(f'small_upper={int(e1)}\nmedium_upper={int(e2)}\n')
    print(f'[bin_edges] {int(e1):,} / {int(e2):,}')

    print(f'\n✅ v2 3-fold stratified split 完成')
    print(f'   yaml 中 split_dir 改指向: {OUT_SCENE}')


if __name__ == '__main__':
    main()
