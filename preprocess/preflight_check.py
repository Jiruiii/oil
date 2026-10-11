"""[F3] 訓練啟動前最小 preflight 驗證器（2026-07-11 Claude×Codex 共識第一批）。

驗證 fold txt + bg_coords.tsv 引用到的每個 unique scene：
  1. VRT 存在且可讀（實際讀一個小 window）
  2. Mask 存在且可讀
  3. VRT 與 Mask 的 shape / transform / CRS 對齊
  4. Mask 像素值 ⊆ {0(oil), 1(bg), 2(other), 255(unlabeled)}
  5. VRT 波段數 = 8

任一項不符 → exit 1 並列出全部問題（fail-fast，不靜默）。

usage:
  python preprocess/preflight_check.py \
      --split-dir /mnt/backup/oil_dataset/new/full_band/data_split/3_fold_stratified_v2/patch_level_gt_expand \
      --bg-tsv    /mnt/backup/oil_dataset/new/full_band/data_split/3_fold_stratified_v2/patch_level_gt_expand_bg/bg_coords.tsv \
      --vrt-dir   /home/alanyh/oil_dataset/new/full_band/MS6_sen2like_vrt \
      --mask-dir  /home/alanyh/oil_dataset/new/full_band/mask
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

VALID_MASK_VALUES = {0, 1, 2, 255}
PATCH_RE = re.compile(r'^(.+)_patch_x(\d+)_y(\d+)$')


def collect_scenes(split_dir: Path, bg_tsv: Path | None) -> set[str]:
    scenes: set[str] = set()
    txts = sorted(split_dir.glob('*_fold*.txt'))
    if not txts:
        sys.exit(f'[F3] {split_dir} 內找不到任何 *_fold*.txt')
    for txt in txts:
        for line in txt.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            m = PATCH_RE.match(Path(line).stem)
            if not m:
                sys.exit(f'[F3] {txt.name} 內有無法解析的 patch 名: {line!r}')
            scenes.add(m.group(1))
    if bg_tsv is not None:
        with open(bg_tsv) as f:
            for row in csv.DictReader(f, delimiter='\t'):
                scenes.add(row['scene'])
    return scenes


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--split-dir', required=True, type=Path)
    ap.add_argument('--bg-tsv', type=Path, default=None)
    ap.add_argument('--vrt-dir', required=True, type=Path)
    ap.add_argument('--mask-dir', required=True, type=Path)
    args = ap.parse_args()

    scenes = collect_scenes(args.split_dir, args.bg_tsv)
    vrt_map = {p.stem: p for p in args.vrt_dir.rglob('*')
               if p.suffix.lower() in ('.tif', '.tiff', '.vrt')}

    problems: list[str] = []
    for scene in sorted(scenes):
        vrt_path = vrt_map.get(scene)
        if vrt_path is None:
            problems.append(f'{scene}: 找不到 VRT')
            continue
        mask_path = args.mask_dir / f'{scene}.tif'
        if not mask_path.exists():
            problems.append(f'{scene}: 找不到 mask {mask_path}')
            continue
        try:
            with rasterio.open(vrt_path) as v:
                if v.count != 8:
                    problems.append(f'{scene}: VRT 波段數 {v.count} != 8')
                v.read(1, window=Window(0, 0, 16, 16))  # 實讀驗證 source band 可用
                v_shape, v_transform, v_crs = (v.height, v.width), v.transform, v.crs
        except Exception as e:
            problems.append(f'{scene}: VRT 讀取失敗: {e}')
            continue
        try:
            with rasterio.open(mask_path) as m:
                mask = m.read(1)
                if (m.height, m.width) != v_shape:
                    problems.append(f'{scene}: mask shape {(m.height, m.width)} != VRT {v_shape}')
                if not np.allclose(tuple(m.transform), tuple(v_transform), atol=1e-6):
                    problems.append(f'{scene}: mask transform 與 VRT 不一致')
                if m.crs != v_crs:
                    problems.append(f'{scene}: mask CRS {m.crs} != VRT {v_crs}')
                bad = set(np.unique(mask).tolist()) - VALID_MASK_VALUES
                if bad:
                    problems.append(f'{scene}: mask 含非法像素值 {sorted(bad)}')
        except Exception as e:
            problems.append(f'{scene}: mask 讀取失敗: {e}')

    print(f'[F3] preflight: {len(scenes)} unique scenes 檢查完畢')
    if problems:
        print(f'[F3] ❌ {len(problems)} 個問題:')
        for p in problems:
            print(f'  - {p}')
        sys.exit(1)
    print('[F3] ✅ ALL PASS')


if __name__ == '__main__':
    main()
