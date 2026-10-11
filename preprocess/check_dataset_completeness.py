"""
資料集完整性檢查：掃描 MS6_sen2like 所有場景，逐一確認：
  [J] LabelMe JSON  — 存在於場景目錄內
  [M] Mask TIF      — 存在於 mask/ 目錄
  [C] COG（已 Tile）— 場景內 7 個波段 TIF 全部為 tiled

執行：
  python check_dataset_completeness.py
  python check_dataset_completeness.py --fix-json   # 補跑缺 JSON 的場景（GPKG → JSON）
"""

import argparse
import json
import re
from pathlib import Path

import rasterio
from tqdm import tqdm

# ── 路徑 ──────────────────────────────────────────────────────────────────────
SCENE_BASE = Path('/home/alanyh/oil_dataset/new/full_band/MS6_sen2like')
MASK_DIR   = Path('/home/alanyh/oil_dataset/new/full_band/mask')
GPKG_TO_LABELME = Path(__file__).parent.parent / 'file_label_temp' / 'gpkg_to_labelme.py'

# 7 個需要 COG 的波段（B01 60m 不列入）
COG_BANDS = ['B02', 'B03', 'B04', 'B08', 'B8A', 'B11', 'B12']


# ── 輔助函式 ──────────────────────────────────────────────────────────────────

def is_tiled(tif_path: Path) -> bool:
    try:
        with rasterio.open(str(tif_path)) as src:
            bs = src.block_shapes
            return bool(bs) and bs[0][0] != src.height
    except Exception:
        return False


def find_all_scenes() -> list[Path]:
    """回傳所有場景目錄（含 *_B02.tif 的目錄）。"""
    scenes = []
    for p in sorted(SCENE_BASE.rglob('*_B02.tif')):
        scenes.append(p.parent)
    return scenes


def check_scene(scene_dir: Path) -> dict:
    name = scene_dir.name

    # JSON
    json_ok = bool(list(scene_dir.glob('*.json')))

    # Mask
    mask_ok = (MASK_DIR / f'{name}.tif').exists()

    # COG
    cog_missing = []
    for band in COG_BANDS:
        tif = scene_dir / f'{name}_{band}.tif'
        if not tif.exists():
            cog_missing.append(f'{band}:missing')
        elif not is_tiled(tif):
            cog_missing.append(f'{band}:not_tiled')
    cog_ok = len(cog_missing) == 0

    return {
        'name':        name,
        'dir':         scene_dir,
        'json_ok':     json_ok,
        'mask_ok':     mask_ok,
        'cog_ok':      cog_ok,
        'cog_missing': cog_missing,
    }


# ── 補跑 JSON（--fix-json 模式）────────────────────────────────────────────────

def fix_missing_json(scenes_no_json: list[dict]):
    import subprocess, sys
    py = sys.executable
    script = str(GPKG_TO_LABELME)
    if not Path(script).exists():
        print(f'⚠ 找不到 gpkg_to_labelme.py：{script}')
        return

    ok = err = 0
    for s in tqdm(scenes_no_json, desc='GPKG→JSON'):
        scene_dir = s['dir']
        name      = s['name']
        ref_tif   = scene_dir / f'{name}_B02.tif'
        out_json  = scene_dir / f'{name}.json'
        if not ref_tif.exists():
            print(f'  ⚠ 無 B02.tif，跳過：{name}')
            err += 1
            continue
        ret = subprocess.run(
            [py, script,
             '--scene_dir', str(scene_dir),
             '--ref_tif',   str(ref_tif),
             '--output',    str(out_json)],
            capture_output=True, text=True)
        if ret.returncode == 0:
            ok += 1
        else:
            print(f'  ✗ {name}\n{ret.stderr.strip()[:200]}')
            err += 1
    print(f'\nGPKG→JSON 完成：{ok} 成功，{err} 失敗')
    print('完成後請再跑：')
    print('  python json_to_mask_tif.py')


# ── 主程式 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fix-json', action='store_true',
                        help='自動補跑缺 JSON 的場景（GPKG→JSON）')
    args = parser.parse_args()

    print(f'掃描：{SCENE_BASE}')
    scenes = find_all_scenes()
    print(f'找到 {len(scenes)} 個場景\n')

    results = []
    for sd in tqdm(scenes, desc='檢查'):
        results.append(check_scene(sd))

    # ── 統計 ──────────────────────────────────────────────────────────────────
    no_json = [r for r in results if not r['json_ok']]
    no_mask = [r for r in results if not r['mask_ok']]
    no_cog  = [r for r in results if not r['cog_ok']]
    all_ok  = [r for r in results if r['json_ok'] and r['mask_ok'] and r['cog_ok']]

    print('\n' + '='*70)
    print(f'總場景：{len(results)}')
    print(f'  全部 OK（JSON+Mask+COG）：{len(all_ok)}')
    print(f'  缺 JSON ：{len(no_json)}')
    print(f'  缺 Mask ：{len(no_mask)}')
    print(f'  非 COG  ：{len(no_cog)}')
    print('='*70)

    if no_json:
        print(f'\n▶ 缺 JSON（{len(no_json)} 個）：')
        for r in no_json:
            print(f'  ✗ {r["name"]}')

    if no_mask:
        print(f'\n▶ 缺 Mask（{len(no_mask)} 個）：')
        for r in no_mask:
            print(f'  ✗ {r["name"]}')

    if no_cog:
        print(f'\n▶ 非 COG（{len(no_cog)} 個）：')
        for r in no_cog:
            print(f'  ✗ {r["name"]}  [{", ".join(r["cog_missing"])}]')

    if not (no_json or no_mask or no_cog):
        print('\n✓ 所有場景完整，無缺漏。')
    else:
        print('\n建議補跑順序：')
        if no_json:
            print('  1. python check_dataset_completeness.py --fix-json')
            print('  2. python json_to_mask_tif.py')
        elif no_mask:
            print('  1. python json_to_mask_tif.py')
        if no_cog:
            print('  （COG）python /home/alanyh/oil_dataset/new/full_band/convert_to_cog.py')

    # --fix-json
    if args.fix_json:
        if no_json:
            print(f'\n▶ 開始補跑 {len(no_json)} 個缺 JSON 場景...')
            fix_missing_json(no_json)
        else:
            print('\n所有場景都已有 JSON，無需補跑。')


if __name__ == '__main__':
    main()
