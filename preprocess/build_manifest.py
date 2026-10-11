"""Dataset manifest builder（單一真相帳本）.

掃描 full_band 資料集,每場景一列,輸出 manifest.csv 到資料集根目錄。
用途:
  1. 增刪改的入口——新場景入庫後重跑本腳本,diff 會列出新增/缺漏;
  2. mask_md5 追蹤標註變更——GT 一改,diff 立刻列出「哪些場景的
     patch 座標已過期、影響哪些 split」;
  3. 完整性檢查——8 band TIF / VRT / mask / JSON 缺一即在 issues 欄標記。

行為:重跑即更新(idempotent)。若既有 manifest 存在,先備份成
manifest.prev.csv 並印出與上一版的差異(新增/移除/mask 變更)。
只讀資料集與 split 檔,唯一寫入 = manifest.csv / manifest.prev.csv。

用法:
  /opt/conda/bin/python preprocess/build_manifest.py
"""
import csv
import hashlib
import re
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

DATA_ROOT   = Path('/home/alanyh/oil_dataset/new/full_band')
IMG_ROOT    = DATA_ROOT / 'MS6_sen2like'
MASK_DIR    = DATA_ROOT / 'mask'
JSON_DIR    = DATA_ROOT / 'MS6_sen2like_JSON'   # 集中目錄;另一存放處=場景資料夾內(兩處都算有)
VRT_HOME    = DATA_ROOT / 'MS6_sen2like_vrt'
VRT_BACKUP  = Path('/mnt/backup/oil_dataset/new/full_band/MS6_sen2like_vrt')
SPLIT_SV2   = Path('/mnt/backup/oil_dataset/new/full_band/data_split/3_fold_stratified_v2')
OUT_CSV     = DATA_ROOT / 'manifest.csv'

EXPECTED_BANDS = ['B01', 'B02', 'B03', 'B04', 'B08', 'B8A', 'B11', 'B12']
DIRTY_MARKERS  = ('dirty.txt', 'blur.txt')
_SCENE_RE = re.compile(r'_(\d{8})_(S2|L8|L9)_([A-Z0-9]+)$')


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _scan_scene_dirs():
    """任何含 *_B02.tif 的資料夾 = 場景資料夾(層級深度不一,以此為準)。"""
    return sorted({p.parent for p in IMG_ROOT.rglob('*_B02.tif')})


def _load_metadata():
    meta = {}
    p = SPLIT_SV2 / 'scene_metadata.csv'
    if p.exists():
        for r in csv.DictReader(open(p)):
            meta[r['scene']] = r
    return meta


def _load_splits():
    """scene → 'train_fold1;test_fold2;...'(3_fold_stratified_v2 scene_level)"""
    member = defaultdict(list)
    d = SPLIT_SV2 / 'scene_level'
    if d.is_dir():
        for txt in sorted(d.glob('*.txt')):
            for line in open(txt):
                s = line.strip()
                if s:
                    member[Path(s).name].append(txt.stem)
    return member


def _row(args):
    scene_dir, meta, splits = args
    sid = scene_dir.name
    m = _SCENE_RE.search(sid)
    date   = m.group(1) if m else ''
    sensor = m.group(2) if m else ''
    tile   = m.group(3) if m else ''

    bands_present = [b for b in EXPECTED_BANDS if (scene_dir / f'{sid}_{b}.tif').exists()]
    mask_p = MASK_DIR / f'{sid}.tif'
    # JSON 實際分佈在兩處:163 個只在集中目錄、1 個只在場景資料夾、其餘兩處都有(2026-07-10 盤點)
    json_p = next((p for p in (scene_dir / f'{sid}.json', JSON_DIR / f'{sid}.json') if p.exists()), None)
    vrt_h  = list(VRT_HOME.rglob(f'{sid}.vrt'))
    vrt_b  = list(VRT_BACKUP.rglob(f'{sid}.vrt')) if VRT_BACKUP.exists() else []
    dirty  = any((scene_dir / mk).exists() for mk in DIRTY_MARKERS)

    issues = []
    if len(bands_present) != len(EXPECTED_BANDS):
        missing = set(EXPECTED_BANDS) - set(bands_present)
        issues.append(f'missing_bands:{"+".join(sorted(missing))}')
    if not mask_p.exists(): issues.append('no_mask')
    if json_p is None:      issues.append('no_json')
    if not vrt_h:           issues.append('no_vrt_home')

    md = meta.get(sid, {})
    return {
        'scene_id':    sid,
        'event_group': scene_dir.relative_to(IMG_ROOT).parts[0],
        'sensor':      sensor or md.get('sensor', ''),
        'date':        date,
        'tile':        tile,
        'image_dir':   str(scene_dir.relative_to(DATA_ROOT)),
        'n_band_tifs': len(bands_present),
        'vrt_home':    str(vrt_h[0].relative_to(DATA_ROOT)) if vrt_h else '',
        'vrt_backup_exists': int(bool(vrt_b)),
        'mask_path':   str(mask_p.relative_to(DATA_ROOT)) if mask_p.exists() else '',
        'mask_md5':    _md5(mask_p) if mask_p.exists() else '',
        'json_exists': int(json_p is not None),
        'dirty':       int(dirty),
        'oil_px':      md.get('oil_px', ''),
        'oil_bin':     md.get('oil_bin_name', ''),
        'splits_sv2':  ';'.join(splits.get(sid, [])),
        'issues':      ';'.join(issues),
    }


def main():
    scene_dirs = _scan_scene_dirs()
    print(f'[scan] {len(scene_dirs)} scene dirs under {IMG_ROOT}')
    meta, splits = _load_metadata(), _load_splits()

    with Pool(8) as pool:
        rows = pool.map(_row, [(d, meta, splits) for d in scene_dirs])
    rows.sort(key=lambda r: r['scene_id'])

    # 與上一版 diff（mask 變更 = patch 座標過期警報）
    if OUT_CSV.exists():
        prev = {r['scene_id']: r for r in csv.DictReader(open(OUT_CSV))}
        cur  = {r['scene_id']: r for r in rows}
        added   = sorted(set(cur) - set(prev))
        removed = sorted(set(prev) - set(cur))
        changed = sorted(s for s in set(cur) & set(prev)
                         if prev[s].get('mask_md5') and cur[s]['mask_md5'] != prev[s]['mask_md5'])
        print(f'[diff] added={len(added)} removed={len(removed)} mask_changed={len(changed)}')
        for s in added:   print(f'  + {s}')
        for s in removed: print(f'  - {s}')
        for s in changed:
            print(f'  ~ MASK CHANGED: {s}  (splits: {cur[s]["splits_sv2"] or "-"})'
                  f'  → 該場景 patch 座標已過期,需重生 patch_level_gt_expand')
        OUT_CSV.replace(OUT_CSV.with_name('manifest.prev.csv'))

    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    n_issue = sum(1 for r in rows if r['issues'])
    n_dirty = sum(r['dirty'] for r in rows)
    print(f'[write] {OUT_CSV}  ({len(rows)} scenes; dirty={n_dirty}; issues={n_issue})')
    if n_issue:
        print('[issues]')
        for r in rows:
            if r['issues']:
                print(f'  {r["scene_id"]}: {r["issues"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
