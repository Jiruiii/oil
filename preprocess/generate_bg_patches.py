"""BG patch coord generator (TODO/01 改動 C).

對 358 場景在「沒有油汙」的區域滑窗採樣 BG patch；輸出 tsv 供 deeplab_adapter
的 bg_coord_txt 機制混進訓練集，補強 GT_expand 缺乏背景知識的問題。

策略 (a)：BG 取自同場景油汙範圍外 → 保留同樣大氣 / 雲況 / 感測器 → 模型學到
「同一張圖內 oil 紋理 vs 非 oil 紋理」的判別。

每個場景滑窗 256×256 stride=512，候選 window 條件：
  1. (mask == 0).sum() == 0  → 零 oil 像素
  2. ((mask == 1) | (mask == 2)).sum() >= 0.30 × 256²  → 至少 30% 是 valid bg
  3. (mask == 255).mean() < 0.50  → 排除幾乎全是 ignore 的 patch

每場景隨機抽 N_PER_SCENE 個，seed 固定。輸出格式跟 deeplab_adapter._load_bg_coords 對齊
（含 scene/x/y 欄位；fold 欄留作向下相容但無效；實際分配靠 adapter 改用 scene_filter，
此檔產出 scene-only）。
"""
from pathlib import Path
import csv, random, sys
from multiprocessing import Pool, cpu_count
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

# ---------- 設定 ----------
import os as _os
# SPLIT_DIR/OUT_DIR 必填:舊版預設指向 5_fold,裸跑會靜默用過期 split 重生資料(footgun),故改為必給
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

PATCH_SIZE   = 256
STRIDE       = 512
N_PER_SCENE  = 10
MIN_VALID_BG_FRAC = 0.30
MAX_IGNORE_FRAC   = 0.50
SEED         = 42

# [P7] 配額模式(選用)。三個變數必須同時給或同時不給；不給時行為與舊版逐位元相同。
#   QUOTA_RATIO   : r，逐景 B_s = min(round(r * G_s), C_s)
#   GT_SPLIT_DIR  : patch-level split 目錄,用以統計逐景 GT patch 數 G_s
#   NEST_BASE_TSV : 既有 bg_coords.tsv,用以維持巢狀(nested)座標
_q = [_os.environ.get(k) for k in ('QUOTA_RATIO', 'GT_SPLIT_DIR', 'NEST_BASE_TSV')]
# [P8] POOL_CAP 模式同樣需要 GT_SPLIT_DIR + NEST_BASE_TSV 但不需要 QUOTA_RATIO,
# 故此處的三缺一檢查改為只在未進入 POOL_CAP 模式時生效(POOL_CAP 自己的檢查見下方)。
if not _os.environ.get('POOL_CAP') and any(_q) and not all(_q):
    sys.exit('[ERROR][P7] QUOTA_RATIO / GT_SPLIT_DIR / NEST_BASE_TSV 必須同時提供')
QUOTA_RATIO   = float(_q[0]) if _q[0] else None
GT_SPLIT_DIR  = Path(_q[1]) if _q[1] else None
NEST_BASE_TSV = Path(_q[2]) if _q[2] else None
GT_COUNT: dict[str, int] = {}
BASE_COORDS: dict[str, list] = {}

# [P8] 擴池模式(選用)。q_s = min(C_s^pass, POOL_CAP),append-only 疊在 NEST_BASE_TSV 之上。
#   與 QUOTA_RATIO 互斥。需 GT_SPLIT_DIR + NEST_BASE_TSV,並必須給 PREFLIGHT_VRT_DIR:
#   extra 座標須通過與 runtime _get_coord_vrt_item 同準則的 NaN 預檢(codex R5-2),
#   否則名目池數 ≠ 實際曝光。base 座標同樣預檢但只記錄、不剔除(Arm A 已凍結)。
POOL_CAP = int(_os.environ['POOL_CAP']) if _os.environ.get('POOL_CAP') else None
PREFLIGHT_VRT_DIR = Path(_os.environ['PREFLIGHT_VRT_DIR']) if _os.environ.get('PREFLIGHT_VRT_DIR') else None
if POOL_CAP is not None:
    if QUOTA_RATIO is not None:
        sys.exit('[ERROR][P8] POOL_CAP 與 QUOTA_RATIO 互斥')
    if not (GT_SPLIT_DIR and NEST_BASE_TSV and PREFLIGHT_VRT_DIR):
        sys.exit('[ERROR][P8] POOL_CAP 模式需同時給 GT_SPLIT_DIR / NEST_BASE_TSV / PREFLIGHT_VRT_DIR')
_VRT_MAP: dict = {}


def _preflight_ok(vrt_path, r: int, c: int) -> bool:
    """與 deeplab_adapter._get_coord_vrt_item 同準則:nodata(NaN) 比例 > 10% 視為無效。
    讀取失敗一律視為無效(不 raise;預檢階段的目的就是把這些座標排除在池外)。"""
    import rasterio
    from rasterio.windows import Window
    try:
        with rasterio.open(str(vrt_path)) as src:
            patch = src.read(window=Window(c, r, PATCH_SIZE, PATCH_SIZE),
                             boundless=True, fill_value=0).astype(np.float32) / 10000.0
    except Exception:
        return False
    return not (np.isnan(patch).mean() > 0.1)


def _vrt_for(scene: str):
    if not _VRT_MAP:
        for p in PREFLIGHT_VRT_DIR.rglob('*'):
            if p.suffix.lower() in ('.tif', '.tiff', '.vrt'):
                _VRT_MAP.setdefault(p.stem, p)
    return _VRT_MAP.get(scene, PREFLIGHT_VRT_DIR / f'{scene}.tif')
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
NUM_WORKERS  = 8


def _load_quota_inputs():
    """[P7] 載入逐景 GT patch 數 G_s 與既有 BG 座標(巢狀基底)。填入模組層全域。"""
    import re as _r
    pat = _r.compile(r'^(.*)_patch_x(\d+)_y(\d+)$')
    # G_s：fold1 的 train/val/test 聯集 = 全部 GT patch,每張剛好一次
    for split in ('train', 'val', 'test'):
        fp = GT_SPLIT_DIR / f'{split}_fold1.txt'
        if not fp.exists():
            sys.exit(f'[ERROR][P7] 缺少 {fp}')
        for line in fp.read_text().splitlines():
            s = line.strip()
            if not s:
                continue
            m = pat.match(s.split('/')[-1])
            if not m:
                sys.exit(f'[ERROR][P7] 無法解析 patch 名稱: {s}')
            GT_COUNT[m.group(1)] = GT_COUNT.get(m.group(1), 0) + 1
    # 巢狀基底：保留檔案順序
    with open(NEST_BASE_TSV) as f:
        for row in csv.DictReader(f, delimiter='\t'):
            BASE_COORDS.setdefault(row['scene'], []).append((int(row['y']), int(row['x'])))
    _mode = f'[P7] 配額模式 r={QUOTA_RATIO}' if QUOTA_RATIO is not None else f'[P8] 擴池模式 cap={POOL_CAP}'
    print(f'{_mode}；G_s 覆蓋 {len(GT_COUNT)} 景，'
          f'巢狀基底 {sum(len(v) for v in BASE_COORDS.values())} 座標 / {len(BASE_COORDS)} 景')


def _load_dirty():
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


def collect_unique_scenes() -> list[str]:
    """掃 scene_level 內所有 fold 的 test_fold{N}.txt（每場景在剛好 1 個 test fold）作為唯一場景清單。"""
    scenes = set()
    for n in range(1, N_FOLDS + 1):
        for split in ('train', 'val', 'test'):
            p = SCENE_SPLIT_DIR / f'{split}_fold{n}.txt'
            if p.exists():
                for line in p.read_text().splitlines():
                    s = line.strip()
                    if s:
                        # scene path 可能含子目錄前綴 (e.g. "NOAA_xxx/NOAA_xxx_20250418_S2_19TDE")
                        scenes.add(s.split('/')[-1])
    return sorted(scenes)


def find_candidates(mask: np.ndarray):
    """滑窗找符合條件的 (row_off, col_off)；回傳 list。"""
    H, W = mask.shape
    cands = []
    for r in range(0, H - PATCH_SIZE + 1, STRIDE):
        for c in range(0, W - PATCH_SIZE + 1, STRIDE):
            win = mask[r:r+PATCH_SIZE, c:c+PATCH_SIZE]
            # 條件 1：零 oil
            if (win == 0).any():
                continue
            # 條件 3：ignore 不能太高
            if (win == 255).mean() >= MAX_IGNORE_FRAC:
                continue
            # 條件 2：valid bg 至少 30%
            valid_bg = ((win == 1) | (win == 2)).sum()
            if valid_bg < MIN_VALID_BG_FRAC * PATCH_SIZE * PATCH_SIZE:
                continue
            cands.append((r, c))
    return cands


def process_scene(scene_name: str):
    mp = MASK_DIR / f'{scene_name}.tif'
    if not mp.exists():
        return scene_name, [], 'mask_not_found'
    try:
        with rasterio.open(mp) as src:
            mask = src.read(1)
    except Exception as e:
        return scene_name, [], f'open_fail: {e}'

    cands = find_candidates(mask)
    if not cands:
        return scene_name, [], 'no_candidate'

    # 用 scene-name-derived seed 確保 reproducible 同時 scene 間獨立
    # [F7] Python 內建 hash() 每個 process 隨機化(PYTHONHASHSEED),重生會抽到不同座標;
    # 改用 blake2b stable digest。注意:與舊版(內建 hash)產出不同,現行凍結資料不受影響,
    # 只影響未來重生 bg_coords.tsv。seed algorithm: blake2b-8byte + SEED, v2026-07-11。
    import hashlib
    _digest = int.from_bytes(hashlib.blake2b(scene_name.encode('utf-8'), digest_size=8).digest(), 'big')
    rng = random.Random(SEED + _digest % (1 << 32))
    if POOL_CAP is not None:
        # [P8] 擴池:base 逐列保留原順序(append-only strict superset),extra 依凍結 shuffle
        # 順序逐一 NaN 預檢,只收通過者,補到 min(C_s^pass, POOL_CAP)。
        vrt = _vrt_for(scene_name)
        cset = set(cands)
        base = [rc for rc in BASE_COORDS.get(scene_name, []) if rc in cset]
        if len(base) != len(BASE_COORDS.get(scene_name, [])):
            return scene_name, [], ('ERROR base_not_subset_of_candidates '
                                    f'(base={len(BASE_COORDS.get(scene_name, []))} '
                                    f'in_cands={len(base)})')
        base_pass = sum(1 for (r, c) in base if _preflight_ok(vrt, r, c))
        extra = [rc for rc in cands if rc not in set(base)]
        rng.shuffle(extra)
        need = POOL_CAP - len(base)
        picked, tested = [], 0
        for (r, c) in extra:
            if len(picked) >= need:
                break
            tested += 1
            if _preflight_ok(vrt, r, c):
                picked.append((r, c))
        sampled = base + picked
        return scene_name, sampled, (
            f'ok ({len(cands)} cands → sampled {len(sampled)}; '
            f'base={len(base)} base_pass={base_pass} extra_tested={tested} '
            f'extra_kept={len(picked)} cap={POOL_CAP})')

    if QUOTA_RATIO is None:
        sampled = rng.sample(cands, min(N_PER_SCENE, len(cands)))
        return scene_name, sampled, f'ok ({len(cands)} cands → sampled {len(sampled)})'

    # [P7] 配額模式：B_s = min(round(r * G_s), C_s)，且與既有座標維持單調嵌套
    g = GT_COUNT.get(scene_name, 0)
    target = int(round(QUOTA_RATIO * g))
    cset = set(cands)
    base = [rc for rc in BASE_COORDS.get(scene_name, []) if rc in cset]
    if target <= len(base):
        sampled = base[:target]
    else:
        extra = [rc for rc in cands if rc not in set(base)]
        rng.shuffle(extra)                      # 決定性：同一 rng 種子
        sampled = base + extra[:target - len(base)]
    return scene_name, sampled, (f'ok ({len(cands)} cands → sampled {len(sampled)}; '
                                 f'G={g} target={target} base={len(base)})')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if QUOTA_RATIO is not None or POOL_CAP is not None:
        _load_quota_inputs()
    dirty = _load_dirty()
    scenes = [s for s in collect_unique_scenes() if s not in dirty]
    print(f'split_dir: {SCENE_SPLIT_DIR}')
    print(f'out_dir:   {OUT_DIR}')
    print(f'n_folds:   {N_FOLDS}')
    print(f'dirty:     {len(dirty)} listed → 已跳過')
    print(f'唯一場景數: {len(scenes)}')
    _quota_desc = (f'QUOTA r={QUOTA_RATIO}' if QUOTA_RATIO is not None else
                   f'POOL_CAP={POOL_CAP}' if POOL_CAP is not None else
                   f'N_PER_SCENE={N_PER_SCENE}')
    print(f'設定: PATCH={PATCH_SIZE}, STRIDE={STRIDE}, {_quota_desc}, '
          f'MIN_VALID_BG={MIN_VALID_BG_FRAC:.0%}, MAX_IGNORE={MAX_IGNORE_FRAC:.0%}')

    rows = []
    stats = []
    with Pool(processes=NUM_WORKERS) as pool:
        for scene_name, sampled, msg in tqdm(
            pool.imap_unordered(process_scene, scenes), total=len(scenes), desc='gen_bg'
        ):
            for (r, c) in sampled:
                rows.append({'scene': scene_name, 'x': c, 'y': r})
            stats.append({'scene': scene_name, 'msg': msg, 'sampled': len(sampled)})

    # 寫主 tsv (scene/x/y 只記座標；adapter 用 scene_filter 機制過濾 fold)
    out_tsv = OUT_DIR / 'bg_coords.tsv'
    with open(out_tsv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['scene', 'x', 'y'], delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    print(f'\n✅ 寫入 {out_tsv}：{len(rows)} 個 BG patch (across {len(scenes)} 場景)')

    # stats.csv
    out_stats = OUT_DIR / 'stats.csv'
    with open(out_stats, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['scene', 'sampled', 'msg'])
        writer.writeheader()
        writer.writerows(stats)
    print(f'   stats: {out_stats}')

    # [P7] quota_freeze.csv：逐景 target / candidate / realized 凍結表
    if QUOTA_RATIO is not None:
        import re as _r
        freeze = []
        for s in stats:
            m = _r.search(r'\((\d+) cands', s['msg'] or '')
            c = int(m.group(1)) if m else 0
            g = GT_COUNT.get(s['scene'], 0)
            tgt = int(round(QUOTA_RATIO * g))
            freeze.append({'scene': s['scene'], 'G_s': g, 'target_B': tgt,
                           'candidate_C': c, 'realized_B': s['sampled'],
                           'capped': int(s['sampled'] < tgt)})
        out_freeze = OUT_DIR / 'quota_freeze.csv'
        with open(out_freeze, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['scene', 'G_s', 'target_B',
                                              'candidate_C', 'realized_B', 'capped'])
            w.writeheader()
            w.writerows(sorted(freeze, key=lambda r: r['scene']))
        n_cap = sum(r['capped'] for r in freeze)
        print(f'   quota_freeze: {out_freeze}')
        print(f'   target 合計 {sum(r["target_B"] for r in freeze)}, '
              f'realized 合計 {sum(r["realized_B"] for r in freeze)}, capped {n_cap} 景')

    # [P8] pool_freeze.csv：逐景池凍結表（codex R5-2/R5-5 要求的分欄記錄）
    #   C_s          = mask 條件下的合法候選數（未含 NaN 預檢）
    #   base_A       = Arm A 既有座標數 = 每 epoch draw 數 d_s（曝光鎖定量）
    #   base_A_pass  = 其中通過 NaN 預檢者（Arm A 已凍結不剔除，僅記錄殘餘不對稱）
    #   extra_tested = 為補滿 cap 而預檢的 extra 個數
    #   realized_q   = 實際池大小 |P^C_s|
    if POOL_CAP is not None:
        import re as _r
        def _f(msg, key, default=0):
            m = _r.search(rf'{key}=(\d+)', msg or '')
            return int(m.group(1)) if m else default
        freeze, errs = [], []
        for s in stats:
            msg = s['msg'] or ''
            # 'no_candidate' 是已知合法狀態（C_s=0 的景，兩臂皆得 0 張），不算異常;
            # 只有 ERROR（base 非候選集子集）與其他非預期 msg 才是巢狀檢查失敗。
            if not (msg.startswith('ok') or msg == 'no_candidate'):
                errs.append((s['scene'], msg))
            m = _r.search(r'\((\d+) cands', msg)
            freeze.append({'scene': s['scene'],
                           'G_s': GT_COUNT.get(s['scene'], 0),
                           'C_s': int(m.group(1)) if m else 0,
                           'base_A': _f(msg, 'base'),
                           'base_A_pass': _f(msg, 'base_pass'),
                           'extra_tested': _f(msg, 'extra_tested'),
                           'extra_kept': _f(msg, 'extra_kept'),
                           'target_q': POOL_CAP,
                           'realized_q': s['sampled'],
                           'capped': int(s['sampled'] < POOL_CAP)})
        out_freeze = OUT_DIR / 'pool_freeze.csv'
        with open(out_freeze, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['scene', 'G_s', 'C_s', 'base_A', 'base_A_pass',
                                              'extra_tested', 'extra_kept', 'target_q',
                                              'realized_q', 'capped'])
            w.writeheader()
            w.writerows(sorted(freeze, key=lambda r: r['scene']))
        tot = lambda k: sum(r[k] for r in freeze)
        print(f'   pool_freeze: {out_freeze}')
        print(f'   base_A 合計 {tot("base_A")}（通過預檢 {tot("base_A_pass")}，'
              f'不對稱 {tot("base_A") - tot("base_A_pass")}）')
        print(f'   extra 預檢 {tot("extra_tested")} → 收錄 {tot("extra_kept")}')
        print(f'   realized 池合計 {tot("realized_q")}，capped {sum(r["capped"] for r in freeze)} 景')
        if errs:
            print(f'   ⚠ {len(errs)} 景異常：{errs[:5]}')
            bad = out_tsv.with_name('bg_coords.INVALID.tsv')
            out_tsv.rename(bad)
            sys.exit(f'[ERROR][P8] {len(errs)} 景未通過 append-only 巢狀檢查；'
                     f'已把 tsv 改名為 {bad.name} 以防誤用')

    # 分佈摘要
    n_zero = sum(1 for s in stats if s['sampled'] == 0)
    n_full = sum(1 for s in stats if s['sampled'] == N_PER_SCENE)
    print(f'\n=== 採樣分佈 ===')
    print(f'  完全沒 candidate: {n_zero} / {len(stats)} 場景')
    print(f'  採到 {N_PER_SCENE} 個 (full): {n_full} / {len(stats)} 場景')
    if rows:
        per_scene_counts = [s['sampled'] for s in stats]
        print(f'  per-scene 平均: {np.mean(per_scene_counts):.1f}, 中位數: {np.median(per_scene_counts):.0f}')


if __name__ == '__main__':
    main()
