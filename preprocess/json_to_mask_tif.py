"""
預處理：JSON polygon 標注 → 全場景 mask TIF（class index）

掃描 MS6_sen2like 目錄結構（LOCATION/YEAR/SCENE/），JSON 位於各場景資料夾內。
mask TIF 以場景資料夾名稱命名，並帶有正確的 CRS 和 geotransform（從 B02 TIF 取得）。

輸出格式：單波段 uint8 TIFF（LZW 壓縮）
  0   = Oil       （明確標注 polygon：label='oil'）
  1   = Background（明確標注 polygon：label='background'）
  2   = Others    （明確標注 polygon：label='others'）
  255 = 未標注     （無任何 polygon 覆蓋的像素）

訓練時以 YAML 的 pixel_mapping 控制各值對應的訓練 class：
  典型設定：{0:0, 1:1, 2:255, 255:255}  → others 與未標注皆作為 ignore_index

⚠ 若 mask/ 目錄已有舊版 TIF，須先刪除再重新執行。
"""
import json
import numpy as np
import cv2
import rasterio
from pathlib import Path
from tqdm import tqdm

SCENE_BASE_DIR = '/home/alanyh/oil_dataset/new/full_band/MS6_sen2like'
MASK_OUT_DIR   = '/home/alanyh/oil_dataset/new/full_band/mask'

_LABEL_MAP = {'oil': 0, 'background': 1, 'others': 2}


def _find_ref_tif(scene_dir: Path) -> Path:
    candidates = sorted(scene_dir.glob('*_B02.tif'))
    return candidates[0] if candidates else None


def _process_scene(scene_dir: Path, out_dir: Path) -> str:
    """回傳 'ok' / 'skip' / 'fail'"""
    out_path = out_dir / f"{scene_dir.name}.tif"
    if out_path.exists():
        return 'skip'

    json_files = sorted(scene_dir.glob('*.json'))
    if not json_files:
        print(f"  ⚠ 無 JSON：{scene_dir.name}")
        return 'fail'

    ref_tif = _find_ref_tif(scene_dir)
    if ref_tif is None:
        print(f"  ⚠ 無 B02 TIF：{scene_dir.name}")
        return 'fail'

    with open(json_files[0], 'r', encoding='utf-8') as f:
        data = json.load(f)

    h = data.get('imageHeight', 10980)
    w = data.get('imageWidth',  10980)

    mask = np.full((h, w), 255, dtype=np.uint8)
    for shape in data.get('shapes', []):
        label  = shape.get('label', '').lower()
        points = np.int32(np.array(shape['points'], dtype=np.float32))
        cv2.fillPoly(mask, [points], color=_LABEL_MAP.get(label, 255))

    with rasterio.open(str(ref_tif)) as src:
        profile = src.profile.copy()

    profile.update({
        'driver': 'GTiff', 'dtype': 'uint8',
        'count': 1, 'compress': 'lzw',
        'width': w, 'height': h,
    })

    with rasterio.open(str(out_path), 'w', **profile) as dst:
        dst.write(mask[np.newaxis, :, :])

    return 'ok'


def main():
    out_dir = Path(MASK_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_dirs = sorted([
        p for p in Path(SCENE_BASE_DIR).glob('*/*/*') if p.is_dir()
    ])
    print(f"找到 {len(scene_dirs)} 個場景，開始轉換...")

    counts = {'ok': 0, 'skip': 0, 'fail': 0}
    for scene_dir in tqdm(scene_dirs, desc="json → mask TIF"):
        counts[_process_scene(scene_dir, out_dir)] += 1

    print(f"\n完成！輸出至 {out_dir}")
    print(f"  新生成: {counts['ok']}  跳過(已存在): {counts['skip']}  失敗: {counts['fail']}")
    print(f"  總檔案數: {len(list(out_dir.glob('*.tif')))}")


if __name__ == '__main__':
    main()
