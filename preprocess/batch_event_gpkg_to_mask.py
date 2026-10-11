"""
事件場景批次處理：GPKG → LabelMe JSON → mask TIF

掃描 MS6_sen2like 中所有非 NOAA 的 2-level 場景目錄，對每個場景執行：
  1. 掃描 *.gpkg，依檔名推斷 label (oil / background / others)
  2. 利用 *_B02.tif 的 affine transform 轉換為 LabelMe JSON
  3. 從 JSON 多邊形填充生成 mask TIF，儲存至 mask/ 目錄

輸出格式：uint8 TIFF（LZW 壓縮）
  0 = Oil   1 = Background   2 = Others   255 = 未標注
"""
import json
import re
import sys
import numpy as np
import cv2
import geopandas as gpd
import rasterio
from pathlib import Path
from rasterio.warp import transform as warp_transform
from tqdm import tqdm

SCENE_BASE_DIR = Path('/home/alanyh/oil_dataset/new/full_band/MS6_sen2like')
MASK_OUT_DIR   = Path('/home/alanyh/oil_dataset/new/full_band/mask')
LABEL_KEYWORDS = ['oil', 'background', 'others']
LABEL_MAP      = {'oil': 0, 'background': 1, 'others': 2}
NOAA_PREFIX    = 'NOAA'


_LABEL_ALIASES = {'bg': 'background'}

def infer_label(path: Path) -> str:
    stem = path.stem.lower()
    for kw in LABEL_KEYWORDS:
        if re.search(rf'_{kw}(\b|_|\d)', stem):
            return kw
    # short-form aliases (e.g. _bg → background)
    for alias, canonical in _LABEL_ALIASES.items():
        if re.search(rf'_{alias}(\b|_|\d|$)', stem):
            return canonical
    return None


def geo_to_pixel(xs, ys, src_crs, dst_crs, img_transform):
    xs_t, ys_t = warp_transform(src_crs, dst_crs, xs, ys)
    cols = [(x - img_transform.c) / img_transform.a for x in xs_t]
    rows = [(y - img_transform.f) / img_transform.e for y in ys_t]
    return list(zip(cols, rows))


def extract_rings(geometry):
    gt = geometry.geom_type
    if gt == 'Polygon':
        return [list(geometry.exterior.coords)]
    elif gt == 'MultiPolygon':
        return [list(p.exterior.coords) for p in geometry.geoms]
    elif gt == 'GeometryCollection':
        rings = []
        for g in geometry.geoms:
            rings.extend(extract_rings(g))
        return rings
    return []


def gpkg_to_shapes(scene_dir: Path, img_crs, img_transform):
    vectors = sorted(scene_dir.glob('*.gpkg')) + sorted(scene_dir.glob('*.shp'))
    all_shapes = []
    for vpath in vectors:
        label = infer_label(vpath)
        if label is None:
            print(f'    跳過（無法推斷 label）：{vpath.name}')
            continue
        gdf = gpd.read_file(str(vpath))
        src_crs = gdf.crs
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            for ring_coords in extract_rings(geom):
                if len(ring_coords) < 3:
                    continue
                xs = [c[0] for c in ring_coords]
                ys = [c[1] for c in ring_coords]
                pts = geo_to_pixel(xs, ys, src_crs, img_crs, img_transform)
                if len(pts) > 1 and pts[0] == pts[-1]:
                    pts = pts[:-1]
                all_shapes.append({'label': label, 'points': pts})
    return all_shapes


def shapes_to_mask(shapes, h, w) -> np.ndarray:
    mask = np.full((h, w), 255, dtype=np.uint8)
    for shape in shapes:
        label = shape['label'].lower()
        pts   = np.int32(np.array(shape['points'], dtype=np.float32))
        cv2.fillPoly(mask, [pts], color=LABEL_MAP.get(label, 255))
    return mask


def write_mask_tif(mask: np.ndarray, ref_tif: Path, out_path: Path):
    with rasterio.open(str(ref_tif)) as src:
        profile = src.profile.copy()
    profile.update({'driver': 'GTiff', 'dtype': 'uint8', 'count': 1, 'compress': 'lzw'})
    with rasterio.open(str(out_path), 'w', **profile) as dst:
        dst.write(mask[np.newaxis, :, :])


def process_scene(scene_dir: Path, out_dir: Path) -> str:
    """回傳 'ok' / 'skip' / 'fail'"""
    out_path = out_dir / f'{scene_dir.name}.tif'
    if out_path.exists():
        return 'skip'

    # Find ref B02 tif
    b02_candidates = sorted(scene_dir.glob('*_B02.tif'))
    if not b02_candidates:
        print(f'  ⚠ 無 B02 TIF：{scene_dir.name}')
        return 'fail'
    ref_tif = b02_candidates[0]

    # Check GPKG availability
    gpkgs = sorted(scene_dir.glob('*.gpkg')) + sorted(scene_dir.glob('*.shp'))
    if not gpkgs:
        print(f'  ⚠ 無 GPKG：{scene_dir.name}')
        return 'fail'

    with rasterio.open(str(ref_tif)) as ds:
        img_transform = ds.transform
        img_crs       = ds.crs
        img_h, img_w  = ds.height, ds.width

    # GPKG → shapes
    shapes = gpkg_to_shapes(scene_dir, img_crs, img_transform)
    if not shapes:
        print(f'  ⚠ 無有效 polygon：{scene_dir.name}')
        return 'fail'

    # Save JSON inside the scene folder (same location as GPKG)
    json_path = scene_dir / f'{scene_dir.name}.json'
    if not json_path.exists():
        labelme_json = {
            'version': '5.2.1', 'flags': {}, 'imagePath': ref_tif.stem + '.png',
            'imageData': None, 'imageHeight': img_h, 'imageWidth': img_w,
            'shapes': [{'label': s['label'],
                        'points': [[round(c, 1), round(r, 1)] for c, r in s['points']],
                        'shape_type': 'polygon', 'flags': {}}
                       for s in shapes],
        }
        json_path.write_text(json.dumps(labelme_json, indent=2, ensure_ascii=False))

    # Shapes → mask TIF
    mask = shapes_to_mask(shapes, img_h, img_w)
    write_mask_tif(mask, ref_tif, out_path)
    return 'ok'


def main():
    MASK_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect 2-level event scenes (non-NOAA top-level directories)
    event_scene_dirs = []
    for loc_dir in sorted(SCENE_BASE_DIR.iterdir()):
        if not loc_dir.is_dir() or loc_dir.name.startswith(NOAA_PREFIX):
            continue
        for scene_dir in sorted(loc_dir.iterdir()):
            if scene_dir.is_dir():
                event_scene_dirs.append(scene_dir)

    print(f'找到 {len(event_scene_dirs)} 個事件場景')

    counts = {'ok': 0, 'skip': 0, 'fail': 0}
    for scene_dir in tqdm(event_scene_dirs, desc='GPKG → mask TIF'):
        result = process_scene(scene_dir, MASK_OUT_DIR)
        counts[result] += 1
        if result == 'fail':
            pass  # already printed warning

    print(f'\n完成！')
    print(f'  新生成: {counts["ok"]}  跳過(已存在): {counts["skip"]}  失敗: {counts["fail"]}')
    total_masks = len(list(MASK_OUT_DIR.glob('*.tif')))
    print(f'  mask/ 目錄現有 TIF 總數: {total_masks}')


if __name__ == '__main__':
    main()
