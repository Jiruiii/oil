"""
建立 MS6_sen2like 8-band VRT

掃描 MS6_sen2like/{LOCATION}/{YEAR}/{SCENE_NAME}/ 目錄，
對每個場景以 rasterio 讀取 metadata 後手動組裝 VRT XML（8 個波段 -separate）。

波段順序（依波長升序）：B01→B02→B03→B04→B08→B8A→B11→B12

VRT 以 B02（10m, 10980×10980）為參考解析度，與 Mask TIF 座標系對齊。
各波段依實際像素尺寸設定 SrcRect/DstRect，GDAL 自動處理 resample：
  - B01 (60m, 1830×1830):   SrcRect=1830×1830 → DstRect=10980×10980 (upsample 6×)
  - B02-B08 (10m):           SrcRect=DstRect=10980×10980 (1:1)
  - B8A/B11/B12 (20m, 5490×5490): SrcRect=5490×5490 → DstRect=10980×10980 (upsample 2×)
"""
import rasterio
from pathlib import Path
from tqdm import tqdm

MS6_DIR     = '/home/alanyh/oil_dataset/new/full_band/MS6_sen2like'
VRT_OUT_DIR = '/home/alanyh/oil_dataset/new/full_band/MS6_sen2like_vrt'

BAND_ORDER = ['B01', 'B02', 'B03', 'B04', 'B08', 'B8A', 'B11', 'B12']

# B02 (index 1) is the 10m reference band used for VRT dimensions
REF_BAND_IDX = 1

_VRT_TMPL = """\
<VRTDataset rasterXSize="{w}" rasterYSize="{h}">
  <SRS dataAxisToSRSAxisMapping="1,2">{wkt}</SRS>
  <GeoTransform>{gt}</GeoTransform>
{bands}
</VRTDataset>
"""

# [F5] resampleAlg="nearest" 明確宣告（GDAL SimpleSource 預設即 nearest,此處只是把
# 隱含選擇寫進 provenance,與既有 VRT 行為一致）;NoDataValue=0 傳遞 source nodata。
_BAND_TMPL = """\
  <VRTRasterBand dataType="Float32" band="{band_idx}">
    <NoDataValue>0</NoDataValue>
    <SimpleSource resampleAlg="nearest">
      <SourceFilename relativeToVRT="0">{src}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="{src_w}" ySize="{src_h}"/>
      <DstRect xOff="0" yOff="0" xSize="{dst_w}" ySize="{dst_h}"/>
    </SimpleSource>
  </VRTRasterBand>"""


def get_band_files(scene_dir: Path) -> list[Path] | None:
    """回傳 8 個波段的絕對路徑，依 BAND_ORDER；任一缺失回傳 None。"""
    result = []
    for band in BAND_ORDER:
        matches = sorted(scene_dir.glob(f'*_{band}.tif'))
        if not matches:
            return None
        result.append(matches[0].resolve())
    return result


def build_vrt_scene(scene_dir: Path, out_root: Path, ms6_root: Path = None) -> tuple[str, bool, str]:
    scene_name = scene_dir.name
    # 3-level: NOAA_LOC/YEAR/SCENE → location = grandparent
    # 2-level: EVENT_NAME/SCENE    → location = parent
    if ms6_root and scene_dir.parent.parent == ms6_root:
        location = scene_dir.parent.name
    else:
        location = scene_dir.parent.parent.name
    out_dir    = out_root / location
    out_dir.mkdir(parents=True, exist_ok=True)
    vrt_path   = out_dir / f"{scene_name}.vrt"

    if vrt_path.exists():
        return scene_name, True, 'skip'

    band_files = get_band_files(scene_dir)
    if band_files is None:
        missing = [b for b in BAND_ORDER if not list(scene_dir.glob(f'*_{b}.tif'))]
        return scene_name, False, f"缺少波段: {missing}"

    # Use REF_BAND_IDX (B02, 10m) as VRT reference to match Mask TIF coordinate system
    try:
        with rasterio.open(str(band_files[REF_BAND_IDX])) as src:
            h, w = src.height, src.width
            ref_crs    = src.crs
            ref_bounds = src.bounds
            wkt  = src.crs.to_wkt() if src.crs else ''
            t    = src.transform
            gt   = f"{t.c}, {t.a}, {t.b}, {t.f}, {t.d}, {t.e}"
    except Exception as e:
        return scene_name, False, f"讀取 metadata 失敗: {e}"

    # [F5] 幾何契約:參考波段不得有旋轉
    if t.b != 0 or t.d != 0:
        return scene_name, False, f"[F5] 參考波段 transform 含旋轉項: b={t.b}, d={t.d}"

    band_xml_parts = []
    for i, p in enumerate(band_files):
        try:
            with rasterio.open(str(p)) as src:
                src_h, src_w = src.height, src.width
                # [F5] 建置前逐波段驗證幾何契約,不符即失敗(過去只讀 width/height,
                # CRS/bounds/nodata/旋轉錯位會靜默混入):
                if src.crs != ref_crs:
                    return scene_name, False, f"[F5] {BAND_ORDER[i]} CRS 不一致: {src.crs} vs {ref_crs}"
                if any(abs(a - b) > 1e-3 for a, b in zip(src.bounds, ref_bounds)):
                    return scene_name, False, f"[F5] {BAND_ORDER[i]} bounds 不一致: {tuple(src.bounds)} vs {tuple(ref_bounds)}"
                if src.transform.b != 0 or src.transform.d != 0:
                    return scene_name, False, f"[F5] {BAND_ORDER[i]} transform 含旋轉項"
                # 等比即可,不要求整數比:S2 為 6×/2×/1×,但 L8/L9 的 B08(15m)是 1.5×
                if abs(w / src_w - h / src_h) > 1e-6:
                    return scene_name, False, f"[F5] {BAND_ORDER[i]} 尺寸 {src_w}×{src_h} 與參考 {w}×{h} 非等比縮放"
                if src.nodata not in (0, None):
                    return scene_name, False, f"[F5] {BAND_ORDER[i]} nodata={src.nodata} 非預期(應為 0 或未設)"
        except Exception as e:
            return scene_name, False, f"讀取波段 {BAND_ORDER[i]} metadata 失敗: {e}"
        band_xml_parts.append(_BAND_TMPL.format(
            band_idx=i + 1, src=str(p),
            src_w=src_w, src_h=src_h,
            dst_w=w, dst_h=h,
        ))

    band_xml = '\n'.join(band_xml_parts)
    vrt_xml = _VRT_TMPL.format(w=w, h=h, wkt=wkt, gt=gt, bands=band_xml)

    vrt_path.write_text(vrt_xml, encoding='utf-8')
    return scene_name, True, 'ok'


def main():
    ms6_root = Path(MS6_DIR)
    out_root  = Path(VRT_OUT_DIR)

    # 3-level NOAA scenes (LOCATION/YEAR/SCENE) + 2-level event scenes (EVENT/SCENE)
    scene_dirs = sorted(
        [p for p in ms6_root.glob('*/*/*') if p.is_dir()] +
        [p for p in ms6_root.glob('*/*') if p.is_dir() and not p.parent.name.startswith('NOAA')]
    )
    print(f"找到 {len(scene_dirs)} 個場景，開始建立 VRT...")

    counts = {'ok': 0, 'skip': 0, 'fail': 0}
    for scene_dir in tqdm(scene_dirs, desc='build VRT'):
        name, ok, msg = build_vrt_scene(scene_dir, out_root, ms6_root)
        if not ok:
            print(f"  ❌ {name}: {msg}")
            counts['fail'] += 1
        elif msg == 'skip':
            counts['skip'] += 1
        else:
            counts['ok'] += 1

    total = sum(1 for _ in out_root.rglob('*.vrt'))
    print(f"\n完成！新建: {counts['ok']}  跳過: {counts['skip']}  失敗: {counts['fail']}")
    print(f"VRT 總數: {total}  輸出目錄: {out_root}")


if __name__ == '__main__':
    main()
