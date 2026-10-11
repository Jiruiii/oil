"""Prepare the delivered 355-scene dataset for the frozen GT-expand folds.

Run from any directory with::

    python preprocess/prepare_local_data.py --data-dir /path/to/oil_dataset

The source images and annotations are never changed. Existing VRT and mask
outputs are skipped, so an interrupted preparation can be resumed safely.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import rasterio

from batch_event_gpkg_to_mask import process_scene as gpkg_to_mask
from batch_event_gpkg_to_mask import gpkg_to_shapes, shapes_to_mask, write_mask_tif
from build_vrt_ms6 import build_vrt_scene
from json_to_mask_tif import _process_scene as json_to_mask


PATCH_RE = re.compile(r"^(.+)_patch_x\d+_y\d+$")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = PROJECT_ROOT / "data_split" / "patch_level_gt_expand"
CURATED_EVENT_SCENE = "2022年祕魯La_Pampilla煉油廠洩油事件_20220202_S2_18LTM"


def required_scenes() -> set[str]:
    scenes = set()
    for path in sorted(SPLIT_DIR.glob("*_fold*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            match = PATCH_RE.fullmatch(line.strip())
            if not match:
                raise ValueError(f"Invalid patch entry in {path}: {line!r}")
            scenes.add(match.group(1))
    if not scenes:
        raise ValueError(f"No fold scenes found under {SPLIT_DIR}")
    bg_path = PROJECT_ROOT / "data_split" / "patch_level_gt_expand_bg" / "bg_coords.tsv"
    with bg_path.open(newline="", encoding="utf-8") as handle:
        bg_scenes = {row["scene"] for row in csv.DictReader(handle, delimiter="\t")}
    if missing := bg_scenes - scenes:
        raise ValueError(f"Background coordinates refer to scenes outside folds: {sorted(missing)}")
    return scenes


def curated_event_mask(scene_dir: Path, out_dir: Path) -> str:
    """Use full-precision GPKG geometry for the polygons retained in its JSON.

    This scene's oil GPKG contains four additional polygons absent from the
    delivered JSON and frozen GT-expand stats. Rasterizing all GPKG features
    changes its oil count from the frozen 8,092 pixels to 14,946 pixels.
    """
    destination = out_dir / f"{scene_dir.name}.tif"
    if destination.exists():
        return "skip"
    reference = next(scene_dir.glob("*_B02.tif"))
    labels = json.loads((scene_dir / f"{scene_dir.name}.json").read_text(encoding="utf-8"))["shapes"]
    with rasterio.open(reference) as image:
        available = gpkg_to_shapes(scene_dir, image.crs, image.transform)
        height, width = image.height, image.width

    def signature(shape):
        return (shape["label"].lower(),
                tuple((round(float(x), 1), round(float(y), 1)) for x, y in shape["points"]))

    selected = []
    for label in labels:
        match = next((index for index, shape in enumerate(available)
                      if signature(shape) == signature(label)), None)
        if match is None:
            raise ValueError(f"Could not match curated JSON polygon to GPKG: {scene_dir}")
        selected.append(available.pop(match))
    mask = shapes_to_mask(selected, height, width)
    write_mask_tif(mask, reference, destination)
    return "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True, type=Path)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    image_dir = data_dir / "MS6_sen2like"
    vrt_dir = data_dir / "MS6_sen2like_vrt"
    mask_dir = data_dir / "mask"
    if not image_dir.is_dir():
        parser.error(f"Source image directory does not exist: {image_dir}")

    expected = required_scenes()
    scene_dirs = {path.parent.name: path.parent for path in image_dir.rglob("*_B02.tif")}
    if missing := expected - scene_dirs.keys():
        parser.error(f"Missing {len(missing)} required scenes: {sorted(missing)[:10]}")
    vrt_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    counts = {"vrt_ok": 0, "vrt_skip": 0, "mask_ok": 0, "mask_skip": 0}
    failures = []
    for index, scene in enumerate(sorted(expected), 1):
        scene_dir = scene_dirs[scene]
        try:
            _, vrt_ok, vrt_status = build_vrt_scene(scene_dir, vrt_dir, image_dir)
            if vrt_ok:
                counts["vrt_skip" if vrt_status == "skip" else "vrt_ok"] += 1
            else:
                failures.append(f"{scene}: VRT: {vrt_status}")
        except Exception as exc:
            failures.append(f"{scene}: VRT: {exc}")
        try:
            if scene == CURATED_EVENT_SCENE:
                mask_status = curated_event_mask(scene_dir, mask_dir)
            elif not scene_dir.parent.name.startswith("NOAA") and list(scene_dir.glob("*.gpkg")):
                mask_status = gpkg_to_mask(scene_dir, mask_dir)
            elif list(scene_dir.glob("*.json")):
                mask_status = json_to_mask(scene_dir, mask_dir)
            elif list(scene_dir.glob("*.gpkg")):
                mask_status = gpkg_to_mask(scene_dir, mask_dir)
            else:
                mask_status = "fail"
            if mask_status == "fail":
                failures.append(f"{scene}: mask conversion failed")
            else:
                counts[f"mask_{mask_status}"] += 1
        except Exception as exc:
            failures.append(f"{scene}: mask: {exc}")
        if index % 25 == 0 or index == len(expected):
            print(f"Processed {index}/{len(expected)} scenes; {counts}; failures={len(failures)}", flush=True)

    for failure in failures:
        print(f"ERROR {failure}")
    print(f"Finished: {counts}; failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
