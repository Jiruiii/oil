"""Check regenerated masks against frozen GT-expand and background statistics."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--stats", required=True, type=Path)
    parser.add_argument("--bg-tsv", required=True, type=Path)
    args = parser.parse_args()

    expected_oil = defaultdict(int)
    with args.stats.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["error"]:
                raise ValueError(f"Frozen stats contain an error: {row}")
            expected_oil[row["scene_name"]] += int(row["n_oil"])

    bg_by_scene = defaultdict(list)
    with args.bg_tsv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            bg_by_scene[row["scene"]].append((int(row["x"]), int(row["y"])))

    errors = []
    checked_bg = 0
    for index, (scene, expected) in enumerate(sorted(expected_oil.items()), 1):
        path = args.mask_dir / f"{scene}.tif"
        if not path.exists():
            errors.append(f"{scene}: mask missing")
            continue
        with rasterio.open(path) as source:
            mask = source.read(1)
        actual = int(np.count_nonzero(mask == 0))
        if actual != expected:
            errors.append(f"{scene}: oil pixels {actual} != frozen {expected}")
        height, width = mask.shape
        for x, y in bg_by_scene.pop(scene, []):
            if not (0 <= x <= width - 256 and 0 <= y <= height - 256):
                errors.append(f"{scene}: background window out of bounds ({x}, {y})")
                continue
            window = mask[y:y + 256, x:x + 256]
            checked_bg += 1
            if np.any(window == 0):
                errors.append(f"{scene}: background window contains oil ({x}, {y})")
        if index % 50 == 0 or index == len(expected_oil):
            print(f"Checked {index}/{len(expected_oil)} masks; errors={len(errors)}", flush=True)

    if bg_by_scene:
        errors.append(f"Background coordinates contain unknown scenes: {sorted(bg_by_scene)[:10]}")
    for error in errors[:100]:
        print(f"ERROR {error}")
    print(f"Finished: {len(expected_oil)} masks, {checked_bg} background windows, {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
