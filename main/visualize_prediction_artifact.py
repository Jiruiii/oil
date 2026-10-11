"""Restore and visualize pixel-exact prediction JSON artifacts on demand.

The authoritative experiment output stays as ``*.pred.json``.  This utility
can recreate a full-resolution uint8 label PNG, a compact colour preview, or a
target-class overlay without requiring the original prediction PNG.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

from predmask_codec import (
    PredMaskError,
    canonical_json_bytes,
    decode_scene_coverage,
    decode_scene_final,
    load_scene_document,
    manifest_roster_errors,
    sha256_bytes,
)


# RGB colours.  Oil/class 0 is intentionally vivid; 255/uncovered is grey.
DEFAULT_PALETTE = np.asarray(
    [
        [255, 64, 32],
        [24, 24, 24],
        [40, 140, 255],
        [80, 200, 120],
        [210, 80, 210],
        [255, 190, 40],
        [80, 220, 220],
        [180, 180, 60],
    ],
    dtype=np.uint8,
)
UNCOVERED_RGB = np.asarray([112, 112, 112], dtype=np.uint8)


def _safe_stem(scene_id: str) -> str:
    value = str(scene_id).replace("/", "__").replace("\\", "__")
    return value if value not in {"", ".", ".."} else "scene"


def _preview_shape(shape: tuple[int, int], max_side: int) -> tuple[int, int]:
    height, width = shape
    if max_side <= 0 or max(height, width) <= max_side:
        return height, width
    scale = float(max_side) / float(max(height, width))
    return max(1, round(height * scale)), max(1, round(width * scale))


def _resize_labels(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    if mask.shape == (height, width):
        return mask
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def _crop_bounds(
    mask: np.ndarray,
    coverage: np.ndarray,
    document: Mapping[str, Any],
    mode: str,
    *,
    padding_fraction: float = 0.15,
    min_side: int = 1024,
) -> tuple[int, int, int, int]:
    """Return y0, x0, y1, x1 for a preview; exact masks stay uncropped."""
    height, width = mask.shape
    resolved = mode
    if resolved == "auto":
        resolved = "full" if document.get("coverage", {}).get("mode") == "full" else "coverage"
    if resolved == "full":
        return 0, 0, height, width
    if resolved == "coverage":
        selected = np.asarray(coverage, dtype=bool)
    elif resolved == "target":
        target = int(document.get("labels", {}).get("target_class_index", 0))
        selected = np.asarray(coverage, dtype=bool) & (np.asarray(mask) == target)
        if not selected.any():
            selected = np.asarray(coverage, dtype=bool)
    else:
        raise ValueError(f"unsupported crop mode: {mode}")

    rows = np.flatnonzero(selected.any(axis=1))
    columns = np.flatnonzero(selected.any(axis=0))
    if rows.size == 0 or columns.size == 0:
        return 0, 0, height, width
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(columns[0]), int(columns[-1]) + 1
    pad_y = max(16, round((y1 - y0) * padding_fraction))
    pad_x = max(16, round((x1 - x0) * padding_fraction))
    y0, y1 = max(0, y0 - pad_y), min(height, y1 + pad_y)
    x0, x1 = max(0, x0 - pad_x), min(width, x1 + pad_x)

    desired_h = min(height, max(min_side, y1 - y0))
    desired_w = min(width, max(min_side, x1 - x0))
    center_y, center_x = (y0 + y1) // 2, (x0 + x1) // 2
    y0 = max(0, min(height - desired_h, center_y - desired_h // 2))
    x0 = max(0, min(width - desired_w, center_x - desired_w // 2))
    return y0, x0, y0 + desired_h, x0 + desired_w


def _write_png(path: Path, rgb_or_labels: np.ndarray, *, rgb: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = rgb_or_labels[..., ::-1] if rgb else rgb_or_labels
    if not cv2.imwrite(str(path), np.ascontiguousarray(output)):
        raise OSError(f"failed to write PNG: {path}")
    return path


def colourise_mask(
    mask: np.ndarray,
    coverage: np.ndarray,
    *,
    palette: np.ndarray = DEFAULT_PALETTE,
) -> np.ndarray:
    labels = np.asarray(mask, dtype=np.uint8)
    valid = np.asarray(coverage, dtype=bool)
    if labels.shape != valid.shape:
        raise ValueError("mask/coverage shape mismatch")
    # A 256-entry uint8 LUT avoids allocating a full-size int64 index raster.
    lut = palette[np.arange(256, dtype=np.uint16) % len(palette)]
    colour = lut[labels]
    colour[~valid] = UNCOVERED_RGB
    return colour


def _stretch_band(band: np.ndarray) -> np.ndarray:
    array = np.asarray(band, dtype=np.float32)
    valid = np.isfinite(array) & (array > 0)
    if not valid.any():
        return np.zeros(array.shape, dtype=np.uint8)
    lo, hi = np.percentile(array[valid], [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(array.shape, dtype=np.uint8)
    stretched = np.clip((array - lo) / (hi - lo), 0.0, 1.0)
    stretched[~valid] = 0.0
    return np.rint(stretched * 255.0).astype(np.uint8)


def _read_background(
    path: Path,
    shape: tuple[int, int],
    bands: tuple[int, int, int],
    crop_bounds: tuple[int, int, int, int],
    raster_shape: tuple[int, int],
) -> np.ndarray:
    height, width = shape
    y0, x0, y1, x1 = crop_bounds
    raster_height, raster_width = raster_shape
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        from PIL import Image

        with Image.open(path) as source:
            source_width, source_height = source.size
        source_pixels = int(source_width) * int(source_height)
        preview_pixels = int(width) * int(height)
        if source_pixels > 50_000_000 and preview_pixels * 4 < source_pixels:
            raise ValueError(
                f"refusing to fully decode oversized {source_width}x{source_height} background "
                f"for a {width}x{height} preview; use the scene VRT/TIF with --background"
            )
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(f"cannot read background image: {path}")
        image_height, image_width = bgr.shape[:2]
        sy0 = max(0, min(image_height - 1, math.floor(y0 * image_height / raster_height)))
        sy1 = max(sy0 + 1, min(image_height, math.ceil(y1 * image_height / raster_height)))
        sx0 = max(0, min(image_width - 1, math.floor(x0 * image_width / raster_width)))
        sx1 = max(sx0 + 1, min(image_width, math.ceil(x1 * image_width / raster_width)))
        bgr = bgr[sy0:sy1, sx0:sx1]
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
        return bgr[..., ::-1]

    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.windows import Window
    except ImportError as exc:
        raise RuntimeError("rasterio is required for VRT/TIF backgrounds") from exc

    with rasterio.open(str(path)) as source:
        selected = bands if max(bands) <= source.count else (1, 2, 3)
        if max(selected) > source.count:
            raise ValueError(f"background has {source.count} bands; cannot select RGB bands")
        source_y0 = y0 * source.height / raster_height
        source_y1 = y1 * source.height / raster_height
        source_x0 = x0 * source.width / raster_width
        source_x1 = x1 * source.width / raster_width
        data = source.read(
            selected,
            window=Window(
                source_x0,
                source_y0,
                source_x1 - source_x0,
                source_y1 - source_y0,
            ),
            out_shape=(3, height, width),
            resampling=Resampling.bilinear,
        )
    return np.stack([_stretch_band(data[index]) for index in range(3)], axis=-1)


def _provenance_path(document: Mapping[str, Any]) -> Path | None:
    provenance = document.get("provenance", {})
    for key in ("scene_source", "vrt", "source_image"):
        value = provenance.get(key)
        if isinstance(value, Mapping):
            value = value.get("path")
        if value:
            candidate = Path(str(value))
            if candidate.exists():
                return candidate
    return None


def _named_provenance_path(document: Mapping[str, Any], key: str) -> Path | None:
    value = document.get("provenance", {}).get(key)
    if isinstance(value, Mapping):
        value = value.get("path")
    if not value:
        return None
    candidate = Path(str(value))
    return candidate if candidate.exists() else None


def _find_background(root: Path, scene_id: str) -> Path | None:
    for extension in (".vrt", ".tif", ".tiff", ".png", ".jpg", ".jpeg"):
        direct = root / f"{scene_id}{extension}"
        if direct.exists():
            return direct
        match = next(root.rglob(f"{scene_id}{extension}"), None)
        if match is not None:
            return match
    return None


def _manifest_artifact_paths(manifest_path: Path) -> list[Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checksum_path = Path(str(manifest_path) + ".sha256")
    if not checksum_path.exists():
        raise PredMaskError(f"manifest checksum sidecar missing: {checksum_path}")
    expected = checksum_path.read_text(encoding="ascii").strip()
    actual = sha256_bytes(canonical_json_bytes(manifest))
    if expected != actual:
        raise PredMaskError(f"manifest checksum mismatch: {manifest_path}")
    roster_errors = manifest_roster_errors(manifest)
    if roster_errors:
        raise PredMaskError(
            f"manifest is incomplete or invalid: {manifest_path}: {roster_errors}"
        )
    return [manifest_path.parent / item["json"] for item in manifest.get("scenes", [])]


def _artifact_paths(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        manifest_path = input_path / "predictions_manifest.json"
        paths = (
            _manifest_artifact_paths(manifest_path)
            if manifest_path.exists()
            else sorted(input_path.glob("*.pred.json"))
        )
    elif input_path.name == "predictions_manifest.json":
        paths = _manifest_artifact_paths(input_path)
    else:
        paths = [input_path]
    if not paths:
        raise FileNotFoundError(f"no *.pred.json artifacts found in {input_path}")
    return paths


def _render_gt_expand_legacy(
    document: Mapping[str, Any],
    final: np.ndarray,
    destination: Path,
    *,
    background: str | Path | None,
    background_root: str | Path | None,
) -> Path:
    """Call the original GT-expand renderer so style/crop/caption stay identical."""
    try:
        from recon_gt_aware_module import (
            _apply_pixel_mapping,
            _compute_scene_iou,
            _draw_overlay_cropped,
            _find_oil_bbox,
            _read_truecolor_window,
        )
    except ImportError as exc:
        if exc.name != "recon_gt_aware_module":
            raise
        raise RuntimeError(
            "gt-expand-legacy rendering must run from the GT_expand repository"
        ) from exc

    scene_id = str(document["scene_id"])
    gt_path = _named_provenance_path(document, "canonical_mask_tif")
    if gt_path is None:
        raise FileNotFoundError(f"canonical_mask_tif provenance is unavailable: {scene_id}")
    source_path = Path(background) if background is not None else None
    if source_path is None and background_root is not None:
        source_path = _find_background(Path(background_root), scene_id)
    if source_path is None:
        source_path = _provenance_path(document)
    if source_path is None or source_path.suffix.lower() not in {".vrt", ".tif", ".tiff"}:
        raise FileNotFoundError(
            f"GT-expand legacy rendering requires the scene VRT/TIF: {scene_id}; "
            "use --background-root or --background"
        )

    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("rasterio is required for GT-expand legacy rendering") from exc
    with rasterio.open(str(gt_path)) as source:
        raw_gt = source.read(1)
    labels = dict(document.get("labels", {}) or {})
    mapping = {int(key): int(value) for key, value in dict(labels.get("pixel_mapping", {})).items()}
    gt_mask = _apply_pixel_mapping(raw_gt, mapping)
    coverage = decode_scene_coverage(document, final)
    target = int(labels.get("target_class_index", 0))
    background_index = int(labels.get("background_class_index", 1))
    num_classes = int(labels.get("num_classes", 2))
    iou = _compute_scene_iou(
        final,
        coverage,
        gt_mask,
        num_classes,
        bg_class=background_index,
    ).get(f"iou_C{target}", 0.0)

    height, width = final.shape
    pred_oil = (final == target) & coverage
    gt_oil = gt_mask == target
    y0, x0, y1, x1 = _find_oil_bbox(pred_oil, gt_oil, H=height, W=width)
    background_crop = _read_truecolor_window(source_path, y0, x0, y1, x1)
    rendered = _draw_overlay_cropped(
        background_crop,
        pred_oil[y0:y1, x0:x1],
        gt_oil[y0:y1, x0:x1],
        scene_id,
        iou,
    )
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / f"{_safe_stem(scene_id)}_iou_{iou:.4f}.jpg"
    if not cv2.imwrite(str(output), rendered, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise OSError(f"failed to write legacy GT-expand JPEG: {output}")
    return output


def render_artifact(
    scene_json: str | Path,
    output_dir: str | Path,
    *,
    kinds: Iterable[str] = ("color",),
    max_side: int = 2048,
    alpha: float = 0.55,
    background: str | Path | None = None,
    background_root: str | Path | None = None,
    bands: tuple[int, int, int] = (4, 3, 2),
    crop: str = "auto",
) -> dict[str, Path]:
    """Render one artifact and return the files that were written."""
    requested = set(kinds)
    unknown = requested - {"mask", "color", "overlay", "gt-expand-legacy"}
    if unknown:
        raise ValueError(f"unknown render kinds: {sorted(unknown)}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if crop not in {"auto", "full", "coverage", "target"}:
        raise ValueError(f"unsupported crop mode: {crop}")

    scene_path = Path(scene_json)
    document = load_scene_document(scene_path)
    final = decode_scene_final(document)
    scene_id = str(document["scene_id"])
    stem = _safe_stem(scene_id)
    destination = Path(output_dir)
    written: dict[str, Path] = {}

    if "mask" in requested:
        written["mask"] = _write_png(destination / f"{stem}.mask.png", final, rgb=False)

    if "gt-expand-legacy" in requested:
        written["gt-expand-legacy"] = _render_gt_expand_legacy(
            document,
            final,
            destination,
            background=background,
            background_root=background_root,
        )

    if requested & {"color", "overlay"}:
        coverage = decode_scene_coverage(document, final)
        bounds = _crop_bounds(final, coverage, document, crop)
        y0, x0, y1, x1 = bounds
        cropped_mask = final[y0:y1, x0:x1]
        cropped_coverage = coverage[y0:y1, x0:x1]
        preview_shape = _preview_shape(cropped_mask.shape, max_side)
        preview_mask = _resize_labels(cropped_mask, preview_shape)
        preview_coverage = _resize_labels(
            cropped_coverage.astype(np.uint8), preview_shape
        ).astype(bool)

    if "color" in requested:
        colour = colourise_mask(preview_mask, preview_coverage)
        written["color"] = _write_png(destination / f"{stem}.color.png", colour, rgb=True)

    if "overlay" in requested:
        background_path = Path(background) if background is not None else None
        if background_path is None and background_root is not None:
            background_path = _find_background(Path(background_root), scene_id)
        if background_path is None:
            background_path = _provenance_path(document)
        if background_path is None or not background_path.exists():
            raise FileNotFoundError(
                f"no source image found for overlay: {scene_id}; use --background or --background-root"
            )
        base = _read_background(background_path, preview_shape, bands, bounds, final.shape)
        labels = document.get("labels", {})
        target_index = int(labels.get("target_class_index", 0))
        selected = preview_coverage & (preview_mask == target_index)
        overlay = base.copy()
        if selected.any():
            blended = np.rint(
                (1.0 - alpha) * base[selected].astype(np.float32)
                + alpha * DEFAULT_PALETTE[target_index % len(DEFAULT_PALETTE)].astype(np.float32)
            ).astype(np.uint8)
            overlay[selected] = blended
        written["overlay"] = _write_png(destination / f"{stem}.overlay.png", overlay, rgb=True)

    return written


def _parse_bands(value: str) -> tuple[int, int, int]:
    parts = tuple(int(item.strip()) for item in value.split(","))
    if len(parts) != 3 or min(parts) < 1:
        raise argparse.ArgumentTypeError("bands must be three positive 1-based indexes, e.g. 4,3,2")
    return parts


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Restore exact masks and on-demand previews from *.pred.json"
    )
    parser.add_argument("input", type=Path, help="one *.pred.json, a manifest, or an artifact directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="default: <artifact-directory>/visualized",
    )
    parser.add_argument(
        "--kind",
        choices=("mask", "color", "overlay", "gt-expand-legacy", "all"),
        default="color",
        help="mask=exact PNG; color/overlay=generic previews; gt-expand-legacy=original small-image style",
    )
    parser.add_argument("--max-side", type=int, default=2048, help="preview long side; 0 keeps full size")
    parser.add_argument("--alpha", type=float, default=0.55, help="target-class overlay opacity")
    parser.add_argument("--background", type=Path, help="source image for a single artifact")
    parser.add_argument("--background-root", type=Path, help="recursively match each scene source")
    parser.add_argument("--bands", type=_parse_bands, default=(4, 3, 2), help="1-based RGB bands")
    parser.add_argument(
        "--crop",
        choices=("auto", "full", "coverage", "target"),
        default="auto",
        help="auto crops partial-coverage GT-expand/windowed artifacts; full-scene artifacts stay full",
    )
    args = parser.parse_args()

    paths = _artifact_paths(args.input)
    if args.background is not None and len(paths) != 1:
        parser.error("--background can only be used with one scene; use --background-root for a batch")
    output_dir = args.output_dir or (
        args.input if args.input.is_dir() else args.input.parent
    ) / "visualized"
    kinds = ("mask", "color", "overlay") if args.kind == "all" else (args.kind,)

    for scene_path in paths:
        written = render_artifact(
            scene_path,
            output_dir,
            kinds=kinds,
            max_side=args.max_side,
            alpha=args.alpha,
            background=args.background,
            background_root=args.background_root,
            bands=args.bands,
            crop=args.crop,
        )
        print(f"{scene_path.name}: " + ", ".join(f"{key}={value}" for key, value in written.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
