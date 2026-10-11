"""Versioned, pixel-exact prediction artifacts for oil-spill segmentation.

The JSON schema stores raster payloads as compressed, base64-encoded label
run streams.  It intentionally keeps probability arrays out of JSON.

This module is vendored byte-for-byte in both oil-spill repositories.  Keep
the copies identical and run the golden tests in both repositories whenever
the codec changes.
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping
import unicodedata
import zlib

import numpy as np


CODEC_NAME = "labelrle"
CODEC_VERSION = "1.0.0"
SCENE_SCHEMA = "oil.predmask/1"
MANIFEST_SCHEMA = "oil.predmanifest/1"
DEFAULT_MAX_RUNS = 10_000_000
DEFAULT_BLOCK_ROWS = 128


class PredMaskError(RuntimeError):
    """Base class for prediction-artifact failures."""


class RunBudgetExceeded(PredMaskError):
    """Raised before an RLE stream can exceed its configured run budget."""

    def __init__(self, actual_runs: int, max_runs: int):
        super().__init__(f"RLE run budget exceeded: {actual_runs:,} > {max_runs:,}")
        self.actual_runs = int(actual_runs)
        self.max_runs = int(max_runs)


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray, block_rows: int = DEFAULT_BLOCK_ROWS) -> str:
    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D raster, got shape={arr.shape}")
    digest = hashlib.sha256()
    for row in range(0, arr.shape[0], block_rows):
        block = np.ascontiguousarray(arr[row : row + block_rows])
        digest.update(memoryview(block).cast("B"))
    return digest.hexdigest()


def module_sha256() -> str:
    return sha256_file(__file__)


def _decompress_array(payload: str, dtype: np.dtype) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(payload.encode("ascii"), validate=True))
    return np.frombuffer(raw, dtype=dtype)


def _validate_uint8_raster(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2-D, got {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"mask must contain integer labels, got {arr.dtype}")
    if arr.size and (int(arr.min()) < 0 or int(arr.max()) > 255):
        raise ValueError("labelrle/1 supports label values in [0, 255]")
    return arr.astype(np.uint8, copy=False)


def encode_label_rle(
    mask: np.ndarray,
    *,
    max_runs: int = DEFAULT_MAX_RUNS,
    block_rows: int = DEFAULT_BLOCK_ROWS,
    compression_level: int = 6,
) -> dict[str, Any]:
    """Encode a 2-D uint8 label raster without per-pixel Python objects."""
    arr = _validate_uint8_raster(mask)
    height, width = map(int, arr.shape)
    if height == 0 or width == 0:
        raise ValueError("empty rasters are not supported")
    if max_runs <= 0 or block_rows <= 0:
        raise ValueError("max_runs and block_rows must be positive")

    value_compressor = zlib.compressobj(compression_level)
    length_compressor = zlib.compressobj(compression_level)
    compressed_values: list[bytes] = []
    compressed_lengths: list[bytes] = []
    total_runs = 0
    emitted_runs = 0
    emitted_pixels = 0
    last_emitted_value: int | None = None
    pending_value: int | None = None
    pending_length = 0
    digest = hashlib.sha256()
    pixel_counts = np.zeros(256, dtype=np.uint64)

    def emit(values: np.ndarray, lengths: np.ndarray) -> None:
        nonlocal emitted_runs, emitted_pixels, last_emitted_value
        if not values.size:
            return
        if last_emitted_value is not None and last_emitted_value == int(values[0]):
            raise AssertionError("non-canonical adjacent equal RLE values")
        value_bytes = value_compressor.compress(np.ascontiguousarray(values, dtype=np.uint8).tobytes())
        length_values = np.ascontiguousarray(lengths, dtype="<u4")
        length_bytes = length_compressor.compress(length_values.tobytes())
        if value_bytes:
            compressed_values.append(value_bytes)
        if length_bytes:
            compressed_lengths.append(length_bytes)
        emitted_runs += int(values.size)
        emitted_pixels += int(length_values.astype(np.uint64).sum())
        last_emitted_value = int(values[-1])

    for row in range(0, height, block_rows):
        block = np.ascontiguousarray(arr[row : row + block_rows], dtype=np.uint8)
        digest.update(memoryview(block).cast("B"))
        flat = block.reshape(-1)
        boundaries = np.flatnonzero(flat[1:] != flat[:-1]) + 1
        starts = np.empty(boundaries.size + 1, dtype=np.int64)
        starts[0] = 0
        starts[1:] = boundaries
        ends = np.empty_like(starts)
        ends[:-1] = boundaries
        ends[-1] = flat.size
        values = flat[starts].astype(np.uint8, copy=True)
        lengths = (ends - starts).astype(np.uint32, copy=False)

        joins_pending = pending_value is not None and pending_value == int(values[0])
        projected = total_runs + int(values.size) - int(joins_pending)
        if projected > max_runs:
            raise RunBudgetExceeded(projected, max_runs)

        if pending_value is not None:
            if joins_pending:
                merged = pending_length + int(lengths[0])
                if merged > np.iinfo(np.uint32).max:
                    raise PredMaskError("single RLE run exceeds uint32")
                lengths[0] = np.uint32(merged)
            else:
                emit(
                    np.asarray([pending_value], dtype=np.uint8),
                    np.asarray([pending_length], dtype="<u4"),
                )
        if values.size > 1:
            emit(values[:-1], lengths[:-1])
        pending_value = int(values[-1])
        pending_length = int(lengths[-1])
        total_runs = projected
        np.add.at(pixel_counts, flat[starts], (ends - starts).astype(np.uint64, copy=False))

    emit(
        np.asarray([pending_value], dtype=np.uint8),
        np.asarray([pending_length], dtype="<u4"),
    )
    compressed_values.append(value_compressor.flush())
    compressed_lengths.append(length_compressor.flush())
    if emitted_runs != total_runs:
        raise AssertionError("RLE values/lengths size mismatch")
    if emitted_pixels != height * width:
        raise AssertionError("RLE lengths do not cover the raster")

    return {
        "encoding": "labelrle-u8-u32-zlib-base64-v1",
        "order": "C",
        "n_runs": int(total_runs),
        "values_b64": base64.b64encode(b"".join(compressed_values)).decode("ascii"),
        "lengths_b64": base64.b64encode(b"".join(compressed_lengths)).decode("ascii"),
        "pixel_counts": {str(index): int(value) for index, value in enumerate(pixel_counts) if value},
        "payload_sha256": digest.hexdigest(),
    }


def decode_label_rle(payload: Mapping[str, Any], shape: Iterable[int]) -> np.ndarray:
    if payload.get("encoding") != "labelrle-u8-u32-zlib-base64-v1":
        raise PredMaskError(f"unsupported encoding: {payload.get('encoding')!r}")
    if payload.get("order") != "C":
        raise PredMaskError(f"unsupported raster order: {payload.get('order')!r}")
    height, width = (int(v) for v in shape)
    values = _decompress_array(str(payload["values_b64"]), np.dtype("u1"))
    lengths = _decompress_array(str(payload["lengths_b64"]), np.dtype("<u4"))
    if values.size != lengths.size or values.size != int(payload["n_runs"]):
        raise PredMaskError("RLE n_runs/stream length mismatch")
    if values.size == 0 or (values.size > 1 and np.any(values[1:] == values[:-1])):
        raise PredMaskError("RLE stream is empty or non-canonical")
    total = int(lengths.astype(np.uint64).sum())
    if total != height * width:
        raise PredMaskError(f"RLE covers {total} pixels, expected {height * width}")
    decoded = np.repeat(values, lengths.astype(np.intp, copy=False)).reshape(height, width)
    if sha256_array(decoded) != payload["payload_sha256"]:
        raise PredMaskError("decoded raster checksum mismatch")
    counts = np.zeros(256, dtype=np.uint64)
    np.add.at(counts, values, lengths.astype(np.uint64, copy=False))
    actual_counts = {str(index): int(value) for index, value in enumerate(counts) if value}
    if actual_counts != {str(k): int(v) for k, v in payload["pixel_counts"].items()}:
        raise PredMaskError("decoded raster pixel_counts mismatch")
    return decoded


def encode_binary_rle(mask: np.ndarray, **kwargs: Any) -> dict[str, Any]:
    arr = np.asarray(mask, dtype=np.uint8)
    if arr.size and np.any((arr != 0) & (arr != 1)):
        raise ValueError("binary RLE accepts only 0/1")
    payload = encode_label_rle(arr, **kwargs)
    payload["semantic_type"] = "binary"
    return payload


def compute_confusion_matrix(
    prediction: np.ndarray,
    reference: np.ndarray,
    num_classes: int,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    pred = np.asarray(prediction)
    ref = np.asarray(reference)
    if pred.shape != ref.shape:
        raise ValueError(f"prediction/reference shape mismatch: {pred.shape} vs {ref.shape}")
    valid = np.ones(pred.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != pred.shape:
        raise ValueError("valid_mask shape mismatch")
    p = pred[valid].astype(np.int64, copy=False)
    g = ref[valid].astype(np.int64, copy=False)
    in_domain = (p >= 0) & (p < num_classes) & (g >= 0) & (g < num_classes)
    if not np.all(in_domain):
        raise ValueError("valid pixels contain labels outside num_classes")
    counts = np.bincount(num_classes * g + p, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes).astype(np.int64, copy=False)


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, Any]:
    cm = np.asarray(confusion, dtype=np.int64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("confusion matrix must be square")
    diag = np.diag(cm)
    gt_totals = cm.sum(axis=1)
    pred_totals = cm.sum(axis=0)
    union = gt_totals + pred_totals - diag
    iou = np.divide(diag, union, out=np.zeros_like(diag, dtype=np.float64), where=union > 0)
    precision = np.divide(diag, pred_totals, out=np.zeros_like(diag, dtype=np.float64), where=pred_totals > 0)
    recall = np.divide(diag, gt_totals, out=np.zeros_like(diag, dtype=np.float64), where=gt_totals > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)
    return {
        "iou": iou.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "accuracy": float(diag.sum() / cm.sum()) if cm.sum() else 0.0,
    }


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: str | Path, value: Any) -> str:
    payload = canonical_json_bytes(value)
    digest = sha256_bytes(payload)
    atomic_write_bytes(path, payload + b"\n")
    atomic_write_bytes(str(path) + ".sha256", (digest + "\n").encode("ascii"))
    return digest


def atomic_save_npz(path: str | Path, **arrays: np.ndarray) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(destination)


def _safe_scene_filename(scene_id: str) -> str:
    name = str(scene_id).replace("/", "__").replace("\\", "__")
    if name in {"", ".", ".."}:
        name = sha256_bytes(str(scene_id).encode("utf-8"))[:24]
    return f"{name}.pred.json"


def _encoded_mask_doc(payload: Mapping[str, Any]) -> dict[str, Any]:
    return dict(payload)


def _count_changed_from_encoded(raw_payload: Mapping[str, Any], final_mask: np.ndarray) -> int:
    if raw_payload.get("encoding") != "labelrle-u8-u32-zlib-base64-v1":
        raise PredMaskError(f"unsupported encoding: {raw_payload.get('encoding')!r}")
    values = _decompress_array(str(raw_payload["values_b64"]), np.dtype("u1"))
    lengths = _decompress_array(str(raw_payload["lengths_b64"]), np.dtype("<u4"))
    if values.size != lengths.size or values.size != int(raw_payload["n_runs"]):
        raise PredMaskError("raw RLE n_runs/stream length mismatch")
    if values.size == 0 or np.any(lengths == 0):
        raise PredMaskError("raw RLE stream is empty or contains zero-length runs")
    ends = np.cumsum(lengths, dtype=np.uint64)
    final_flat = np.asarray(final_mask).reshape(-1)
    if int(ends[-1]) != final_flat.size:
        raise PredMaskError("raw RLE shape mismatch")

    changed = 0
    digest = hashlib.sha256()
    block_pixels = max(1, DEFAULT_BLOCK_ROWS * int(final_mask.shape[1]))
    for start in range(0, final_flat.size, block_pixels):
        stop = min(final_flat.size, start + block_pixels)
        first_run = int(np.searchsorted(ends, start, side="right"))
        last_run = int(np.searchsorted(ends, stop - 1, side="right"))
        boundaries = np.empty(last_run - first_run + 2, dtype=np.uint64)
        boundaries[0] = np.uint64(start)
        boundaries[1:] = np.minimum(ends[first_run : last_run + 1], np.uint64(stop))
        segment_lengths = np.diff(boundaries).astype(np.intp, copy=False)
        raw_block = np.repeat(values[first_run : last_run + 1], segment_lengths)
        if raw_block.size != stop - start:
            raise PredMaskError("raw RLE block decode length mismatch")
        digest.update(memoryview(raw_block).cast("B"))
        changed += int(np.count_nonzero(raw_block != final_flat[start:stop]))
    if digest.hexdigest() != raw_payload["payload_sha256"]:
        raise PredMaskError("raw RLE checksum mismatch")
    return changed


def _normalise_matrix_atoms(metric_atoms: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, atom in metric_atoms.items():
        if isinstance(atom, Mapping) and "confusion_matrix" in atom:
            matrix = np.asarray(atom["confusion_matrix"], dtype=np.int64)
            entry = dict(atom)
            entry["confusion_matrix"] = matrix.tolist()
            entry.setdefault("valid_px", int(matrix.sum()))
            result[str(name)] = entry
        else:
            result[str(name)] = atom
    return result


class PredictionRunWriter:
    """Atomic scene sidecars plus a run-level completion manifest."""

    def __init__(
        self,
        output_dir: str | Path,
        run_metadata: Mapping[str, Any],
        *,
        required: bool = False,
        max_runs: int = DEFAULT_MAX_RUNS,
        block_rows: int = DEFAULT_BLOCK_ROWS,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_metadata = dict(run_metadata)
        self.required = bool(required)
        self.max_runs = int(max_runs)
        self.block_rows = int(block_rows)
        self.scenes: list[dict[str, Any]] = []
        self.skipped: list[dict[str, Any]] = []
        self.failed: list[dict[str, Any]] = []
        self.started_utc = utc_now()

    def encode_raw(self, mask: np.ndarray) -> dict[str, Any]:
        return encode_label_rle(mask, max_runs=self.max_runs, block_rows=self.block_rows)

    def record_skip(self, scene_id: str, status: str, detail: str = "") -> None:
        self.skipped.append({"scene_id": str(scene_id), "status": str(status), "detail": str(detail)})

    def record_failure(self, scene_id: str, status: str, detail: str) -> None:
        self.failed.append({"scene_id": str(scene_id), "status": str(status), "detail": str(detail)})

    def adopt_scene(self, path: str | Path) -> dict[str, Any]:
        """Adopt an already verified sidecar when resuming an interrupted backfill."""
        scene_path = Path(path)
        document = load_scene_document(scene_path)
        final = document["masks"]["final"]
        coverage = document["coverage"]
        record = {
            "scene_id": document["scene_id"],
            "status": "ok",
            "json": scene_path.name,
            "json_sha256": sha256_bytes(canonical_json_bytes(document)),
            "payload_sha256": final["payload_sha256"],
            "height": int(document["raster"]["height"]),
            "width": int(document["raster"]["width"]),
            "coverage_mode": coverage["mode"],
            "covered_px": int(coverage["covered_px"]),
            "pixel_counts": final["pixel_counts"],
            "raw_final_changed_px": int(document["raw_final_changed_px"]),
            "metric_atoms": document["metric_atoms"],
        }
        self.scenes.append(record)
        return record

    def write_scene(
        self,
        *,
        scene_id: str,
        final_mask: np.ndarray,
        raw_payload: Mapping[str, Any],
        coverage: Mapping[str, Any],
        labels: Mapping[str, Any],
        reconstruction: Mapping[str, Any],
        postprocess: list[Mapping[str, Any]],
        probability: Mapping[str, Any],
        metric_semantics: Mapping[str, Any],
        metric_atoms: Mapping[str, Any],
        provenance: Mapping[str, Any],
    ) -> dict[str, Any]:
        final_arr = _validate_uint8_raster(final_mask)
        final_payload = encode_label_rle(
            final_arr, max_runs=self.max_runs, block_rows=self.block_rows
        )
        if raw_payload["payload_sha256"] == final_payload["payload_sha256"]:
            raw_doc: dict[str, Any] = {"same_as": "final"}
            changed_px = 0
        else:
            raw_doc = _encoded_mask_doc(raw_payload)
            changed_px = _count_changed_from_encoded(raw_payload, final_arr)

        postprocess_changed = sum(int(item.get("pixels_changed", 0)) for item in postprocess)
        if changed_px > postprocess_changed:
            raise PredMaskError(
                f"raw/final changed_px={changed_px} exceeds postprocess total={postprocess_changed}"
            )

        coverage_doc = dict(coverage)
        coverage_mode = coverage_doc.get("mode")
        if coverage_mode == "derived_from_untested":
            untested = int(coverage_doc["untested_value"])
            num_classes = int(labels["num_classes"])
            if 0 <= untested < num_classes:
                raise PredMaskError("untested_value overlaps the class-index domain")
            covered_px = sum(
                int(np.count_nonzero(final_arr[row : row + self.block_rows] != untested))
                for row in range(0, final_arr.shape[0], self.block_rows)
            )
            coverage_doc["covered_px"] = covered_px
        elif coverage_mode == "full":
            coverage_doc["covered_px"] = int(final_arr.size)
        elif coverage_mode == "explicit":
            explicit = np.asarray(coverage_doc.pop("mask"), dtype=bool)
            if explicit.shape != final_arr.shape:
                raise PredMaskError("explicit coverage shape mismatch")
            coverage_doc["payload"] = encode_binary_rle(
                explicit, max_runs=self.max_runs, block_rows=self.block_rows
            )
            coverage_doc["covered_px"] = int(explicit.sum())
        else:
            raise PredMaskError(f"unsupported coverage mode: {coverage_mode!r}")

        scene = str(scene_id)
        document = {
            "schema": SCENE_SCHEMA,
            "codec": {
                "name": CODEC_NAME,
                "version": CODEC_VERSION,
                "module_sha256": module_sha256(),
            },
            "scene_id": scene,
            "scene_id_nfc_sha256": sha256_bytes(
                unicodedata.normalize("NFC", scene).encode("utf-8")
            ),
            "raster": {"height": int(final_arr.shape[0]), "width": int(final_arr.shape[1]), "order": "C"},
            "labels": dict(labels),
            "masks": {"raw": raw_doc, "final": _encoded_mask_doc(final_payload)},
            "coverage": coverage_doc,
            "postprocess": [dict(item) for item in postprocess],
            "raw_final_changed_px": int(changed_px),
            "probability": dict(probability),
            "metric_semantics": dict(metric_semantics),
            "metric_atoms": _normalise_matrix_atoms(metric_atoms),
            "reconstruction": dict(reconstruction),
            "provenance": dict(provenance),
        }
        relative = _safe_scene_filename(scene)
        path = self.output_dir / relative
        json_sha = atomic_write_json(path, document)
        record = {
            "scene_id": scene,
            "status": "ok",
            "json": relative,
            "json_sha256": json_sha,
            "payload_sha256": final_payload["payload_sha256"],
            "height": int(final_arr.shape[0]),
            "width": int(final_arr.shape[1]),
            "coverage_mode": coverage_mode,
            "covered_px": int(coverage_doc["covered_px"]),
            "pixel_counts": final_payload["pixel_counts"],
            "raw_final_changed_px": int(changed_px),
            "metric_atoms": document["metric_atoms"],
        }
        self.scenes.append(record)
        return record

    def write_scene_safe(self, **kwargs: Any) -> dict[str, Any] | None:
        scene_id = str(kwargs.get("scene_id", "<unknown>"))
        try:
            return self.write_scene(**kwargs)
        except RunBudgetExceeded as exc:
            self.record_failure(scene_id, "exceeds_run_budget", str(exc))
            print(f"  ⚠ prediction artifact skipped for {scene_id}: {exc}")
        except Exception as exc:  # artifact failures must not erase completed inference
            self.record_failure(scene_id, "json_write_failed", repr(exc))
            print(f"  ⚠ prediction artifact failed for {scene_id}: {exc}")
        return None

    def finalize(self) -> Path:
        expected = self.run_metadata.get("expected_scenes")
        roster = self.scenes + self.skipped + self.failed
        roster_ids = [str(item.get("scene_id")) for item in roster]
        roster_errors: list[str] = []
        if len(set(roster_ids)) != len(roster_ids):
            roster_errors.append("duplicate_scene_ids")
        if expected is not None and len(roster) != int(expected):
            roster_errors.append(f"roster_count={len(roster)} expected={int(expected)}")
        complete = not self.failed and not self.skipped and not roster_errors
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "codec": {
                "name": CODEC_NAME,
                "version": CODEC_VERSION,
                "module_sha256": module_sha256(),
            },
            "run": self.run_metadata,
            "started_utc": self.started_utc,
            "finished_utc": utc_now(),
            "complete": complete,
            "roster_integrity": {
                "ok": not roster_errors,
                "errors": roster_errors,
                "unique_scene_ids": len(set(roster_ids)),
                "records": len(roster),
            },
            "scenes": self.scenes,
            "skipped": self.skipped,
            "failed": self.failed,
            "counts": {
                "ok": len(self.scenes),
                "skipped": len(self.skipped),
                "failed": len(self.failed),
                "expected": self.run_metadata.get("expected_scenes"),
            },
        }
        path = self.output_dir / "predictions_manifest.json"
        atomic_write_json(path, manifest)
        if self.required and not complete:
            raise PredMaskError(
                "required prediction artifacts incomplete: "
                f"{len(self.failed)} failed, {len(self.skipped)} skipped, "
                f"roster_errors={roster_errors}; "
                f"see {path}"
            )
        return path

    def finalize_safe(self) -> Path | None:
        """Finalize fail-soft unless this writer was configured as required."""
        try:
            return self.finalize()
        except Exception as exc:
            if self.required:
                raise
            print(f"  ⚠ optional prediction manifest finalization failed: {exc}")
            return None


def load_scene_document(path: str | Path, verify_sidecar: bool = True) -> dict[str, Any]:
    scene_path = Path(path)
    raw = scene_path.read_bytes()
    document = json.loads(raw.decode("utf-8"))
    if document.get("schema") != SCENE_SCHEMA:
        raise PredMaskError(f"unsupported scene schema: {document.get('schema')!r}")
    if verify_sidecar:
        expected = Path(str(scene_path) + ".sha256").read_text().strip()
        actual = sha256_bytes(canonical_json_bytes(document))
        if expected != actual:
            raise PredMaskError(f"JSON checksum mismatch: {scene_path}")
    return document


def decode_scene_final(document: Mapping[str, Any]) -> np.ndarray:
    raster = document["raster"]
    return decode_label_rle(
        document["masks"]["final"],
        (int(raster["height"]), int(raster["width"])),
    )


def decode_scene_coverage(document: Mapping[str, Any], final_mask: np.ndarray | None = None) -> np.ndarray:
    raster = document["raster"]
    shape = (int(raster["height"]), int(raster["width"]))
    coverage = document["coverage"]
    if coverage["mode"] == "full":
        return np.ones(shape, dtype=bool)
    if coverage["mode"] == "derived_from_untested":
        if final_mask is None:
            final_mask = decode_scene_final(document)
        return final_mask != int(coverage["untested_value"])
    if coverage["mode"] == "explicit":
        return decode_label_rle(coverage["payload"], shape).astype(bool)
    raise PredMaskError(f"unsupported coverage mode: {coverage['mode']!r}")


def manifest_roster_errors(manifest: Mapping[str, Any]) -> list[str]:
    """Validate completeness independently, including manifests from older writers."""
    errors: list[str] = []
    scenes = list(manifest.get("scenes", []))
    skipped = list(manifest.get("skipped", []))
    failed = list(manifest.get("failed", []))
    roster = scenes + skipped + failed
    scene_ids = [str(item.get("scene_id")) for item in roster]
    expected = manifest.get("counts", {}).get(
        "expected", manifest.get("run", {}).get("expected_scenes")
    )
    if not bool(manifest.get("complete", False)):
        errors.append("manifest_complete_false")
    if skipped:
        errors.append(f"skipped={len(skipped)}")
    if failed:
        errors.append(f"failed={len(failed)}")
    if len(set(scene_ids)) != len(scene_ids):
        errors.append("duplicate_scene_ids")
    if expected is not None and len(roster) != int(expected):
        errors.append(f"roster_count={len(roster)} expected={int(expected)}")
    return errors


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": _package_version("torch"),
        "opencv": _package_version("opencv-python") or _package_version("opencv-python-headless"),
        "rasterio": _package_version("rasterio"),
    }


def collect_git_state(repo_root: str | Path, code_files: Iterable[str | Path] = ()) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    base = ["git", "-c", f"safe.directory={root}"]

    def run(*args: str) -> str:
        result = subprocess.run(base + list(args), cwd=root, check=True, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        status = run("status", "--porcelain=v1")
    except Exception as exc:
        return {"available": False, "error": repr(exc)}
    hashes: dict[str, str] = {}
    for item in code_files:
        path = Path(item)
        if not path.is_absolute():
            path = root / path
        if path.exists() and path.is_file():
            try:
                key = str(path.resolve().relative_to(root))
            except ValueError:
                key = str(path.resolve())
            hashes[key] = sha256_file(path)
    return {
        "available": True,
        "commit": commit,
        "dirty": bool(status),
        "status_sha256": sha256_bytes(status.encode("utf-8")),
        "code_files": hashes,
    }


def build_run_metadata(
    *,
    repo_name: str,
    repo_root: str | Path,
    arm: str,
    fold: int | str | None,
    model_path: str | Path,
    resolved_config: Mapping[str, Any],
    entrypoint: str,
    recon_callable: str,
    expected_scenes: int | None,
    inputs: Mapping[str, Any],
    code_files: Iterable[str | Path],
    unsupported_harnesses: Iterable[str] = (),
) -> dict[str, Any]:
    model = Path(model_path)
    model_sha = sha256_file(model) if model.exists() else None
    fold_label = f"fold{fold}" if fold is not None else "fold-unknown"
    stable_model = model_sha[:12] if model_sha else "model-unknown"
    return {
        "repo": str(repo_name),
        "arm": str(arm),
        "fold": fold,
        "run_id": f"{arm}/{fold_label}/{stable_model}",
        "entrypoint": entrypoint,
        "recon_callable": recon_callable,
        "resolved_config_sha256": sha256_bytes(canonical_json_bytes(resolved_config)),
        "model": {
            "path": str(model),
            "sha256": model_sha,
            "size": model.stat().st_size if model.exists() else None,
        },
        "inputs": dict(inputs),
        "expected_scenes": expected_scenes,
        "git": collect_git_state(repo_root, code_files),
        "environment": collect_environment(),
        "unsupported_harnesses": list(unsupported_harnesses),
    }


def probability_reference(
    *,
    status: str,
    path: str | Path | None = None,
    key: str | None = None,
    dtype: str | None = None,
    shape: Iterable[int] | None = None,
    semantics: str | None = None,
    uncovered_sentinel: str | None = None,
    coverage_ref: str | None = None,
) -> dict[str, Any]:
    if status not in {"persisted", "not_persisted", "unknown"}:
        raise ValueError(f"invalid probability status: {status}")
    result: dict[str, Any] = {"status": status}
    if status == "persisted":
        if path is None:
            raise ValueError("persisted probability requires a path")
        probability_path = Path(path)
        result.update({
            "path": str(probability_path),
            "sha256": sha256_file(probability_path),
            "key": key,
            "dtype": dtype,
            "shape": list(shape) if shape is not None else None,
            "semantics": semantics,
            "uncovered_sentinel": uncovered_sentinel,
            "coverage_ref": coverage_ref,
            "authoritative_for": ["threshold_sweep", "pr_curve", "calibration"],
            "not_authoritative_for": ["published_hard_metrics"],
        })
    return result


def verify_artifact_dir(directory: str | Path) -> dict[str, int]:
    root = Path(directory)
    manifest_path = root / "predictions_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    ok = 0
    failed = 0
    for path in sorted(root.glob("*.pred.json")):
        try:
            document = load_scene_document(path)
            final = decode_scene_final(document)
            coverage = decode_scene_coverage(document, final)
            if coverage.shape != final.shape:
                raise PredMaskError("coverage/final shape mismatch")
            ok += 1
        except Exception as exc:
            failed += 1
            print(f"FAIL {path}: {exc}", file=sys.stderr)
    if manifest is not None and len(manifest.get("scenes", [])) != ok:
        raise PredMaskError(
            f"manifest lists {len(manifest.get('scenes', []))} scenes, verified {ok}"
        )
    return {"ok": ok, "failed": failed}


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Verify and inspect oil prediction artifacts")
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify", help="decode and verify every artifact in a directory")
    verify.add_argument("directory")
    stat = sub.add_parser("stat", help="print a sidecar summary without decoding the raster")
    stat.add_argument("scene_json")
    args = parser.parse_args()
    if args.command == "verify":
        result = verify_artifact_dir(args.directory)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["failed"] == 0 else 1
    document = load_scene_document(args.scene_json)
    summary = {
        "scene_id": document["scene_id"],
        "shape": [document["raster"]["height"], document["raster"]["width"]],
        "n_runs": document["masks"]["final"]["n_runs"],
        "pixel_counts": document["masks"]["final"]["pixel_counts"],
        "coverage": document["coverage"],
        "probability_status": document["probability"]["status"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
