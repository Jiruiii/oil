#!/usr/bin/env python3
"""Backfill patch metrics from stored confusion matrices.

Historical DeepLab runs stored Oil-only IoU/F1 under the aggregate names
``mIoU`` and ``F1-macro``. The four confusion-matrix cells are authoritative,
so this migration can correct the labels without rerunning inference.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


CM_KEYS = {
    "tn": "CM_GT(Background)_Pred(Background)",
    "fp": "CM_GT(Background)_Pred(Oil)",
    "fn": "CM_GT(Oil)_Pred(Background)",
    "tp": "CM_GT(Oil)_Pred(Oil)",
}
SEMANTICS = "macro_all_classes_from_confusion_matrix_v2"


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def metrics_from_confusion_matrix(data: dict) -> dict[str, str]:
    tn, fp, fn, tp = (float(data[CM_KEYS[key]]) for key in ("tn", "fp", "fn", "tp"))

    oil_precision = _safe_div(tp, tp + fp)
    oil_recall = _safe_div(tp, tp + fn)
    oil_iou = _safe_div(tp, tp + fp + fn)
    oil_f1 = _safe_div(2 * tp, 2 * tp + fp + fn)

    bg_precision = _safe_div(tn, tn + fn)
    bg_recall = _safe_div(tn, tn + fp)
    bg_iou = _safe_div(tn, tn + fp + fn)
    bg_f1 = _safe_div(2 * tn, 2 * tn + fp + fn)

    return {
        "mIoU": f"{(oil_iou + bg_iou) / 2:.4f}",
        "Accuracy": f"{_safe_div(tp + tn, tp + tn + fp + fn):.4f}",
        "F1-macro": f"{(oil_f1 + bg_f1) / 2:.4f}",
        "IoU(Oil)": f"{oil_iou:.4f}",
        "Precision(Oil)": f"{oil_precision:.4f}",
        "Recall(Oil)": f"{oil_recall:.4f}",
        "F1(Oil)": f"{oil_f1:.4f}",
        "IoU(Background)": f"{bg_iou:.4f}",
        "Precision(Background)": f"{bg_precision:.4f}",
        "Recall(Background)": f"{bg_recall:.4f}",
        "F1(Background)": f"{bg_f1:.4f}",
        "metric_semantics": SEMANTICS,
    }


def _atomic_dump(path: Path, data: dict) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False,
    ) as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def backfill(result_base: Path, apply: bool) -> tuple[int, int, int]:
    eligible = changed = skipped = 0
    for path in sorted(result_base.rglob("run_metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("_kind") != "patch" or any(key not in data for key in CM_KEYS.values()):
            skipped += 1
            continue

        eligible += 1
        corrected = metrics_from_confusion_matrix(data)
        if all(data.get(key) == value for key, value in corrected.items()):
            continue

        changed += 1
        if apply:
            data.update(corrected)
            data["_metrics_backfilled_from_cm"] = True
            _atomic_dump(path, data)

    return eligible, changed, skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-base", type=Path,
        default=Path(__file__).resolve().parents[1] / "result-seg",
    )
    parser.add_argument("--apply", action="store_true", help="Write corrected metrics atomically")
    args = parser.parse_args()

    eligible, changed, skipped = backfill(args.result_base, args.apply)
    mode = "applied" if args.apply else "dry-run"
    print(f"{mode}: eligible={eligible}, changed={changed}, skipped={skipped}")


if __name__ == "__main__":
    main()
