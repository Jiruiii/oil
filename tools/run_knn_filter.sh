#!/usr/bin/env bash
# G hnm_knn / G0 hnm_swap: A encoder vectors of C's candidate pool, build both arms
# (tools/filter_hard_negatives.py), write their yaml, then train both (3 folds each, ~8 h GPU).
# Folds whose pool vectors exist are not re-extracted.
# Usage: setsid nohup tools/run_knn_filter.sh > hnm_runs/knn_filter.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
export NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=.venv/bin/python

echo "[$(date -Is)] extract start"
$PY tools/filter_hard_negatives.py --stage extract
echo "[$(date -Is)] select start"
$PY tools/filter_hard_negatives.py --stage select
$PY tools/make_hnm_arm_configs.py --arm hnm_knn --arm hnm_swap
echo "[$(date -Is)] training start"
tools/run_hnm_step2.sh hnm_knn hnm_swap
echo "[$(date -Is)] ALL DONE"
