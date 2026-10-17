#!/usr/bin/env bash
# Full HNM_SI run: mine all folds, select the hnm_si arm, export reviewable images.
set -euo pipefail
cd "$(dirname "$0")/.."
export NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1
CFG=preprocess/hnm_si/config.example.yaml
PY=.venv/bin/python
OUT=hnm_runs/hnm_si_v1
for f in 1 2 3; do
  echo "[$(date -Is)] mine fold $f"
  $PY -W ignore -m preprocess.hnm_si.pipeline --config $CFG --stage mine --fold $f
done
for f in 1 2 3; do
  echo "[$(date -Is)] select fold $f"
  $PY -W ignore -m preprocess.hnm_si.pipeline --config $CFG --stage select --arm hnm_si --fold $f
done
for f in 1 2 3; do
  echo "[$(date -Is)] export fold $f"
  $PY -W ignore -m preprocess.export_hnm_selection --fold-dir $OUT/arms/hnm_si/fold_$f --out $OUT/hnm_si_selected_images/fold_$f
done
echo "[$(date -Is)] ALL DONE"
