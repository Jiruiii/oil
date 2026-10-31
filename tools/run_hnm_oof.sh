#!/usr/bin/env bash
# OOF arms (E oof, F oof_si): scene-OOF EfficientNet scoring of the HNM_SI pool, selection,
# training configs, then segmentation training of both arms (~6 h scoring + ~7.5 h training).
# Needs the completed HNM_SI mining in hnm_runs/hnm_si_v1/mining.
# Usage: nohup tools/run_hnm_oof.sh > hnm_runs/oof_training.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
export NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1
CFG=preprocess/hnm_si/config.example.yaml
PY=.venv/bin/python
for f in 1 2 3; do
  echo "[$(date -Is)] oof scoring fold $f"
  $PY -W ignore -m preprocess.hnm_si.pipeline --config $CFG --stage oof --fold $f
done
for arm in oof oof_si; do
  for f in 1 2 3; do
    echo "[$(date -Is)] select $arm fold $f"
    $PY -W ignore -m preprocess.hnm_si.pipeline --config $CFG --stage select --arm $arm --fold $f
  done
done
echo "[$(date -Is)] training configs"
$PY tools/make_hnm_arm_configs.py --arm oof --arm oof_si
tools/run_hnm_step2.sh oof oof_si
echo "[$(date -Is)] ALL DONE"
