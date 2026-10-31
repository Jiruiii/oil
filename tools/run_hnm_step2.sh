#!/usr/bin/env bash
# Step 2 of HNM_EXPERIMENT_PLAN.md: train the random, hnm and hnm_si arms (3 folds each, ~12 h GPU).
# Needs main/local_hnm_<arm>.yaml from tools/make_hnm_arm_configs.py. Arms run one after
# another; pass arm names to run a subset, e.g. tools/run_hnm_step2.sh hnm_si.
# Usage: nohup tools/run_hnm_step2.sh > hnm_runs/step2_training.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=.venv/bin/python
arms=("$@")
[ ${#arms[@]} -eq 0 ] && arms=(random hnm hnm_si)  # oof, oof_si: tools/run_hnm_oof.sh
status=0
for arm in "${arms[@]}"; do
  cfg=main/local_hnm_${arm}.yaml
  if [ ! -f "$cfg" ]; then
    echo "[$(date -Is)] missing $cfg; run tools/make_hnm_arm_configs.py first"; exit 1
  fi
  echo "[$(date -Is)] start arm $arm ($cfg)"
  $PY main/main_runner.py --config "$cfg"
  rc=$?
  echo "[$(date -Is)] arm $arm finished with exit code $rc"
  [ $rc -ne 0 ] && status=$rc
done
echo "[$(date -Is)] ALL DONE (status $status)"
exit $status
