#!/usr/bin/env bash
set -uo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir" || exit 1

export PYTHONUNBUFFERED=1
export NO_ALBUMENTATIONS_UPDATE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

exec >> "$project_dir/training_3fold.log" 2>&1
printf '\n[%s] Starting GT-expand 3-fold reproduction\n' "$(date -Is)"
printf 'Python: %s\nConfig: %s\n' "$project_dir/.venv/bin/python" "$project_dir/main/local_gt_expand.yaml"

"$project_dir/.venv/bin/python" main/main_runner.py --config main/local_gt_expand.yaml
result=$?
printf '[%s] Finished with exit code %s\n' "$(date -Is)" "$result"
exit "$result"
