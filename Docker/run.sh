#!/usr/bin/env bash
# =====================================================================
# 啟動訓練用 container 的小工具（乾淨版，不含任何個人 agent/SSH 狀態）
#   * 第一次執行：docker run -d 建立背景常駐 container，再 exec 進去
#   * 已停止：docker start 拉起，再 exec 進去
#   * 執行中：直接 exec 進去
#   關掉終端機/斷線只會結束當下這個 shell，container 跟裡面的訓練照跑；
#   要真正停掉 container，另外下：docker stop "${NAME}"
#
# 用法：
#   DATA_DIR=/path/to/oil_dataset ./Docker/run.sh
#   （在本目錄根目錄下執行，$(pwd) 會被掛成容器裡的 /workspace）
# =====================================================================
set -euo pipefail

IMAGE="${IMAGE:-oil-detect:latest}"
NAME="${NAME:-oil_detect}"
DATA_DIR="${DATA_DIR:?請先設定 DATA_DIR=你的資料集路徑，例如：DATA_DIR=/path/to/oil_dataset ./Docker/run.sh}"

if docker ps --format '{{.Names}}' | grep -Fqx -- "${NAME}"; then
    echo "[run.sh] container '${NAME}' 正在執行中，docker exec 進去"
    exec docker exec -it "${NAME}" /bin/bash
fi

if docker ps -a --format '{{.Names}}' | grep -Fqx -- "${NAME}"; then
    echo "[run.sh] container '${NAME}' 已停止，docker start（detached）後 exec 進去"
    docker start "${NAME}" >/dev/null
    exec docker exec -it "${NAME}" /bin/bash
fi

echo "[run.sh] 建立新 container '${NAME}' (image=${IMAGE})，detached 背景常駐"
docker run -d -it \
  --name "${NAME}" \
  --gpus all \
  --shm-size=32g \
  -w /workspace \
  -v "$(pwd):/workspace" \
  -v "${DATA_DIR}:/data" \
  "${IMAGE}" /bin/bash >/dev/null
exec docker exec -it "${NAME}" /bin/bash
