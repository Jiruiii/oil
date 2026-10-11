#!/usr/bin/env bash
# ------------------------------------------------------------------
# 一次性「host 端」設定：讓 Docker 容器能存取 GPU（nvidia-container-toolkit）
#
# 這支腳本要在「host 主機」上執行（不是在容器裡面），且需要 sudo。
# 適用 Ubuntu/Debian（apt-based）。跑之前請先確認：
#   1. host 已經裝好 NVIDIA driver，`nvidia-smi` 跑得出東西
#   2. host 已經裝好 Docker Engine（`docker --version` 跑得出來）
#
# 跑完這支腳本之後，才需要去 build/run Docker/Dockerfile 定義的訓練環境。
# ------------------------------------------------------------------
set -euo pipefail

echo "[1/4] 檢查 host 的 NVIDIA driver ..."
nvidia-smi || {
    echo "找不到可用的 NVIDIA driver，請先在 host 上裝好 driver 再跑這支腳本。" >&2
    exit 1
}

echo "[2/4] 加入 NVIDIA Container Toolkit 官方 apt 來源 ..."
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg2
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

echo "[3/4] 安裝 nvidia-container-toolkit 並接上 docker runtime ..."
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "[4/4] 驗證 GPU 能不能穿透進容器 ..."
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

echo
echo "完成。如果上面有印出你的 GPU 資訊，代表 GPU passthrough 設定成功，"
echo "接下來可以照 README 的 Docker 章節去 build/run 訓練環境。"
