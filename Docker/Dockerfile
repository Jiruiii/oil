# =====================================================================
# 油汙偵測 pipeline — 可攜式訓練環境（供交接使用）
#   * 不含個人 git 身份 / SSH / Obsidian vault 掛載
#   * base image 是官方 PyTorch image（已含 conda、CUDA 12.8 runtime、cuDNN）
#   * cu128 wheel 是多架構編譯，涵蓋 Ampere(sm_86，例如 RTX 3090)
#     到 Blackwell(sm_120，RTX 50 系)。換到別張卡不需要改 base image，
#     只要對方 host 的 NVIDIA driver 支援 CUDA 12.8
#     （Linux 需 driver >= 570.26，這是 CUDA 12.8 GA 對應門檻；550.x 只夠
#     CUDA 12.4，且 Blackwell sm_120 本身也是 570 系列驅動才開始支援，
#     用 `nvidia-smi` 查得到版本）
# =====================================================================
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

# ---------------------------------------------------------------------
# 1. 系統層級工具（rasterio / opencv / GUI lib + tzdata）
# ---------------------------------------------------------------------
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libexpat1 \
        libgl1 \
        libglib2.0-0 \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------
# 1b. 時區設定為台北（UTC+8）
#     不設的話容器是 UTC，date/檔案 timestamp 都比實際慢 8 小時
# ---------------------------------------------------------------------
ENV TZ=Asia/Taipei
RUN ln -sf /usr/share/zoneinfo/Asia/Taipei /etc/localtime && \
    echo "Asia/Taipei" > /etc/timezone && \
    dpkg-reconfigure -f noninteractive tzdata

# ---------------------------------------------------------------------
# 2. Node.js + Claude Code CLI
# ---------------------------------------------------------------------
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g @anthropic-ai/claude-code

# ---------------------------------------------------------------------
# 2b. Codex CLI
# ---------------------------------------------------------------------
ENV CODEX_HOME=/root/.codex
RUN mkdir -p /opt/codex /usr/local/bin && \
    curl -fsSL https://chatgpt.com/codex/install.sh -o /tmp/install-codex.sh && \
    CODEX_HOME=/opt/codex \
    CODEX_INSTALL_DIR=/usr/local/bin \
    CODEX_NON_INTERACTIVE=1 \
    sh /tmp/install-codex.sh && \
    rm -f /tmp/install-codex.sh && \
    /usr/local/bin/codex --version

WORKDIR /workspace

# ---------------------------------------------------------------------
# 3. Python 套件（pipeline 真正用到的，清單見 requirements.txt）
# ---------------------------------------------------------------------
# requirements.txt 在專案根目錄，build 時 context 要是根目錄
# （例：docker build -f Docker/Dockerfile -t oil-detect:latest .）
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
    && rm /tmp/requirements.txt

# ---------------------------------------------------------------------
# 4. 環境變數 / Shell alias
# ---------------------------------------------------------------------
ENV TERM=xterm-256color
ENV COLORTERM=truecolor
ENV FORCE_COLOR=3

RUN echo "" >> /root/.bashrc && \
    echo "# 自訂別名" >> /root/.bashrc && \
    echo "alias nv='watch -n 5 nvidia-smi'" >> /root/.bashrc && \
    echo "alias ll='ls -lah'" >> /root/.bashrc && \
    echo "alias python=python3" >> /root/.bashrc

# ---------------------------------------------------------------------
# 5. 提示符（方便辨識 Docker 環境）
# ---------------------------------------------------------------------
RUN echo "PS1='\[\033[01;36m\][DeepLabV3+]\[\033[00m\] \u@\h:\w\$ '" >> /root/.bashrc

CMD ["/bin/bash"]
