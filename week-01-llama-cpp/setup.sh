#!/usr/bin/env bash
# Lumen Atelier · Week 01 setup
#
# 做什么：
#   1) brew install llama.cpp（提供 llama-cli / llama-bench / llama-perplexity 等）
#   2) 安装 uv（现代 Python 包管理器，比 pip 快 10x）
#   3) 在本目录建 .venv 并装 huggingface_hub / gguf / numpy / tabulate
#   4) 下载 Qwen2.5-7B-Instruct-Q4_K_M.gguf（~4.7GB）到 ./models/
#
# 这个脚本是幂等的——重复运行不会重复安装。
# 每一步开始前会打印当前步骤，失败立即停止。

set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
MODEL_DIR="$ROOT/models"
MODEL_FILE="Qwen2.5-7B-Instruct-Q4_K_M.gguf"
# 国内访问 HuggingFace 速度通常只有 100KB/s 量级；ModelScope（魔搭，阿里）
# 是 Qwen 官方镜像，速度通常在 1MB/s+。文件内容完全一致。
MS_MODEL_ID="Qwen/Qwen2.5-7B-Instruct-GGUF"
MS_FILE="qwen2.5-7b-instruct-q4_k_m.gguf"
MS_URL="https://modelscope.cn/api/v1/models/${MS_MODEL_ID}/repo?Revision=master&FilePath=${MS_FILE}"

say() { printf "\n\033[1;36m▶ %s\033[0m\n" "$*"; }
ok()  { printf "  \033[1;32m✓\033[0m %s\n" "$*"; }
skip(){ printf "  \033[1;33m·\033[0m %s\n" "$*"; }

# ── 1. llama.cpp ───────────────────────────────────────────────────────
say "Step 1/4 · llama.cpp"
if command -v llama-cli >/dev/null 2>&1; then
  skip "已安装：$(llama-cli --version 2>&1 | head -1)"
else
  brew install llama.cpp
  ok "llama-cli / llama-bench / llama-perplexity / llama-server 就绪"
fi

# ── 2. uv ──────────────────────────────────────────────────────────────
say "Step 2/4 · uv (Python 管理器)"
if command -v uv >/dev/null 2>&1; then
  skip "已安装：$(uv --version)"
else
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # uv 装到 ~/.local/bin，本会话需要把它加到 PATH
  export PATH="$HOME/.local/bin:$PATH"
  ok "uv 已安装。新终端窗口里 uv 命令会自动可用。"
fi

# ── 3. venv + Python 依赖 ─────────────────────────────────────────────
say "Step 3/4 · Python 环境 (.venv)"
if [ ! -d ".venv" ]; then
  uv venv --python 3.11
  ok "建好 .venv (Python 3.11)"
else
  skip ".venv 已存在"
fi

uv pip install --quiet \
  huggingface_hub \
  gguf \
  numpy \
  tabulate \
  transformers
ok "依赖装好：huggingface_hub / gguf / numpy / tabulate / transformers"

# ── 4. 下载模型 ────────────────────────────────────────────────────────
say "Step 4/4 · 下载 Qwen2.5-7B-Instruct-Q4_K_M (~4.5GB · 走魔搭)"
mkdir -p "$MODEL_DIR"
TARGET="$MODEL_DIR/$MODEL_FILE"
if [ -f "$TARGET" ]; then
  size=$(du -h "$TARGET" | cut -f1)
  skip "模型已在 ./models/$MODEL_FILE ($size)"
else
  echo "  来源：ModelScope ($MS_MODEL_ID)"
  echo "  目标：$TARGET"
  echo ""
  # -L 跟随重定向；-C - 断点续传；--retry 网络抖动时重试；--progress-bar 看进度
  curl -L --progress-bar -C - --retry 5 --retry-delay 3 \
    -o "$TARGET" "$MS_URL"
  # 校验最小尺寸（防止下到 HTML 错误页）
  actual=$(stat -f%z "$TARGET" 2>/dev/null || stat -c%s "$TARGET")
  if [ "$actual" -lt 1000000000 ]; then
    echo ""
    echo "  ⚠️  下载到的文件只有 $actual 字节，应该 ≥ 4GB。可能下到错误页。删了重试。"
    rm -f "$TARGET"
    exit 1
  fi
  ok "模型就绪：./models/$MODEL_FILE ($(du -h "$TARGET" | cut -f1))"
fi

# ── 收尾 ───────────────────────────────────────────────────────────────
cat <<EOF


┌─────────────────────────────────────────────────────────────┐
│  Setup 完成。下面几条命令做第一次冒烟测试：                       │
└─────────────────────────────────────────────────────────────┘

  # ① 第一次对话（出 token 即成功）
  llama-cli -m ./models/$MODEL_FILE \\
    -p "用一句话解释 KV cache 是什么。" \\
    -n 128 --temp 0.7

  # ② 看看 GGUF 文件内部
  source .venv/bin/activate
  python tools/gguf_inspect.py ./models/$MODEL_FILE

  # ③ 看看 tokenizer 怎么切句子
  python tools/tokenizer_play.py "AI Infra 工程师在 2026 年的核心技能是什么？"

  # ④ 打开你的实验本
  open lab.html

EOF
