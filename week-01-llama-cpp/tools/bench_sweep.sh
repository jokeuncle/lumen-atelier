#!/usr/bin/env bash
# bench_sweep.sh — 自动跑 llama-bench 多组配置，结果存到 reports/
#
# 你能学到什么：
#   • prefill (pp) 是吞吐受限：tokens/sec 高，并行度高
#   • decode (tg) 是 memory-bandwidth 受限：tokens/sec 低很多
#   • -ngl 0 (纯 CPU) vs -ngl 99 (尽量 GPU/Metal) 的差别 → unified memory 的威力
#
# 用法：
#   bash bench_sweep.sh                              # 用默认模型
#   bash bench_sweep.sh ./models/your.gguf

set -euo pipefail

cd "$(dirname "$0")/.."
MODEL="${1:-./models/Qwen2.5-7B-Instruct-Q4_K_M.gguf}"

if [ ! -f "$MODEL" ]; then
  echo "找不到模型：$MODEL"
  echo "先跑 setup.sh，或手动指定 .gguf 路径"
  exit 1
fi

if ! command -v llama-bench >/dev/null 2>&1; then
  echo "llama-bench 没装。先跑 setup.sh。"
  exit 1
fi

mkdir -p reports
TS=$(date +%Y%m%d-%H%M%S)
OUT="reports/bench-$TS.csv"

echo "▶ benchmark 写到 $OUT"
echo ""

# llama-bench 默认指标：
#   pp512   prefill 512 token 的吞吐
#   tg128   decode 128 token 的吞吐
# 这里扫描 -ngl（GPU layer 数）

# llama-bench 的 -o csv 输出包含表头，把每次结果都拼起来
HEADER_WRITTEN=0
for NGL in 0 16 32 99; do
  for THREADS in 6 12; do
    echo "── ngl=$NGL threads=$THREADS ──"
    if [ "$HEADER_WRITTEN" -eq 0 ]; then
      llama-bench -m "$MODEL" -ngl $NGL -t $THREADS -p 512 -n 128 -o csv \
        | tee -a "$OUT"
      HEADER_WRITTEN=1
    else
      # 跳过表头那一行
      llama-bench -m "$MODEL" -ngl $NGL -t $THREADS -p 512 -n 128 -o csv \
        | tail -n +2 | tee -a "$OUT"
    fi
    echo ""
  done
done

echo ""
echo "✓ 完成。结果在：$OUT"
echo ""
echo "💡 看完想想："
echo "   1) ngl=0 (纯 CPU) vs ngl=99 的 tg 提速比？unified memory 让 GPU 直接吃 weight，没有 PCIe copy"
echo "   2) threads 从 6 → 12 在 ngl=99 时还有意义吗？为什么？"
echo "   3) pp / tg 的差距大概多少倍？理解 prefill 与 decode 的瓶颈差异"
