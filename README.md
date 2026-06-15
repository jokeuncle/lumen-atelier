# Lumen Atelier · 光之工坊

> A 90-day, source-code-first journey into AI Infrastructure on Apple Silicon.
> 一份 90 天的 AI Infra 领跑路线 —— 从源码读起，从 M5 Pro 跑起。

```
machine    │ MacBook Pro · M5 Pro · 18C CPU · 20C GPU · 24GB unified · 2TB
status     │ Week 01 in progress
output     │ blogs · PRs · benchmarks (公开才算完成)
```

## What this is

Three parallel tracks, twelve weeks, one MacBook + occasional cloud GPU:

| Track | Focus | Why |
|---|---|---|
| **Mainline** | vLLM / SGLang internals — paged attention, continuous batching, KV quant, spec decoding | 业界最缺的技能 |
| **Sideline** | MLX & Apple Silicon ML stack — custom ops, Metal kernels | 中文圈最少人卷的差异化赛道 |
| **Cloud** | A100 / H100 sanity checks — Triton kernels, tensor parallel | Mac 跑不动的部分，按需补 |

→ Open [`roadmap.html`](roadmap.html) in any browser. Click any milestone to read the full syllabus for that node.

## Layout

```
lumen-atelier/
├── roadmap.html              # 交互式 90 天路线图
├── week-01-llama-cpp/        # 当前进度
│   ├── README.md             # 本周学习地图
│   ├── setup.sh              # 一键搭环境
│   ├── lab.html              # 交互式实验本
│   ├── notes.md              # 交付物草稿
│   └── tools/
│       ├── gguf_inspect.py   # 拆 GGUF 文件
│       ├── tokenizer_play.py # 玩 BPE
│       └── bench_sweep.sh    # 自动 benchmark
└── ...                       # 后续 11 周陆续加入
```

## Principles

```
01 · 读代码 > 看视频     vLLM / MLX 源码即教材
02 · 本地实验 + 云补完   Mac 跑 7B 找手感 · A100 跑真实负载
03 · 产出大于消化       博客 / PR / benchmark · 公开才算完成
04 · 差异化优先         MLX 是别人没卷的赛道 · 优先占位
```

## Why "Lumen Atelier"

**Lumen** = 光通量，象征洞察。**Atelier** = 艺术家工坊。
不是 bootcamp，不是 syllabus —— 是匠人在工坊里造东西，作品会留下来。

---

_Started 2026-06 · License: code MIT, writings CC-BY-SA_
