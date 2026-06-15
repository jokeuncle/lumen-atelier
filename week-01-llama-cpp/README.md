# Week 01 — llama.cpp 推理流程

> 主线 W1–W2 · vLLM / SGLang · 推理引擎内核
> 起点节点：在 M5 Pro 上跑通 Qwen2.5-7B-Q4，读懂 ggml 张量流，建立 LLM 推理的端到端心智模型。

---

## 这两周想要的转变

不是"会用 llama-cli"，而是：**给你一段 prompt，你能在白板上画出每一个张量从输入到输出的形状变化，并指出 prefill 与 decode 在哪一步分叉。**

---

## 学习地图

```
Day 1–2   ┃ 跑通      llama.cpp 装好 · Qwen2.5-7B-Q4 出第一个 token
Day 3–4   ┃ 拆 GGUF   用 tools/gguf_inspect.py 看张量布局
Day 5–6   ┃ 拆推理路径 读 src/llama.cpp + ggml.c 主循环
Day 7     ┃ 拆 token   tools/tokenizer_play.py 看 BPE 切分
─────────────────────────────────────
Day 8–9   ┃ 测吞吐    llama-bench / tools/bench_sweep.sh
Day 10    ┃ 测质量    llama-perplexity 跑 wiki 评估
Day 11–12 ┃ 改采样    --temp / --top-p 看分布
Day 13–14 ┃ 写笔记    输出 notes.md 这篇 2000 字
```

---

## 必修知识点（验收时要能说清楚）

1. **GGUF 文件格式** — header / metadata / tensor data 三段布局
2. **ggml 张量结构** — `ne[]` 维度、`nb[]` stride、type 量化类型
3. **推理主循环** — tokenize → embed → N × (RMSNorm + attn + FFN) → unembed → sample
4. **Attention 内部** — QKV 投影 → RoPE → masked softmax → out 投影
5. **KV cache 形状** — `[layer][batch, head, seq, head_dim]`，decode 每步追加 1
6. **量化方案** — Q4_K_M / Q5_K_M / Q8_0 的位宽与质量取舍
7. **Prefill vs Decode** — 一个吞吐受限、一个延迟受限，KV cache 在两阶段扮演不同角色
8. **采样策略** — temperature / top-k / top-p / repetition penalty 各抑制什么

---

## 文件说明

| 文件 | 用途 |
|---|---|
| `setup.sh` | 一键装环境 + 下模型（运行前请过目） |
| `lab.html` | 交互式实验本，进度与笔记自动保存 |
| `tools/gguf_inspect.py` | 读 GGUF 文件，打印 metadata 与所有张量 |
| `tools/tokenizer_play.py` | 输入文本看 BPE 切成什么 token |
| `tools/bench_sweep.sh` | 自动跑多组 llama-bench 配置，输出 CSV |
| `notes.md` | 这周的交付物草稿（请你写，我给了骨架） |
| `models/` | 模型文件目录 |
| `reports/` | benchmark / perplexity 结果 |

---

## 必读资料（按这个顺序）

1. **先粗读** — [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) （Jay Alammar）
2. **配合源码** — [llama.cpp 仓库](https://github.com/ggml-org/llama.cpp)：
   - `src/llama.cpp` 的 `llama_decode_internal` 函数（主推理循环）
   - `ggml/src/ggml.c` 顶部的 `struct ggml_tensor` 定义
   - `examples/main/main.cpp` 的 token 生成循环
3. **GGUF 格式** — [官方 spec](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
4. **进阶配菜** — Lilian Weng [The Transformer Family v2.0](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)

---

## 开始

```bash
cd week-01-llama-cpp
bash setup.sh           # 先看一眼内容再跑
open lab.html           # 在浏览器里打开实验本
```
