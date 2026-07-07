# 学习台账 Ledger

> 单一事实源。每次学习后追加一行，**没有证据链接的条目不算数**。
> 信心自评：1=完全不懂 2=听过 3=能复述 4=能推导/能改代码 5=能教别人
> 此文件由每日 digest 读取判断进度，请保持格式。

## 格式

```
| 日期 | 课程日 | 做了什么 | 证据 | 信心 | 验收状态 |
```

## 记录

| 日期 | 课程日 | 做了什么 | 证据 | 信心 | 验收状态 |
|---|---|---|---|---|---|
| 2026-06-18 | Day 1-2 | 环境搭建 + 模型下载启动 | commit f9093ce · setup.sh 幂等可重跑 | 4 | - |
| 2026-07-04 | Day 3-4 | 模型续传补全 4.4GB · llama-completion 冒烟通过（prefill 110-257 t/s / decode 25 t/s）· gguf_inspect + 手写 gguf_handparse.py 拆完 GGUF（GQA 28:4 · 非 tied embedding · Q4_K/Q6_K/F32 分布） | tools/gguf_handparse.py · 全景图+数学图两篇博客已发布 | 3 | W1-Q1 ✅（经提示） |
| 2026-07-04 | Day 5 | 追通 main→decode 调用链（讲义 Claude 代产）。晚间消化：拼车/KV cache 两个检查问题自主答对；验收 W1-Q3 部分通过、W1-Q1 不通过 | week-01-llama-cpp/notes/day5-call-chain.md · Day5 技术总结博客 https://jokeuncle.github.io/blog/llama-cpp-day5-prefill-decode-call-chain/ · 本次验收记录 | 2 | W1-Q3 ✅（7/6 追问通过） · W1-Q1 ✅（7/6 经提示） |
| 2026-07-06 | Day 6 | 计算图与 ggml 深挖：伪代码实现计算图，拆 ggml_tensor / ggml_cgraph / Qwen2 graph builder / KV cache 写入路径，并发布 Day6 博客 | docs/blog/2026-07-06-ggml-computation-graph.md · https://jokeuncle.github.io/blog/lumen-atelier-2026-07-06-ggml-computation-graph/ · blog commit 4b4381a | 3 | W1-Q4 待验收 |
| 2026-07-07 | Day 6 补强 | QKV 投影专项讲解：从 Query/Key/Value 角色、矩阵乘、shape 推导、GQA 28:4 到 llama.cpp build_qkv 源码对应，写成小白向博客 | docs/blog/2026-07-07-qkv-projection-math.md · https://jokeuncle.github.io/blog/lumen-atelier-2026-07-07-qkv-projection-math/ · source commit bafc679 · blog commit dbb43e7 | 3 | W1-Q4 待验收 |
| 2026-07-07 | Day 6 补强 | RoPE 专项讲解：从二维旋转公式、128 维 head 拆成 64 对坐标、位置角度 theta、相对位置进入 QK 点积，到 llama.cpp ggml_rope_ext 源码对应，写成小白向博客 | docs/blog/2026-07-07-rope-rotation-math.md · https://jokeuncle.github.io/blog/lumen-atelier-2026-07-07-rope-rotation-math/ · source commit 038a6ee · blog commit 60c5d50 | 3 | W1-Q4 待验收 |

## 欠账区（跳过待补）

| 记入日期 | 欠什么 | 薄弱点 | 还账方式 |
|---|---|---|---|

## 验收通过记录

| 日期 | 题号 | 评语 |
|---|---|---|
| 2026-07-06 | W1-Q1 | 经提示通过：能说清 header/metadata/tensor info/tensor data 顺序、metadata 前置原因、alignment 作用和 tensor offset 相对 tensor data 区。 |
| 2026-07-06 | W1-Q3 | 追问通过：能分清 vocab 是文本片段到 token id 的字典，token_embd.weight 是 token id 到 3584 维向量的权重表，二者共享 vocab_size 编号空间。 |
