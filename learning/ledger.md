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
| 2026-07-06 | Day 6 | 计算图与 ggml 深挖：伪代码实现计算图，拆 ggml_tensor / ggml_cgraph / Qwen2 graph builder / KV cache 写入路径，并发布 Day6 博客 | docs/blog/2026-07-06-ggml-computation-graph.md · https://jokeuncle.github.io/blog/lumen-atelier-2026-07-06-ggml-computation-graph/ · blog commit 4b4381a | 3 | W1-Q4 ✅（7/8 追问通过） |
| 2026-07-07 | Day 6 补强 | QKV 投影专项讲解：从 Query/Key/Value 角色、矩阵乘、shape 推导、GQA 28:4 到 llama.cpp build_qkv 源码对应，写成小白向博客 | docs/blog/2026-07-07-qkv-projection-math.md · https://jokeuncle.github.io/blog/lumen-atelier-2026-07-07-qkv-projection-math/ · source commit bafc679 · blog commit dbb43e7 | 3 | W1-Q4 ✅（7/8 追问通过） |
| 2026-07-07 | Day 6 补强 | RoPE 专项讲解：从二维旋转公式、128 维 head 拆成 64 对坐标、位置角度 theta、相对位置进入 QK 点积，到 llama.cpp ggml_rope_ext 源码对应，写成小白向博客 | docs/blog/2026-07-07-rope-rotation-math.md · https://jokeuncle.github.io/blog/lumen-atelier-2026-07-07-rope-rotation-math/ · source commit 038a6ee · blog commit 60c5d50 | 3 | W1-Q4 ✅（7/8 追问通过） |
| 2026-07-07 | Day 6 补强 | RoPE 二维旋转图补强：新增 SVG 图解释二维旋转不是分别缩放，而是 x/y 交叉混合，并同步更新公开博客 | docs/blog/assets/rope-2d-rotation.svg · docs/blog/2026-07-07-rope-rotation-math.md · https://jokeuncle.github.io/blog/lumen-atelier-2026-07-07-rope-rotation-math/ · source commit 3a90493 · blog commit d5fb81f | 3 | W1-Q4 ✅（7/8 追问通过） |
| 2026-07-07 | Day 6 补强 | RoPE 数学地基补强：在博客中先补 cos/sin 的横向/纵向直觉，再从基础箭头推导二维旋转公式；同时更新发布 skill，要求数学类学习博客先补最小前置知识 | docs/blog/2026-07-07-rope-rotation-math.md · https://jokeuncle.github.io/blog/lumen-atelier-2026-07-07-rope-rotation-math/ · source commit e6683ea · blog commit 6466cd4 · local skill publish-to-lei-blog/SKILL.md | 3 | W1-Q4 ✅（7/8 追问通过） |
| 2026-07-07 | Day 6 补强 | masked softmax 专项讲解：从 softmax 的 exp/求和/归一化数学地基、causal mask 禁止未来 token、手算权重，到 GQA 下 score/weight/V 的 shape 和 ggml attention graph 对应，写成小白向博客 | docs/blog/2026-07-07-masked-softmax-attention.md · https://jokeuncle.github.io/blog/lumen-atelier-2026-07-07-masked-softmax-attention/ · source commit ca92743 · blog commit fa623c6 | 3 | W1-Q4 ✅（7/8 追问通过） |
| 2026-07-08 | Day 6 补强 | out 投影专项讲解：从多头 attention 结果的混音台直觉、维度不变但信息混合的小数字矩阵乘，到 Qwen2.5-7B 的 attn_output.weight 与最终 output.weight 区分，补齐 W1-Q4 最后一块 | docs/blog/2026-07-08-attention-out-projection.md · https://jokeuncle.github.io/blog/lumen-atelier-2026-07-08-attention-out-projection/ · source commit e8c9959 · blog commit bf1b82f | 3 | W1-Q4 ✅（7/8 追问通过） |
| 2026-07-08 | Day 7 | tokenizer_play.py 跑中文/英文/代码/emoji 4 组样本，记录字符数、UTF-8 字节、token 数、压缩比；发布 BPE 与 token 计费深度讲解 | week-01-llama-cpp/notes.md 第 3 节 · docs/blog/2026-07-08-bpe-tokenizer-billing.md · https://jokeuncle.github.io/blog/lumen-atelier-2026-07-08-bpe-tokenizer-billing/ · source commit 50ce2f0 · blog commit eab4849 | 3 | W1-Q2 ✅（经提示） |
| 2026-07-09 | Day 8 | KV cache 账专项讲解：拆清 layers × 2(K+V) × batch × KV_heads × seq × head_dim × bytes，手算 Qwen2.5-7B @2048 的 117.4MB、无 GQA 约 822MB、100 并发约 11.7GB | docs/blog/2026-07-09-kv-cache-memory.md · https://jokeuncle.github.io/blog/lumen-atelier-2026-07-09-kv-cache-memory/ · source commit 99d2561 · blog commit c32fd87 | 3 | W1-Q5 ✅（经提示） |
| 2026-07-09 | Day 8 | decode 速度上限公式专项讲解：用 bench 的 4.677GB 模型大小和 16.99 tok/s 反推有效权重读取约 79.5GB/s，并解释 150GB/s 带宽下理想上限约 32.1 tok/s、真实开销来自反量化/KV cache/kernel 调度/小 batch | docs/blog/2026-07-09-decode-bandwidth-limit.md · https://jokeuncle.github.io/blog/lumen-atelier-2026-07-09-decode-bandwidth-limit/ · source commit 03ca922 · blog commit 8f45a4e | 3 | W2-Q1 ✅（经提示） |
| 2026-07-09 | 网络排障补充 | DNS / fake-ip / TUN / HTTP 代理 / TLS 握手 / GitHub Pages 节点链路排障复盘：用本机 jokeuncle.github.io 无法访问案例拆清“DNS 污染”和代理节点链路问题的区别 | docs/blog/2026-07-09-dns-fake-ip-tun-github-pages-debug.md · https://jokeuncle.github.io/blog/lumen-atelier-2026-07-09-dns-fake-ip-tun-github-pages-debug/ · source commit ad9b827 · blog commit be14038 | 4 | - |
| 2026-07-10 | Day 9 | bench 聚焦扫描：对比 ngl=0/99 与 threads=6/12，观察 Metal offload 下 prefill 从约 31→511 tok/s 暴涨、decode 仅约 15→18 tok/s，小 batch decode 仍受权重/KV 读取带宽限制 | week-01-llama-cpp/reports/bench-day9-focused-20260710-142718.csv · week-01-llama-cpp/reports/bench-day9-threads-20260710-142800.csv | 3 | W2-Q2 ✅（经提示） |
| 2026-07-10 | Day 10 | llama-perplexity 跑 WikiText-2 validation 小基线：Qwen2.5-7B Q4_K_M 在 ctx=512/chunks=16 下 PPL=6.8790±0.30335；补写 PPL/交叉熵/量化质量评估小白向讲解并发布博客 | week-01-llama-cpp/reports/ppl-day10-summary.md · docs/blog/2026-07-10-perplexity-ppl-explained.md · https://jokeuncle.github.io/blog/lumen-atelier-2026-07-10-perplexity-ppl-explained/ · source commit 428c15b · blog commit 1c9790a | 2 | W2-Q3 ✅（经提示） |
| 2026-07-10 | Day 10 补强 | 交叉熵数学动机补强：从整段文本概率连乘、浮点下溢、log 把乘法变加法、负 log 变 loss、ln/exp 与 softmax/梯度优化配合，到 PPL 还原为平均有效岔路数 | docs/blog/2026-07-10-why-log-cross-entropy-ppl.md · https://jokeuncle.github.io/blog/lumen-atelier-2026-07-10-why-log-cross-entropy-ppl/ · source commit 0b71006 · blog commit 9bebe93 | 2 | W2-Q3 ✅（经提示） |
| 2026-07-10 | Day 10 补强 | “换空间”数学方法论补强：把 log、embedding、RoPE、量化、LoRA 统一到“原空间痛点→目标空间收益→代价→下游使用”的思维框架，并整理成十大思维检查表和日常练习法 | docs/blog/2026-07-10-change-of-space-math-muscle-memory.md · https://jokeuncle.github.io/blog/lumen-atelier-2026-07-10-change-of-space-math-muscle-memory/ · source commit fb92fb8 · blog commit 65210f4 | 3 | W2-Q3 ✅（经提示） |
| 2026-07-10 | Day 10 补强 | 数学恐惧与探索路径方法论：从 Euclid/Euler/Gauss/Riemann/Noether/Poincare/Hilbert/Grothendieck/von Neumann/Ramanujan/Tao 的思维动作中，提炼“问题→对象→空间→不变量→证明/计算→验证”的可训练路径 | docs/blog/2026-07-10-mathematical-mindset-path.md · https://jokeuncle.github.io/blog/lumen-atelier-2026-07-10-mathematical-mindset-path/ · source commit 63ac730 · blog commit f289c37 | 3 | W2-Q3 ✅（经提示） |
| 2026-07-13 | Day 11 | 采样实验：同一 prompt 对比 temp=0/0.7/1.5、top-k=10、top-p=0.3，确认 fixed seed 可复现、短强约束 prompt 主要在近义表达内摆动，并发布 logits/softmax/temperature/top-k/top-p 深度讲解 | docs/blog/2026-07-13-sampling-temperature-topk-topp.md · https://jokeuncle.github.io/blog/lumen-atelier-2026-07-13-sampling-temperature-topk-topp/ · source commit 41c2dda · blog commit 4f657bf | 3 | W2-Q4 ✅（经提示） |
| 2026-07-13 | Day 8 补强 | decode 访存开销补强：系统拆解为什么理论 32 tok/s 跑不到满值，讲清 KV cache 读写、activation 中间结果、Q4_K_M 反量化 metadata、kernel 调度和 cache miss 如何降低有效带宽 | docs/blog/2026-07-13-decode-memory-overheads.md · https://jokeuncle.github.io/blog/lumen-atelier-2026-07-13-decode-memory-overheads/ · source commit 90f091f · blog commit 02ea44f | 3 | W2-Q1 ✅（经提示） |
| 2026-07-13 | Day 13-14 | Week 01-02 旗舰总结：用 Scrapling 消化 BPE、Attention、RoPE、GQA、FlashAttention、AWQ、PagedAttention 等论文，以论文纵轴串联 Qwen2.5-7B 在 llama.cpp 上的 tokenizer、forward、KV cache、benchmark、PPL、sampling 实验，并补两张 SVG 主线图 | docs/blog/2026-07-13-attention-to-pagedattention-llama-cpp-summary.md · https://jokeuncle.github.io/blog/lumen-atelier-2026-07-13-attention-to-pagedattention-llama-cpp-summary/ · source commit 6d17e79 · blog commit 9e3e047 | 3 | Week 01-02 quiz 全绿 |

## 欠账区（跳过待补）

| 记入日期 | 欠什么 | 薄弱点 | 还账方式 |
|---|---|---|---|

## 验收通过记录

| 日期 | 题号 | 评语 |
|---|---|---|
| 2026-07-06 | W1-Q1 | 经提示通过：能说清 header/metadata/tensor info/tensor data 顺序、metadata 前置原因、alignment 作用和 tensor offset 相对 tensor data 区。 |
| 2026-07-06 | W1-Q3 | 追问通过：能分清 vocab 是文本片段到 token id 的字典，token_embd.weight 是 token id 到 3584 维向量的权重表，二者共享 vocab_size 编号空间。 |
| 2026-07-08 | W1-Q4 | 追问通过：能说清 GQA 下 Q=28 头、K/V=4 头、head_dim=128，RoPE 旋转 Q/K 的二维坐标对，masked softmax 将未来 token 权重归零，out 投影混合各 head 输出向量。 |
| 2026-07-10 | W2-Q2 | 经提示通过：能说出 prefill 是大矩阵并行计算、decode 有自回归依赖且小 batch 下每 token 需重读大量权重/KV，continuous batching 通过拼多个请求当前 decode 步提高算术强度和吞吐。 |
| 2026-07-13 | W2-Q4 | 经提示通过：能说清 T 越小分布越尖、T 越大分布越平；T->0 近似 argmax/one-hot，T->∞ 近似均匀分布；top-k 按前 k 名截断，top-p 按累计概率达到 p 的最小集合截断。 |
| 2026-07-13 | W2-Q3 | 经提示通过：能把 PPL 理解为平均有效岔路数；知道 PPL 越低代表模型给真实 token 的概率越高；能说出交叉熵在 log 加法空间平均负 log 概率，PPL = exp(cross_entropy) 是还原后的有效候选数。 |
| 2026-07-13 | W2-Q1 | 经提示通过：能说出 decode t/s 上限约为内存带宽除以每 token 总读取字节数；小 batch 下权重读取是大头；用 150GB/s ÷ 4.677GB 估算约 32 token/s，实测差值来自反量化、KV cache 读写、kernel 调度和带宽利用率。 |
| 2026-07-13 | W1-Q5 | 经提示通过：能写出 KV cache = layers × 2(K+V) × batch × KV_heads × seq × head_dim × bytes，算出 Qwen2.5-7B @2048 FP16 约 117.4MB；经提示把无 GQA 倍数从 4 修正为 28/4=7，得到约 821.8MB；100 并发约 11.7GB。 |
| 2026-07-13 | W1-Q2 | 经提示通过：能说清 BPE/tokenizer 基于训练语料里的高频片段合并，不是一个汉字固定一个 token；能引用中文实验 21 字符 -> 9 token、压缩比 2.33 字符/token；能说明 token 是 embedding、attention、KV cache 和 decode 的基本推理/计费单位。 |
