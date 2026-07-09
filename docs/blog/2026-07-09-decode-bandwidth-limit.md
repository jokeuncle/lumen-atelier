---
title: "Decode 速度上限公式：为什么 4.677GB 模型、16.99 tok/s 反推约 79.5GB/s"
date: 2026-07-09
source: "Lumen Atelier decode bandwidth limit deep dive"
---

# Decode 速度上限公式：为什么 4.677GB 模型、16.99 tok/s 反推约 79.5GB/s

上一篇把 KV cache 的内存账算清楚了：

```text
Qwen2.5-7B @ seq=2048, fp16 KV cache
≈ 117.4 MB / request
```

这篇继续 Day 8 的第二块：**decode 速度上限公式**。

这次 `llama-bench` 的 decode 行给了两个关键数字：

```text
model_size = 4,677,120,000 bytes ≈ 4.677 GB
decode speed = 16.989110 tok/s
```

于是可以做一笔反推：

```text
4.677 GB/token × 16.99 token/s
≈ 79.5 GB/s
```

这 79.5GB/s 不是机器的峰值带宽，也不是说机器“只有”这个带宽。它的意思更窄：

**如果 decode 每生成 1 个 token 至少要读一遍模型权重，那么 16.99 tok/s 这个实测速度，已经对应了每秒约 79.5GB 的有效权重读取量。**

真实推理还要付出更多东西：反量化、KV cache 读写、kernel 调度、小 batch 利用率低、采样和同步开销。所以这笔账是一个下限视角，不是完整 profiler。

## 1. 先分清 prefill 和 decode：为什么只给 decode 算这条公式

LLM 推理分两段：

```text
prefill: 一次性处理整段 prompt
decode: 每次只生成 1 个新 token
```

这次 benchmark 里同一个模型有两行：

```text
prefill: n_prompt=128, avg_ts≈539.93 tok/s
decode:  n_gen=32,    avg_ts≈16.99 tok/s
```

两个数字不能直接比较，因为它们的计算形态不一样。

prefill 一次喂进来很多 token。比如 128 个 token 同时过一层线性层时，权重矩阵读进来以后，可以被这 128 个 token 共享使用。矩阵乘更像：

```text
[128, hidden] × [hidden, out]
```

这更容易把计算单元喂饱。

decode 每步通常只有 1 个新 token。线性层更像：

```text
[1, hidden] × [hidden, out]
```

权重仍然很大，但这次只服务 1 个 token。权重读进来，用一下就过去了，复用少得多。

所以 decode 常见瓶颈不是“算力不够”，而是：

```text
每生成 1 个 token，都要把大量权重从内存搬到计算单元附近。
```

这就是 decode 速度上限公式的来源。

## 2. 最小数学地基：速度 = 带宽 / 每件东西多大

先不用模型，用搬箱子理解。

假设仓库传送带每秒最多搬：

```text
100 GB/s
```

每生成 1 个 token，至少要搬：

```text
5 GB 权重
```

那么理想情况下每秒最多生成：

```text
100 GB/s / 5 GB/token = 20 token/s
```

这就是公式：

```text
decode tok/s <= memory_bandwidth / bytes_per_token
```

而在最粗的心算里：

```text
bytes_per_token ≈ model_size
```

所以：

```text
decode tok/s <= memory_bandwidth / model_size
```

这不是精确模拟器，只是第一性原理的上限估算。

如果你只知道机器内存带宽和模型大小，就可以先心算：

```text
这台机器最多大概能 decode 多少 tok/s？
```

如果你已经有实测 tok/s，也可以反过来算：

```text
这个实测速度至少对应多少 GB/s 的权重读取？
```

## 3. 正向估算：150GB/s 带宽下，4.677GB 模型理论上限是多少

先按一个粗略的 M5 Pro 统一内存带宽数字：

```text
memory_bandwidth ≈ 150 GB/s
```

模型大小来自 `llama-bench`：

```text
model_size = 4,677,120,000 bytes
           ≈ 4.677 GB
```

注意这里用的是十进制 GB：

```text
1 GB = 1,000,000,000 bytes
```

如果按二进制 GiB：

```text
4,677,120,000 / 1024 / 1024 / 1024
≈ 4.356 GiB
```

心算时不要混用 GB 和 GiB。这里沿用 benchmark 讨论里常用的十进制 GB。

代入上限公式：

```text
decode tok/s <= 150 GB/s / 4.677 GB/token
              ≈ 32.1 token/s
```

这表示：如果每个 token 只需要完美地读一遍 4.677GB 权重，而且硬件带宽能被完全吃满，那么 decode 大约可以到 32 tok/s。

但这只是理想上限。真实跑出来是：

```text
16.99 tok/s
```

它低于 32 tok/s 很正常。因为真实系统不只搬权重，还要做很多别的事。

## 4. 反向估算：16.99 tok/s 对应约 79.5GB/s

现在换个方向。

已知实测：

```text
decode speed = 16.989110 token/s
model_size   = 4.677120 GB
```

如果每生成 1 个 token 至少读一遍模型权重，那么每秒至少读：

```text
4.677120 GB/token × 16.989110 token/s
= 79.460106 GB/s
≈ 79.5 GB/s
```

这就是那句：

```text
4.677 GB/token × 16.99 token/s ≈ 79.5 GB/s
```

它的含义不是：

```text
机器极限带宽 = 79.5GB/s
```

而是：

```text
只按“每 token 读一遍模型权重”这件事看，
16.99 tok/s 已经要求系统每秒交付约 79.5GB 的有效权重数据。
```

如果拿 150GB/s 当粗略峰值带宽，比例是：

```text
79.5 / 150 ≈ 53%
```

这可以理解成：这次 decode 把峰值带宽的一部分转化成了有效权重读取吞吐。但它不是完整带宽利用率，因为真实内存流量里还有很多不在 `model_size × tok/s` 这笔账里的东西。

## 5. 为什么说这是“至少读了多少权重”

Transformer decode 每生成一个 token，需要走完整个模型：

```text
embedding / 当前 hidden
-> 28 层 decoder block
   -> RMSNorm
   -> QKV 投影
   -> attention
   -> out 投影
   -> FFN
-> final norm
-> output projection
-> sample
```

其中大量时间花在线性层和 FFN 上。权重矩阵已经训练好，decode 当前 token 时要拿当前 hidden 去乘这些权重。

对 batch=1 的 decode 来说，权重复用很差。一个权重值被读进来，通常只服务当前这个 token 的一次计算；下一步生成新 token，又要再来一遍。

所以可以粗略说：

```text
每生成 1 个 token，大致要扫一遍模型权重。
```

这句话是近似，但它非常有用。它让我们能先不用 profiler，就抓住 decode 的主瓶颈：

```text
模型越大，每 token 要搬的数据越多；
带宽越高，每秒能搬的数据越多；
所以 decode tok/s 首先被 bandwidth / model_size 卡住。
```

这里的“至少”来自两个方向。

第一，`model_size` 只是在算模型权重这块主要 payload。真实运行还会读写激活、临时 buffer、KV cache、采样相关数据。

第二，硬件实际内存事务不一定等于理想 payload。缓存行、对齐、kernel 实现、数据布局、反量化元数据，都可能让真实搬运量和有效权重字节不同。

因此：

```text
model_size × tok/s
```

更像“有效权重吞吐”的下限估计，而不是总内存流量。

## 6. 真实开销还包括什么

### 6.1 反量化开销

这次模型是：

```text
Qwen2.5-7B-Instruct-Q4_K_M.gguf
model_type = qwen2 7B Q4_K - Medium
model_size ≈ 4.677 GB
```

Q4_K_M 权重不是 fp16 原样存放，而是量化后的格式。推理时不能直接拿 4bit 权重做普通矩阵乘，通常要在 kernel 里读取量化块、scale/min 等元数据，并把它们还原到可计算的数值路径里。

所以每个 token 不只是：

```text
读权重 -> 乘加
```

还包括：

```text
读量化块
读量化元数据
反量化
再参与乘加
```

量化减少了内存读取量，但增加了解码和还原的计算/指令开销。decode 是否更快，取决于省下来的带宽是否大于额外反量化成本。对大模型 decode 来说，通常值得，因为瓶颈主要在搬权重。

### 6.2 KV cache 读写

上一篇算过：

```text
Qwen2.5-7B @ seq=2048 的 fp16 KV cache
≈ 117.4 MB / request
```

decode 当前 token 时，attention 要读取历史 K/V。粗略看，seq=2048 时这块是百 MB 量级，比 4.677GB 权重小不少，但不是 0。

每生成一个新 token，还要把当前 token 的 K/V 追加进去：

```text
layers × 2(K+V) × KV_heads × head_dim × 2 bytes
= 28 × 2 × 4 × 128 × 2
= 57,344 bytes
≈ 56 KiB
```

写入新 K/V 本身不大，但读取历史 K/V 会随着上下文长度线性增长。

如果上下文从 2048 增到 32768，KV cache 读取压力大约也乘 16。那时候 decode 就不再只是“权重扫描”这一个主角，长上下文 attention 的 cache 读也会明显上桌。

### 6.3 小 batch 利用率低

decode batch=1 时，矩阵乘更像 GEMV：

```text
vector × matrix
```

它的特点是权重大、输出小、复用少。硬件很难把所有计算单元都喂饱。

prefill 更像 GEMM：

```text
matrix × matrix
```

同一份权重可以服务很多 token，算术强度更高，硬件更容易跑满。

这就是为什么同一个模型里：

```text
prefill: 539.93 tok/s
decode:   16.99 tok/s
```

看起来差距巨大。不是 decode 的代码“坏了”，而是它天生处在更难利用硬件的形态。

### 6.4 kernel 调度、同步和采样

一个 token 的生成不是一个单独的大 kernel 结束。它经过很多层、很多算子，还要做：

```text
kernel launch / command scheduling
中间 buffer 管理
CPU/GPU 或运行时同步
最后 logits 处理
temperature / top-k / top-p / sampler
token 输出
```

这些东西单独看可能不大，但 decode 每步只生成 1 个 token，小开销会被放大。

所以实际 16.99 tok/s 低于 32.1 tok/s 的理想带宽上限，是正常结果。

## 7. 为什么 prefill 的 539 tok/s 不能拿来反推同一条带宽账

如果把 prefill 也粗暴套：

```text
4.677 GB × 539.93 tok/s
```

会得到一个离谱的大数。这个数没有同样的解释意义，因为 prefill 不是“每个 token 独立扫一遍权重”。

prefill 处理 128 个 prompt token 时，权重读入后可以在这 128 个 token 上复用。站在每个 token 平摊的角度，权重读取成本被摊薄了。

所以 prefill 更应该问：

```text
一次处理 S 个 token 时，矩阵乘的算术强度是多少？
算力能不能吃满？
```

decode 更应该问：

```text
每生成 1 个 token 要搬多少权重？
内存带宽能不能跟上？
```

这就是 Week 2 里那道题的核心：

```text
prefill 更偏 compute-bound
decode 更偏 memory-bandwidth-bound
```

不要背这个结论，要从“权重能不能被很多 token 复用”推出它。

## 8. 这条公式能怎么用

### 8.1 看机器适不适合跑某个模型

假设模型是 20GB，机器可用带宽粗略是 200GB/s：

```text
200 / 20 = 10 tok/s
```

这说明单请求 decode 理想上限大概就是 10 tok/s 级别。真实可能更低。

如果你想要 50 tok/s，就不能只靠这台机器直接跑这个 20GB 模型。要么换更高带宽硬件，要么减少每 token 搬运量，比如更强量化、裁小模型、投机解码、batch 调度等。

### 8.2 解释为什么量化能提速

同一台机器，带宽固定。

如果模型从 9GB 量化到 4.5GB，理想 decode 上限会接近翻倍：

```text
bandwidth / 9GB
-> bandwidth / 4.5GB
```

当然真实不会完美翻倍，因为量化有反量化开销，kernel 实现也不一样。但方向是清楚的：

```text
decode memory-bound 时，减少权重字节数通常能提高 tok/s。
```

### 8.3 解释为什么 continuous batching 有意义

单请求 decode 复用差。多个请求一起 decode 时，batch 变大，同一批权重可以服务多个序列的当前 token。

这会把：

```text
[1, hidden] × [hidden, out]
```

变成更接近：

```text
[B, hidden] × [hidden, out]
```

权重读取被 B 个 token 共享，算术强度上升，硬件利用率更好。

这就是 continuous batching 的核心动机之一：不是让单条请求的数学变了，而是把很多请求的 decode 步拼在一起，让昂贵的权重读取更值得。

## 9. 用这次 benchmark 重新说一遍

真实数据：

```text
model_size = 4,677,120,000 bytes = 4.677 GB
decode = 16.989110 tok/s
```

正向理论上限：

```text
150 GB/s / 4.677 GB/token
≈ 32.1 tok/s
```

反向有效权重吞吐：

```text
4.677 GB/token × 16.989 tok/s
≈ 79.5 GB/s
```

解释：

```text
79.5GB/s 不是机器峰值带宽；
它是按“每 token 至少读一遍模型权重”反推出来的有效权重读取量。
```

实际 tok/s 低于理想值，是因为真实 decode 还包括：

```text
反量化
KV cache 读写
小 batch 利用率低
kernel 调度和同步
采样与输出处理
```

这条公式不是为了替代 profiler，而是为了让你看到数量级。看到一个 decode 速度时，你马上能问：

```text
这个速度对应的有效权重带宽是多少？
离机器带宽上限还有多远？
瓶颈更像带宽、算力，还是调度？
```

这就是 AI Infra 里最有用的一类心算。

## 自测

不看上文，试着回答：

```text
1. 为什么 decode tok/s 可以先用 bandwidth / model_size 估上限？
2. 4.677GB 模型跑 16.99 tok/s，为什么反推是 79.5GB/s？
3. 为什么这个 79.5GB/s 不是机器峰值带宽？
4. 为什么 prefill 的 tok/s 不能直接套同一条“每 token 扫一遍权重”的账？
```

能把这四问讲顺，W2-Q1 的核心就过了。
