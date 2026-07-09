---
title: "KV cache 那笔账：为什么 Qwen2.5-7B 在 2048 长度下大约是 117MB"
date: 2026-07-09
source: "Lumen Atelier KV cache memory deep dive"
---

# KV cache 那笔账：为什么 Qwen2.5-7B 在 2048 长度下大约是 117MB

今天只解决一个问题：

```text
KV cache = layers × 2(K+V) × batch × KV_heads × seq × head_dim × bytes
```

这行公式到底是什么意思？为什么截图里会算成：

```text
28 × 2 × 1 × 4 × 2048 × 128 × 2
= 117,440,512 bytes
≈ 117.4 MB
```

如果能把这笔账讲清楚，后面看 vLLM PagedAttention、continuous batching、长上下文显存瓶颈，都会顺很多。因为 KV cache 不是一个抽象概念，它就是推理服务里会真实膨胀的一块内存。

## 1. 人话直觉：KV cache 是模型给历史 token 做的索引卡片

在 attention 里，Q/K/V 可以先这样理解：

```text
Q: 当前 token 想找什么
K: 历史 token 能被什么问题匹配
V: 历史 token 被选中后贡献什么内容
```

模型生成文本时分两段：

```text
prefill: 一次性读完整个 prompt
decode: 之后每次只生成 1 个新 token
```

假设前面已经有 2048 个 token 的历史。继续生成时，当前 token 的 Query 要去匹配这些历史位置的 Key，然后用注意力权重去加权这些历史位置的 Value。

如果没有 KV cache，每生成一个新 token，都要把前面所有历史 token 重新跑过模型前缀，重新得到每一层的历史 K/V。这里先只看 K/V 这笔账：

```text
第 2049 步：重算 2048 个历史 token 的 K/V
第 2050 步：重算 2049 个历史 token 的 K/V
第 2051 步：重算 2050 个历史 token 的 K/V
...
```

这会把 decode 阶段拖死。

KV cache 的做法很直接：

```text
历史 token 的 K/V 算过一次，就存起来。
下一步 decode 只给新 token 算新的 K/V，再追加到 cache 末尾。
```

所以 KV cache 可以理解成：每一层 attention 都给历史 token 存了一套已经算好的 Key 和 Value。当前 token 来的时候，直接拿它们用，不重算历史。

## 2. 最小数学地基：一个张量形状就是一组格子

先不急着看 28 层、2048 长度。用小数字。

假设一个玩具模型：

```text
layers = 2
batch = 1
KV_heads = 2
seq = 3
head_dim = 4
每个数用 fp16 = 2 bytes
```

只看 K cache，它的逻辑形状可以写成：

```text
[layers, batch, KV_heads, seq, head_dim]
= [2, 1, 2, 3, 4]
```

这不是公式玄学，而是在数格子：

```text
2 层
每层 1 个请求
每个请求 2 个 KV head
每个 head 存 3 个历史位置
每个位置是 4 个数
```

所以 K cache 里有：

```text
2 × 1 × 2 × 3 × 4 = 48 个数
```

每个数是 fp16，占 2 bytes：

```text
K cache = 48 × 2 = 96 bytes
```

但 attention 不只需要 K，还需要 V。K 和 V 形状一样，所以总共乘 2：

```text
KV cache = 2(K+V) × 96 = 192 bytes
```

这就是大公式的核心。所有复杂版本，本质都只是在数这块多维表里有多少个数。

## 3. 逐项拆公式：每一项都在数什么

完整公式：

```text
KV cache bytes
= layers × 2 × batch × KV_heads × seq × head_dim × bytes_per_value
```

逐项解释：

```text
layers
```

每一层 Transformer 都有自己的 K/V。第 0 层的 hidden state、K/V 投影权重和第 1 层不同，所以不能共用。Qwen2.5-7B 有 28 层，因此这里是 28。

```text
2
```

这个 2 不是 batch，也不是 fp16。它表示 K 和 V 两份缓存。K 一份，V 一份。

```text
batch
```

一次同时处理多少条独立序列。单请求就是 1。并发 100 个请求时，粗略上就是乘 100，因为每个请求的历史上下文不同，KV cache 不能互相复用。

```text
KV_heads
```

存的是 Key/Value 的 head 数，不是 Query 的 head 数。Qwen2.5-7B 用 GQA，Query 有 28 个头，但 K/V 只有 4 个头，所以这里填 4，不填 28。

```text
seq
```

缓存里已经存了多少个位置。上下文越长，KV cache 线性增长。2048 token 就是 2048 个历史位置。

```text
head_dim
```

每个 head 的向量维度。Qwen2.5-7B 的 hidden size 是 3584，Query heads 是 28：

```text
3584 / 28 = 128
```

所以每个 head 是 128 维。

```text
bytes_per_value
```

每个缓存数值占多少字节。fp16/bf16 都是 2 bytes，fp32 是 4 bytes，int8 是 1 byte。这里按 llama-bench 里的 `type_k=f16`、`type_v=f16` 算，所以是 2。

## 4. 回到截图：Qwen2.5-7B 的 117MB 怎么来

Qwen2.5-7B 这组数字是：

```text
layers = 28
batch = 1
KV_heads = 4
seq = 2048
head_dim = 128
type_k/type_v = fp16 = 2 bytes
```

代入公式：

```text
KV cache
= layers × 2(K+V) × batch × KV_heads × seq × head_dim × 2 bytes
= 28 × 2 × 1 × 4 × 2048 × 128 × 2
= 117,440,512 bytes
```

如果按十进制 MB：

```text
117,440,512 / 1,000,000
= 117.4 MB
```

如果按二进制 MiB：

```text
117,440,512 / 1024 / 1024
= 112.0 MiB
```

所以截图里的 `117.4 MB` 没错，只是它用的是十进制 MB。很多系统工具显示的是 MiB/GiB，数字会略小一点。

## 5. 为什么公式里没有 Q

这是最容易问错的一点：

```text
既然 attention 叫 Q/K/V，为什么 cache 只存 K/V，不存 Q？
```

原因是 decode 时只需要当前 token 的 Query。

生成下一个 token 时，当前 token 会拿自己的 Q 去查所有历史 K：

```text
当前 Q × 历史 K -> attention score
attention score -> softmax -> weight
weight × 历史 V -> 当前 token 的 attention 输出
```

过去 token 的 Q 已经完成了它们自己的那一步输出。后面生成新 token 时，不需要再拿过去 token 的 Q 去查别人。真正会被未来反复访问的是过去 token 的 K 和 V。

所以每一步 decode 做的是：

```text
1. 给当前 token 算 Q/K/V
2. 把当前 token 的 K/V 追加进 KV cache
3. 用当前 Q 读取所有历史 K/V
4. 产出当前 token 的 attention 结果
```

Q 是临时工作区，K/V 才是要长期保存的历史资料。

## 6. GQA 为什么能把 KV cache 从 820MB 降到 117MB

如果没有 GQA，K/V head 数通常会和 Query head 数一样。对 Qwen2.5-7B 来说，就是把 `KV_heads=4` 换成 `KV_heads=28`。

重新算：

```text
28 × 2 × 1 × 28 × 2048 × 128 × 2
= 822,083,584 bytes
≈ 822.1 MB
```

而使用 GQA 时：

```text
28 × 2 × 1 × 4 × 2048 × 128 × 2
= 117,440,512 bytes
≈ 117.4 MB
```

两者比例正好是：

```text
28 / 4 = 7
```

也就是说，GQA 在这笔 KV cache 账上省了 7 倍。

直觉上可以这样理解：

```text
28 个 Query head 仍然保留，负责从不同角度提问；
但 Key/Value 只准备 4 组，被多组 Query head 共享。
```

这就是为什么前面 QKV 投影里会看到：

```text
Q: [S, 28, 128]
K: [S,  4, 128]
V: [S,  4, 128]
```

Q 头多，表达能力还在；K/V 头少，cache 和带宽压力降下来。

## 7. 100 并发为什么会变成 11.7GB

单请求、2048 长度、fp16 KV cache 是：

```text
117.4 MB
```

如果同时有 100 个请求，每个请求都有自己的上下文：

```text
117.4 MB × 100
= 11,740 MB
≈ 11.7 GB
```

这还只是 2048 长度。如果上下文变成 8192，seq 直接乘 4：

```text
117.4 MB × 4
≈ 469.8 MB / request
```

100 并发就是：

```text
469.8 MB × 100
≈ 47.0 GB
```

注意，模型权重大小基本是固定的。比如你手上的 Q4_K_M GGUF 大约 4.7GB，加载后权重不会因为上下文变长而线性增长。但 KV cache 会随着：

```text
上下文长度 seq
并发 batch/request 数
层数 layers
KV head 数
KV 精度
```

一起增长。

这就是长上下文服务里 KV cache 会变成一等公民的原因。

## 8. 这笔账和 PagedAttention 有什么关系

上面公式算的是理想 payload：真正有用的 K/V 数值占多少内存。实际推理框架还会遇到几个工程问题：

```text
不同请求长度不同，cache 容易碎片化
请求会来来走走，连续大块内存不好复用
有些请求提前结束，空出来的 cache 要回收
并发调度时，新 token 要不断追加 cache
```

如果每个请求都按最大上下文长度预留连续 KV cache，浪费会很大。vLLM 的 PagedAttention 就是为了解这个问题：把 KV cache 切成块，像操作系统管理内存页一样管理它。

所以这篇的 117MB 不是终点，而是下一步的起点。

你先知道单条请求的 KV cache 为什么是：

```text
layers × 2 × batch × KV_heads × seq × head_dim × bytes
```

才能继续追问：

```text
当请求很多、长度不同、不断进出时，这些 cache 应该怎么分配、复用和回收？
```

这就是 PagedAttention 要回答的问题。

## 9. 最后一遍心算模板

以后看到任何模型，都按这几个问题填空：

```text
1. 多少层？layers
2. K/V 几个头？KV_heads
3. 每个 head 多宽？head_dim
4. 当前上下文多长？seq
5. 几个请求一起算？batch 或并发数
6. KV cache 用什么精度？bytes_per_value
```

然后套：

```text
KV cache bytes
= layers × 2 × batch × KV_heads × seq × head_dim × bytes_per_value
```

Qwen2.5-7B @ seq=2048：

```text
28 × 2 × 1 × 4 × 2048 × 128 × 2
= 117,440,512 bytes
≈ 117.4 MB
```

没有 GQA：

```text
28 × 2 × 1 × 28 × 2048 × 128 × 2
≈ 822.1 MB
```

100 并发：

```text
117.4 MB × 100
≈ 11.7 GB
```

这三个数字就是 W1-Q5 的核心答案。

## 自测

不看上文，试着回答三件事：

```text
1. KV cache 为什么乘 2？
2. Qwen2.5-7B 为什么用 KV_heads=4，而不是 heads=28？
3. 为什么 Q 不进 cache？
```

如果这三问能答顺，KV cache 这笔账就不是背公式了。
