---
title: "Decode 为什么跑不到理论带宽：KV cache、activation、反量化和调度开销"
date: 2026-07-13
source: "Lumen Atelier decode memory overhead learning note"
---

# Decode 为什么跑不到理论带宽：KV cache、activation、反量化和调度开销

前面我们用一条心算公式估过 decode 速度上限：

```text
decode tok/s <= 内存带宽 / 每 token 总读取字节数
```

在最粗的第一阶估算里：

```text
每 token 总读取字节数 ≈ 模型权重大小
```

所以 Qwen2.5-7B Q4_K_M 这类 4.677GB 模型，在 150GB/s 内存带宽下，理想上限大约是：

```text
150GB/s / 4.677GB ≈ 32 token/s
```

这个公式很有用，因为它能先把问题钉住：

```text
decode 慢，首先不是因为模型不会算，而是因为每生成一个 token 都要搬大量数据。
```

但真实速度通常跑不到这个理论数。你问到的四个东西：

```text
KV cache 读写
activation 中间结果
反量化 metadata
kernel 调度和 cache miss
```

就是这条心算公式背后的“下半场”。

这篇把它们逐个拆开。目标不是让你会做 profiler，而是让你看到一句“decode memory-bound”时，脑子里能自动展开成一条真实的数据流。

## 1. 先建立一张最小地图：一个 token 生成时发生什么

decode 阶段每次只生成一个新 token。这个新 token 不是直接从模型里跳出来的，它要完整走一遍 Transformer：

```text
当前 token
-> embedding / hidden state
-> 第 1 层：norm -> attention -> FFN
-> 第 2 层：norm -> attention -> FFN
...
-> 第 28 层
-> final norm
-> output projection
-> logits
-> sample 下一个 token
```

每一层都会用到模型权重。权重就是模型训练后学出来的参数表，例如：

```text
q_proj.weight
k_proj.weight
v_proj.weight
o_proj.weight
ffn_gate.weight
ffn_up.weight
ffn_down.weight
norm.weight
```

这些权重存在模型文件里。你的 GGUF 模型大小约 4.677GB，粗略说就是模型本体的大小。

所以第一层心算是：

```text
生成 1 个 token，要把模型里大部分权重读出来参与计算。
```

但真实 decode 不只读权重。更完整的心智模型应该是：

```text
每 token 总开销
≈ 权重读取
+ KV cache 历史读写
+ activation 中间结果读写
+ 量化权重解码和 metadata 处理
+ kernel 调度、同步、cache miss 造成的有效带宽损失
```

注意最后一项不是一个简单的“多读多少 GB”。它更像是：硬件明明标称 150GB/s，但你的这条计算路径不可能把每一秒都用来完美搬权重。

下面逐项拆。

## 2. KV cache 读写：历史 token 的 K/V 不是白放着

先复习 attention 里的 Q/K/V：

```text
Q: 当前 token 想找什么
K: 历史 token 能被什么问题匹配
V: 历史 token 被选中后贡献什么内容
```

decode 时，新 token 来了。它在每一层都会生成自己的 Query，然后拿这个 Query 去匹配历史上所有 token 的 Key，再根据匹配结果加权读取 Value。

如果没有 KV cache，每生成一步都要重算全部历史 token 的 K/V：

```text
第 2049 个 token：重算前 2048 个 token 的 K/V
第 2050 个 token：重算前 2049 个 token 的 K/V
...
```

这会很浪费。所以推理程序把历史 token 的 K/V 存起来。下一步只需要：

```text
1. 给当前新 token 计算新的 K/V
2. 把新的 K/V 追加进 cache
3. 读取历史 K/V，做当前 token 的 attention
```

这就是 KV cache 的作用。

但缓存不是免费午餐。它省掉了“重算历史”，同时引入了“读取历史缓存”。

对 Qwen2.5-7B，在 seq=2048、fp16 KV cache、batch=1 时，前面算过这笔账：

```text
KV cache
= layers × 2(K+V) × batch × KV_heads × seq × head_dim × bytes
= 28 × 2 × 1 × 4 × 2048 × 128 × 2
= 117,440,512 bytes
≈ 117.4 MB
```

这 117.4MB 是一个请求在 2048 长度下、所有层合起来的 K/V 缓存规模。

decode 新 token 时，每层 attention 至少要和历史 K/V 打交道：

```text
读历史 K：当前 Q 和历史 K 做相似度
读历史 V：根据 attention 权重加权求和
写新 K/V：把当前 token 的 K/V 追加到 cache
```

其中“写新 K/V”每步只追加一个位置，量不大：

```text
28 × 2 × 1 × 4 × 1 × 128 × 2
= 57,344 bytes
≈ 0.057 MB
```

但“读历史 K/V”会随 seq 变长。上下文越长，当前 token 要看的历史越长，KV cache 读流量越大。

所以 KV cache 对性能的影响有两个方向：

```text
好处：不用重算历史 token，decode 才能跑起来。
代价：历史越长，每步 attention 要读的缓存越多。
```

这也是为什么长上下文推理不只是“内存能不能装下”，还要看每步能不能快速读完这些历史 K/V。

## 3. activation：中间结果不是模型参数，但也要在硬件里流动

权重是模型固定的参数。activation 是当前这次 forward 临时产生的中间结果。

用小白版说：

```text
权重 = 菜谱
输入 token = 食材
activation = 做菜过程中切好的菜、半成品、锅里的汤汁
```

它不是永久保存的模型知识，但每一步计算都要产生它、读取它、传给下一步。

举一层 Transformer 里的简化流程：

```text
hidden
-> norm 后的 hidden
-> q 向量
-> k 向量
-> v 向量
-> attention score
-> attention output
-> FFN gate/up 中间向量
-> FFN output
-> residual 相加后的 hidden
```

这些都可以叫 activation。

decode 一次只有 1 个新 token，所以 activation 规模通常比模型权重小很多。比如 hidden size 是 3584，一个 fp16 hidden 向量只有：

```text
3584 × 2 bytes ≈ 7 KB
```

看起来不大。但问题是：

```text
activation 会在很多层、很多算子之间反复产生、读取、写回。
```

有些 activation 可以一直放在寄存器或高速缓存里，很快；有些中间结果会写到更慢的内存里；有些算子之间还要重新读回来。

所以 activation 的影响不是“单个 hidden 向量很大”，而是：

```text
模型的一次 forward 不是一个完美连续的大矩阵乘。
它由很多小步骤组成，每一步都有临时数据流动。
```

这会降低“理论带宽 / 模型大小”这个公式的完美程度。理论公式假设硬件只做一件事：

```text
连续、完美地读权重。
```

真实情况是：

```text
读权重的同时，还要搬当前 token 的中间结果。
```

在小 batch decode 里，这些小张量和小算子尤其容易让硬件吃不满。

## 4. 反量化 metadata：Q4 不是直接拿 4bit 数字就能乘

你的模型是 Q4_K_M。听起来像是：

```text
每个权重只用 4 bit 存
```

但真实计算时，硬件不能直接拿一个 4bit 整数当原始浮点权重用。

量化的核心思想是：把一组浮点权重压缩成低 bit 表示，同时保存还原所需的缩放信息。

用一个玩具例子：

```text
原始权重：[-0.8, -0.2, 0.1, 0.7]
```

可以近似存成：

```text
低 bit 编码：[0, 5, 8, 15]
scale: 0.1
zero / min 信息：用于把整数映射回近似浮点值
```

计算时不能只读 `[0, 5, 8, 15]`，还要读对应的 `scale`、`min` 或其他块级 metadata，然后做近似还原：

```text
真实参与计算的近似权重 ≈ scale × (低 bit 编码 + 偏移)
```

Q4_K_M 这类 K-quant 格式会按块组织权重。一个块里不只是权重编码，还有 scale、min、超级块信息等 metadata。不同张量也可能不是同一种量化类型，有些是 Q4_K，有些是 Q6_K，有些保留 F32。

这带来两个后果。

第一，模型文件大小已经包含了量化权重和 metadata，所以 `4.677GB` 不是纯 4bit 裸数据。

第二，推理时还要付出解码成本：

```text
读低 bit 权重
读 scale/min 等 metadata
把压缩编码还原成可乘的数
再参与矩阵乘
```

这就是反量化开销。

它不一定让内存流量翻倍，但会让硬件做更多工作，也会让访存模式更复杂。对 decode 这种小 batch 场景，权重读出来只服务很少 token，反量化开销就更不容易被摊薄。

所以量化有一个现实权衡：

```text
好处：模型变小，权重读取字节数下降，memory-bound 场景通常会变快。
代价：每次使用权重前，要解码压缩格式；质量也可能略降。
```

为什么 Q4_K_M 仍然值得用？因为对本地推理来说，少搬几 GB 权重通常比多做一点解码更划算。

## 5. kernel 调度：一次 forward 不是一个神奇大函数

很多初学者会误以为：

```text
模型 forward = CPU/GPU 执行一个巨大函数
```

更真实的情况是：

```text
forward = 很多 kernel / 算子按顺序或按图执行
```

kernel 可以理解成硬件上的一个小任务，比如：

```text
做一次矩阵乘
做一次 RMSNorm
做一次 RoPE
做一次 attention
做一次 FFN 激活
做一次 residual add
```

每个 kernel 启动、排队、拿输入、写输出，都有成本。

prefill 的时候，一个 kernel 往往处理很多 token：

```text
[128, hidden] × weight
```

工作量大，硬件更容易被喂饱。启动一次 kernel 的成本可以摊到很多 token 上。

decode 的时候，一个 kernel 常常只服务很小的 batch：

```text
[1, hidden] × weight
```

同样要调度，但干的活少。于是调度、同步、等待数据这些固定成本就更显眼。

这也是为什么 continuous batching 重要。它想做的事就是把很多请求当前 decode 步拼在一起：

```text
1 个请求 × 1 token
变成
N 个请求 × 1 token
```

这样同一个 kernel 启动后能处理更多 token，权重读出来能服务更多请求，算术强度变高。

## 6. cache miss：不是所有数据都能待在最快的地方

现代硬件不是只有一种内存。它大概有多层：

```text
寄存器：最快，但很小
L1/L2 cache：很快，但也有限
统一内存 / 显存：大，但慢很多
磁盘：更大，但推理时不能靠它
```

cache 的作用是：把刚用过、马上可能还会用的数据放在更快的位置。

cache hit：

```text
要用的数据已经在高速缓存里，很快拿到。
```

cache miss：

```text
要用的数据不在高速缓存里，只能去更慢的内存拿。
```

decode 小 batch 对 cache 不友好，原因很直接：

```text
模型权重太大，不可能整模型都待在高速缓存里。
当前 token 用完某块权重后，下一层又要去读另一大块权重。
```

有些数据会有局部复用，比如当前层内部的小中间结果；但整套权重和长 KV cache 的体量远大于高速缓存。

所以理论带宽公式里的理想世界是：

```text
连续读取、完美预取、没有等待、没有浪费。
```

真实世界会有：

```text
数据不连续
cache line 没完全用满
等待内存返回
算子之间读写中间结果
不同 kernel 之间同步
```

这些都会让“标称 150GB/s”变成某条实际推理路径上的较低有效带宽。

## 7. 把四类开销放回 32 tok/s vs 实测速度

现在再看这条公式：

```text
150GB/s / 4.677GB ≈ 32 token/s
```

它隐含了几个理想假设：

```text
每 token 只读模型权重
没有 KV cache 历史读取
没有 activation 额外搬运
反量化没有成本
kernel 调度没有等待
cache 永远命中或完美流式读取
内存带宽能 100% 吃满
```

真实 decode 更接近：

```text
实际 tok/s
≈ 有效带宽 / 实际每 token 总开销
```

其中：

```text
实际每 token 总开销
= 权重读取
+ KV cache 读写
+ activation 流量
+ 量化 metadata 和反量化处理
+ 各种不能完全摊薄的调度与等待
```

如果你某次实测约 25 tok/s，那么只按权重读取反推：

```text
4.677GB/token × 25 token/s
≈ 116.9GB/s
```

这表示：光是“有效读取权重”这件事，已经相当于每秒交付约 117GB 数据。

如果另一组实验是 16.99 tok/s：

```text
4.677GB/token × 16.99 token/s
≈ 79.5GB/s
```

这两个数字不矛盾。不同参数、线程、Metal offload、batch、上下文长度、后台负载，都会改变实际有效吞吐。

关键不是死记某一次速度，而是掌握解释框架：

```text
理论上限 = 峰值带宽 / 模型大小
反推有效权重带宽 = 模型大小 × 实测 tok/s
真实差距 = KV cache、activation、反量化、kernel/cache 等开销吃掉了完美假设
```

## 8. 一张表记住四个词

| 名字 | 人话解释 | 在 decode 里怎么出现 | 为什么拖慢 |
|---|---|---|---|
| KV cache 读写 | 历史 token 的 K/V 索引卡片 | 当前 Q 要读历史 K/V，并追加新 K/V | 上下文越长，每步读的历史越多 |
| activation | 当前 forward 的临时中间结果 | norm、QKV、attention、FFN 都产生中间张量 | 小张量在算子间流动，增加读写和同步 |
| 反量化 metadata | 压缩权重的 scale/min 等还原信息 | Q4 权重要先解码成可计算的近似值 | 需要额外 metadata 和解码指令 |
| kernel / cache 开销 | 硬件任务调度和缓存不命中 | 很多小 kernel 串起来执行 | 小 batch 难摊薄固定成本，权重太大难缓存 |

## 9. 最小结论

如果只记一句话：

```text
decode 的第一瓶颈是读权重，但真实每 token 不只读权重。
```

更完整一点：

```text
权重决定理论上限；
KV cache 随上下文增长；
activation 是 forward 过程里的临时数据流；
量化降低权重大小，但引入解码成本；
kernel 调度和 cache miss 决定你能不能接近峰值带宽。
```

所以看到：

```text
理论 32 tok/s，实测 25 tok/s
```

不要只说“机器没跑满”。应该能说：

```text
32 tok/s 是只按权重读取和峰值带宽估出来的天花板。
实测低一些，是因为 decode 还要读写 KV cache、搬 activation、处理量化 metadata，
并且小 batch 下 kernel 调度和 cache miss 会降低有效带宽。
```

这就是从“背公式”到“能解释系统行为”的分界线。

## 10. 自测

1. 为什么 decode 的第一阶公式可以近似写成 `带宽 / 模型大小`？
2. KV cache 为什么既加速 decode，又会在长上下文下变成新的内存压力？
3. activation 和权重的区别是什么？
4. Q4_K_M 为什么不能理解成“直接拿 4bit 权重做矩阵乘”？
5. 为什么小 batch decode 比 prefill 更难吃满硬件？

这五题答得出来，后面看 vLLM 的 PagedAttention、continuous batching、chunked prefill，就不会只是在记名词。
