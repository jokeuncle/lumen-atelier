---
title: "Masked Softmax 到底在 mask 什么：把 QK 分数变成注意力权重"
date: 2026-07-07
source: "Lumen Atelier masked softmax deep dive"
---

# Masked Softmax 到底在 mask 什么：把 QK 分数变成注意力权重

前两篇已经把 attention 里的两步拆开了：

```text
1. QKV 投影
   hidden -> Q/K/V

2. RoPE
   Q/K 按 token 位置旋转
   V 不旋转
```

这篇继续往后走一步，只讲：

```text
3. masked softmax
```

目标很窄：读完以后，你要能用自己的话讲清楚这条链：

```text
Q/K 点积得到分数
-> 加 mask，禁止看到未来 token
-> softmax，把分数变成权重
-> 权重乘 V，汇总内容
```

如果你能讲清楚这一步，`W1-Q4 attention 内部四步` 就只剩最后的 out 投影。

## 1. 人话直觉：masked softmax 是注意力里的选择器

QKV 里我们说过：

```text
Q = 我想找什么
K = 我能被什么找到
V = 如果我被选中，我贡献什么内容
```

RoPE 之后，Q/K 已经带上了位置信息。

现在来到 attention 真正让 token 互相看的地方。

对某个当前 token 来说，它会拿自己的 Q 去和一堆历史 token 的 K 做匹配：

```text
当前 token 的 Q
  去匹配
所有可见 token 的 K
```

匹配结果是一串分数：

```text
score = [2.0, 1.0, -1.0, 3.0]
```

这些分数的意思是：

```text
第 0 个 token：比较相关
第 1 个 token：有点相关
第 2 个 token：不太相关
第 3 个 token：最相关
```

但注意：**分数还不是概率，也不是权重。**

它只是模型粗略打出来的“相关程度”。下一步 softmax 会把它变成权重：

```text
weight = [0.24, 0.09, 0.01, 0.66]
```

权重有两个特点：

```text
每个数都 >= 0
所有数加起来 = 1
```

所以你可以把 masked softmax 理解成：

```text
先把不能看的位置删掉，
再把剩下的匹配分数变成一组分配比例。
```

## 2. 数学地基：softmax 是什么

先不讲 mask，只讲 softmax。

softmax 的输入是一串任意分数：

```text
[2, 1, -1]
```

这些分数可以是正数、负数、很大、很小。它们还不是概率，因为：

```text
2 + 1 + (-1) = 2
```

不是 1，而且还有负数。

softmax 做三步：

```text
第一步：每个分数做 exp
第二步：把 exp 后的数加起来
第三步：每个数除以总和
```

这里的 `exp(x)` 可以先粗略理解成：

```text
把分数变成一个正数，而且高分会被放大得更多。
```

几个常用值：

```text
exp(2)  ≈ 7.389
exp(1)  ≈ 2.718
exp(0)  = 1
exp(-1) ≈ 0.368
```

现在对 `[2, 1, -1]` 做 softmax。

第一步，做 exp：

```text
2  -> exp(2)  ≈ 7.389
1  -> exp(1)  ≈ 2.718
-1 -> exp(-1) ≈ 0.368
```

第二步，加起来：

```text
sum = 7.389 + 2.718 + 0.368 = 10.475
```

第三步，每个数除以总和：

```text
7.389 / 10.475 ≈ 0.705
2.718 / 10.475 ≈ 0.260
0.368 / 10.475 ≈ 0.035
```

所以：

```text
softmax([2, 1, -1]) ≈ [0.705, 0.260, 0.035]
```

这三个数加起来约等于 1：

```text
0.705 + 0.260 + 0.035 = 1.000
```

这就是 softmax 的核心：

```text
把一串任意分数，变成一串可分配的权重。
```

## 3. 为什么 attention 需要 softmax

Q/K 点积得到的是匹配分数：

```text
score_i = Q · K_i
```

点积越大，说明当前 Query 和第 `i` 个 Key 越匹配。

但真正汇总 V 的时候，模型需要的是“比例”：

```text
我应该从第 0 个 token 拿多少内容？
从第 1 个 token 拿多少内容？
从第 2 个 token 拿多少内容？
```

所以要把 score 变成 weight：

```text
score  -> softmax -> weight
```

然后再做：

```text
output = weight_0 * V_0 + weight_1 * V_1 + weight_2 * V_2 + ...
```

举个一维小例子。

假设三个 token 的 V 是：

```text
V = [10, 20, 100]
```

刚才算出的权重是：

```text
weight = [0.705, 0.260, 0.035]
```

那汇总结果就是：

```text
output = 0.705*10 + 0.260*20 + 0.035*100
       = 7.05 + 5.20 + 3.50
       = 15.75
```

真实模型里，V 不是一个数字，而是一个 128 维向量。计算方式一样，只是每个维度都做一遍加权求和。

## 4. mask 是什么：不能偷看未来

现在加上 mask。

语言模型是预测下一个 token 的模型。训练或推理时，它不能在生成当前位置的时候偷看未来 token。

比如句子是：

```text
我 / 爱 / 北京 / 天安门
```

当模型处理第 1 个 token “爱” 时，它可以看：

```text
我
爱
```

但不能看：

```text
北京
天安门
```

否则就相当于考试时提前看答案。

所以 causal mask 的规则是：

```text
当前位置只能看自己和自己之前的 token。
不能看自己之后的 token。
```

如果有 4 个 token，mask 矩阵可以这样理解：

```text
query 0 可以看：0
query 1 可以看：0, 1
query 2 可以看：0, 1, 2
query 3 可以看：0, 1, 2, 3
```

写成表：

```text
        key0   key1   key2   key3
q0       看    禁止   禁止   禁止
q1       看     看    禁止   禁止
q2       看     看     看    禁止
q3       看     看     看     看
```

实现时，禁止的位置会加一个非常大的负数，数学上常写成：

```text
-∞
```

也就是说：

```text
未来位置的 score -> -∞
```

为什么是 `-∞`？

因为：

```text
exp(-∞) = 0
```

softmax 时，这个位置的权重就会变成 0。

## 5. masked softmax 手算一遍

假设当前 query 对 4 个 key 算出的分数是：

```text
score = [2, 1, -1, 3]
```

但第 3 个 token 是未来 token，不能看。

mask 后：

```text
masked_score = [2, 1, -1, -∞]
```

现在做 softmax。

第一步 exp：

```text
exp(2)   ≈ 7.389
exp(1)   ≈ 2.718
exp(-1)  ≈ 0.368
exp(-∞)  = 0
```

第二步求和：

```text
sum = 7.389 + 2.718 + 0.368 + 0 = 10.475
```

第三步除以总和：

```text
7.389 / 10.475 ≈ 0.705
2.718 / 10.475 ≈ 0.260
0.368 / 10.475 ≈ 0.035
0     / 10.475 = 0
```

所以：

```text
masked_softmax([2, 1, -1, 3], mask_last)
≈ [0.705, 0.260, 0.035, 0]
```

注意第 3 个 token 原来的分数是 3，最高。

如果不 mask，它会拿到最大权重。

但因为它是未来 token，被 mask 成 `-∞`，最后权重变成 0：

```text
分数高也没用，未来 token 不许看。
```

这就是 masked softmax 的本质。

## 6. 为什么要除以 sqrt(head_dim)

attention 分数通常不是直接：

```text
Q @ K^T
```

而是：

```text
(Q @ K^T) / sqrt(head_dim)
```

Qwen2.5-7B 里：

```text
head_dim = 128
sqrt(128) ≈ 11.31
```

所以 attention score 更准确地写成：

```text
score = (Q @ K^T) / sqrt(128)
```

为什么要除？

先用人话理解：

```text
Q/K 是 128 维向量。
点积会把 128 个乘法结果加起来。
维度越多，点积分数越容易变得很大。
```

如果分数太大，softmax 会变得太尖。

比如：

```text
softmax([2, 1, -1]) ≈ [0.705, 0.260, 0.035]
```

还比较柔和。

但如果分数放大 10 倍：

```text
[20, 10, -10]
```

softmax 会几乎把全部权重给第一个位置：

```text
接近 [0.99995, 0.00005, 0]
```

这会让训练不稳定，也会让注意力过早变得极端。

除以 `sqrt(head_dim)` 的作用是：

```text
把 QK 点积的尺度压回来，让 softmax 不要太尖。
```

这就是 scaled dot-product attention 里的 `scaled`。

## 7. 对上 Qwen2.5-7B 的 shape

前两篇已经得到：

```text
Q: [S, 28, 128]
K: [S,  4, 128]
V: [S,  4, 128]
```

这里的 `S` 可以先理解成本轮 token 数。实际推理时，K/V 会进入 KV cache，所以被查询的 key/value 长度通常是：

```text
T = 过去 token 数 + 当前 token 数
```

更完整一点可以写成：

```text
Q: [S, 28, 128]
K: [T,  4, 128]
V: [T,  4, 128]
```

GQA 的关键是：

```text
28 个 Q head
4 个 K/V head
每 7 个 Q head 共享 1 个 K/V head
```

因为：

```text
28 / 4 = 7
```

所以对某个 Q head 来说，它只会去找自己 group 对应的 K head。

逻辑上可以想成：

```text
Q head 0-6   用 K/V head 0
Q head 7-13  用 K/V head 1
Q head 14-20 用 K/V head 2
Q head 21-27 用 K/V head 3
```

对每个 Q head，会做：

```text
score: [S, T]
```

意思是：

```text
每个 query token
对每个可见 key token
算一个匹配分数
```

全部 Q head 合起来，逻辑 shape 可以理解成：

```text
score: [28, S, T]
weight: [28, S, T]
```

masked softmax 作用在最后的 `T` 这个维度上：

```text
对每个 query token、每个 Q head，
把它对所有 key token 的分数变成一组权重。
```

## 8. masked softmax 之后怎么乘 V

softmax 得到权重后，下一步是乘 V。

对一个 head 来说：

```text
weight: [S, T]
V:      [T, 128]
```

矩阵乘出来：

```text
out_head: [S, 128]
```

人话解释：

```text
每个 query token
用一排权重
从 T 个 value 向量里加权汇总出一个新的 128 维向量
```

28 个 Q head 都做完后：

```text
out_heads: [S, 28, 128]
```

然后把 head 维度拼回 hidden 维度：

```text
[S, 28, 128] -> [S, 3584]
```

因为：

```text
28 * 128 = 3584
```

这之后才进入 out 投影：

```text
[S, 3584] -> W_o -> [S, 3584]
```

所以 masked softmax 本身负责的是：

```text
算权重。
```

它后面的 `weight @ V` 才负责：

```text
按权重汇总内容。
```

## 9. prefill 和 decode 里的 mask 有什么区别

prefill 时，一次输入很多 token：

```text
S = prompt 长度
```

比如 prompt 有 4 个 token，就会同时构建 4 行 query：

```text
q0, q1, q2, q3
```

这时必须用三角 mask：

```text
q0 只能看 k0
q1 能看 k0, k1
q2 能看 k0, k1, k2
q3 能看 k0, k1, k2, k3
```

decode 时，通常一次只生成一个新 token：

```text
S = 1
```

这个新 token 可以看所有历史 K/V cache：

```text
过去 token + 当前 token
```

因为未来 token 还没生成出来，所以 decode 单步里看起来 mask 没那么明显。

但规则没有变：

```text
永远不能看未来。
```

只是 decode 时未来还不存在。

## 10. 和 llama.cpp / ggml 的 graph 对上

在 ggml 计算图里，attention 这段可以粗略看成：

```text
Q, K, V = build_qkv(...)
Q = rope(Q)
K = rope(K)

score = Q @ K^T
score = score / sqrt(head_dim)
score = score + mask
weight = softmax(score)
out = weight @ V
```

在 llama.cpp 的实际实现里，为了性能，这些步骤可能会被融合成 flash attention 风格的算子，或者走不同 backend 的 kernel。

但概念上仍然是这一条链：

```text
QK 分数
-> scale
-> mask
-> softmax
-> 乘 V
```

看源码时不要被名字吓住。你只要抓住：

```text
mask 发生在 softmax 之前
softmax 的输出是 attention weight
weight 再去乘 V
```

这三句话就不会迷路。

## 11. W1-Q4 里该怎么讲 masked softmax

验收时可以这样说：

```text
Q/K 经过 RoPE 后，模型用 Q @ K^T 算每个 query token 对每个 key token 的匹配分数。
因为 head_dim 是 128，所以分数会除以 sqrt(128) 做缩放。
然后加 causal mask，把未来 token 的位置变成 -∞。
softmax 后，这些未来位置的权重变成 0，剩下可见 token 的权重加起来等于 1。
最后这些权重会乘 V，得到每个 head 的输出。
```

如果要带上 shape：

```text
Q: [S, 28, 128]
K: [T,  4, 128]
V: [T,  4, 128]

GQA: 每 7 个 Q head 共享 1 个 K/V head

score/weight 逻辑上可以看成：
[28, S, T]

weight @ V 后：
[S, 28, 128]
```

这就够回答 `masked softmax` 这一步。

## 12. 最后用一句话总结

Masked softmax 做的事情是：

```text
把 Q/K 点积得到的匹配分数，
先用 mask 禁止未来 token，
再用 softmax 变成一组加起来等于 1 的注意力权重。
```

它在 attention 里的位置是：

```text
QKV 投影
-> RoPE(Q/K)
-> QK score
-> masked softmax 得到 weight
-> weight @ V
-> out 投影
```

## 自检题

1. softmax 为什么要先 `exp` 再除以总和？
2. mask 为什么要加在 softmax 之前？
3. 未来 token 的 score 被 mask 成 `-∞` 后，softmax 权重为什么是 0？
4. 为什么 attention score 要除以 `sqrt(head_dim)`？
5. GQA 下 `Q: [S, 28, 128]`，`K/V: [T, 4, 128]`，每几个 Q head 共享一个 K/V head？
6. masked softmax 的输出是最终内容吗？如果不是，它下一步要乘什么？

能答出这六题，`W1-Q4` 的 masked softmax 部分就过关了。
