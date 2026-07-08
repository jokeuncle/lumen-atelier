---
title: "Out 投影到底在输出什么：把多头 attention 结果重新混回 hidden state"
date: 2026-07-08
source: "Lumen Atelier attention out projection deep dive"
---

# Out 投影到底在输出什么：把多头 attention 结果重新混回 hidden state

前面三篇已经把 attention 里的前三步拆开了：

```text
1. QKV 投影
   hidden -> Q/K/V

2. RoPE
   Q/K 按 token 位置旋转
   V 不旋转

3. masked softmax
   QK 分数 -> mask -> softmax -> 权重乘 V
```

这篇补最后一步：

```text
4. out 投影
```

目标很窄：读完以后，你要能分清两件名字很像、但位置完全不同的东西：

```text
blk.N.attn_output.weight  每一层 attention 里面的 out 投影
output.weight             整个模型最后的 unembed / logits 投影
```

一句话先说结论：

**attention 的 out 投影不是“输出下一个词”，而是把多头 attention 汇总出来的结果重新混回 hidden state，让它能接上残差和后面的 FFN。**

## 1. 人话直觉：out 投影是 attention 里的混音台

多头 attention 为什么要有很多 head？

你可以先把每个 head 想成一个独立的观察角度：

```text
head 0：可能偏向看语法依赖
head 1：可能偏向看实体指代
head 2：可能偏向看局部搭配
...
head 27：可能学到另一个有用模式
```

注意，这不是说每个 head 一定有人类能命名的功能。真实模型里 head 的分工是训练学出来的，不一定能被我们干净解释。但从工程形状上看，每个 head 确实是在较小的子空间里独立算 attention。

以 Qwen2.5-7B 为例：

```text
hidden size H = 3584
attention heads = 28
head_dim = 128

28 × 128 = 3584
```

masked softmax 结束后，每个 token 会得到 28 份 head 结果。每份是 128 维：

```text
[S, 28, 128]
```

接下来可以直接把 28 个 head 拼起来：

```text
[S, 28, 128] -> [S, 3584]
```

但问题来了：**拼起来只是把 28 路声音排成一排，还没有把它们混成一首歌。**

out 投影就是这个混音台。

它用一个矩阵 `W_o` 把 3584 个数重新线性组合一次：

```text
[S, 3584] -> W_o -> [S, 3584]
```

输入和输出维度一样，都是 3584。但它不是“什么都没变”。它会把不同 head、不同 head 内部维度的信息重新混合，让 attention 的结果回到 Transformer 层通用的 hidden 表示空间里。

## 2. 最小数学：维度不变，也可以发生信息混合

很多人第一次看到 out 投影会困惑：

```text
[S, 3584] -> [S, 3584]
```

既然维度没变，为什么还要乘一个矩阵？

先用小数字看。

假设只有 2 个 head，每个 head 2 维。某个 token 做完 attention 后得到：

```text
head_0 = [10, 1]
head_1 = [2,  8]
```

把它们拼起来：

```text
a = [10, 1, 2, 8]
```

现在我们想输出一个新的 4 维 hidden 向量。最简单的做法是原样返回：

```text
y = [10, 1, 2, 8]
```

但这等于说：

```text
第 0 维只能来自 head_0 的第 0 维
第 1 维只能来自 head_0 的第 1 维
第 2 维只能来自 head_1 的第 0 维
第 3 维只能来自 head_1 的第 1 维
```

这太死了。

out 投影允许每个输出维度从所有输入维度里取料。比如：

```text
y0 = 0.5 * a0 + 0.5 * a2
y1 = 1.0 * a1 - 0.2 * a3
y2 = 0.1 * a0 + 0.9 * a3
y3 = 0.3 * a1 + 0.7 * a2
```

代入 `a = [10, 1, 2, 8]`：

```text
y0 = 0.5 * 10 + 0.5 * 2 = 6.0
y1 = 1.0 * 1  - 0.2 * 8 = -0.6
y2 = 0.1 * 10 + 0.9 * 8 = 8.2
y3 = 0.3 * 1  + 0.7 * 2 = 1.7
```

输出仍然是 4 维：

```text
y = [6.0, -0.6, 8.2, 1.7]
```

但每个位置已经混进了不同 head 的信息。

这就是矩阵乘的意义：

```text
y = a W_o
```

它不一定改变维度。它更重要的作用是：

```text
重新组合特征
重新分配信息
把一个表示空间变换到另一个表示空间
```

所以 out 投影里的“投影”，和 QKV 投影里的“投影”是一类东西：都是用训练学出来的权重矩阵，把向量变成更适合下一步使用的表示。

## 3. 回到 Qwen2.5-7B：真实 shape 是什么

Qwen2.5-7B 的关键数字：

```text
hidden size H = 3584
attention heads = 28
KV heads = 4
head_dim = 128
layer count = 28
```

前面 QKV 投影阶段因为 GQA，会出现 Q 和 K/V head 数不同：

```text
Q: [S, 28, 128]
K: [S,  4, 128]
V: [S,  4, 128]
```

masked softmax 阶段，Q 的 28 个 head 会分组复用 K/V 的 4 个 head。算完注意力权重乘 V 以后，输出仍然要回到 Q head 的数量：

```text
attention result: [S, 28, 128]
```

然后把 head 维度拼回 hidden 维度：

```text
[S, 28, 128] -> [S, 3584]
```

最后乘 out 投影矩阵：

```text
A:   [S, 3584]
W_o: [3584, 3584]
Y:   [S, 3584]
```

这一步在 GGUF 里对应的张量是：

```text
blk.0.attn_output.weight   3584 × 3584
blk.1.attn_output.weight   3584 × 3584
...
blk.27.attn_output.weight  3584 × 3584
```

每一层都有自己的一份 `attn_output.weight`。因为每层学到的表示空间不同，第 0 层的 out 投影和第 27 层的 out 投影不是共享的。

## 4. out 投影前后，token 之间还会互相看吗？

不会。

这一点很关键。

一层 attention 里可以粗略分成两种混合：

```text
token mixing：不同 token 之间互相看
feature mixing：同一个 token 内部的维度重新组合
```

QK 点积、mask、softmax、权重乘 V，是 token mixing：

```text
当前 token 从历史 token 里按权重取信息
```

out 投影是 feature mixing：

```text
每个 token 已经拿到自己的 attention 汇总结果
然后在这个 token 自己的 3584 个维度里重新混合
```

用 shape 看更清楚。

masked softmax 里的注意力权重有 token 对 token 的关系：

```text
weights: [S, S]
```

它回答的是：

```text
第 i 个 token 应该看第 j 个 token 多少？
```

out 投影的矩阵没有 `S × S`，它是：

```text
W_o: [3584, 3584]
```

它回答的是：

```text
这个 token 的第 k 个 hidden 维度，应该由哪些 head/feature 混出来？
```

所以别把 out 投影理解成“又做了一次 attention”。它不再决定看哪个 token，而是在每个 token 内部整理刚刚看来的信息。

## 5. 为什么不能省掉 out 投影？

理论上，如果没有 out 投影，多头 attention 也能拼出一个 `[S, 3584]` 的结果。那为什么 Transformer 还要加这一步？

原因一：**head 之间需要交流。**

如果只是 concat，每个 head 的结果被固定放在自己的 128 维槽位里。后面的层当然也可以慢慢混合，但 attention 模块自己就失去了把 head 信息整合起来的机会。

out 投影允许：

```text
输出第 10 维 = head 0 的一点 + head 7 的一点 + head 19 的一点
```

也允许：

```text
某些 head 的信号被放大
某些 head 的信号被压低
某些 head 的组合被送到残差流里的特定方向
```

原因二：**attention 的输出要回到 residual stream。**

Transformer 每层通常有残差连接：

```text
x = x + attention_out
```

这里的 `x` 是这一层进来时的 hidden state：

```text
x: [S, 3584]
```

所以 attention 模块最终也必须产出：

```text
attention_out: [S, 3584]
```

但“维度对上”还不够。它还要在语义上适合加回 residual stream。out 投影就是训练出来的适配器：把多头 attention 的内部表示，变成这一层 hidden state 能接住的增量。

原因三：**后面的 FFN 期待的是通用 hidden 表示。**

attention 之后通常会接：

```text
residual -> RMSNorm -> FFN
```

FFN 不知道前面每个 head 的边界，也不应该被迫理解“第几个 128 维来自哪个 head”。out 投影先把 head 结果混成普通 hidden state，后面的模块才可以按统一接口继续处理。

## 6. 它和最终 output.weight 不是一回事

这是最容易混的点。

Qwen2.5-7B 的 GGUF 里同时有：

```text
blk.N.attn_output.weight
output.weight
```

名字都带 output，但位置完全不同。

### 6.1 attention out 投影：每一层内部都有

`blk.N.attn_output.weight` 属于第 `N` 层 decoder 的 attention 模块：

```text
attention result: [S, 3584]
-> blk.N.attn_output.weight
attention out:    [S, 3584]
```

它的工作是：

```text
把多头 attention 结果混回 hidden state
```

它不产生词表概率，不决定下一个 token 的 id。

### 6.2 最后的 output.weight：整模型末尾才用

`output.weight` 是整个模型最后的 unembed / logits 投影：

```text
last hidden: [S, 3584]
-> output.weight
logits:      [S, vocab_size]
```

在这份 Qwen2.5-7B GGUF 里能看到：

```text
output.weight  152064 × 3584
```

它的工作是：

```text
把 hidden state 投到词表空间
```

也就是给每个候选 token 一个 logit 分数。后面再经过 temperature、top-k、top-p 等采样策略，才会选出下一个 token。

所以两者区别可以一行记住：

```text
attn_output.weight: [3584 -> 3584]，层内 attention 收尾
output.weight:      [3584 -> vocab]，整模型生成 token 前的最后一步
```

## 7. 把 W1-Q4 串起来

现在可以完整回答 `W1-Q4 attention 内部四步` 了。

输入是一层 decoder 里的 hidden states：

```text
X: [S, 3584]
```

第一步，QKV 投影：

```text
Q: [S, 3584] -> [S, 28, 128]
K: [S, 3584] -> [S,  4, 128]
V: [S, 3584] -> [S,  4, 128]
```

这里 28:4 是 GQA。Q 有 28 个 head，K/V 只有 4 个 head，多个 Q head 共享一组 K/V，用来省 KV cache 和计算/带宽。

第二步，RoPE：

```text
Q/K 按位置旋转
V 不旋转
```

RoPE 的作用是把位置信息塞进 Q/K 的角度里，让后面的 QK 点积能感知相对位置。

第三步，masked softmax：

```text
scores = Q K^T / sqrt(128)
scores + causal mask
weights = softmax(scores)
context = weights V
```

这一步让每个 token 从它能看到的历史 token 里按权重取信息。输出回到：

```text
context: [S, 28, 128]
```

第四步，out 投影：

```text
[S, 28, 128]
-> reshape/concat
[S, 3584]
-> W_o
[S, 3584]
```

这一步把多头 attention 的结果重新混合成标准 hidden state，然后才能加回残差、继续走 FFN。

把四步压成一句话：

```text
QKV 负责造出查询/匹配/内容三种视角；
RoPE 给 Q/K 加位置；
masked softmax 决定每个 token 看历史 token 的权重；
out 投影把 28 个 head 的结果重新混回 3584 维 hidden state。
```

## 8. 自测题

1. 为什么 `out 投影`不是“输出下一个 token”？
2. Qwen2.5-7B 里 `28 × 128 = 3584` 说明了什么？
3. masked softmax 结束后的 context 是 `[S, 28, 128]`，为什么还要变成 `[S, 3584]`？
4. `blk.0.attn_output.weight` 和 `output.weight` 的 shape 分别是什么？它们分别位于模型的哪里？
5. out 投影是 token mixing 还是 feature mixing？为什么？

能不用稿子答出这五题，`W1-Q4 attention 内部四步` 就完整了。

