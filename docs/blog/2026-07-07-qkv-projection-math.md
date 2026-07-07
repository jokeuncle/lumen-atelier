---
title: "QKV 投影到底在投什么：从 3584 维 hidden 到 28 个 Query 头、4 个 KV 头"
date: 2026-07-07
source: "Lumen Atelier QKV projection deep dive"
---

# QKV 投影到底在投什么：从 3584 维 hidden 到 28 个 Query 头、4 个 KV 头

上一篇写 ggml 计算图时，我把 Qwen2 的一层 decoder 拆成了：

```text
RMSNorm -> QKV 投影 -> RoPE -> attention -> FFN -> residual
```

这篇只专注其中一步：**QKV 投影**。

目标很窄：读完以后，你应该能不用背答案，自己写出下面三件事：

```text
Q = X Wq
K = X Wk
V = X Wv
```

以及 Qwen2.5-7B 的真实形状：

```text
X:  [S, 3584]
Q:  [S, 3584] -> [S, 28, 128]
K:  [S,  512] -> [S,  4, 128]
V:  [S,  512] -> [S,  4, 128]
```

这里的 `S` 是本轮 token 数。prefill 时可能是几百，decode 时通常是 1。

## 1. 先别急着算：Q/K/V 各自是什么角色

Attention 可以先用一个信息检索系统来理解。

假设你在图书馆找书：

- Query：你手上的问题，表示“我想找什么”
- Key：每本书贴在外面的标签，表示“我能被什么问题匹配”
- Value：书里面真正的内容，表示“如果我被选中，我贡献什么信息”

放到 Transformer 里，一个 token 的 hidden 向量本来只有一份。比如 token “苹果”经过 embedding 和前面层处理后，得到一个 3584 维向量。这个向量同时包含很多信息：它可能是水果，也可能是公司，也可能出现在一句和手机有关的话里。

但 attention 需要这个 token 扮演三种不同角色：

```text
当它主动找别人时：它需要一个 Query
当它被别人匹配时：它需要一个 Key
当它被别人选中后：它需要一个 Value
```

所以模型不会直接拿同一个 hidden 向量去做所有事，而是用三套不同的权重矩阵，把同一个输入投影成三份：

```text
hidden -> Q
hidden -> K
hidden -> V
```

这里“投影”不是几何课里把影子投到墙上那么窄的意思。在线性代数里，它更泛指：**用一个矩阵把向量从一个表示空间变换到另一个表示空间**。

这三个空间的分工不同：

```text
Q 空间：适合表达“我想找什么”
K 空间：适合表达“我可以被什么找到”
V 空间：适合表达“我被找到后提供什么内容”
```

权重矩阵 `Wq/Wk/Wv` 是训练学出来的，不是人工规定的。训练过程中，模型发现“这样变换 hidden 向量，有利于预测下一个 token”，这些矩阵就被梯度下降慢慢调出来。

## 2. 输入 X 是什么：一排 token 的 hidden states

进入 QKV 投影之前，每个 token 已经有一个 hidden 向量。

用 Qwen2.5-7B 的数字：

```text
hidden size H = 3584
```

如果本轮有 `S` 个 token，就把它们排成一个矩阵：

```text
X: [S, H] = [S, 3584]
```

把 `S=3` 写开，就是：

```text
X =
[
  token_0 的 3584 维向量,
  token_1 的 3584 维向量,
  token_2 的 3584 维向量
]
```

每一行是一个 token。每一列是 hidden 向量的一个特征维度。

这里先记住一个重要事实：**QKV 投影本身只在每个 token 的 3584 个特征维度之间做混合，还没有让 token 之间互相看。**

也就是说，`X Wq` 是对每一行 token 独立做同一个线性变换。token 之间真正互相看，是后面的 `Q @ K^T` 和 softmax。

## 3. 最小数学：矩阵乘到底在算什么

先用小数字，不用 3584。

假设一个 token 的 hidden 向量只有 3 维：

```text
x = [2, 1, 3]
```

我们想把它投影成 2 维的 Query，于是准备一个权重矩阵：

```text
Wq =
[
  [1,  0],
  [0,  2],
  [1, -1]
]
```

形状是：

```text
x:  [1, 3]
Wq: [3, 2]
q:  [1, 2]
```

计算：

```text
q = x Wq
```

第一维：

```text
q0 = 2*1 + 1*0 + 3*1 = 5
```

第二维：

```text
q1 = 2*0 + 1*2 + 3*(-1) = -1
```

所以：

```text
q = [5, -1]
```

这就是投影的核心。新的每一维，都是旧向量所有维度的加权求和。

如果有多个 token，比如：

```text
X =
[
  [2, 1, 3],
  [0, 4, 1],
  [5, 2, 2]
]
```

那 `X Wq` 就是对每一行都做同样的变换：

```text
Q = X Wq
```

形状：

```text
X:  [3, 3]
Wq: [3, 2]
Q:  [3, 2]
```

这就是大模型里的 QKV 投影，只是数字从 3 维、2 维变成了 3584 维、512 维或 3584 维。

## 4. 回到 Qwen2.5-7B：三次矩阵乘

Qwen2.5-7B 的输入是：

```text
X: [S, 3584]
```

Q 投影：

```text
Q = X Wq
Wq: [3584, 3584]
Q:  [S, 3584]
```

K 投影：

```text
K = X Wk
Wk: [3584, 512]
K:  [S, 512]
```

V 投影：

```text
V = X Wv
Wv: [3584, 512]
V:  [S, 512]
```

如果模型有 bias，公式会写成：

```text
Q = X Wq + bq
K = X Wk + bk
V = X Wv + bv
```

但主干逻辑还是矩阵乘。bias 只是给每个输出维度再加一个偏移量。

这里最容易疑惑的是：为什么 Q 是 3584 维，K/V 只有 512 维？

答案是 GQA。

## 5. head_dim、n_head、n_kv_head 三个数字

先看 Qwen2.5-7B 的三个关键超参数：

```text
n_head    = 28
n_kv_head = 4
head_dim  = 128
```

`head_dim` 是每个 attention head 的宽度。Qwen2.5-7B 里每个头 128 维。

Q 有 28 个头：

```text
Q 总维度 = n_head * head_dim
        = 28 * 128
        = 3584
```

K/V 只有 4 个头：

```text
K 总维度 = n_kv_head * head_dim
        = 4 * 128
        = 512

V 总维度 = n_kv_head * head_dim
        = 4 * 128
        = 512
```

所以你会看到：

```text
Wq: [3584, 3584]
Wk: [3584, 512]
Wv: [3584, 512]
```

不是 K/V 少了一步，也不是写错了。Qwen2.5-7B 本来就让 Query 头数多，让 Key/Value 头数少。

这叫 GQA，Grouped-Query Attention。

## 6. reshape：从一整条向量切成很多个 head

矩阵乘刚算完时，Q 是：

```text
Q: [S, 3584]
```

但 attention 不把它当成一整条 3584 维向量用，而是切成 28 个 head：

```text
Q: [S, 3584]
 -> [S, 28, 128]
```

意思是：

```text
每个 token
有 28 个 query head
每个 head 是 128 维
```

K/V 同理：

```text
K: [S, 512] -> [S, 4, 128]
V: [S, 512] -> [S, 4, 128]
```

这一步通常叫 reshape。它不一定复制数据，只是换一种方式解释同一串数字：

```text
3584 = 28 * 128
512  =  4 * 128
```

举个小例子，假设一个向量长度是 6：

```text
[a, b, c, d, e, f]
```

如果切成 2 个 head，每个 head 3 维：

```text
[
  [a, b, c],
  [d, e, f]
]
```

数字没有变，只是从“长度 6 的向量”变成了“2 个长度 3 的头”。

## 7. GQA：28 个 Q head 怎么共享 4 个 K/V head

标准 Multi-Head Attention 里，Q/K/V head 数量一样。

如果 Qwen2.5-7B 不用 GQA，而是让 K/V 也有 28 个头：

```text
Q: [S, 28, 128]
K: [S, 28, 128]
V: [S, 28, 128]
```

这样最直观：第 0 个 Q head 看第 0 个 K/V head，第 1 个 Q head 看第 1 个 K/V head，以此类推。

但 Qwen2.5-7B 用 GQA：

```text
Q: [S, 28, 128]
K: [S,  4, 128]
V: [S,  4, 128]
```

28 个 Q head 要分组共享 4 组 K/V。比例是：

```text
28 / 4 = 7
```

也就是每 7 个 Q head 共享 1 个 K/V head：

```text
Q head  0-6   -> KV head 0
Q head  7-13  -> KV head 1
Q head 14-20  -> KV head 2
Q head 21-27  -> KV head 3
```

人话解释：

```text
模型保留 28 种“提问方式”
但只保留 4 组“可被匹配的标签”和“内容载荷”
```

这样做的最大工程收益是省 KV cache。

decode 阶段每生成一个新 token，都要把这个 token 的 K/V 追加到 KV cache。KV cache 的大小跟 `n_kv_head` 成正比：

```text
KV cache ~= layers * 2(K+V) * batch * n_kv_head * seq * head_dim * bytes
```

如果 `n_kv_head` 从 4 变成 28，KV cache 直接变 7 倍。对长上下文和高并发来说，这个差距非常大。

所以 GQA 的核心取舍是：

```text
Q head 保持多：保留丰富的查询角度
K/V head 变少：降低 KV cache 体积和带宽压力
```

## 8. 投影之后发生了什么

QKV 投影只完成了 attention 的准备工作。

投影之后，流程继续：

```text
Q -> RoPE
K -> RoPE
V -> 不做 RoPE
K/V -> 写入 KV cache
Q 和 cache 里的 K 做匹配
softmax 得到权重
权重乘 V 得到 attention 输出
```

为什么 Q/K 做 RoPE，V 不做？

因为 attention score 来自：

```text
Q @ K^T
```

位置关系需要影响“谁和谁匹配”，所以位置编码进入 Q/K。V 是被加权汇总的内容，它不负责决定匹配分数。

QKV 投影和 attention 的分工可以这样记：

```text
QKV 投影：每个 token 自己内部做特征变换
QK 匹配：不同 token 之间开始互相看
softmax：把匹配分数变成权重
乘 V：按权重汇总内容
```

这也是一个很关键的边界：**QKV 投影阶段还没有 token 间通信。attention score 阶段才开始 token 间通信。**

## 9. 和 llama.cpp 源码对上

当前 llama.cpp 版本里，Qwen2 的 graph 在：

```text
src/models/qwen2.cpp
```

其中一层 self-attention 会调用：

```text
build_qkv(model.layers[il], cur, n_embd_head, n_head, n_head_kv, il)
```

这里的参数对应：

```text
n_embd_head = 128
n_head      = 28
n_head_kv   = 4
```

`build_qkv` 的实现位于：

```text
src/llama-graph.cpp
```

它先算：

```text
n_embd_q  = n_embd_head * n_head
n_embd_kv = n_embd_head * n_head_kv
```

放进 Qwen2.5-7B 的数字：

```text
n_embd_q  = 128 * 28 = 3584
n_embd_kv = 128 *  4 = 512
```

然后有两条路径：

第一种是 fused QKV：

```text
一次大矩阵乘算出 Q/K/V 拼在一起的 qkv
再用 view 切出 Q、K、V
```

第二种是 separate Q/K/V：

```text
Qcur = Wq * cur
Kcur = Wk * cur
Vcur = Wv * cur
再 reshape 成 3D
```

源码里 separate 路径最后会做类似：

```text
Qcur -> [n_embd_head, n_head,    n_tokens]
Kcur -> [n_embd_head, n_head_kv, n_tokens]
Vcur -> [n_embd_head, n_head_kv, n_tokens]
```

这和文章前面写的数学形状：

```text
Q: [S, 28, 128]
K: [S,  4, 128]
V: [S,  4, 128]
```

是同一件事，只是维度顺序不同。

数学讲解通常把 token 放前面：

```text
[tokens, heads, head_dim]
```

ggml 源码里常看到：

```text
[head_dim, heads, tokens]
```

不要被顺序吓到。核心数字没变：

```text
Q 有 28 个头，每头 128 维
K/V 有 4 个头，每头 128 维
```

## 10. 一张从输入到 Q/K/V 的全流程图

把所有步骤串起来：

```text
输入 hidden states
X: [S, 3584]
        |
        | 乘 Wq: [3584, 3584]
        v
Q flat: [S, 3584]
        |
        | reshape: 3584 = 28 * 128
        v
Q heads: [S, 28, 128]


输入 hidden states
X: [S, 3584]
        |
        | 乘 Wk: [3584, 512]
        v
K flat: [S, 512]
        |
        | reshape: 512 = 4 * 128
        v
K heads: [S, 4, 128]


输入 hidden states
X: [S, 3584]
        |
        | 乘 Wv: [3584, 512]
        v
V flat: [S, 512]
        |
        | reshape: 512 = 4 * 128
        v
V heads: [S, 4, 128]
```

如果你能从这张图自己推出三组 shape，QKV 投影就基本过关。

## 11. 最后用一句话总结

QKV 投影做的事情是：**把每个 token 当前的 hidden 向量，分别变换成“用来找别人”的 Query、“用来被别人匹配”的 Key、以及“被选中后贡献内容”的 Value。**

对 Qwen2.5-7B 来说，具体数学就是：

```text
X: [S, 3584]

Q = X Wq, Wq: [3584, 3584], Q: [S, 3584] -> [S, 28, 128]
K = X Wk, Wk: [3584,  512], K: [S,  512] -> [S,  4, 128]
V = X Wv, Wv: [3584,  512], V: [S,  512] -> [S,  4, 128]
```

再压缩成验收答案：

```text
Q 头数 28，K/V 头数 4，每头 128 维。
GQA 让 28 个 Q head 分组共享 4 组 K/V head。
这样保留多查询角度，同时把 KV cache 降到原来的 1/7。
```

这就是 QKV 投影。

## 自检题

1. 为什么 Q/K/V 要用三套不同矩阵，而不是直接复制同一个 hidden 向量？
2. Qwen2.5-7B 里 `head_dim=128`、`n_head=28`，为什么 Q 的总维度是 3584？
3. `n_kv_head=4` 时，K/V 为什么是 512 维？
4. QKV 投影阶段有没有 token 之间的信息交换？
5. 数学里的 `[S, 28, 128]` 和 ggml 里的 `[128, 28, S]` 为什么不矛盾？

能答出这五题，再回去看 `W1-Q4 attention 内部四步`，就不会只是在背“QKV 投影 -> RoPE -> softmax -> out 投影”。
