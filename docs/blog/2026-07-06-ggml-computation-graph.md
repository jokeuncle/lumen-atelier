---
title: "计算图不是在算，而是在排活：从伪代码到 ggml 看懂 llama.cpp 的 Qwen2 forward"
date: 2026-07-06
source: "Lumen Atelier Day 6"
---

# 计算图不是在算，而是在排活：从伪代码到 ggml 看懂 llama.cpp 的 Qwen2 forward

Day 5 看清楚了 llama.cpp 的外层循环：准备 token，调用 `llama_decode`，从 logits 采样下一个 token，再把新 token 塞回下一轮。Day 6 要拆的是更里面的问题：一次 `llama_decode` 到底怎么把 Transformer 的 forward pass 组织起来。

答案是计算图。

很多人第一次看到“计算图”会把它想成一张流程图。这个理解不算错，但还不够工程化。对推理引擎来说，计算图首先不是画给人看的图，而是一份可执行的算子清单。它告诉 backend：有哪些 tensor，每个 tensor 由哪个 op 产生，依赖哪些输入，最终要算出哪个输出。

所以计算图最重要的一句话是：**它不是在算，而是在排活。**

## 1. 什么是计算图

假设我们要算一个很小的表达式：

```text
y = (a * x + b) * x
```

普通程序可以马上执行：

```python
t1 = a * x
t2 = t1 + b
y = t2 * x
```

计算图的思路不同。它先不急着算，而是把每一步登记成节点：

```text
a ----\
      MUL ----\
x ----/        \
               ADD ----\
b -------------/        \
                         MUL ---- y
x ----------------------/
```

这里每个节点都记录：

- `op`：我要做什么操作，比如 `MUL`、`ADD`、`SOFTMAX`
- `src`：我的输入来自哪些 tensor
- `shape`：输出 tensor 长什么样
- `data`：如果已经分配，数据放在哪里

真正执行时，runtime 只要按依赖顺序跑：

```text
先算 t1 = a * x
再算 t2 = t1 + b
最后算 y = t2 * x
```

这件事在深度学习框架里很常见。PyTorch eager mode 看起来像是立即执行，但 autograd 也会为反向传播记录图。TensorFlow 早期更明显，先 define graph，再 run session。llama.cpp 里的 ggml 更贴近推理引擎的需求：它用 C/C++ 的 tensor 结构描述 forward graph，再交给 backend scheduler 执行。

## 2. 一个 30 行伪代码实现

下面是一个极简计算图。它不处理内存分配、不做优化、不支持自动微分，但能表达 ggml 的基本味道。

```python
class Tensor:
    def __init__(self, shape, op="NONE", src=None, data=None, name=""):
        self.shape = shape
        self.op = op
        self.src = src or []
        self.data = data
        self.name = name


def input_tensor(name, shape):
    return Tensor(shape=shape, op="NONE", name=name)


def mul_mat(a, b):
    # 只写直觉，不处理广播和真实 layout
    out_shape = [a.shape[0], b.shape[1]]
    return Tensor(shape=out_shape, op="MUL_MAT", src=[a, b])


def add(a, b):
    return Tensor(shape=a.shape, op="ADD", src=[a, b])


def rms_norm(x):
    return Tensor(shape=x.shape, op="RMS_NORM", src=[x])


def build_forward(output):
    visited = set()
    nodes = []
    leafs = []

    def dfs(t):
        if id(t) in visited:
            return
        visited.add(id(t))

        for parent in t.src:
            dfs(parent)

        if t.op == "NONE":
            leafs.append(t)
        else:
            nodes.append(t)

    dfs(output)
    return nodes, leafs


def execute(nodes):
    for node in nodes:
        inputs = [src.data for src in node.src]
        node.data = run_kernel(node.op, inputs)
```

用它搭一层简化版网络：

```python
x = input_tensor("hidden", [16, 3584])
w = input_tensor("attn_q.weight", [3584, 3584])
b = input_tensor("bias", [16, 3584])

q = mul_mat(x, w)
h = add(rms_norm(q), b)

nodes, leafs = build_forward(h)
execute(nodes)
```

注意 `mul_mat()`、`add()`、`rms_norm()` 在这个模型里没有立刻算出结果。它们只是返回新的 `Tensor`，并把“这个 tensor 是怎么来的”记录下来。`build_forward()` 从最终输出 `h` 往回追依赖，把所有父节点排成一个可执行顺序。

这正是 ggml 的核心直觉。

## 3. ggml_tensor：张量也是图节点

ggml 里最重要的数据结构是 `struct ggml_tensor`，定义在 `ggml/include/ggml.h`。

它不是一个只装数据的数组。一个 `ggml_tensor` 同时包含两类信息。

第一类是张量本身：

```c
enum ggml_type type;
int64_t ne[GGML_MAX_DIMS];
size_t  nb[GGML_MAX_DIMS];
void * data;
char name[GGML_MAX_NAME];
```

`ne[]` 是每一维的元素数量。你可以粗略把它理解成 shape。比如一个 hidden states tensor 可以是 `[3584, n_tokens]`。

`nb[]` 是 stride，单位是字节。它回答的问题不是“这一维有多长”，而是“这一维下标加 1，内存地址要跳多少字节”。这对 `view`、`transpose`、`permute` 很关键，因为很多 layout 变化可以只改 stride，不复制数据。

`data` 是真实数据地址。有些 tensor 是权重，有现成数据。有些 tensor 是中间结果，只有执行 graph 后才会被写入。

第二类是图节点信息：

```c
enum ggml_op op;
int32_t op_params[...];
struct ggml_tensor * src[GGML_MAX_SRC];
struct ggml_tensor * view_src;
size_t view_offs;
```

`op` 说明这个 tensor 是哪个操作的输出，例如：

- `GGML_OP_GET_ROWS`：embedding 查表
- `GGML_OP_RMS_NORM`：RMSNorm
- `GGML_OP_MUL_MAT`：矩阵乘
- `GGML_OP_ROPE`：旋转位置编码
- `GGML_OP_SOFT_MAX`：softmax
- `GGML_OP_FLASH_ATTN_EXT`：flash attention
- `GGML_OP_GLU`：门控 FFN 相关操作

`src[]` 指向输入 tensor。比如：

```text
cur = ggml_mul_mat(ctx, weight, x)
```

得到的 `cur` 不是一块已经算好的结果，而是一个 tensor 节点：

```text
cur.op     = GGML_OP_MUL_MAT
cur.src[0] = weight
cur.src[1] = x
```

所以在 ggml 里，“tensor”和“graph node”不是完全分开的概念。很多 tensor 本身就是某个 op 的输出节点。

## 4. ggml_cgraph：把输出反向追成执行表

另一个关键结构是 `struct ggml_cgraph`，定义在 `ggml/src/ggml-impl.h`。

它里面有几组重要字段：

```c
int n_nodes;
int n_leafs;
struct ggml_tensor ** nodes;
struct ggml_tensor ** leafs;
int32_t * use_counts;
struct ggml_hash_set visited_hash_set;
```

`nodes` 是需要计算的节点。`leafs` 是叶子节点，通常是输入、权重、常量，自己不是由别的 op 算出来的。`visited_hash_set` 用来避免重复访问。`use_counts` 帮助内存规划和复用。

把一个输出 tensor 放进 graph 的入口是：

```c
ggml_build_forward_expand(gf, cur);
```

它内部会调用 forward build 逻辑，从 `cur` 开始递归访问父节点：

```text
最终 logits
  依赖 output projection
    依赖 output norm
      依赖第 28 层输出
        依赖第 27 层输出
          ...
            依赖 embedding
              依赖 input tokens 和 token_embd.weight
```

访问完成后，`gf->nodes` 里就是一个可以顺序执行的节点列表。这里还是没有真正计算，它只是把“要算什么、先算什么、后算什么”排好了。

真正执行在后面：

```text
ggml_backend_sched_graph_compute_async(sched, gf)
```

这一步才会把 graph 交给 backend scheduler，由它决定节点跑在哪个 backend 上，怎么分配中间 buffer，怎么把数据从设备取回来。

## 5. llama.cpp 里从 decode 到 graph 的路径

Day 5 我们已经看到，外层生成循环会反复调用 `llama_decode`。在当前 llama.cpp 版本里，主要路径可以简化成：

```text
llama_context::decode
  -> memory->init_batch
  -> process_ubatch
      -> model.build_graph
          -> build_arch_graph
              -> llama_model_qwen2::graph
      -> res->set_inputs
      -> graph_compute
          -> ggml_backend_sched_graph_compute_async
```

这里有两个容易混的点。

第一，`decode` 不是只处理“生成阶段的一个 token”。从 llama.cpp 外层看，prefill 和 decode 都可以走 `llama_decode`。区别是本轮 batch 里装了多少 token：prompt 阶段可能是几十、几百个 token，生成阶段通常是 1 个 token。

第二，`model.build_graph` 不等于“每次都从零做一切”。`process_ubatch` 会检查 graph 参数是否兼容，如果能复用，就更新输入 tensor，复用上一轮 graph。这就是日志里经常看到 graph reused 的原因。计算图的拓扑一样时，没有必要重新分配和重建所有元信息。

这条路径的工程意义很大：

```text
decode 负责调度一次 forward
model.build_graph 负责搭出这次 forward 的计算图
res->set_inputs 负责把本轮 token、position、mask 等输入填进去
graph_compute 负责让 backend 真正执行
```

把这几层分开后，llama.cpp 就可以支持不同模型架构、不同 backend、不同 batch 形态，而外层生成循环不用知道每种模型的每个算子细节。

## 6. Qwen2 的一层 graph 长什么样

当前 llama.cpp 已经把模型架构拆到 `src/models/` 下。Qwen2 的 forward graph 在 `src/models/qwen2.cpp`。

Qwen2.5-7B 的关键数字是：

```text
hidden size = 3584
layers      = 28
q heads     = 28
kv heads    = 4
head dim    = 128
vocab size  = 152064
```

`llama_model_qwen2::graph` 的主体是一个 layer 循环。每一层大致是：

```text
inpL
  -> RMSNorm
  -> Q/K/V projection
  -> RoPE(Q), RoPE(K)
  -> attention
  -> residual
  -> RMSNorm
  -> FFN(SwiGLU)
  -> residual
```

把它翻译成更贴近源码的伪代码：

```text
inpL = embedding(input_tokens)
inp_pos = positions()
inp_attn = attention_kv_inputs()

for layer in layers:
    residual = inpL

    x = rms_norm(inpL, attn_norm.weight)

    Q, K, V = qkv_projection(x)
    Q = rope(Q, inp_pos)
    K = rope(K, inp_pos)

    attn_out = attention(Q, K, V, KV_cache, mask)
    x = attn_out + residual

    residual = x
    x = rms_norm(x, ffn_norm.weight)
    x = swiglu_ffn(x)
    inpL = x + residual

x = rms_norm(inpL, output_norm.weight)
logits = output.weight @ x
build_forward_expand(graph, logits)
```

最后这句非常关键：

```c
ggml_build_forward_expand(gf, cur);
```

`cur` 是 logits。ggml 从 logits 往回追，就会把 28 层 decoder 的所有依赖节点追出来，形成完整 graph。

## 7. QKV 和 GQA：为什么 Q 是 28 头，K/V 是 4 头

Qwen2.5-7B 使用 GQA，也就是 grouped-query attention。

普通多头注意力里，Q、K、V 的头数通常相同：

```text
Q heads = 28
K heads = 28
V heads = 28
```

GQA 则让多个 Q head 共享较少的 K/V head：

```text
Q heads  = 28
KV heads = 4
```

因为 `hidden = 3584`，`head_dim = 128`：

```text
Q 维度 = 28 * 128 = 3584
K 维度 =  4 * 128 = 512
V 维度 =  4 * 128 = 512
```

在 llama.cpp 的 `build_qkv` 里，Q/K/V projection 后会 reshape 成类似这样的布局：

```text
Q: [head_dim, n_head,    n_tokens] = [128, 28, S]
K: [head_dim, n_head_kv, n_tokens] = [128,  4, S]
V: [head_dim, n_head_kv, n_tokens] = [128,  4, S]
```

这就是你在 GGUF tensor 里看到 `attn_q` 和 `attn_k`、`attn_v` 形状不同的原因。不是 K/V 少算了，而是架构有意减少 K/V 头数，用更小的 KV cache 换取接近的质量。

这对推理很重要。KV cache 大小大致按下面算：

```text
layers * 2(K+V) * batch * kv_heads * seq * head_dim * bytes_per_elem
```

如果不用 GQA，`kv_heads` 从 4 变 28，KV cache 会变成 7 倍。对长上下文和高并发来说，这不是小优化，而是生死线。

## 8. RoPE 不是“加位置向量”，而是旋转 Q/K

在 Qwen2 graph 中，Q/K projection 后会调用 RoPE：

```text
Q = rope(Q, position)
K = rope(K, position)
```

V 不做 RoPE。原因是 attention score 来自 Q 和 K 的相似度：

```text
score = Q @ K^T
```

位置关系需要影响“谁关注谁”，所以进 Q/K；V 是被加权汇总的内容，不负责决定位置相似度。

从数学直觉看，RoPE 把每两个维度看成一个二维平面坐标：

```text
(x0, x1)
```

然后根据 token position 旋转一个角度：

```text
x0' = x0 * cos(theta) - x1 * sin(theta)
x1' = x0 * sin(theta) + x1 * cos(theta)
```

这就是为什么 RoPE 用高中三角函数就能解释。它没有给 hidden state 拼接一个位置编号，而是把 Q/K 的方向按位置旋转。不同位置之间做点积时，相对角度就会进入 attention score。

## 9. Attention：KV cache 写入也是 graph 的一部分

Qwen2 的 attention build 里有一个很值得注意的细节：Q/K/V 节点会先被加入 graph，然后 K/V 会写入 KV cache。

简化后是：

```text
expand(Q)
expand(V)
expand(K)

expand(copy K into KV cache)
expand(copy V into KV cache)

k = cache.get_k(layer)
v = cache.get_v(layer)
out = attention(Q, k, v, mask)
```

这说明 KV cache 写入不是在 graph 外面随便做的副作用。它也被组织进计算图里，变成需要按顺序执行的节点。

为什么要这样？

因为 decode 阶段本轮只输入一个新 token，但 attention 需要看见历史所有 token 的 K/V：

```text
本轮新 K/V：由当前 token 算出来
历史 K/V：已经在 cache 里
attention：Q_current 和 cache 中完整 K/V 做注意力
```

所以一轮 decode 的 attention 不是只对当前 token 做小计算。它虽然只新增一个位置，却要读取越来越长的 KV cache。这正是 decode 阶段容易 memory-bandwidth-bound 的原因之一。

如果启用 flash attention，`build_attn_mha` 会走 `GGML_OP_FLASH_ATTN_EXT`。如果不走 flash attention，就会显式搭出：

```text
kq = K @ Q
kq = softmax(kq + mask)
kqv = V @ kq
```

不管走哪条路径，计算图表达的都是同一件事：

```text
softmax((Q @ K^T) / sqrt(head_dim) + mask) @ V
```

只是 backend 可以选择不同 kernel 来执行。

## 10. FFN：SwiGLU 是两条支路再合流

Qwen2 的 FFN 不是简单的一层 MLP。它是 SwiGLU 结构。

直觉上有两条并行支路：

```text
up   = x @ W_up
gate = x @ W_gate
gate = SiLU(gate)
mid  = gate * up
out  = mid @ W_down
```

在 llama.cpp 的 graph builder 里，`build_ffn` 会构造 `ffn_up`、`ffn_gate`、`ffn_swiglu`、`ffn_down` 这些节点。它们最终再和 residual 相加：

```text
layer_out = ffn_out + ffn_inp
```

这也解释了为什么 GGUF 里每层会看到：

```text
blk.N.ffn_gate.weight
blk.N.ffn_up.weight
blk.N.ffn_down.weight
```

`gate` 和 `up` 不是重复，它们是 SwiGLU 的两条不同投影。

## 11. 三个容易误解的点

**第一，graph build 不等于 graph compute。**

`ggml_mul_mat`、`ggml_rope_ext`、`ggml_soft_max_ext` 这些调用大多是在创建 tensor 节点。真正执行是在 backend scheduler compute graph 时发生。

**第二，tensor 不一定拥有独立数据。**

很多 tensor 是 view、reshape、permute 的结果。它们可能只是换了一种方式解释同一块内存。理解 `ne[]` 和 `nb[]`，比只看 shape 更接近 ggml 的真实工作方式。

**第三，decode 每轮只输入 1 个 token，不代表只读很少数据。**

decode 阶段仍然要经过所有层的权重，还要读历史 KV cache。输入小，不等于访问小。这也是为什么 decode 常常受内存带宽限制。

## 12. 今天应该带走什么

如果只记三句话：

第一，计算图是“待执行的依赖表”，不是已经算好的结果。

第二，ggml 的 tensor 同时记录 shape、stride、op、src，所以 tensor 本身就能成为 graph node。

第三，Qwen2 forward 的 graph 从最终 logits 往回追，会串起 embedding、28 层 RMSNorm/QKV/RoPE/attention/FFN、output projection；KV cache 写入也在这张图里。

学到这里，`W1-Q4` 就不该再背答案了。你应该能白板画出 attention 内部四步：

```text
QKV projection
→ RoPE(Q, K)
→ masked softmax / flash attention
→ output projection
```

并且能说出 GQA 下的形状差异：

```text
Q: [128, 28, S]
K: [128,  4, S]
V: [128,  4, S]
```

这就是 Day 6 的验收线。

## 参考源码

- `ggml/include/ggml.h`: `struct ggml_tensor`、`enum ggml_op`
- `ggml/src/ggml-impl.h`: `struct ggml_cgraph`
- `ggml/src/ggml.c`: `ggml_build_forward_expand`
- `src/llama-context.cpp`: `llama_context::decode`、`process_ubatch`、`graph_compute`
- `src/llama-model.cpp`: `llama_model::build_graph`
- `src/models/qwen2.cpp`: `llama_model_qwen2::graph`
- `src/llama-graph.cpp`: `build_qkv`、`build_attn`、`build_attn_mha`、`build_ffn`
