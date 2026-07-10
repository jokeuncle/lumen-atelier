# 本机实操一次最小大模型蒸馏：用 GGUF Qwen 当 teacher，训练一个 NumPy student

日期：2026-07-10  
状态：发布源稿

## 一句话结论

大模型蒸馏不是让小模型只抄 teacher 最终吐出的文字，而是让 student 学 teacher 在每一步“下一个 token 可能是谁”的概率分布。

这次我没有用玩具 teacher，也没有下载新模型。teacher 就是本机已经跑通的 `week-01-llama-cpp/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf`。实现上用 `llama-server` 暴露本地 HTTP 接口，向 `/v1/completions` 请求 `logprobs`，拿到每个生成 token 的 top-k 候选和 log probability，再训练一个极小的 NumPy student。

这个 student 不是聊天模型。它只是一个教学用的条件 softmax 表：

```text
输入：上一个生成 token id
输出：teacher top-k token 的概率分布
```

但它已经具备蒸馏最核心的闭环：

```text
真实 GGUF teacher
  -> teacher top-k logprobs
  -> soft labels
  -> student softmax
  -> KL loss
  -> 训练前后指标对比
```

## 先讲直觉：蒸馏学的是“老师的犹豫”

普通监督学习像这样：

```text
问题：Explain knowledge distillation...
标准答案 token：Knowledge
```

student 只知道正确 token 是 `Knowledge`，其他 token 都被当成错。

蒸馏更细一点。teacher 可能给出这样的分布：

```text
Knowledge      0.917
\n\n           0.053
\n             0.021
**             0.002
Trans          0.001
```

这就多了一层信息：teacher 不只是说“选 Knowledge”，还告诉 student “我几乎确定是 Knowledge，但换行也有一点可能”。这种软概率就是 soft label。

在大模型里，这些概率来自 logits：

```text
logits -> softmax -> probability distribution
```

logits 是模型最后一层对整个词表打的分数。softmax 把分数变成概率。真正完整的蒸馏通常会用全词表分布；这次用 `llama-server` 能直接拿到的 top-k `logprobs`，所以做的是 top-k 截断分布蒸馏。

## 最小数学基础：softmax 和 KL 在这里干什么

假设 teacher 对三个候选 token 的分数是：

```text
A: logprob = -0.3
B: logprob = -1.2
C: logprob = -2.0
```

这些是对数概率。为了训练 student，我先把它们重新归一化成 top-k 内部的概率：

```python
def normalize_top_logprobs(rows):
    logprobs = np.array([float(row["logprob"]) for row in rows], dtype=np.float64)
    probs = softmax(logprobs)
    return [
        {
            "id": int(row["id"]),
            "token": str(row.get("token", "")),
            "logprob": float(row["logprob"]),
            "prob": float(prob),
        }
        for row, prob in zip(rows, probs)
    ]
```

这里的 `softmax` 做了一个稳定性处理：

```python
def softmax(values):
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    return weights / np.sum(weights)
```

减掉最大值不会改变 softmax 结果，但可以避免 `exp` 爆掉。

student 也输出一个概率分布。训练目标是让 student 分布靠近 teacher 分布。我用 KL divergence 记录距离：

```text
KL(teacher || student)
```

可以把它理解成：如果 teacher 的概率分布才是真正想学的答案，student 现在的分布还浪费了多少概率质量。

训练时真正优化的是 cross entropy：

```text
loss = - sum(teacher_prob * log(student_prob))
```

teacher 的熵不随 student 变化，所以 cross entropy 下降时，KL 也会下降。

## 这次的最小 student 长什么样

为了把流程做小，我没有引入 PyTorch，也没有训练一个小 Transformer。student 是一个 NumPy 参数表：

```text
logits: [context_count, vocab_size]
```

其中：

- `context_id` 是上一个生成 token id；每个 prompt 的第一个生成 token 用 `-1` 当起始上下文。
- `vocab_ids` 是这次采样过程中 teacher top-k 里出现过的 token id 集合。
- 每一行经过 softmax 后，就是 student 对“下一个 token”的预测分布。

样本解析的关键逻辑是：

```python
def parse_completion_samples(payload, prompt, start_context=-1):
    content = _completion_content(payload)
    samples = []
    context_id = start_context

    for position, item in enumerate(content):
        teacher_topk = [
            TeacherProb(
                id=int(row["id"]),
                token=str(row.get("token", "")),
                logprob=float(row["logprob"]),
                prob=float(row["prob"]),
            )
            for row in normalize_top_logprobs(item["top_logprobs"])
        ]
        token_id = int(item["id"])
        samples.append(
            TrainingSample(
                prompt=prompt,
                position=position,
                context_id=context_id,
                token_id=token_id,
                token_text=str(item.get("token", "")),
                token_bytes=[int(value) for value in item.get("bytes", [])],
                teacher_topk=teacher_topk,
            )
        )
        context_id = token_id

    return samples
```

这段代码的核心是把一条 teacher completion 拆成多条训练样本：

```text
START -> 第 1 个生成 token 的 teacher 分布
第 1 个生成 token -> 第 2 个生成 token 的 teacher 分布
第 2 个生成 token -> 第 3 个生成 token 的 teacher 分布
...
```

训练循环也很小：

```python
for _ in range(epochs):
    grad = np.zeros_like(logits)
    for sample in samples:
        row = context_to_row[sample.context_id]
        target = target_vector(sample, vocab_to_col, len(vocab_ids))
        grad[row] += softmax(logits[row]) - target
    grad /= len(samples)
    logits -= lr * grad
```

这一行是整个蒸馏训练的核心：

```text
gradient = student_prob - teacher_prob
```

如果 student 给某个 token 的概率比 teacher 高，梯度会把它压下去；如果 student 给得太低，梯度会把它拉上来。

## 怎么从真实 GGUF teacher 拿 soft labels

脚本会启动本地 `llama-server`：

```bash
llama-server \
  -m models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  -c 512 \
  -ngl 99 \
  --host 127.0.0.1 \
  --port 18087 \
  --no-ui
```

然后对 `/v1/completions` 发请求：

```python
payload = {
    "model": "local-gguf-teacher",
    "prompt": prompt,
    "max_tokens": n_predict,
    "temperature": temperature,
    "top_k": top_k,
    "logprobs": top_k,
    "seed": seed,
}
```

关键是 `logprobs`。它让 server 在返回生成 token 的同时，也返回该步的 top-k 候选 token 和 log probability。

本次运行命令：

```bash
cd week-01-llama-cpp
./.venv/bin/python tools/distill_tiny_from_gguf.py \
  --n-predict 8 \
  --top-k 8 \
  --epochs 240 \
  --lr 0.5 \
  --out-dir reports/distill
```

输出：

```json
{
  "samples": 48,
  "contexts": 34,
  "vocab_size": 247,
  "initial_kl": 4.970174616885221,
  "final_kl": 2.9020281663309966,
  "top1_accuracy": 0.7916666666666666,
  "out_dir": "reports/distill"
}
```

这表示：

- 从 6 个 prompt 中采到 48 个生成 token 位置。
- 这些位置对应 34 种上下文 token。
- top-k 候选合起来覆盖 247 个 token id。
- student 训练前 KL 是 4.97，训练后降到 2.90。
- student 在这些样本上的 top-1 token 和 teacher top-1 对齐率约 79%。

`reports/distill/teacher_samples.jsonl` 里可以看到真实样本：

```json
{
  "prompt": "Explain knowledge distillation in one concise sentence.",
  "position": 0,
  "context_id": -1,
  "token_id": 31925,
  "token_text": " Knowledge",
  "teacher_topk": [
    {"id": 31925, "token": " Knowledge", "prob": 0.9172},
    {"id": 4710, "token": " \n\n", "prob": 0.0535},
    {"id": 715, "token": " \n", "prob": 0.0211}
  ]
}
```

这个样本说明 teacher 在第一步强烈倾向 `Knowledge`，但仍保留了换行等候选。这就是 hard label 看不到的信息。

## 这个 demo 为什么算真实，又为什么仍然很小

它真实在三点：

1. teacher 是本机真实 GGUF Qwen，不是手写规则或随机分布。
2. 监督信号来自 teacher 的概率分布，不只是生成文本。
3. student 通过 KL/cross entropy 训练，训练后指标确实下降。

它很小也很明显：

1. student 不是 Transformer，只是 `P(next_token | previous_token)` 的表。
2. 只学习生成出来的位置，不学习 prompt 内每个 token 的 teacher 分布。
3. 只用 top-k 截断分布，不是全词表 logits。
4. prompt 数量很少，指标只能说明流程跑通，不能说明模型能力。

所以这篇的重点不是“我训练出了一个可用小模型”，而是把蒸馏流程拆到能在本机完整看见：

```text
teacher 怎么给软标签
soft labels 怎么变成训练目标
student 怎么更新
指标怎么证明它学近了一点
```

## 下一步可以怎么扩展

如果要更接近生产训练，可以沿三条路扩展。

第一，把 student 换成 PyTorch 小模型，比如 tiny GRU 或 tiny Transformer。这样就可以学习更长上下文，而不是只看上一个 token。

第二，改 teacher 采样方式，拿更多 prompt、更多 token，并固定 prompt 模板。这样 student 学到的不是几个例子的局部模式。

第三，如果能通过更底层的 llama.cpp API 拿到全词表 logits，就可以做完整分布蒸馏，而不是 top-k 截断蒸馏。

但作为第一步，这个最小 demo 已经足够回答一个核心问题：

```text
蒸馏不是神秘的“压缩模型”按钮。
它是一条可执行的数据管线：
大模型概率分布 -> 小模型概率分布 -> 分布距离 -> 参数更新。
```

## 自查问题

1. hard label 和 soft label 的区别是什么？
2. 为什么 teacher 的第二、第三候选 token 也有学习价值？
3. 这次为什么要把 top-k logprobs 重新归一化？
4. `student_prob - teacher_prob` 这个梯度直觉上在做什么？
5. 这个 demo 和真正训练一个小语言模型还差哪些步骤？
