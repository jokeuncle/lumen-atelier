# Day 5 讲义 · 一个 token 的诞生：从命令行到 llama_decode

> 产出者：Claude（代读源码），2026-07-04 · 源码版本：llama.cpp master (shallow clone)
> 用法：读完后接受 W1-Q3 验收。看不懂的段落标出来，验收时一起过。
> 所有行号可直接在 ~/self-projects/llama.cpp 里验证。

---

## 0. 一图流总览

你敲下 `llama-completion -m model.gguf -p "你好" -n 128` 之后：

```
tools/completion/completion.cpp
│
│  ① tokenize（一次性）
│  L320   embd_inp = common_tokenize(ctx, prompt, true, true)
│         "你好" → [108386, ...]  （文本变成 token ID 数组）
│
│  ② 生成主循环
│  L586   while ((n_remain != 0 && !is_antiprompt) || interactive) {
│  │
│  │  ③ 喂料（本轮要 forward 的 token 装进 embd）
│  │  L731   把 prompt 里还没消化的 token 搬进 embd（最多 n_batch 个）
│  │         ↑ 第一轮走这里 = prefill（一次装几百个）
│  │  L715   embd.push_back(id)  ← 之后每轮只装上一轮采样出的 1 个 = decode
│  │
│  │  ④ forward pass（真正的计算）
│  │  L688   common_prompt_batch_decode(ctx, ..., n_past, n_batch, ...)
│  │           └→ common/common.cpp L1995
│  │                └→ L2038  llama_decode(ctx, llama_batch_get_one(tokens, n_new))
│  │                     └→ src/llama-context.cpp L4058（薄壳，只有 4 行）
│  │                          └→ ctx->decode(batch)
│  │                               └→ llama_context::decode  L1680 ★明天 Day 6 的正文
│  │
│  │  ⑤ 采样（logits → 下一个 token）
│  │  L709   id = common_sampler_sample(smpl, ctx, -1)
│  │         L725   --n_remain   ← 你的 -n 128 预算在这里一格格扣
│  │
│  └──⑥ 新 token 塞回 embd → 回到 ④，直到预算耗尽或遇到停止词
```

**一句话总结**：整个"聊天"就是一个 while 循环，每轮做一次 forward + 一次采样，唯一的分叉是"这轮喂几个 token"。

---

## 1. 六个关键点的人话解读

### ① tokenize 只发生一次（L320）
`common_tokenize` 把整个 prompt 一次性变成 ID 数组 `embd_inp`。之后循环里再也不碰文本——模型的世界里只有整数。你 Day 7 会玩的 tokenizer_play.py 就是单独把这一步拎出来看。

### ② while 的退出条件（L586）
```c
while ((n_remain != 0 && !is_antiprompt) || params.interactive)
```
两条退出路径：`n_remain` 是 `-n 128` 的倒计时；`is_antiprompt` 是撞上停止词（比如聊天模板里的 `<|im_end|>`）。`||  interactive` 意味着交互模式下永不退出——这就是你冒烟测试时 llama-cli 一直挂着 `>` 等输入的原因。

### ③ prefill 和 decode 的分叉不在函数上，在"装几个"上（L707 的 if/else）
这是今天最重要的发现，很多教程讲错：**llama.cpp 里没有单独的 prefill 函数**。分叉是这个 if：

```c
if (embd_inp.size() <= n_consumed && !is_interacting) {
    // prompt 已消化完 → 采样 1 个新 token 放进 embd   ← decode 路径
} else {
    // prompt 还有剩 → 一次搬最多 n_batch 个进 embd    ← prefill 路径
}
```

同一个 `llama_decode`，喂 512 个 token 就是 prefill，喂 1 个就是 decode。你实测的 257 t/s vs 25 t/s 的差距，就源于这一个数字的差别（数学图 6.2：算术强度 ×512 vs ×1）。

### ④ batch：装 token 的"托盘"（llama-batch.cpp L863）
```c
struct llama_batch { n_tokens; tokens; embd; pos; n_seq_id; seq_id; logits; }
```
`llama_batch_get_one(tokens, n)` 就是把 n 个 token 放上托盘，其他字段全 nullptr（让引擎自己推断位置）。vLLM 的 continuous batching 本质上就是把**多个用户的 token 装上同一个托盘**——W5 会回到这里。

### ⑤ n_past：KV cache 的游标
`common_prompt_batch_decode`（common.cpp L1995）里每次成功 decode 后 `n_past += n_new`。这个数字是"KV cache 里已经存了多少个位置"——每轮 +1（decode）或 +几百（prefill）。你昨天算的 117MB 那笔账里的 `seq`，运行时就是这个 `n_past`。

### ⑥ llama_decode 是个薄壳（llama-context.cpp L4058）
```c
int32_t llama_decode(llama_context * ctx, llama_batch batch) {
    const int ret = ctx->decode(batch);   // 就这一行
    ...
}
```
C API 只是门面，真正干活的是 `llama_context::decode`（L1680，约几百行）——里面才有"把 batch 塞进计算图、跑 backend、取回 logits"。**这是明天 Day 6 的正文**。

---

## 2. 数学配菜的答案（形状追踪第一步）

prefill 一轮（假设 prompt 16 个 token，Qwen2.5-7B）：

```
输入 token IDs:  [16]              ← 16 个整数
   ↓ embedding 查表（token_embd.weight [3584, 152064]）
hidden states:   [16, 3584]        ← 每个 token 换成一根 3584 维向量
   ↓ 28 层 decoder（明天拆）
   ↓ 只取最后一个位置
logits:          [1, 152064]       ← 对 152064 个候选 token 的打分
```

decode 一轮：输入是 `[1]`，中间是 `[1, 3584]`，输出同样 `[1, 152064]`。
注意 embedding 不是矩阵乘——是**查表**（第 108386 号 token 就取矩阵第 108386 列）。

---

## 3. 今天留下的钩子（明天从这里继续）

- `llama_context::decode`（llama-context.cpp L1680）内部长什么样？
- 计算图（ggml graph）是每轮重建还是复用？（提示：冒烟日志里那句 `graphs reused = 14`）
- K/V 是在哪一步写进 cache 的？

---

## 4. 自检（读完讲义现在就试）

1. 不看上文，画出 ①-⑥ 的调用链（文件名可以不精确，顺序和分工要对）
2. 用自己的话说：prefill 和 decode 在代码层面的唯一区别是什么？
3. `-n 128` 的 128 在代码里叫什么变量、在哪一行扣减？
