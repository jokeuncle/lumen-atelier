# llama.cpp 一次 forward 都做了什么

> Week 01 交付物 · 目标 ≥ 2000 字 · 配图至少 2 张
> 这是你的，不是我的。下面的每一节是骨架，自己用学到的填满。

---

## 0. 写在前面：为什么从 llama.cpp 开始

<!-- 50–150 字 -->
- 不从 transformers 开始的理由（black box / overhead）
- 不从 vLLM 开始的理由（一上来就太多调度抽象）
- llama.cpp 的甜点：可读、可跑、可改、能上 M5 Pro

---

## 1. 从一行命令说起

<!-- 200–400 字 -->
- 贴你 `llama-cli -m ... -p "..."` 的真实命令
- 解释每个参数（-m / -p / -n / -ngl / --temp / --top-p）
- 末尾的 `prompt eval time` 与 `eval time` 数字分别是什么 → 引出 prefill / decode 两阶段

**配图建议**：截一张 llama-cli 输出末尾性能统计。

---

## 2. GGUF：一个 .gguf 文件里有什么

<!-- 300–500 字 -->
- 文件布局图：header (magic+version) → metadata KV → tensor table → tensor data
- 用 `gguf_inspect.py` 实际看到的内容
- 关键 metadata：architecture / context_length / block_count / head_count / head_count_kv
- 关键张量：token_embd / blk.N.attn_q / blk.N.attn_k / blk.N.attn_v / blk.N.attn_output / blk.N.ffn_gate / blk.N.ffn_up / blk.N.ffn_down / output
- **GQA 的痕迹**：head_count vs head_count_kv 的差异，attn_k/v 形状如何小一圈

**配图建议**：自己画一张 GGUF 文件分层图。

---

## 3. Tokenizer：从文本到 ID

<!-- 200–300 字 -->
- BPE 简述（不展开算法，讲直觉）
- 你用 `tokenizer_play.py` 测的 3-5 个例子（中文 / 英文 / 代码 / emoji）的压缩比
- 为什么 LLM 用 token 计费而不是字符
- 一个 token 的"语义大小"是上下文相关的

### 3.1 tokenizer_play 实测数据

| 样本 | 字符数 | UTF-8 字节 | token 数 | BPE 切碎字符 | 压缩比 |
|---|---:|---:|---:|---:|---:|
| `你好，今天我们继续学习 AI Infra。` | 21 | 45 | 9 | 0 | 2.33 字符/token |
| `Today we continue learning AI infrastructure.` | 45 | 45 | 7 | 0 | 6.43 字符/token |
| `def hello(name): return f"hello {name}"` | 39 | 39 | 11 | 0 | 3.55 字符/token |
| `AI Infra 🚀🔥` | 11 | 17 | 7 | 1 | 1.57 字符/token |

备注：`AutoTokenizer.vocab_size` 输出为 151643；模型配置/输出矩阵里的 vocab 行数可能是 152064，后面写 `output.weight` 时要区分这两个口径。

---

## 4. Forward Pass：张量从输入到输出

<!-- 600–800 字，本节是重点 -->

### 4.1 Embed

- input_ids: `[batch, seq]` → embed → `[batch, seq, hidden]`
- Qwen2.5-7B 的 hidden = ___（填上你查到的值）

### 4.2 Decoder Layer × N

把一层拆成 6 步写：

1. **RMSNorm 1**：`[B, S, H]` → `[B, S, H]`
2. **QKV 投影**：
   - Q: `[B, S, H]` × `[H, n_heads × head_dim]` → `[B, S, n_heads, head_dim]`
   - K: `[B, S, H]` × `[H, n_kv_heads × head_dim]` → `[B, S, n_kv_heads, head_dim]`（GQA）
   - V 同 K
3. **RoPE**：在 Q 与 K 上施加旋转位置编码
4. **Attention**：
   - K/V 写入 KV cache
   - softmax((Q @ K^T) / sqrt(d) + mask) @ V → `[B, S, H]`
5. **RMSNorm 2 → FFN（SwiGLU）**：
   - gate = SiLU(x @ W_gate)
   - up = x @ W_up
   - down = (gate * up) @ W_down
6. **残差连接**包住整层

**配图建议**：把上面 6 步画成 box diagram。

### 4.3 Unembed + Sample

- 最后一个 token 的 hidden state → output 矩阵 → logits: `[B, vocab_size]`
- 采样得到下一个 token
- 是否 tied embedding？验证 token_embd 与 output 的关系

---

## 5. Prefill vs Decode：同一个 forward 的两副面孔

<!-- 400–500 字，本节决定深度 -->

|  | Prefill | Decode |
|---|---|---|
| 输入 seq 长度 | 整个 prompt（如 512） | 1 |
| KV cache | 一次性写入 prompt 的 K/V | 每步追加 1 个位置 |
| 算力使用 | 高（大矩阵乘） | 低（向量乘） |
| 瓶颈 | compute-bound | memory-bandwidth-bound |
| 指标 | TTFT | TPOT / ITL |

- 为什么 decode 慢？因为每步要把全部 weight 过一遍但只算 1 个 token —— 算力闲置，带宽吃满
- 这就是 vLLM continuous batching 的存在理由：用 batch 把 decode 的带宽吃满
- 这就是投机解码的存在理由：用 draft 多算几个 token 把算力填上

---

## 6. KV Cache：为什么不能没有它

<!-- 250–400 字 -->
- 如果没有 KV cache，每个新 token 都要重算之前所有位置的 K/V，O(n²) 复杂度
- 有了 KV cache，decode 阶段每步只算 1 个位置的 K/V，O(n) 复杂度
- KV cache 形状：`[layer, batch, n_kv_heads, seq, head_dim]`
- 算一笔账：Qwen2.5-7B 在 seq=2048 下 KV cache 占多少显存？
  - layer × 2 (K+V) × batch × n_kv_heads × seq × head_dim × 2 (fp16)
  - 算出来填这里：___ MB
- 这个数字 × 并发数 = 为什么 PagedAttention 必须存在

---

## 7. 量化：Q4_K_M 到底是什么

<!-- 200–300 字 -->
- Q4_K_M ≈ 4.5 bits per weight，含 super-block 元数据
- 反量化代价：每次访问 weight 都要解码（但 memory-bound 算子值得）
- Q4 vs Q5 vs Q8 的 PPL 退化（贴你 perplexity 实验的数据）
- 为什么 RMSNorm 这种张量保留 F32（数值稳定性）

---

## 8. 采样：从 logits 到 token

<!-- 150–250 字 -->
- 贪心 vs temperature vs top-k vs top-p 的物理含义
- 贴你 W3 第 11 项实验的 3-5 段输出对比

---

## 9. 我学到的最反直觉的一件事

<!-- 100–200 字 -->
留一节给自己。这周做完，写一个让你"啊原来是这样"的瞬间。这是博客最值钱的部分。

---

## 10. 下周预告

<!-- 50–100 字 -->
W3-W4 我要开始读 vLLM 的 PagedAttention —— 这周建立的张量直觉就是为了下周看懂 vLLM 为什么这么设计 KV cache 管理。

---

## 附录 · 我的实验数据

- `reports/bench-*.csv` 的关键数字
- gguf_inspect 输出的张量总览
- tokenizer_play 的 3-5 个例子结果

---

_完成后：贴个人博客 / 知乎专栏 / GitHub Gist，把链接放进 lab.html 的对应实验里。_
