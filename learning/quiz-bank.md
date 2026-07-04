# 验收题库 Quiz Bank

> 规则：由 AI（Claude/Codex）口头考，答不上 = 未验收，记入 ledger 欠账区。
> 白板题 = 不看资料、不跑代码，讲清楚为止。
> 每题标注对应框架图节点，答错时回图找前置。

## Week 01-02

### W1-Q1 GGUF 三段布局 〔全景 4〕
一个 .gguf 文件从第 0 个字节开始依次是什么？为什么残缺文件也能读出全部元数据？
（追问：alignment 是干嘛的？tensor offset 是相对谁的？）

### W1-Q2 BPE 直觉 〔全景 4 · 数学 3.1〕
为什么"一个汉字=一个 token"是错的？举一个你实验里中文压缩比的真实数字。
（追问：为什么 LLM 按 token 计费而不是字符？）

### W1-Q3 推理主循环六步 〔全景 4 · 数学 1.4〕
白板画出：tokenize → embed → N×(...) → unembed → sample，每步张量形状标出来（用 Qwen2.5-7B 的真实数字：3584/28层/152064）。

### W1-Q4 attention 内部四步 〔全景 4 · 数学 1.6〕
QKV 投影 → RoPE → masked softmax → out 投影，写出 GQA 下 Q 和 K/V 的形状差异（28 头 vs 4 头）。
（追问：RoPE 为什么高中三角函数就够了？）

### W1-Q5 KV cache 那笔账 〔数学 5.3〕
现场算：Qwen2.5-7B @ seq=2048 FP16 的 KV cache 多少 MB？没有 GQA 是多少？100 并发是多少？
（标准答案：117MB / 820MB / 11.7GB）

### W2-Q1 decode 速度上限 〔数学 6.3〕
decode t/s 的理论上限公式是什么？用 M5 Pro（~150GB/s）+ 4.4GB 模型算出理论值，和你实测的 25 t/s 比，差值可能来自哪？

### W2-Q2 两副面孔 〔全景 4 · 数学 6.2〕
prefill 和 decode 谁 compute-bound 谁 memory-bound？用算术强度解释（不许背结论，要推）。
（追问：batch=8 的 decode 算术强度变几倍？这解释了什么机制的存在？）

### W2-Q3 perplexity 人话 〔数学 3.3〕
PPL=8 是什么意思？为什么量化实验用 PPL 当质量标尺？
（追问：交叉熵和 PPL 的关系式？）

### W2-Q4 temperature 数学行为 〔数学 3.2〕
T→0 和 T→∞ 时 softmax(x/T) 各退化成什么？top-k 和 top-p 本质区别一句话？

## 面试题精选 · Week 01-02（llama.cpp / 推理基础）

> 规则：每周主题配 5 道该领域高频面试题（AI Infra 推理方向），比验收题更接近真实面试的问法。
> 验收通过后再刷这 5 题 = 把"学会了"升级成"面试能打"。

### 面试-W1/2-1 讲讲 LLM 推理为什么分 prefill 和 decode 两个阶段？各自的瓶颈是什么？
关联：W2-Q2 + Day 5 讲义（分叉只在 batch 装几个 token）
追问方向：TTFT/TPOT 分别由哪个阶段决定？chunked prefill 解决什么问题？

### 面试-W1/2-2 KV cache 是什么？不用它会怎样？它的显存怎么估算？
关联：W1-Q5（117MB 那笔账现场算）
追问方向：GQA/MQA 怎么省 KV cache？长上下文下 KV cache 和权重谁先成为瓶颈？

### 面试-W1/2-3 Q4_K_M 量化是怎么回事？为什么量化能加速推理？质量损失怎么评估？
关联：Day 3-4 GGUF 拆解（Q4_K/Q6_K/F32 混合分布）+ W2-Q3（PPL）
追问方向：为什么 norm 层保 F32？权重量化和 KV cache 量化的区别？

### 面试-W1/2-4 给你一台 192GB 统一内存的 Mac 和一张 24GB 4090，各适合跑什么推理场景？
关联：W2-Q1（带宽上限公式）+ 数学图 6.3 心算公式
追问方向：统一内存 vs 独立显存的取舍？batch 场景下谁赢？

### 面试-W1/2-5 从你敲下 prompt 到看见第一个字，中间发生了什么？（系统设计式开放题）
关联：W1-Q3 主循环六步 + Day 5 调用链笔记
追问方向：面试官会顺着你的回答往深挖——tokenize 细节、embedding 查表、采样策略任选一路；答题时主动把形状数字（3584/152064）带出来是加分项

## Week 03-04（预告，到时细化）

### W3-Q1 online softmax
数据分批到来时如何增量维护 max 和分母？为什么这是 FlashAttention 的核心？

### W3-Q2 分页的账
PagedAttention 的页大小权衡：页大了浪费什么，页小了浪费什么？

### W3-Q3 为什么 vLLM 必须存在
用 W1-Q5 和 W2-Q2 的结论，30 秒推出 PagedAttention + continuous batching 的必然性。
