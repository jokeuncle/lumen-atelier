# Lumen Atelier 课程表 · 两张框架图 → 90 天

> 本文件是每日 digest 的内容源。digest 生成器读它 + ledger.md 的进度状态，
> 生成"今天学什么"。每个学习日映射到框架图的具体节点。
>
> 图例：〔全景〕= ai-infra-mindmap.md 节点 · 〔数学〕= ai-infra-math-mindmap.md 节点
> 每日结构：主任务（≤45min 核心）+ 数学配菜（≤15min）+ 验收题引用（quiz-bank.md）

---

## 规划原则

1. **主线拉动数学**：不单独学数学，每天的数学配菜服务当天主任务
2. **连续性**：每天开头必须回答"昨天学了 X，所以今天学 Y"
3. **周节拍**：周一至周五推进，周六缓冲/补欠账，周日复盘+博客
4. **粒度**：W1-W2 按天排（当前进行中）；W3 起按周排，digest 生成器在周内自行细化，避免远期计划腐烂

---

## Week 01-02 · llama.cpp 推理流程（按天）

### 状态注记（2026-07-04）
Day 1-4 已完成：环境 ✓ 冒烟 ✓ GGUF 拆解 ✓（含手写解析器加分项）
从 Day 5 继续。

### Day 5 · llama.cpp 主循环源码（上）：从 main 到 decode
- 主任务：clone llama.cpp，读 tools/main/main.cpp 的 token 生成循环 + llama_decode 的入口路径，画出调用链
- 〔全景〕4.推理引擎.llama-cpp 〔数学〕1.4 GEMM 形状规则（读代码时验证每个矩阵乘）
- 数学配菜：1.1-1.3 向量/矩阵/矩阵乘向量（🟢🟡，读源码前热身 15 分钟）
- 验收：quiz W1-Q3（推理主循环六步）
- 证据：调用链笔记 push 到 week-01-llama-cpp/notes/

### Day 6 · llama.cpp 主循环源码（下）：ggml 计算图
- 主任务：读 build_qwen2 的图构建（找 RMSNorm→QKV→RoPE→attn→FFN 六步），对照 gguf_inspect 看到的张量名
- 〔全景〕2.编译栈.ggml计算图 〔数学〕1.5 张量与形状演算
- 数学配菜：1.6 RoPE=旋转矩阵（🟡，高中三角函数）
- 验收：quiz W1-Q4（attention 内部四步）
- 证据：六步各自的源码行号 + 形状标注

### Day 7 · tokenizer 实验日
- 主任务：tokenizer_play.py 跑中文/英文/代码/emoji 4 组例子，记录压缩比
- 〔全景〕4.推理引擎（tokenize 入口）〔数学〕3.1 概率分布（vocab=152064 的含义）
- 验收：quiz W1-Q2（BPE 直觉）
- 证据：4 组数据进 notes.md 第 3 节

### Day 8 · KV cache 账 + benchmark 开箱
- 主任务：手算 KV cache 账（117MB @2048），然后 llama-bench 首跑，验证 25 t/s 与带宽公式
- 〔全景〕4.核心机制.KV cache 〔数学〕5.3 KV cache 账 + 6.3 心算公式（本课程最重要的一天）
- 验收：quiz W1-Q5（KV cache 形状与增长）+ W2-Q1（decode 上限公式）
- 证据：计算过程 + bench 首份 CSV

### Day 9 · bench_sweep 全参数扫描
- 主任务：bench_sweep.sh 扫 batch/thread/ngl 组合，找出 M5 Pro 最优配置
- 〔全景〕8.benchmark 方法论 〔数学〕6.2 Roofline（解释扫描结果为什么长这样）
- 验收：quiz W2-Q2（prefill/decode 谁 compute-bound 为什么）
- 证据：reports/bench-*.csv + 一段"结果为什么长这样"的解读

### Day 10 · perplexity 质量评估
- 主任务：llama-perplexity 跑 wiki 语料，对比 Q4_K_M 的 PPL 基线
- 〔全景〕8.模型质量.perplexity 〔数学〕3.3 信息论三件套（跑之前先懂公式）
- 验收：quiz W2-Q3（PPL=8 是什么含义）
- 证据：PPL 数字 + 一句人话解释进 notes.md 第 7 节

### Day 11 · 采样实验日
- 主任务：--temp/--top-p/--top-k 组合实验，同 prompt 生成 5 组对比
- 〔全景〕4.采样与输出 〔数学〕3.2 采样的数学
- 验收：quiz W2-Q4（temperature 的数学行为）
- 证据：5 组输出对比进 notes.md 第 8 节

### Day 12 · 机动缓冲日
- 主任务：补前面欠的账（未验收的 quiz、没跑完的实验）；全绿则预读 vLLM PagedAttention 论文摘要
- 无新增节点

### Day 13-14 · 交付物冲刺
- 主任务：写完 notes.md 全文 ≥2000 字 + 2 张配图，发博客
- 验收：把 quiz W1-Q1 到 W2-Q4 全部口头过一遍（白板验收周）
- 证据：博客 URL 进 ledger

---

## Week 03-04 · vLLM PagedAttention（按周，digest 周内细化）

- 本周问题：单机 llama.cpp 的 KV cache 是连续大块，vLLM 为什么按页管理？
- 节点：〔全景〕4.核心机制.PagedAttention/连续批处理 · 2.核心算子.PagedAttention kernel
- 数学前置（周一必修）：〔数学〕2.4 online softmax（🔴 W3 门票）+ 5.4 分页的数学
- 实验载体：vLLM 源码（block_manager / scheduler）+ 论文
- 交付物：博客《PagedAttention：把操作系统的分页搬进 KV cache》

## Week 05-06 · 连续批处理与调度
- 节点：〔全景〕4.连续批处理/Chunked prefill · 5.SLO 与调度
- 数学：〔数学〕5.5 Little's Law · 6.2 batch 强度分析 · 6.4 MFU
- 交付物：本地复现 continuous batching 吞吐曲线 + 博客

## Week 07-08 · 量化深入 + 投机解码
- 节点：〔全景〕4.量化（GPTQ/AWQ）· 4.投机解码
- 数学：〔数学〕2.5 GPTQ/AWQ 数学 · 3.4 投机解码接受概率
- 交付物：Q4/Q5/Q8 PPL-速度权衡实测博客

## Week 09-10 · MLX 与 Metal（副线高峰）
- 节点：〔全景〕10.端侧.MLX · 2.内核编程.Metal
- 数学：〔数学〕6 全章复习（用 MLX 重算所有心算公式）
- 交付物：MLX 自定义算子一个 + 博客（差异化内容）

## Week 11 · 云端周（A100 冲刺）
- 节点：〔全景〕2.Triton · 3.并行策略（TP 概念）· 7.并行数学
- 数学：〔数学〕7 全章（Amdahl/allreduce/气泡率）
- 交付物：A100 vs M5 Pro 同模型 benchmark 对比博客

## Week 12 · 收官整合
- 全景图全图复盘：每个 ★ 节点口头验收
- 交付物：《90 天 AI Infra 学习总复盘》长文

---

## digest 生成器使用本文件的规则

1. 读 ledger.md 最后一条记录确定"上一个完成日"
2. 在本文件找到下一个未完成的学习日
3. 若 ledger 显示 >5 天未活动 → 今日任务替换为"重新入场微任务"（15 分钟：重跑上次的实验/重看上次的笔记）
4. 生成卡片必含五段：昨天回顾 → 今天节点（引用框架图路径）→ 主任务 ≤45min → 数学配菜 ≤15min → 今日验收题
5. 周日卡片替换为复盘模板引导
6. **周包检查**：若下一个学习日属于尚无 `week-NN-*/` 目录的周 → 卡片头部加醒目提醒："下周学习包未生成，请开 Claude Code 会话说『生成下周学习包』"（周包必须在会话中生成并跑通工具，不能无人值守生成）
