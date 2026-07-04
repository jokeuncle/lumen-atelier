# AI Infra 全景图

## 1. 硬件层 Hardware

### 计算芯片
- NVIDIA GPU（Hopper H100/H200 · Blackwell B200 · 消费卡 4090/5090）
- 国产芯片（昇腾 910B · 寒武纪 · 摩尔线程）
- Google TPU（v5p/v6e · SparseCore）
- AMD（MI300X · ROCm 生态）
- Apple Silicon（M 系列 · 统一内存 · AMX/ANE）★90天副线
- 推理专用（Groq LPU · Cerebras · SambaNova）

### 存储层级
- SRAM / L2 cache（FlashAttention 存在的理由）
- HBM 带宽（decode 瓶颈的物理根源）★W1 正在体感
- 统一内存 vs 独立显存
- NVMe offload（权重/KV 换出）

### 互联
- 节点内：NVLink / NVSwitch · PCIe
- 节点间：InfiniBand · RoCE · 以太网
- 拓扑感知（ring vs tree allreduce）

## 2. 算子与编译层 Kernels & Compilers

### 内核编程
- CUDA（SM 架构 · warp · shared memory · Tensor Core / WGMMA）
- Triton（Python 写 kernel · vLLM/SGLang 的主力）★90天云线
- Metal / MPS（Apple GPU 内核）★90天副线
- CUTLASS / CuTe（模板化 GEMM）

### 核心算子
- GEMM（一切的地基 · roofline 模型）
- FlashAttention v1/v2/v3（tiling + online softmax）
- PagedAttention kernel ★W3-W4
- 融合算子（RMSNorm+residual · SwiGLU fused）
- MoE 算子（grouped GEMM · expert 路由）

### 编译栈
- torch.compile（Dynamo + Inductor）
- XLA / TVM / MLIR
- CUDA Graph（消 decode 阶段 launch 开销）
- ggml 计算图 ★W1 正在读

## 3. 训练基础设施 Training Infra

### 并行策略
- 数据并行 DP · ZeRO-1/2/3 / FSDP
- 张量并行 TP（Megatron 切法）
- 流水并行 PP（1F1B · interleaved）
- 序列/上下文并行 SP/CP（RingAttention）
- 专家并行 EP（MoE）
- 混合并行（3D/4D 组合与自动搜索）

### 训练效率
- 混合精度（BF16 · FP8 训练）
- 梯度检查点 / 重计算
- MFU（Model FLOPs Utilization 指标）
- 通信计算重叠

### 容错与规模
- checkpoint 保存/恢复（异步 · 分片）
- 硬件故障检测（straggler · ECC error）
- 万卡集群实践（Llama 3 / DeepSeek 技术报告）

## 4. 推理基础设施 Inference Infra ★90天主线

### 推理引擎
- vLLM（学术+工业标准）★W3-W6
- SGLang（RadixAttention · 结构化输出）★W7+
- TensorRT-LLM（NV 官方极致优化）
- llama.cpp / ggml（端侧 · 可读性最好的教材）★W1-W2
- MLX（Apple 官方 ML 框架）★90天副线

### 核心机制
- Prefill vs Decode（compute vs bandwidth bound）★W1
- KV cache 管理（PagedAttention · prefix caching · RadixAttention)
- 连续批处理 Continuous Batching
- Chunked prefill · P/D 分离部署（Mooncake · DistServe）
- 投机解码（draft model · Medusa · EAGLE）
- 长上下文（YaRN · KV 压缩 · sliding window）

### 量化
- 权重量化（GPTQ · AWQ · GGUF K-quants）★W1 已拆
- 激活量化（SmoothQuant · FP8 · W8A8）
- KV cache 量化（FP8/INT4 KV）
- 2-bit 前沿（AQLM · QuIP#）

### 采样与输出
- 采样策略（temp · top-k/p · min-p · 惩罚项）★W1
- 结构化输出（JSON schema · 语法约束 FSM）
- logits processor 流水线

## 5. 服务与调度层 Serving & Scheduling

### 服务形态
- 推理网关（路由 · 限流 · 计费）
- 多模型/多租户（LoRA 热插拔 · S-LoRA）
- 弹性伸缩（冷启动 · 模型预热）
- Serverless GPU（Modal · RunPod · 冷启动优化）

### SLO 与调度
- 指标定义：TTFT · TPOT/ITL · goodput
- 请求调度（优先级 · 抢占 · fairness）
- 前缀感知路由（cache hit 最大化）

## 6. 数据基础设施 Data Infra

### 训练数据管线
- 大规模清洗去重（MinHash · 质量分类器)
- tokenization at scale
- 数据配比与课程（data mixture）

### 检索基础设施
- 向量数据库（HNSW · IVF-PQ · Milvus/Qdrant/pgvector）
- Embedding 服务（批处理 · 缓存）
- RAG 管线（chunking · rerank · hybrid search）

## 7. 后训练基础设施 Post-training Infra

- SFT 框架（TRL · Axolotl · LLaMA-Factory）
- RLHF/RLVR 系统（PPO/GRPO 的 rollout+train 双引擎架构 · verl · OpenRLHF）
- 偏好优化（DPO 及变体）
- rollout 加速（推理引擎复用 · 异步采样）
- LoRA/QLoRA（低资源微调）

## 8. 可观测性与评估 Observability & Eval

### 性能剖析
- GPU 侧：Nsight Systems/Compute · torch.profiler
- 服务侧：Prometheus + Grafana · OpenTelemetry
- benchmark 方法论（llama-bench · vLLM bench serving）★W2

### 模型质量
- perplexity ★W2
- eval harness（lm-eval · 自建评估集）
- 量化质量回归检测

## 9. 集群与编排 Cluster & Orchestration

- Kubernetes（device plugin · topology aware）
- Ray（vLLM 分布式的底座）
- Slurm（HPC 传统 · 科研集群标配）
- GPU 共享（MIG · time-slicing · vGPU）
- 存储（对象存储拉权重 · 分布式缓存加速加载）

## 10. 端侧与边缘 On-device ★90天副线

- llama.cpp 生态（GGUF · Metal/Vulkan 后端）★W1-W2
- MLX（unified memory 优势 · 自定义算子）
- CoreML / ANE · Android NNAPI / QNN
- WebGPU（浏览器内推理 · WebLLM）
- 端云协同（小模型路由 · 投机解码端云分工）

## 11. 成本工程 Cost Engineering

- 单位经济学（$/1M tokens 的拆解）
- 硬件选型（H100 vs 4090 vs Mac 的场景边界）
- Spot 实例与抢占容错
- Prompt caching / batch API（应用侧省钱）
- 量化-质量-成本三角 ★W2 perplexity 实验

## 12. 前沿方向 Frontier

- MoE 推理（DeepSeek-V3 架构 · expert offload）
- 多模态 serving（vision encoder 与 LLM 的流水协同）
- Agent infra（KV 复用跨请求 · sandbox 执行环境 · 长程记忆）
- diffusion LM / 并行解码
- 训推一体（RL 时代训练与推理引擎融合）
