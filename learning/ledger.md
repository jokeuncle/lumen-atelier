# 学习台账 Ledger

> 单一事实源。每次学习后追加一行，**没有证据链接的条目不算数**。
> 信心自评：1=完全不懂 2=听过 3=能复述 4=能推导/能改代码 5=能教别人
> 此文件由每日 digest 读取判断进度，请保持格式。

## 格式

```
| 日期 | 课程日 | 做了什么 | 证据 | 信心 | 验收状态 |
```

## 记录

| 日期 | 课程日 | 做了什么 | 证据 | 信心 | 验收状态 |
|---|---|---|---|---|---|
| 2026-06-18 | Day 1-2 | 环境搭建 + 模型下载启动 | commit f9093ce · setup.sh 幂等可重跑 | 4 | - |
| 2026-07-04 | Day 3-4 | 模型续传补全 4.4GB · llama-completion 冒烟通过（prefill 110-257 t/s / decode 25 t/s）· gguf_inspect + 手写 gguf_handparse.py 拆完 GGUF（GQA 28:4 · 非 tied embedding · Q4_K/Q6_K/F32 分布） | tools/gguf_handparse.py · 全景图+数学图两篇博客已发布 | 3 | W1-Q1 ✅（经提示） |
| 2026-07-04 | Day 5 | 追通 main→decode 调用链（讲义 Claude 代产）。晚间消化：拼车/KV cache 两个检查问题自主答对；验收 W1-Q3 部分通过、W1-Q1 不通过 | week-01-llama-cpp/notes/day5-call-chain.md · 本次验收记录 | 2 | W1-Q3 🟡 · W1-Q1 ✅（7/6 经提示） |

## 欠账区（跳过待补）

| 记入日期 | 欠什么 | 薄弱点 | 还账方式 |
|---|---|---|---|
| 2026-07-04 | W1-Q3（部分） | tokenize 的词表 vs embedding 的权重表分不清（"先查字典拿号，再凭号取衣服"） | 下次验收复问追问 1，脱口而出即销账 |

## 验收通过记录

| 日期 | 题号 | 评语 |
|---|---|---|
| 2026-07-06 | W1-Q1 | 经提示通过：能说清 header/metadata/tensor info/tensor data 顺序、metadata 前置原因、alignment 作用和 tensor offset 相对 tensor data 区。 |
