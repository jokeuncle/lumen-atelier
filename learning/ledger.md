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
| 2026-07-04 | Day 3-4 | 模型续传补全 4.4GB · llama-completion 冒烟通过（prefill 110-257 t/s / decode 25 t/s）· gguf_inspect + 手写 gguf_handparse.py 拆完 GGUF（GQA 28:4 · 非 tied embedding · Q4_K/Q6_K/F32 分布） | tools/gguf_handparse.py · 全景图+数学图两篇博客已发布 | 3 | W1-Q1 待验收 |
| 2026-07-04 | Day 5 | 追通 main→decode 调用链。**注：讲义由 Claude 代读源码产出，用户尚未消化** — 用户读讲义 + 通过 W1-Q3 验收后此条才算数 | week-01-llama-cpp/notes/day5-call-chain.md | 待定 | W1-Q3 未验收 |

## 欠账区（跳过待补）

（空）

## 验收通过记录

（空 —— 由验收人 Claude/Codex 在口头验收通过后填写，自己不能给自己签字）
