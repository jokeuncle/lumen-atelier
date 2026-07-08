---
title: "BPE 不是一个汉字一个 Token：从 Qwen tokenizer 实验看懂 token 计费"
date: 2026-07-08
source: "Lumen Atelier Day 7 tokenizer experiment"
---

# BPE 不是一个汉字一个 Token：从 Qwen tokenizer 实验看懂 token 计费

今天的实验只想回答两个问题：

```text
1. 为什么“一个汉字 = 一个 token”是错的？
2. 为什么 LLM 按 token 计费，而不是按字符计费？
```

这两个问题看起来像产品计费规则，其实是模型输入层的工程事实。LLM 看不懂人类眼里的“字”“词”“emoji”。它真正吃进去的是一串 token id，然后用这些 id 去查 embedding 表，再进入 attention、KV cache、采样和生成循环。

所以这篇先不讲复杂算法，先用一组 Qwen2.5 tokenizer 的真实输出，把直觉打牢。

## 1. 人话直觉：token 是模型学出来的文本碎片

最容易误解的一句话是：

```text
中文一个字一个 token，英文一个词一个 token。
```

这句话只是偶尔看起来像对的。更准确的说法是：

**token 是 tokenizer 从大量文本里学出来的常见碎片。它可以是一个汉字、多个汉字、一个英文单词、半个英文单词、一个空格加单词，也可能是一个 emoji 的一部分字节。**

例如这次实验里：

```text
你好，今天我们继续学习 AI Infra。
```

被切成 9 个 token：

```text
"你好"
"，"
"今天我们"
"继续"
"学习"
" AI"
" Inf"
"ra"
"。"
```

这里有几个很反直觉的点：

```text
"你好"      不是 2 个 token，而是 1 个 token
"今天我们"  不是 4 个 token，而是 1 个 token
"Infra"     没有整体命中，而是被切成 " Inf" + "ra"
```

所以 token 不是“字符”。它更像 tokenizer 训练出来的一张切片表：常见片段尽量合并，不常见片段就拆小一点。

## 2. 最小地基：从字符到 UTF-8 字节，再到 vocab

先把三个概念分清：

```text
字符     人看到的单位，比如“你”、"A"、"🚀"
字节     计算机存储文本的单位，UTF-8 里一个字符可能占多个字节
token    tokenizer 输出的模型输入单位，每个 token 对应一个整数 id
```

英文字符通常很便宜：

```text
A
```

在 UTF-8 里是 1 个字节。

常见汉字通常是 3 个字节：

```text
你
```

在 UTF-8 里是 3 个字节。

emoji 往往更长：

```text
🚀
```

在 UTF-8 里是 4 个字节。

byte-level BPE 的关键点是：它不是直接从“人类字符”开始工作，而是先把文本转成 UTF-8 字节，再在字节序列上学习哪些片段经常一起出现。经常一起出现的片段可以合并成一个 token，不常见的片段就保留为更小的 token。

模型还会有一张词表，也就是 vocab。词表里每个 token 片段对应一个 id：

```text
token 片段 -> token id
```

模型真正收到的是 id 序列：

```text
[108386, 3837, 114854, ...]
```

后面的 embedding 层再把每个 id 查成向量。

这就是第一层地基：

```text
文本 -> tokenizer -> token id 序列 -> embedding 向量 -> transformer
```

## 3. BPE 到底在“合并”什么

BPE 的全名是 Byte Pair Encoding。名字里的 pair 可以先理解成“相邻片段”。

用一个很小的例子讲。假设训练语料里经常出现：

```text
今 天 我 们
今 天 我 们
今 天 我 们
```

一开始可以把它看成更细的片段：

```text
今 / 天 / 我 / 们
```

如果“今”和“天”总是挨着出现，tokenizer 训练时可能先把它合并成：

```text
今天 / 我 / 们
```

如果“今天”和“我们”也经常挨着出现，可能继续合并成：

```text
今天我们
```

真实训练比这个复杂得多，但核心直觉就是这件事：

**常见组合被合并，少见组合保留得更碎。**

所以中文里会出现一个 token 包住多个汉字；英文里也会出现一个 token 包住完整单词；代码里会出现 `def`、` return`、`(name` 这种带语法痕迹的片段；emoji 如果不够常见，可能被拆成几个字节 token。

## 4. 这次实验：四组文本的真实 token 数

实验脚本是：

```text
week-01-llama-cpp/tools/tokenizer_play.py
```

使用的 tokenizer 是：

```text
Qwen/Qwen2.5-7B-Instruct
```

脚本输出的 `AutoTokenizer.vocab_size` 是：

```text
151643
```

注意：后面读模型权重时可能看到 `output.weight` 有 152064 行。这通常是模型配置或权重矩阵为了对齐、padding、特殊 token 预留造成的口径差异，不要把 tokenizer 的 `vocab_size` 和模型输出矩阵行数粗暴当成同一个数字。

四组实测数据如下：

| 样本 | 字符数 | UTF-8 字节 | token 数 | BPE 切碎字符 | 压缩比 |
|---|---:|---:|---:|---:|---:|
| `你好，今天我们继续学习 AI Infra。` | 21 | 45 | 9 | 0 | 2.33 字符/token |
| `Today we continue learning AI infrastructure.` | 45 | 45 | 7 | 0 | 6.43 字符/token |
| `def hello(name): return f"hello {name}"` | 39 | 39 | 11 | 0 | 3.55 字符/token |
| `AI Infra 🚀🔥` | 11 | 17 | 7 | 1 | 1.57 字符/token |

这张表已经足够推翻“一个汉字一个 token”：

```text
中文样本：21 个字符 -> 9 个 token
```

如果真是一个汉字或字符一个 token，这里应该接近 21 个 token。但真实结果是 9 个，因为：

```text
"你好"      合成了 1 个 token
"今天我们"  合成了 1 个 token
"继续"      合成了 1 个 token
"学习"      合成了 1 个 token
```

同样，英文样本也不是“一个英文单词固定一个 token”的简单规则：

```text
Today we continue learning AI infrastructure.
```

45 个字符只用了 7 个 token。`infrastructure` 这种常见英文片段整体命中了一个 token，所以压缩比很高。

代码样本则体现了 tokenizer 会学到程序文本里的常见结构：

```text
def hello(name): return f"hello {name}"
```

它不是按字符切，也不是严格按单词切，而是出现了：

```text
"def"
" hello"
"(name"
"):"
" return"
" f"
"hello"
" {"
"name"
"}\""
```

这说明代码里的空格、括号、关键字和变量名也会影响切分方式。

emoji 样本最能说明 byte-level BPE 的底层事实：

```text
AI Infra 🚀🔥
```

这里 `🚀` 被切成了 3 个 token，而 `🔥` 命中了 1 个 token。也就是说，同样是 emoji，不同符号的 tokenizer 命运也不一样。常见程度、训练语料、字节组合都会影响结果。

## 5. 问题一：为什么“一个汉字 = 一个 token”是错的

现在可以把答案说严谨一点：

**因为 tokenizer 的单位不是汉字，而是从训练语料里学出来的 token 片段。一个 token 可以覆盖一个汉字、多个汉字，也可以只覆盖某个字符的一部分字节。**

用这次实验的真实数字说：

```text
"你好，今天我们继续学习 AI Infra。"

字符数：21
token 数：9
压缩比：2.33 字符/token
```

其中：

```text
"今天我们" = 1 个 token
```

这一个例子已经说明“一个汉字一个 token”不成立。

但反过来也不能说“中文总是多个汉字一个 token”。遇到生僻字、罕见符号、特殊组合时，中文字符也可能被拆得更碎。正确说法不是固定比例，而是：

```text
token 数取决于 tokenizer 词表和具体文本。
```

所以任何关于 token 成本的估算，都应该跑 tokenizer 或使用模型厂商提供的 token counter，而不是按字符数硬猜。

## 6. 问题二：为什么 LLM 按 token 计费，而不是按字符计费

因为模型的计算不是按字符发生的，而是按 token 序列发生的。

从推理流程看：

```text
文本
  -> tokenizer
  -> token id 序列
  -> embedding 查表
  -> N 层 transformer
  -> logits
  -> sample 下一个 token
```

从模型角度看，输入长度不是“多少个字符”，而是：

```text
多少个 token
```

上下文窗口也是 token 口径：

```text
最多放 4096 token
最多放 32768 token
最多放 128K token
```

不是最多放多少汉字或多少英文字符。

生成也是 token 口径。模型每一步通常生成一个新 token，然后把这个 token 接回上下文，再继续算下一步。你看到屏幕上出现的可能是一个字、一个词、半个词、一个标点，甚至某个 emoji 的一部分，但模型内部的生成步是 token。

计费按 token 而不是字符，至少有三个工程原因。

第一，embedding 查表按 token id 发生。

模型没有直接处理字符。它看到的是：

```text
token id 108386
token id 3837
token id 114854
```

每个 id 去 embedding 表里查一行向量。输入有多少 token，就要查多少次。

第二，attention 的序列长度按 token 算。

在 prefill 阶段，prompt 的 token 数决定了 attention 要处理多长的序列。序列越长，计算和 KV cache 写入都越多。

第三，decode 阶段每生成一个 token 都要跑一次生成循环。

输出 100 个 token，大体就意味着 decode 循环推进 100 步。至于这 100 个 token 最后在屏幕上显示成多少个汉字、英文词或符号，是 tokenizer decode 之后的人类可见结果，不是模型内部的计量单位。

所以按 token 计费更贴近真实成本：

```text
输入成本：prompt token 数
输出成本：generated token 数
上下文成本：KV cache 里保存的 token 数
```

字符只是人类读文本的单位，不是 transformer 算东西的单位。

## 7. 一个容易踩的坑：字符少不等于便宜

这次最典型的是：

```text
AI Infra 🚀🔥
```

它只有 11 个字符，却用了 7 个 token，压缩比只有：

```text
1.57 字符/token
```

因为 `🚀` 被拆成了 3 个 token。

反过来，英文句子：

```text
Today we continue learning AI infrastructure.
```

有 45 个字符，却只有 7 个 token，压缩比是：

```text
6.43 字符/token
```

所以“看起来短”不等于“token 少”。短文本里如果有罕见符号、emoji、生僻字、混合语言、奇怪格式，可能并不便宜。长一点的英文常见句子，因为很多片段被词表命中，反而 token 压缩比更高。

这也是为什么真实工程里要做 token 预算：

```text
系统提示词
用户输入
检索出来的上下文
工具返回结果
模型输出上限
```

这些都应该按 token 估算，而不是按字符数估算。

## 8. 和 llama.cpp 推理主线怎么接上

Day 5 和 Day 6 已经拆过主循环：

```text
tokenize -> embed -> N 层 transformer -> unembed -> sample
```

tokenizer 是第一步，但它会影响后面的每一步。

如果 prompt 是 9 个 token，embedding 阶段就是 9 个 token id 查表。如果 prompt 是 900 个 token，后面的 attention 就要处理 900 长度的序列，并把对应的 K/V 写入 KV cache。

因此 token 不是“前处理小细节”。它直接决定：

```text
prompt 能不能塞进上下文窗口
prefill 要算多长的序列
KV cache 要保存多少位置
decode 生成了多少步
账单按多少输入/输出 token 计费
```

这就是 tokenizer 必须亲手跑一次的原因。没有跑过，很多人会把“我看到的字数”和“模型实际处理的长度”混在一起。

## 9. 小结

今天这组实验可以压成三句话：

1. token 不是字符，也不是固定的词；它是 tokenizer 从语料里学出来的文本碎片。
2. “一个汉字 = 一个 token”是错的；这次中文样本是 21 字符变 9 token，`今天我们` 这种多个汉字可以合成一个 token。
3. LLM 按 token 计费，是因为 embedding、attention、KV cache、decode 生成循环都按 token 序列工作，而不是按人类字符工作。

最后给自己一个白板自测：

```text
为什么“一个汉字一个 token”是错的？
请用一个真实数字回答。

为什么 LLM 按 token 计费，不按字符计费？
请从 tokenize -> embed -> transformer -> sample 这条链路回答。
```

如果这两个问题能不用资料讲清楚，Day 7 的 tokenizer 直觉就算立住了。
