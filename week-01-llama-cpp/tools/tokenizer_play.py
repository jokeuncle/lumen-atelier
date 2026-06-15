#!/usr/bin/env python3
"""Tokenizer playground — 看看 Qwen 的 BPE tokenizer 把你输入的文本切成什么样。

教学意图：
  推理第一步是 tokenize。你以为的"一个汉字 = 一个 token"几乎都是错的，
  亲眼看看才知道为什么 prompt 长度计费、上下文窗口都按 token 算。

用法：
    python tokenizer_play.py "你的文本"
    python tokenizer_play.py             # 进入交互模式
    python tokenizer_play.py --model Qwen/Qwen2.5-7B-Instruct "你好"
"""
from __future__ import annotations

import argparse
import sys

import os
# 我们只用 tokenizer，不需要 PyTorch；屏蔽 transformers 启动时那条无害的 advisory
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    # 即使设了 env var，transformers 启动期还是会往 stderr 打那条消息——临时拦掉
    import io as _io
    _saved_stderr = sys.stderr
    sys.stderr = _io.StringIO()
    try:
        from transformers import AutoTokenizer
    finally:
        sys.stderr = _saved_stderr
    from tabulate import tabulate
except ImportError:
    sys.stderr.write(
        "缺依赖：先 `source .venv/bin/activate` 再跑，或 `uv pip install transformers tabulate`\n"
    )
    sys.exit(1)


# 终端配色循环，让相邻 token 颜色不同便于肉眼分段
COLORS = ["\033[48;5;24m", "\033[48;5;28m", "\033[48;5;94m", "\033[48;5;55m",
          "\033[48;5;130m", "\033[48;5;22m", "\033[48;5;52m", "\033[48;5;90m"]
RESET = "\033[0m"


def compute_char_groups(tok, ids):
    """把"单 token 解码就 UTF-8 残缺"的相邻 token 合成一组。
    返回 [(start_idx, end_idx_exclusive, decoded_text), ...]。
    这是 byte-level BPE 的本质：一个汉字可能被拆到多个 token。"""
    groups = []
    i = 0
    n = len(ids)
    while i < n:
        j = i + 1
        text = tok.decode([ids[i]])
        # � 是 Unicode 替换字符 (�)，说明这段字节序列不能单独解成完整 UTF-8
        while "�" in text and j < n:
            j += 1
            text = tok.decode(ids[i:j])
        groups.append((i, j, text))
        i = j
    return groups


def show(tok, text: str) -> None:
    ids = tok.encode(text, add_special_tokens=False)
    groups = compute_char_groups(tok, ids)

    # 1. 着色：每个色块 = 一个完整字符（可能跨多 token）
    print("\n色块视图（每个色块 = 一个完整字符；[N] 表示该字符由 N 个 token 拼成）：")
    print("  ", end="")
    for gi, (start, end, gtext) in enumerate(groups):
        color = COLORS[gi % len(COLORS)]
        visible = gtext.replace("\n", "↵").replace("\t", "→")
        marker = f"\033[1;33m[{end-start}]\033[0m{color}" if end - start > 1 else ""
        print(f"{color} {visible}{marker} {RESET}", end="")
    print()

    # 2. 表格：逐 token 真实样貌
    print(f"\n切成 {len(ids)} 个 token：")

    # 索引 → (groupIdx, partIdx, span, groupText)
    info = {}
    for gi, (start, end, gtext) in enumerate(groups):
        for k, idx in enumerate(range(start, end)):
            info[idx] = (gi, k, end - start, gtext)

    rows = []
    for i, tid in enumerate(ids):
        raw = tok.convert_ids_to_tokens([tid])[0]
        single = tok.decode([tid])
        _, k, span, gtext = info[i]
        if span == 1:
            decoded_col = f'"{single}"'
        else:
            # 多 token 拼出一个字：第一格显示拼完的字+(part/total)，其余只显示 (part/total)
            decoded_col = f'"{gtext}" ({k+1}/{span})' if k == 0 else f'      ({k+1}/{span})'
        rows.append([i, tid, raw, decoded_col])
    print(tabulate(rows, headers=["#", "id", "raw", "decoded"], tablefmt="simple"))

    # 3. 统计直觉
    chars = len(text)
    bytes_ = len(text.encode("utf-8"))
    multi = sum(1 for (s, e, _) in groups if e - s > 1)
    print(f"\n字符数: {chars}    UTF-8 字节: {bytes_}    token 数: {len(ids)}    "
          f"BPE 切碎字符: {multi}")
    if chars:
        print(f"压缩比：{chars/len(ids):.2f} 字符 / token   ·   {bytes_/len(ids):.2f} 字节 / token")
    if multi:
        print(f"\n💡 这次有 {multi} 个字符被 byte-level BPE 切成多 token。")
        print("   这是 Qwen / Llama / GPT 系 BPE 的共同特征：先转 UTF-8 字节，再在字节级做合并。")
        print("   罕用汉字、emoji、生僻符号经常吃这个亏 — 这是中文 prompt 比英文费 token 的原因之一。")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("text", nargs="?", help="要切的文本；不给就进入交互模式")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                   help="HF 上的 tokenizer 名（默认 Qwen2.5）")
    args = p.parse_args()

    print(f"加载 tokenizer: {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    print(f"✓ vocab_size = {tok.vocab_size}\n")

    if args.text:
        show(tok, args.text)
        return

    print("交互模式（空行或 Ctrl-D 退出）：")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        show(tok, line)


if __name__ == "__main__":
    main()
