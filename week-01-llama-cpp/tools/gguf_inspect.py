#!/usr/bin/env python3
"""GGUF inspector — 把一个 GGUF 文件的元数据 + 所有张量都打印出来。

教学意图：让你亲眼看到 GGUF 不是黑盒，而是
  header (magic + version) → metadata KV → tensor table → tensor data
四段拼起来的二进制布局。

用法：
    python gguf_inspect.py path/to/model.gguf
    python gguf_inspect.py path/to/model.gguf --filter "attn"   # 只看 attn 张量
    python gguf_inspect.py path/to/model.gguf --summary          # 只看汇总
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

try:
    from gguf import GGUFReader
    from tabulate import tabulate
except ImportError:
    sys.stderr.write(
        "缺依赖：先 `source .venv/bin/activate` 再跑，或 `uv pip install gguf tabulate`\n"
    )
    sys.exit(1)


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path, help="GGUF 文件路径")
    p.add_argument("--filter", help="只显示名字含此关键字的张量")
    p.add_argument("--summary", action="store_true", help="只看 metadata + 汇总，不展开张量")
    args = p.parse_args()

    if not args.path.exists():
        sys.exit(f"找不到文件：{args.path}")

    print(f"\n📦 {args.path}  ({human_bytes(args.path.stat().st_size)})\n")
    reader = GGUFReader(args.path, "r")

    # ─── Metadata ────────────────────────────────────────────────
    print("─── METADATA ─────────────────────────────────────────")

    def read_value(field):
        """把一个 ReaderField 转成 Python 值。GGUF 里值可能是字符串、标量或数组。"""
        if not field.parts:
            return None
        # 多个 types 一般表示数组（第一个是 ARRAY 类型，后面是元素类型）
        if len(field.types) > 1:
            return f"<array, {len(field.data)} entries>"
        raw = field.parts[field.data[-1]]
        # 单字节序列 → UTF-8 字符串
        if hasattr(raw, "dtype") and str(raw.dtype) in ("uint8", "int8"):
            try:
                return bytes(raw).decode("utf-8")
            except UnicodeDecodeError:
                return list(raw)
        # 数值标量（numpy array of 1 element）
        if hasattr(raw, "tolist"):
            v = raw.tolist()
            return v[0] if isinstance(v, list) and len(v) == 1 else v
        return raw

    interesting = [
        "general.architecture", "general.name", "general.file_type",
        "general.quantization_version", "general.size_label",
    ]

    arch = None
    if "general.architecture" in reader.fields:
        arch = read_value(reader.fields["general.architecture"])
    if isinstance(arch, str):
        interesting += [
            f"{arch}.context_length",
            f"{arch}.embedding_length",
            f"{arch}.block_count",
            f"{arch}.attention.head_count",
            f"{arch}.attention.head_count_kv",
            f"{arch}.attention.layer_norm_rms_epsilon",
            f"{arch}.rope.freq_base",
            f"{arch}.feed_forward_length",
            f"{arch}.vocab_size",
        ]

    rows = []
    for key in interesting:
        if key not in reader.fields:
            continue
        try:
            v = read_value(reader.fields[key])
            rows.append([key, str(v)[:80]])
        except Exception as e:
            rows.append([key, f"<err: {e}>"])

    print(tabulate(rows, headers=["key", "value"], tablefmt="simple"))

    # ─── Tensors ─────────────────────────────────────────────────
    tensors = reader.tensors
    print(f"\n─── TENSORS ({len(tensors)} 个) ──────────────────────────")

    # 按 type 分类的汇总
    type_count: Counter = Counter()
    type_bytes: Counter = Counter()
    for t in tensors:
        type_count[t.tensor_type.name] += 1
        type_bytes[t.tensor_type.name] += t.n_bytes
    summary = sorted(
        ((name, type_count[name], human_bytes(type_bytes[name])) for name in type_count),
        key=lambda r: -type_bytes[r[0]],
    )
    print("\n按量化类型分布：")
    print(tabulate(summary, headers=["quant", "count", "size"], tablefmt="simple"))
    print(f"\n总张量字节：{human_bytes(sum(type_bytes.values()))}")

    if args.summary:
        return

    # 逐个张量
    flt = args.filter
    print(f"\n张量列表{'（filter: ' + flt + '）' if flt else ''}：")
    rows = []
    for t in tensors:
        if flt and flt not in t.name:
            continue
        shape = " × ".join(str(s) for s in reversed(t.shape))  # gguf 是 little-endian shape
        rows.append([t.name, shape, t.tensor_type.name, human_bytes(t.n_bytes)])
    print(tabulate(rows, headers=["name", "shape", "type", "bytes"], tablefmt="simple"))

    # ─── 解读提示 ────────────────────────────────────────────────
    print("\n💡 看完想想：")
    print("  • 哪些张量是每层一份？哪些只有一份？")
    print("  • token_embd 和 output 的形状有什么关系？(tied embedding?)")
    print("  • attn_q.weight 的形状 = (hidden, head_count × head_dim)，验证一下")
    print("  • 为什么 norm 类张量都是 F32 而不是 Q4_K？")


if __name__ == "__main__":
    main()
