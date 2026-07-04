#!/usr/bin/env python3
"""手撕 GGUF —— 只用 struct，不依赖 gguf 库，从字节流里解析 header + metadata + tensor table。

教学意图：
  gguf.GGUFReader 帮你做了所有事，但也藏起了所有事。
  这里每一个 read 都对应 spec 里的一个字段，读完你就"拥有"这个格式了。
  额外收获：它能解析残缺文件——因为我们只读文件头部的三段，不碰 tensor data。

用法：
    python gguf_handparse.py path/to/model.gguf
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

# spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
VALUE_TYPES = {
    0: ("UINT8", "<B", 1), 1: ("INT8", "<b", 1),
    2: ("UINT16", "<H", 2), 3: ("INT16", "<h", 2),
    4: ("UINT32", "<I", 4), 5: ("INT32", "<i", 4),
    6: ("FLOAT32", "<f", 4), 7: ("BOOL", "<?", 1),
    8: ("STRING", None, None), 9: ("ARRAY", None, None),
    10: ("UINT64", "<Q", 8), 11: ("INT64", "<q", 8),
    12: ("FLOAT64", "<d", 8),
}

# ggml_type 枚举（节选常见的）
GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 30: "BF16",
}


class Cursor:
    """在 bytes 上顺序前进的读指针——手写解析器的全部状态就是一个 offset。"""

    def __init__(self, buf: bytes):
        self.buf = buf
        self.off = 0

    def read(self, fmt: str):
        (v,) = struct.unpack_from(fmt, self.buf, self.off)
        self.off += struct.calcsize(fmt)
        return v

    def read_string(self) -> str:
        n = self.read("<Q")          # 字符串 = u64 长度前缀 + UTF-8 字节，无 \0 结尾
        s = self.buf[self.off : self.off + n].decode("utf-8")
        self.off += n
        return s

    def read_value(self, vtype: int):
        name, fmt, _ = VALUE_TYPES[vtype]
        if name == "STRING":
            return self.read_string()
        if name == "ARRAY":
            elem_type = self.read("<I")
            count = self.read("<Q")
            elem_name = VALUE_TYPES[elem_type][0]
            if count > 8:            # 大数组（如 152k 词表）只看形状不展开
                # 仍要把字节跳过去，否则后面全错位
                for _ in range(count):
                    self.read_value(elem_type)
                return f"<array of {elem_name} × {count}>"
            return [self.read_value(elem_type) for _ in range(count)]
        return self.read(fmt)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if not path or not path.exists():
        sys.exit("用法：python gguf_handparse.py model.gguf")

    file_size = path.stat().st_size
    # 前三段（header+metadata+tensor table）在 7B 模型上 ~3MB，读 8MB 足够
    with open(path, "rb") as f:
        buf = f.read(8 * 1024 * 1024)
    c = Cursor(buf)

    # ── 1. header：固定 4 + 4 + 8 + 8 = 24 字节 ─────────────────────
    magic = bytes(c.buf[:4]); c.off = 4
    version = c.read("<I")
    n_tensors = c.read("<Q")
    n_kv = c.read("<Q")
    assert magic == b"GGUF", f"不是 GGUF 文件：magic={magic!r}"
    print(f"── HEADER (24 bytes) ─────────────────────────")
    print(f"magic={magic.decode()}  version={version}  n_tensors={n_tensors}  n_kv={n_kv}")

    # ── 2. metadata：n_kv 个 (key, type, value) ─────────────────────
    print(f"\n── METADATA ({n_kv} 对，起始 offset=24) ──────")
    alignment = 32                   # 默认值，可被 general.alignment 覆盖
    meta = {}
    for _ in range(n_kv):
        key = c.read_string()
        vtype = c.read("<I")
        val = c.read_value(vtype)
        meta[key] = val
        if key == "general.alignment":
            alignment = val
        print(f"  {key:52s} {VALUE_TYPES[vtype][0]:8s} {str(val)[:60]}")

    # ── 3. tensor table：n_tensors 个描述符（不含数据本体）───────────
    table_start = c.off
    tensors = []
    for _ in range(n_tensors):
        name = c.read_string()
        n_dims = c.read("<I")
        dims = [c.read("<Q") for _ in range(n_dims)]
        ggml_type = c.read("<I")
        rel_offset = c.read("<Q")    # 相对 data 段起点，不是相对文件头！
        tensors.append((name, dims, GGML_TYPES.get(ggml_type, f"?{ggml_type}"), rel_offset))

    print(f"\n── TENSOR TABLE ({n_tensors} 条，offset {table_start} → {c.off}) ──")
    for name, dims, ttype, off in tensors[:12]:
        print(f"  {name:28s} {str(dims):22s} {ttype:6s} @data+{off}")
    print(f"  ... 共 {n_tensors} 条")

    # ── 4. data 段起点 = tensor table 结束后按 alignment 对齐 ────────
    data_start = (c.off + alignment - 1) // alignment * alignment
    print(f"\n── 布局总结 ──────────────────────────────────")
    print(f"header+metadata+table 共 {c.off:,} bytes（{c.off/1024/1024:.2f} MB）")
    print(f"data 段从 {data_start:,} 开始（对齐到 {alignment}B）")
    print(f"当前文件 {file_size:,} bytes → data 段只有 {file_size - data_start:,} bytes")

    last_name, last_dims, last_type, last_off = tensors[-1]
    print(f"最后一个张量 {last_name} 在 data+{last_off:,} → 完整文件应远大于当前大小" if data_start + last_off > file_size else "文件看起来是完整的")


if __name__ == "__main__":
    main()
