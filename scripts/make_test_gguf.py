#!/usr/bin/env python3
"""
生成一个手工合成的、极小的的gguf文件
"""

import struct
import argparse
import os


def write_string(buf: bytearray, s: str):
    b = s.encode("utf-8")
    buf.extend(struct.pack("<Q", len(b)))
    buf.extend(b)


def make_test_gguf(output_path: str):
    # Tensor data: 128 * 32 floats = 16384 bytes
    n_floats = 128 * 32
    tensor_data = bytes(4 * n_floats)  # all zeros for simplicity

    # Build header + info section in memory first to compute offsets
    info = bytearray()

    # Magic
    info.extend(b"GGUF")
    # Version
    info.extend(struct.pack("<I", 3))
    # n_tensors
    info.extend(struct.pack("<Q", 1))
    # n_kv
    info.extend(struct.pack("<Q", 3))

    # Metadata KV 1: general.architecture = "llama"
    write_string(info, "general.architecture")
    info.extend(struct.pack("<I", 8))  # GGUF_TYPE_STRING
    write_string(info, "llama")

    # Metadata KV 2: general.name = "tiny-test"
    write_string(info, "general.name")
    info.extend(struct.pack("<I", 8))  # GGUF_TYPE_STRING
    write_string(info, "tiny-test")

    # Metadata KV 3: general.alignment = 64
    write_string(info, "general.alignment")
    info.extend(struct.pack("<I", 4))  # GGUF_TYPE_UINT32
    info.extend(struct.pack("<I", 64))

    # Tensor info: token_embd.weight
    write_string(info, "token_embd.weight")
    info.extend(struct.pack("<I", 2))  # n_dims = 2
    info.extend(struct.pack("<Q", 128))  # dim[0]
    info.extend(struct.pack("<Q", 32))   # dim[1]
    info.extend(struct.pack("<I", 0))    # type = GGML_TYPE_F32
    info.extend(struct.pack("<Q", 0))    # offset = 0 (relative to data start)

    # Compute padding to align to general.alignment
    alignment = 64
    data_start = len(info)
    padding = (alignment - (data_start % alignment)) % alignment

    with open(output_path, "wb") as f:
        f.write(info)
        f.write(b"\x00" * padding)
        f.write(tensor_data)

    print(f"Wrote {output_path}")
    print(f"  info size: {data_start} bytes")
    print(f"  padding: {padding} bytes")
    print(f"  data size: {len(tensor_data)} bytes")
    print(f"  total: {data_start + padding + len(tensor_data)} bytes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="models/tiny/test.gguf")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    make_test_gguf(args.output)


if __name__ == "__main__":
    main()
