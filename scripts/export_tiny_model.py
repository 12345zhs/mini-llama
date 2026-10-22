#!/usr/bin/env python3
"""
生成一个tiny模型,方便后面调试
"""

import json
import struct
import argparse
import os
import numpy as np


def export_model(config: dict, output_dir: str, seed: int = 42):
    np.random.seed(seed)

    vocab_size = config["vocab_size"]
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    n_kv_heads = config["n_kv_heads"]
    head_dim = dim // n_heads

    tensors = []

    def add_tensor(name: str, shape: tuple, data: np.ndarray):
        tensors.append((name, shape, data.astype(np.float32)))

    # Token embedding
    token_embedding = np.random.randn(vocab_size, dim).astype(np.float32) * 0.02
    add_tensor("token_embedding", (vocab_size, dim), token_embedding)

    # Layers
    for layer in range(n_layers):
        prefix = f"layers.{layer}."
        # attention_norm
        add_tensor(prefix + "attention_norm", (dim,), np.ones(dim, dtype=np.float32))
        # wq
        add_tensor(prefix + "wq", (n_heads * head_dim, dim),
                   np.random.randn(n_heads * head_dim, dim).astype(np.float32) * 0.02)
        # wk
        add_tensor(prefix + "wk", (n_kv_heads * head_dim, dim),
                   np.random.randn(n_kv_heads * head_dim, dim).astype(np.float32) * 0.02)
        # wv
        add_tensor(prefix + "wv", (n_kv_heads * head_dim, dim),
                   np.random.randn(n_kv_heads * head_dim, dim).astype(np.float32) * 0.02)
        # wo
        add_tensor(prefix + "wo", (dim, n_heads * head_dim),
                   np.random.randn(dim, n_heads * head_dim).astype(np.float32) * 0.02)
        # ffn_norm
        add_tensor(prefix + "ffn_norm", (dim,), np.ones(dim, dtype=np.float32))
        # w_gate
        add_tensor(prefix + "w_gate", (hidden_dim, dim),
                   np.random.randn(hidden_dim, dim).astype(np.float32) * 0.02)
        # w_up
        add_tensor(prefix + "w_up", (hidden_dim, dim),
                   np.random.randn(hidden_dim, dim).astype(np.float32) * 0.02)
        # w_down
        add_tensor(prefix + "w_down", (dim, hidden_dim),
                   np.random.randn(dim, hidden_dim).astype(np.float32) * 0.02)

    # final_norm
    add_tensor("final_norm", (dim,), np.ones(dim, dtype=np.float32))
    # lm_head
    add_tensor("lm_head", (vocab_size, dim),
               np.random.randn(vocab_size, dim).astype(np.float32) * 0.02)

    # Compute offsets and byte sizes
    offset = 0
    tensor_metadata = []
    for name, shape, data in tensors:
        byte_size = data.nbytes
        tensor_metadata.append({
            "name": name,
            "shape": list(shape),
            "dtype": "float32",
            "offset": offset,
            "byte_size": byte_size,
        })
        offset += byte_size

    # Write JSON config + tensor metadata
    json_data = {
        "config": config,
        "tokenizer": {
            "type": "json_vocab",
            "path": "vocab.json",
            "bos_id": 1,
            "eos_id": 2,
            "unk_id": 0,
        },
        "tensors": tensor_metadata,
    }

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "model.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Wrote {json_path}")

    # Write binary weights
    bin_path = os.path.join(output_dir, "model.bin")
    with open(bin_path, "wb") as f:
        for name, shape, data in tensors:
            f.write(data.tobytes())
    print(f"Wrote {bin_path}")

    # Print total size
    total_params = sum(np.prod(shape) for _, shape, _ in tensors)
    print(f"Total parameters: {total_params}")
    print(f"Binary size: {os.path.getsize(bin_path)} bytes")


def main():
    parser = argparse.ArgumentParser(description="Export tiny LLaMA model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="models/tiny", help="Output directory")
    args = parser.parse_args()

    config = {
        "vocab_size": 128,
        "dim": 32,
        "hidden_dim": 86,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 4,
        "max_seq_len": 128,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
    }

    export_model(config, args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
