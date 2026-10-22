#!/usr/bin/env python3
"""
把.gguf中的tokenizer metadata导出vocab.json  merges.txt  special_tokens.json 
再复用BPETokenizer
"""

import argparse
import json
import os
import struct
import sys


def read_gguf_string(f):
    slen = struct.unpack('<Q', f.read(8))[0]
    return f.read(slen).decode('utf-8', errors='replace')


def skip_gguf_value(f, vtype):
    if vtype == 0 or vtype == 1:  # uint8/int8
        f.read(1)
    elif vtype == 2 or vtype == 3:  # uint16/int16
        f.read(2)
    elif vtype == 4 or vtype == 5 or vtype == 6:  # uint32/int32/float32
        f.read(4)
    elif vtype == 7:  # bool
        f.read(1)
    elif vtype == 8:  # string
        read_gguf_string(f)
    elif vtype == 9:  # array
        etype = struct.unpack('<I', f.read(4))[0]
        alen = struct.unpack('<Q', f.read(8))[0]
        for _ in range(alen):
            skip_gguf_value(f, etype)
    elif vtype == 10 or vtype == 11 or vtype == 12:  # uint64/int64/float64
        f.read(8)


def read_gguf_metadata(path):
    with open(path, 'rb') as f:
        f.read(4)  # magic
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_metadata = struct.unpack('<Q', f.read(8))[0]

        tokens = None
        merges = None
        token_types = None
        bos_id = None
        eos_id = None
        unk_id = None
        pad_id = None
        chat_template = None

        for _ in range(n_metadata):
            key = read_gguf_string(f)
            vtype = struct.unpack('<I', f.read(4))[0]

            if key == 'tokenizer.ggml.tokens' and vtype == 9:
                etype = struct.unpack('<I', f.read(4))[0]
                alen = struct.unpack('<Q', f.read(8))[0]
                tokens = [read_gguf_string(f) for _ in range(alen)]
            elif key == 'tokenizer.ggml.merges' and vtype == 9:
                etype = struct.unpack('<I', f.read(4))[0]
                alen = struct.unpack('<Q', f.read(8))[0]
                merges = [read_gguf_string(f) for _ in range(alen)]
            elif key == 'tokenizer.ggml.token_type' and vtype == 9:
                etype = struct.unpack('<I', f.read(4))[0]
                alen = struct.unpack('<Q', f.read(8))[0]
                token_types = [struct.unpack('<i', f.read(4))[0] for _ in range(alen)]
            elif key == 'tokenizer.ggml.bos_token_id' and vtype == 4:
                bos_id = struct.unpack('<I', f.read(4))[0]
            elif key == 'tokenizer.ggml.eos_token_id' and vtype == 4:
                eos_id = struct.unpack('<I', f.read(4))[0]
            elif key == 'tokenizer.ggml.unknown_token_id' and vtype == 4:
                unk_id = struct.unpack('<I', f.read(4))[0]
            elif key == 'tokenizer.ggml.padding_token_id' and vtype == 4:
                pad_id = struct.unpack('<I', f.read(4))[0]
            elif key == 'tokenizer.chat_template' and vtype == 8:
                chat_template = read_gguf_string(f)
            else:
                skip_gguf_value(f, vtype)

    return {
        'tokens': tokens,
        'merges': merges,
        'token_types': token_types,
        'bos_id': bos_id,
        'eos_id': eos_id,
        'unk_id': unk_id,
        'pad_id': pad_id,
        'chat_template': chat_template,
    }


def main():
    parser = argparse.ArgumentParser(description="Export GGUF tokenizer data to JSON")
    parser.add_argument("gguf", help="Input GGUF file")
    parser.add_argument("--out-dir", default="models/chat", help="Output directory")
    args = parser.parse_args()

    print(f"Reading tokenizer from {args.gguf}")
    meta = read_gguf_metadata(args.gguf)

    os.makedirs(args.out_dir, exist_ok=True)

    # vocab.json
    vocab = {}
    if meta['tokens']:
        for i, t in enumerate(meta['tokens']):
            vocab[t] = i
        vocab_path = os.path.join(args.out_dir, "vocab.json")
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, separators=(',', ':'))
        print(f"  Wrote {len(vocab)} entries to {vocab_path}")

    # merges.txt
    if meta['merges']:
        merges_path = os.path.join(args.out_dir, "merges.txt")
        with open(merges_path, 'w', encoding='utf-8') as f:
            for m in meta['merges']:
                f.write(m + '\n')
        print(f"  Wrote {len(meta['merges'])} merges to {merges_path}")

    # special_tokens.json
    special = {
        'bos_id': meta['bos_id'] if meta['bos_id'] is not None else -1,
        'eos_id': meta['eos_id'] if meta['eos_id'] is not None else -1,
        'unk_id': meta['unk_id'] if meta['unk_id'] is not None else -1,
        'pad_id': meta['pad_id'] if meta['pad_id'] is not None else -1,
    }
    special_path = os.path.join(args.out_dir, "special_tokens.json")
    with open(special_path, 'w', encoding='utf-8') as f:
        json.dump(special, f, indent=2)
    print(f"  Wrote special tokens to {special_path}")

    # chat_template.txt
    if meta['chat_template']:
        template_path = os.path.join(args.out_dir, "chat_template.txt")
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(meta['chat_template'])
        print(f"  Wrote chat template to {template_path}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
