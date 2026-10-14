#pragma once

#include  "mini_llama/tensor.h"

namespace mini_llama {

struct KvCache {
    Tensor keys;
    Tensor values;

    KvCache() = default;
    KvCache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim);


    // 写入内存，写入一个层的pos位置
    // k: [n_kv_heads, head_dim]
    // v: [n_kv_heads, head_dim]
    void Write(int layer, int pos, const Tensor& k, const Tensor& v);

    // 零拷贝读取
    const float* KeyPtr(int layer, int pos, int kv_head) const;
    const float* ValuePtr(int layer, int pos, int kv_head) const;
};

}