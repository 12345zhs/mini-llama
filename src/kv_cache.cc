#include "mini_llama/kv_cache.h"

#include <stdexcept>
#include <cstring>

namespace mini_llama {

static void Require4DCacheSHape(const Tensor& t, const char* name) {
    if (t.num_dims() != 4) {
        throw std::runtime_error(std::string(name) + " must be a 4D tensor");
    }
}


KvCache::KvCache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim) {
    keys = MakeTensor4D(n_layers, max_seq_len, n_kv_heads, head_dim, 0.0f);
    values = MakeTensor4D(n_layers, max_seq_len, n_kv_heads, head_dim, 0.0f);
}


// 写入内存，写入一个层的pos位置
// k: [n_kv_heads, head_dim]
// v: [n_kv_heads, head_dim]
void KvCache::Write(int layer, int pos, const Tensor& k, const Tensor& v) {
    Require4DCacheSHape(keys, "keys");
    Require4DCacheSHape(values, "values");
    if (k.num_dims() != 2 || v.num_dims() != 2 || k.shape != v.shape) {
        throw std::runtime_error("KvCache::Write expected matching [n_kv_heads, head_dim] tensors");
    }

    int n_kv_heads = k.shape[0];
    int head_dim = k.shape[1];

    if (layer < 0 || layer >= keys.shape[0]) {
        throw std::runtime_error("KvCache::Write layer out of range");
    }
    if (pos < 0 || pos >= keys.shape[1]) {
        throw std::runtime_error("KvCache::Write position out of range");
    }
    if (n_kv_heads != keys.shape[2] || head_dim != keys.shape[3]) {
        throw std::runtime_error("KvCache::Write tensor shape does not match cache shape");
    }

    size_t layer_stride = static_cast<size_t>(keys.shape[1] * keys.shape[2] * keys.shape[3]);
    size_t pos_stride = static_cast<size_t>(keys.shape[2] * keys.shape[3]);
    size_t head_stride = static_cast<size_t>(keys.shape[3]);

    // 相对地址偏移
    size_t base = layer * layer_stride + pos * pos_stride;
    for (int h = 0; h < n_kv_heads; ++h) {
        size_t offset = base + h * head_stride;
        std::memcpy(&keys.data[offset], &k.data[h * head_dim], head_dim * sizeof(float));
        std::memcpy(&values.data[offset], &v.data[h * head_dim], head_dim * sizeof(float));
    }

}

// 零拷贝读取
const float* KvCache::KeyPtr(int layer, int pos, int kv_head) const {
    Require4DCacheSHape(keys, "keys");
    if (layer < 0 || layer >= keys.shape[0] || pos < 0 || pos >= keys.shape[1] || kv_head < 0 || kv_head >= keys.shape[2]) {
        throw std::out_of_range("KvCache::KeyPtr out of range");
    }

    size_t layer_stride = static_cast<size_t>(keys.shape[1] * keys.shape[2] * keys.shape[3]);
    size_t pos_stride = static_cast<size_t>(keys.shape[2] * keys.shape[3]);
    size_t head_stride = static_cast<size_t>(keys.shape[3]);

    return &keys.data[layer * layer_stride + pos * pos_stride + kv_head * head_stride];
}


const float* KvCache::ValuePtr(int layer, int pos, int kv_head) const {
    Require4DCacheSHape(values, "values");
    if (layer < 0 || layer >= keys.shape[0] || pos < 0 || pos >= keys.shape[1] || kv_head < 0 || kv_head >= keys.shape[2]) {
        throw std::out_of_range("KvCache::ValuePtr out of range");
    }

    size_t layer_stride = static_cast<size_t>(values.shape[1] * values.shape[2] * values.shape[3]);
    size_t pos_stride = static_cast<size_t>(values.shape[2] * values.shape[3]);
    size_t head_stride = static_cast<size_t>(values.shape[3]);

    return &values.data[layer * layer_stride + pos * pos_stride + kv_head * head_stride];
}

}