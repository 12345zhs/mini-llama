#pragma once

#include "mini_llama/kv_cache.h"
#include "mini_llama/model.h"

namespace mini_llama {

struct MiniLlamaContext {

    const MiniLlamaModel* model = nullptr;
    KvCache kv_cache;
    int pos = 0;

    // 记录已经处理的token
    std::vector<int> token_history;

    // 计数器
    int n_prefill_tokens = 0;
    int n_decode_tokens = 0;
    
    MiniLlamaContext() = default;

    // 禁止拷贝构造和拷贝赋值
    MiniLlamaContext(const MiniLlamaContext&) = delete;
    MiniLlamaContext& operator=(const MiniLlamaContext&) = delete;

    MiniLlamaContext(MiniLlamaContext&&) noexcept = default;
    MiniLlamaContext& operator=(MiniLlamaContext&&) noexcept = default;

    explicit MiniLlamaContext(const MiniLlamaModel* model);
};

}