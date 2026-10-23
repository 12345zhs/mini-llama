#pragma once

#include "mini_llama/tensor.h"
#include "mini_llama/kv_cache.h"
#include "mini_llama/model.h"


#include <string>
#include <vector>

namespace mini_llama {

// 性能指标
struct BenchmarkResult {

    int n_prompt_tokens = 0;
    int n_generated_tokens = 0;
    int n_decode_tokens = 0;

    double prefill_ms = 0.0;
    double decode_ms = 0.0;

    double tokens_per_sec() const;
    double decode_tokens_per_sec() const;
};

void DumpTensorShape(const Tensor& t, const std::string& name);
void DumpLogitsTopK(const Tensor& logits, int k);
void DumpKvCacheInfo(const KvCache& cache, int current_pos = -1);


BenchmarkResult RunBenchmark(const MiniLlamaModel& model, const std::vector<int>& prompt_tokens, int n_predict, unsigned int seed, bool verbose);


}