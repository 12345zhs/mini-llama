#include "mini_llama/debug.h"
#include "mini_llama/context.h"
#include "mini_llama/sampler.h"
#include "mini_llama/batch.h"
#include "mini_llama/forward.h"

#include <iostream>
#include <algorithm>
#include <iomanip>
#include <chrono>

namespace mini_llama {

// 总体生成速度
double BenchmarkResult::tokens_pre_sec() const {
    double total_ms = prefill_ms + decode_ms;
    if (total_ms <= 0.0) {
        return 0.0;
    }
    return n_generated_tokens * 1000.0 / total_ms;

}

// 生成阶段的素的
double BenchmarkResult::decode_tokens_pre_sec() const {
    if (decode_ms <= 0.0 || n_decode_tokens <= 0) {
        return 0.0;
    }
    return n_decode_tokens * 1000.0 / decode_ms;
}


// 打印一个tensor的名字、形状、元素总数
void DumpTensorShape(const Tensor& t, const std::string& name) {
    std::cout << "  tensor: " << name;
    std::cout << "  shape=[";
    for (size_t i = 0; i < t.shape.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << t.shape[i];
    }
    std::cout << "] dtype =f32 size=" << t.size() << std::endl;
}

// 找出分数最高的前k个词，打印加载ID和分数
void DumpLogitsTopK(const Tensor& logits, int k) {
    if (logits.num_dims() != 1) {
        std::cout << "  logits: invalid shape for top-k dump\n";
        return;
    }

    int vocab_size = logits.shape[0];
    if (k <= 0 || k > vocab_size) {
        k = vocab_size;
    }

    std::vector<std::pair<float, int>> pairs;
    pairs.reserve(vocab_size);
    for (size_t i = 0; i < logits.num_elements(); ++i) {
        pairs.push_back({logits.data[i], i});
    }

    std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    for (int i = 0; i < k; ++i) {
        std::cout << "    [" << std::setw(3) << i << "] id=" << std::setw(4)
                  << pairs[i].second << " value=" << std::fixed
                  << std::setprecision(6) << pairs[i].first << "\n";
    }

}

// 打印KvCache信息
void DumpKvCacheInfo(const KvCache& cache, int current_pos = -1) {
    int n_layers = cache.keys.shape[0];
    int seq_len = cache.keys.shape[1];
    int n_kv_heads = cache.keys.shape[2];
    int head_dim = cache.keys.shape[3];

    std::cout << "  KV cache: layers=" << n_layers << " seq_len=" << seq_len << " n_kv_heads=" << n_kv_heads << " head_dim=" << head_dim << "\n";

    if (current_pos >= 0) {
        std::cout << "    current_pos: " << current_pos << "\n";
    }

    std::cout << "    keys shape: [" << cache.keys.shape[0] << ", " << cache.keys.shape[1] << ", " << cache.keys.shape[2] << ", " << cache.keys.shape[3] << "]\n";
    std::cout << "    values shape: [" << cache.values.shape[0] << ", " << cache.values.shape[1] << ", " << cache.values.shape[2] << ", " << cache.values.shape[3] << "]\n";

}


BenchmarkResult RunBenchmark(const MiniLlamaModel& model, const std::vector<int>& prompt_tokens, int n_predict, unsigned int seed, bool verbose) {
    BenchmarkResult result;
    result.n_prompt_tokens = static_cast<int>(prompt_tokens.size());  // 记录提示词的长度
    
    MiniLlamaContext ctx(&model);      // 初始化kvcache     
    MiniSampler sampler(seed);         // 初始化采样器
    SamplingParams params;
    params.seed = seed;                

    if (verbose) {
        std::cout << "[verbose] prefill " << prompt_tokens.size() << " tokens\n";
    }

    // 记录prefill时间
    auto prefill_start = std::chrono::steady_clock::now();
    MiniBatch prefill = MiniBatch::FromTokens(prompt_tokens, 0);  // 把token合成一个batch
    Tensor logits = ForwardBatch(ctx, model, prefill);            // 前向传播
    auto prefill_end = std::chrono::steady_clock::now();
    result.prefill_ms = std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();
    
    if (verbose) {
        DumpTensorShape(logits, "logits");
        DumpLogitsTopK(logits, 5);
        DumpKvCacheInfo(ctx.kv_cache, ctx.pos);
    }


    //decode
    if (n_predict > 0 && result.n_prompt_tokens < model.config.max_seq_len) {
        std::vector<int> tokens = prompt_tokens;

        int next_token = sampler.Sample(logits, params);
        tokens.push_back(next_token);
        ++result.n_generated_tokens;

        if (verbose) {
            std::cout << "[verbose] Decode step 0: token=" << next_token << "\n";
        }

        auto decode_start = std::chrono::steady_clock::now();

        // decode forward
        for (int i = 0; i < n_predict && static_cast<int>(tokens.size()) < model.config.max_seq_len; ++i) {
            MiniBatch decode = MiniBatch::Single(tokens.back(), static_cast<int>(tokens.size() - 1));        // 把新加入的token形成一个batch
            logits = ForwardBatch(ctx, model, decode);           // forward
            ++result.n_decode_tokens;
            next_token = sampler.Sample(logits, params);         // 选出下一个token
            tokens.push_back(next_token);
            ++result.n_generated_tokens;

            if (verbose) {
                std::cout << "[verbose] Decode step " << i << ": token=" << next_token << "\n";
                DumpLogitsTopK(logits, 5);
            }
        }

        auto decode_end = std::chrono::steady_clock::now();
        if (result.n_decode_tokens > 0) {
            result.decode_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
        }
 
    }

    if (verbose) {
        DumpKvCacheInfo(ctx.kv_cache, ctx.pos);
    }
}

}