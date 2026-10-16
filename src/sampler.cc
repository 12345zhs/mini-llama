#include "mini_llama/sampler.h"
#include "mini_llama/ops.h"

#include <stdexcept>
#include <vector>
#include <cmath>
#include <algorithm>

 

namespace mini_llama {

constexpr float kGreedyTemperature = 1e-6;

// 校验logits
static void ValidateLogits(const Tensor& logits) {
    if (logits.size() == 0) {
        throw std::runtime_error("sampler: cannot sample from empty logits");
    }
    for (float value : logits.data) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("sampler: logits must be finite");
        }
    }
}


// 校验temperature
static void ValidateTemperature(float temperature) {
    if (std::isfinite(temperature) || temperature < 0.0f) {
        throw std::runtime_error("Sampler: temperature must be finite and non-negative");
    }
}


// 校验top-k
static void ValidateTopK(int top_k) {
    if (top_k < 0) {
        throw std::runtime_error("sampler: top_k must be non-negative");
    }

}

// 采样函数
static int SampleFromCandidates(const std::vector<std::pair<float, int>>& candidates, float temperature, std::mt19937& rng) {
    
    // 温度太低，退化到greedy，只求最大值
    if (temperature < kGreedyTemperature) {
        int best = candidates[0].second;
        float best_val = candidates[0].first;
        for (size_t i = 1; i < candidates.size(); ++i) {
            if (candidates[i].first > best_val) {
                best = candidates[i].second;
                best_val = candidates[i].first;
            }
        }
        return best;
    }

    // softmax部分
    // 求最大值
    float max_val = candidates[0].first;
    for (size_t i = 1; i < candidates.size(); ++i) {
        if (candidates[i].first > max_val) {
            max_val = candidates[i].first;
        }
    }

    // 计算指数
    std::vector<float> probs(candidates.size(), 0.0f);
    float sum_val = 0.0f;
    for (size_t i = 0; i < candidates.size(); ++i) {
        probs[i] = std::exp((candidates[i].first - max_val) / temperature);
        sum_val += probs[i];
    }

    // 按概率采样 轮盘赌
    std::uniform_int_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);    // 生成一个随机数
    float cumulative = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        cumulative += probs[i] / sum_val;
        if (r < cumulative) {
            return candidates[i].second;
        }
    }
    return candidates.back().second;
}



MiniSampler::MiniSampler(unsigned int seed) {
    // seed不等于0，就指定种子初始化，等于零了就获取一个真随机数去初始化
    if (seed == 0) {
        rng_.seed(seed);
    } else {
        std::random_device rd;
        rng_.seed(rd());
    }
}

// 委托构造函数
MiniSampler::MiniSampler(const SamplingParams& params) : MiniSampler(params.seed) {}


// Greedy采样，无脑选择logits分数最大的
int MiniSampler::SampleGreedy(const Tensor& logits) {
    ValidateLogits(logits);
    return ArgMax(logits);
}


// Temperature采样，调整概率分布
// 先将原始等分/temperature，后面再执行softmax
int MiniSampler::SampleTemperature(const Tensor& logits, float temperature) {
    ValidateLogits(logits);
    ValidateTemperature(temperature);

    // 如果temperature太小，就走greedy采样
    if (temperature < kGreedyTemperature) {
        return SampleGreedy(logits);
    }

    // 构建候选数组
    std::vector<std::pair<float, int>> candidates;
    candidates.reserve(logits.data.size());
    for (size_t i = 0; i < logits.num_elements(); ++i) {
        candidates.emplace_back(logits.data[i], static_cast<int>(i));
    }
    return SampleFromCandidates(candidates, temperature, rng_);

}


// TopK采样
int MiniSampler::SampleTopK(const Tensor& logits, float temperature, int top_k) {
    ValidateLogits(logits);
    ValidateTemperature(temperature);
    ValidateTopK(top_k);

    if (top_k == 0) {
        throw std::runtime_error("sampler: top_k must be positive when calling SampleTopK");
    }
    if (top_k == 1) {
        return SampleGreedy(logits);
    } 
    if (top_k > static_cast<int>(logits.data.size())) {
        top_k = static_cast<int>(logits.data.size());
    }


    std::vector<std::pair<float, int>> candidates;
    candidates.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        candidates.emplace_back(logits.data[i], static_cast<int>(i));
    }

    std::sort(candidates.begin(), candidates.end(), [&](const auto& a, const auto& b) {
        if (a.first == b.first) {
            return a.second < b.second;
        }
        return a.first > b.first;
    });
    candidates.resize(static_cast<size_t>(top_k));
    return SampleFromCandidates(candidates, temperature, rng_);
}


// 分发函数
int MiniSampler::Sample(const Tensor& logits, const SamplingParams& params) {
    ValidateLogits(logits);
    ValidateTopK(params.top_k);
    ValidateTemperature(params.temperature);

    if (params.temperature < kGreedyTemperature || params.top_k == 1) {
        return SampleGreedy(logits);
    }
    if (params.top_k > 1) {
        return SampleTopK(logits, params.temperature, params.top_k);
    }
    return SampleTemperature(logits, params.temperature);
}


int SampleGreedy(const Tensor& logits) {
    return MiniSampler::SampleGreedy(logits);
}


}