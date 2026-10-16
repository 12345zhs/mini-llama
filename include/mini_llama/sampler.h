#pragma once

#include "mini_llama/tensor.h"

#include <random>

namespace mini_llama {

struct SamplingParams {
    float temperature = 0.0f;    // 温度,温度越高，输出的词概率越平滑，越具有随机性
    int top_k = 0;               // top_k中选几个
    unsigned int seed = 0;       // 随机数种子
};

class MiniSampler {
public:
    explicit MiniSampler(unsigned int seed = 0);
    explicit MiniSampler(const SamplingParams& params);

    int Sample(const Tensor& logits, const SamplingParams& params);

    static int SampleGreedy(const Tensor& logits);

    int SampleTemperature(const Tensor& logits, float temperature);

    int SampleTopK(const Tensor& logits, float temperature, int top_k);

private:
    std::mt19937 rng_;

};

int SampleGreedy(const Tensor& logits);

}