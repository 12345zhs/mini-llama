#pragma once

#include "mini_llama/tensor.h"
#include "mini_llama/quantized_tensor.h"

#include <memory>

namespace mini_llama {

// RoPE的两种实现
enum class RopeType {
    kNormal,
    kNeoX,
};


// 模型超参数
struct ModelConfig {
    int vocab_size = 128;                   // 词表大小
    int dim = 32;                           // 隐藏层维度
    int hidden_dim = 86;                    // FFN中间层的维度
    int n_layers = 2;                       // Transform的层数
    int n_heads = 4;                        // Query的头数
    int n_kv_heads = 4;                     // kv的头数
    int head_dim = 8;                       // 每个头的维度
    int max_seq_len = 128;                  // 最大上下文的长度
    float rope_theta = 10000.0f;            // Rope的参数
    float rms_norm_eps = 1e-5f;             // rms中防止开方为0
    RopeType rope_type = RopeType::kNormal; // 用哪种Rope
};

// 单层权重
struct LayerWeights {
    Tensor attention_norm;      // attention前的RMSnorm权重

    // [n_heads * head_dim, dim]
    QuantizedTensor wq;         // Q投影矩阵
    // [n_kv_heads * head_dim, dim]
    QuantizedTensor wk;         // k投影矩阵
    QuantizedTensor wv;         // V投影矩阵

    // 可选偏执
    Tensor bq;
    Tensor bk;
    Tensor bv;

    QuantizedTensor wo;          // [dim, n_heads * head_dim]

    Tensor ffn_norm;              // FFN前的RMSnorm权重
    QuantizedTensor w_gate;       // [hidden_dim, dim]
    QuantizedTensor w_up;         // [hidden_dim, dim]
    QuantizedTensor w_dowm;       // [dim, hidden_dim]
};


struct CudaModelWeights {

};


// 整个模型
struct MiniLlamaModel {
    ModelConfig config;                     // 模型的配置
    bool loaded = false;                    // 是否加载完成
    std::string load_error;                 // 加载时的错误信息
    Tensor token_embedding;                  // token嵌入表
    Tensor final_norm;                      // 最后一层rms的权重
    std::vector<LayerWeights> layers;       // 所有层的权重数组
    std::shared_ptr<CudaModelWeights> cuda_weights;   // cuda权重
    QuantizedTensor lm_head;                // [vocab_size, dim]
};

}