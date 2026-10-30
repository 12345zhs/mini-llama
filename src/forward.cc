#pragma once

#include "mini_llama/forward.h"
#include "mini_llama/context.h"
#include "mini_llama/model.h"
#include "mini_llama/ops.h"
#include "mini_llama/kv_cache.h"
#include "mini_llama/thread_pool.h"
#include "mini_llama/cuda_matmul.h"
#include "mini_llama/cuda_quant.h"

#include <stdexcept>
#include <cstring>
#include <cmath>
#include <vector>

namespace mini_llama {


// 防御性校验，前向传播输入校验函数
static void ValidateForwardInputs(const MiniLlamaContext& ctx, const MiniLlamaModel& model, int token) {
    const ModelConfig& c = model.config;
    // 模型是否加载
    if (!model.loaded) {
        throw std::runtime_error("ForwardToken called with an unload model");
    }

    // token是否越界
    if (token < 0 || token >= c.vocab_size) {
        throw std::out_of_range("ForwardToken token id out of range");
    }

    // 查看当前token在序列中的位置是否越界
    if (ctx.pos < 0 || ctx.pos >= c.max_seq_len) {
        throw std::out_of_range("ForwardToken position out of range");
    }

    // 层数是否匹配
    if (model.layers.size() != static_cast<size_t>(c.n_layers)) {
        throw std::runtime_error("ForwardToken layer count does not match model config");
    }

    // token嵌入表大小是否足够
    if (model.token_embedding.data.size() < static_cast<size_t>(c.vocab_size * c.dim)) {
        throw std::runtime_error("ForwardToken token embedding tensor is smaller than config");
    }
}

// 形状检查
static void RequireShape(const Tensor& t, const std::vector<int>& excepted, const char* caller) {
    t.AssertShape(excepted, caller);
}


// 可选的偏置加法函数，有bias就加上，没有就返回
static Tensor AddOptionalBias(const Tensor& x, const Tensor& bias, const char* caller) {
    if (bias.data.empty()) {
        return x;
    }

    if (x.num_dims() != 1 || bias.num_dims() != 1 || x.shape[0] != bias.num_dims()) {
        throw std::runtime_error(std::string(caller) + " : bias shape mismatch x = " + x.ShapeToStringShort() + " bias =" + bias.ShapeToStringShort());
    }

    Tensor y = x;
    for (int i = 0; i < bias.num_elements(); ++i) {
        y.data[i] += bias.data[i];
    }
    return y;
}


// 检查权重可不可以支持计算
static bool CudaLinearSupported(const QuantizedTensor& weight) {
    return weight.type == QuantType::kF32 || weight.type == QuantType::kQ40 || weight.type == QuantType::kQ41 || weight.type == QuantType::kQ80;
}



// 从minillamaModel.weights中找对应的weight
static const CudaUploadedWeight* FindCudaWeight(const MiniLlamaModel& model, const std::string& name, QuantType type) {
    if (!model.cuda_weights) {
        return nullptr;
    }
    for (const auto& weight : model.cuda_weights->weights) {
        if (weight.name == name && weight.type == type) {
            return &weight;
        }
    }
    return nullptr;
}

// 记录linear中cuda权重的数据搬运情况
static void RecordCudaLinearCopy(const MiniLlamaModel& model, const Tensor& x, const Tensor& y) {
    if (!model.cuda_weights) {
        return;
    }
    model.cuda_weights->linear_calls += 1;
    model.cuda_weights->host_to_device_copies += 1;
    model.cuda_weights->device_to_host_copies += 1;
    model.cuda_weights->host_to_devide_bytes += x.size() * sizeof(float);
    model.cuda_weights->devide_to_host_bytes += y.size() * sizeof(float);

}


// 记录Linear的调用次数
static void RecordCudaLinearCall(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return;
    }
    model.cuda_weights->linear_calls += 1;
}


// 记录linear中的数据搬运情况host->device
static void RecordCudaHostToDeviceCopy(const MiniLlamaModel& model, size_t bytes) {
    if (!model.cuda_weights) {
        return;
    }
    model.cuda_weights->host_to_device_copies += 1;
    model.cuda_weights->host_to_devide_bytes += bytes;
}


// 记录linear中的数据搬运情况device->host
static void RecordCudaDeviceToHostCopy(const MiniLlamaModel& model, size_t bytes) {
    if (!model.cuda_weights) {
        return;
    }
    model.cuda_weights->device_to_host_copies += 1;
    model.cuda_weights->devide_to_host_bytes += bytes;
}


// 统计调用了几次激活函数
static void RecordCudaActivationCall(const MiniLlamaModel& model, size_t calls = 1) {
    if (!model.cuda_weights) {
        return;
    }
    model.cuda_weights->activation_calls += calls;
}


// 统计调用了多少次注意力计算
static void RecordCudaAttentionCall(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return;
    }
    model.cuda_weights->attention_calls += 1;
}


// 统计多少次回退到cpu上做注意力计算
static void RecordCudaAttentionCpuFallback (const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return;
    }
    model.cuda_weights->attention_cpu_fallbacks += 1;
}


// kvcache写了多少字节
static void RecordCudaKvCacheWrite(const MiniLlamaModel& model, size_t bytes) {
    if (!model.cuda_weights) {
        return;
    }
    model.cuda_weights->kv_cache_write_bytes += bytes;
}



// kvcache读了多少字节
static void RecordCudaKvCacheRead(const MiniLlamaModel& model, size_t bytes) {
    if (!model.cuda_weights) {
        return;
    }
    model.cuda_weights->kv_cache_read_bytes += bytes;
}



// 获取model的device_id
static int CudaDeviceId(const MiniLlamaModel& model) {
    return model.cuda_weights ? model.cuda_weights->device_id : 0;
}



// linear的cuda权重是否准备好
static bool CudaLinearReady(const MiniLlamaModel& model, const std::string& name, const QuantizedTensor& weight) {
    return model.cuda_weights && CudaLinearSupported(weight) && FindCudaWeight(model, name, weight.type) != nullptr;
}


// Tensor转变为cuda_tensor,并记录数据搬运
static CudaTensor UploadCudaTensor(const MiniLlamaModel& model, const Tensor& x) {
    CudaTensor x_dev = CudaTensorFromHost(x, CudaDeviceId(model));
    RecordCudaHostToDeviceCopy(model, x.size() * sizeof(float));
    return x_dev;
}


// cuda_tensor转变为tensor，并记录数据搬运
static Tensor DownloadCudaTensor(const MiniLlamaModel& model, const CudaTensor& x) {
  Tensor y = x.Download();
  RecordCudaDeviceToHostCopy(model, y.size() * sizeof(float));
  return y;
}


// cpu的linear
static Tensor ForwardLinear(const MiniLlamaModel& model, const std::string& name, const Tensor& x, const QuantizedTensor& weight) {
    // CPU
    if (!model.cuda_weights) {
        return Linear(x, weight);
    }


    if (!CudaLinearSupported(weight)) {
        return Linear(x, weight);
    }

    
    // GPU
    const CudaUploadedWeight* cuda_weight = FindCudaWeight(model, name, weight.type);
    if (cuda_weight == nullptr) {
        throw std::runtime_error("CUDA forward Linear missing uploaded weight: " + name);
    }

    Tensor y;
    switch (weight.type) {
        case QuantType::kF32 : {
            y = CudaLinearDeviceWeight(x, cuda_weight->buffer.data(), cuda_weight->shape, nullptr, model.cuda_weights->device_id);
            break;
        }
        case QuantType::kQ40 : {
            y = CudaQ40LinearDeviceWeight(x, cuda_weight->buffer.data(), cuda_weight->block_count, cuda_weight->shape, nullptr, model.cuda_weights->device_id);
            break;
        }
        case QuantType::kQ41 : {
            y = CudaQ41LinearDeviceWeight(x, cuda_weight->buffer.data(), cuda_weight->block_count, cuda_weight->shape, nullptr, model.cuda_weights->device_id);
            break;
        }
        case QuantType::kQ80 : {
            y = CudaQ80LinearDeviceWeight(x, cuda_weight->buffer.data(), cuda_weight->block_count, cuda_weight->shape, nullptr, model.cuda_weights->device_id);
            break;
        }
        break;
    }
    RecordCudaLinearCall(model);
    return y;
}


// 在cuda上的linear
static CudaTensor ForwardLinearDevice(const MiniLlamaModel& model, const std::string& name, const CudaTensor& x, const QuantizedTensor& weight) {
    if (!CudaLinearSupported(weight)) {
        throw std::runtime_error("CUDA forward Linear unsupported weight type: " + name);
    }

    const CudaUploadedWeight* cuda_weight = FindCudaWeight(model, name, weight.type);
    if (cuda_weight == nullptr) {
        throw std::runtime_error("CUDA forward Linear missing uploaded weight: " + name);
    }

    CudaTensor y;
    switch (weight.type) {
        case QuantType::kF32 : {
            y = CudaLinearDeviceInput(x, cuda_weight->buffer.data(), cuda_weight->shape, CudaDeviceId(model));
            break;
        }
        case QuantType::kQ80 : {
            y = CudaQ80LinearDeviceInput(x, cuda_weight->buffer.data(), cuda_weight->block_count, cuda_weight->shape, CudaDeviceId(model));
            break;
        }
        case QuantType::kQ40 : {
            y = CudaQ40LinearDeviceInput(x, cuda_weight->buffer.data(), cuda_weight->block_count, cuda_weight->shape, CudaDeviceId(model));
            break;
        }
        case QuantType::kQ41 : {
            y = CudaQ41LinearDeviceInput(x, cuda_weight->buffer.data(), cuda_weight->block_count, cuda_weight->shape, CudaDeviceId(model));
            break;
        }
    }
    RecordCudaLinearCall(model);
    return y;
}

// Q\K\V投影矩阵 + bias
static void ForwardQkvProjection(const MiniLlamaModel& model, const std::string& layer_prefix, const Tensor& h, const LayerWeights& lw, Tensor& q_flat, Tensor& k_flat, Tensor& v_flat) {
    const std::string q_name = layer_prefix + "wq";
    const std::string k_name = layer_prefix + "wk";
    const std::string v_name = layer_prefix + "wv";

    // GPU路径
    if (CudaLinearReady(model, q_name, lw.wq) && CudaLinearReady(model, k_name, lw.wk) && CudaLinearReady(model, v_name, lw.wv)) {
        CudaTensor h_dev = UploadCudaTensor(model, h);
        CudaTensor q_dev = ForwardLinearDevice(model, q_name, h_dev, lw.wq);
        CudaTensor k_dev = ForwardLinearDevice(model, k_name, h_dev, lw.wk);
        CudaTensor v_dev = ForwardLinearDevice(model, v_name, h_dev, lw.wv);

        q_flat = AddOptionalBias(DownloadCudaTensor(model, q_dev), lw.bq, "forward_layer q");
        k_flat = AddOptionalBias(DownloadCudaTensor(model, k_dev), lw.bk, "forward_layer k");
        v_flat = AddOptionalBias(DownloadCudaTensor(model, v_dev), lw.bv, "forward_layer v");
    
        return;
    }

    q_flat = AddOptionalBias(ForwardLinear(model, q_name, h, lw.wq), lw.bq, "forward_layer q");
    k_flat = AddOptionalBias(ForwardLinear(model, q_name, h, lw.wk), lw.bk, "forward_layer k");
    v_flat = AddOptionalBias(ForwardLinear(model, q_name, h, lw.wv), lw.bv, "forward_layer v");
    return;
}



// token嵌入查表
static Tensor EmbedToken(const MiniLlamaModel& model, int token_id) {
    int dim = model.config.dim;

    // 创建输出向量
    Tensor x({dim}, 0.0f);
    // token_embedding [vocab_size, dim]
    for (int i = 0; i < dim; ++i) {
        x.data[i] = model.token_embedding[token_id * dim + i];
    }
    return x;
}


// RmsNorm
static Tensor ForwardRmsNorm(const MiniLlamaModel& model, const Tensor& x, const Tensor& weight, float eps) {
    if (!model.cuda_weights) {
        return RmsNorm(x, weight, eps);
    }

    // cuda权重，GPU实现
}



// Rope
static void ForwardRope(const MiniLlamaModel& model, Tensor& q, Tensor& k, int pos, float theta, RopeType rope_type) {
    // cpu
    if (!model.cuda_weights) {
        Rope(q, k, pos, theta, rope_type);
        return;
    }

    //gpu
}

// GQA映射
static int MapQHeadToKvHead(int q_head, int n_heads, int n_kv_heads) {
    return q_head / (n_heads / n_kv_heads);
}


// Softmax
static Tensor ForwardSoftmax(const MiniLlamaModel& model, const Tensor& x) {
    //cpu
    if (!model.cuda_weights) {
        return Softmax(x);
    }
    // GPU

}


// Attention计算
static Tensor AttentionForward(const MiniLlamaModel& model, const Tensor& q, const Tensor& k, const Tensor& v, int pos, 
                               int layer, KvCache& kv_cache, CudaKvCache& cuda_kv_cache, int n_heads, int n_kv_heads, int head_dim) {

    kv_cache.Write(layer, pos, k, v);
    // 如果有CudaKvCache,并调用cuda attention计算
    if () {

    }

    // 有cuda_weight
    if () {

    }

    // CPU注意力计算
    Tensor attn_out({n_heads, head_dim}, 0.0f);
    // 计算缩放因子
    float scale = 1.0 / std::sqrt(static_cast<float>(head_dim));  // qk点积会饱和，缩放因子让分数更平稳

    // 定义每个头的匿名函数
    // QK    --->   缩放  ---->   softmax    ----> v
    // [head_dim]
    auto compute_head = [&](int h) {
        int kv_head = MapQHeadToKvHead(h, n_heads, n_kv_heads);

        // 计算
        std::vector<float> scores_data(pos + 1, 0.0f);  // 对历史的每一个位置进行打分
        
        // q[h]和每一个历史K做点积
        for (int t = 0; t <= pos; ++t) {
            const float* k_ptr = kv_cache.KeyPtr(layer, t, kv_head);
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q.data[h * head_dim + d] * k_ptr[d];
            }
            scores_data[t] = dot * scale;
        }

        Tensor scores({pos + 1}, 0.0f);
        scores.data = std::move(scores_data);
        Tensor probs = ForwardSoftmax(model, scores);       // 转化为概率


        // 加权求和v
        for (int d = 0; d < head_dim; ++d) {
            float val = 0.0f;
            for (int t = 0; t < pos; ++t) {
                const float* v_ptr = kv_cache.ValuePtr(layer, t, kv_head);
                val += probs[t] * v_ptr[d];
            }
            attn_out.data[h * head_dim + d] = val;
        }

    };

    // GPU
    

    // cpu 并行执
    ParallelFor(n_heads, [&](int begin, int end) {
        for (int h = begin; h < end; ++h) {
            compute_head(h);
        }
    });


    return attn_out;
}

// 残差相加
static Tensor ForwardAdd(const MiniLlamaModel& model, const Tensor& a, const Tensor& b) {
    // cpu
    if (!model.cuda_weights) {
        if (a.shape != b.shape) {
            throw std::runtime_error("forward_add: shape mismatch " + a.ShapeToStringShort() + " vs " + b.ShapeToStringShort());
        }
        Tensor y(a.shape, 0.0f);
        for (size_t i = 0; i < a.size(); ++i) {
            y.data[i] = a.data[i] + b.data[i];
        }
        return y;
    }

    // GPU

}

// SwiGlu
static Tensor ForwardSwiGlu(const MiniLlamaModel& model, const Tensor& gate, const Tensor& up) {
    // cpu  
    if (!model.cuda_weights) {
        return SwiGlu(gate, up);
    }
    // gpu
}

// FFN
static Tensor FfnForward(const MiniLlamaModel& model, const std::string& layer_prefix, const Tensor& h, const LayerWeights& lw) {
    const std::string gate_name = layer_prefix + "w_gate";
    const std::string up_name = layer_prefix + "w_up";
    const std::string down_name = layer_prefix + "w_down";

    // GPU


    // cpu
    Tensor gate = ForwardLinear(model, gate_name, h, lw.w_gate);       // gate = w_gate(h)          // [dim] → [hidden_dim]
    Tensor up  = ForwardLinear(model, up_name, h, lw.w_up);            // up   = w_up(h)            // [dim] → [hidden_dim]
    
    Tensor act = ForwardSwiGlu(model, gate, up);                       // act  = SiLU(gate) * up    // [hidden_dim]
    Tensor ff = ForwardLinear(model, down_name, lw.w_down);            // ff   = w_down(act)        // [hidden_dim] → [dim]
    
    return ff;
}

// transform层级计算
static Tensor ForwardLayer(MiniLlamaContext& ctx, const MiniLlamaModel& model, const Tensor& x, int layer, const LayerWeights& lw, const ModelConfig& c) {
    int dim = c.dim;              // 隐藏层维度
    int n_heads = c.n_heads;      // Q的头数
    int n_kv_heads = c.n_kv_heads;// kv的头数
    int head_dim = c.head_dim;    // 每个头的维度
    int pos = ctx.pos;            // 当前token在seq的位置
    const std::string layer_prefix = "layers." + std::to_string(layer) + ".";

    // 首先Rmsnorm归一化，在层数较深的Transform中，能有效控制方差，防止梯度计算爆炸活消失
    //                                     [dim]
    Tensor h = ForwaradRmsNorm(model, x, lw.attention_norm, c.rms_norm_eps);

    // QKV投影
    Tensor q_flat;
    Tensor k_flat;
    Tensor v_flat;
    ForwardQKVProjection(model, layer_prefix, h, lw, q_flat, k_flat, v_flat);

    // 形状转化
    Tensor q = q_flat.ReshapeChecked({n_heads, head_dim}, "forward_layer q");
    Tensor k = k_flat.ReshapeChecked({n_kv_heads, head_dim}, "forward_layer k");
    Tensor v = v_flat.ReshapeChecked({n_kv_heads, head_dim}, "forward_layer v");


    // 旋转位置编码
    ForwardRope(model, q, k, pos, c.rope_theta, c.rope_type);


    // 算注意力 + 读写kvcache
    // [n_heads, head_dim]
    Tensor attn_out = AttentionForward(model, q, k ,v, pos, layer, ctx.kv_cache, 
                      model.cuda_weights ? &ctx.cuda_kv_cache : nullptr, n_heads, n_kv_heads, head_dim);


    // 方便后面映射 [1, n_heads* head_dim]
    Tensor attn_out_flat = attn_out.ReshapeChecked({1, n_heads * head_dim}, "forward_layer attn_out");

    // 把多头的输出融合成一个dim维向量，让不同头的信息混合 [1, dim]
    Tensor attn_proj = ForwardLinear(model, layer_prefix + "wo", attn_out_flat, lw.wo);   

    // 形状校验
    RequireShape(attn_proj, {1, dim}, "forward_layer attn_proj 1d");

    // 在reshape回[dim], 后面方便与attention分数相加
    Tensor attn_proj_1d = attn_proj.ReshapeChecked({dim}, "forward_layer attn_proj 1d");

    // 残差相加
    Tensor x_attn = ForwardAdd(model, x, attn_proj_1d);

    // FFN
    Tensor h2 = ForwardRmsNorm(model, x_attn, lw.ffn_norm, c.rms_norm_eps);
    Tensor ff = FfnForward(model, layer_prefix, h2, lw);
    // 形状检查
    RequireShape(ff, {dim}, "forward_layer ffn");

    // 残差相加
    return ForwardAdd(model, x_attn, ff);
}


// 把最后一层[dim]转化为logits，得到每个候选token的原始分数
static Tensor ComputeLogits(const Tensor& x, const MiniLlamaModel& model, const ModelConfig& c) {
    Tensor normed = ForwardRmsNorm(model, x, model.final_norm, c.rms_norm_eps);
    Tensor logits_flat = ForwardLinear(model, "lm_head", normed, model.lm_head);  // [vacab_size]
    return logits_flat.ReshapeChecked({c.vocab_size}, "compute_logits");
}

Tensor ForwardToken(MiniLlamaContext& ctx, const MiniLlamaModel& model, int token) {
    ValidateForwardInputs(ctx, model, token);

    const ModelConfig& c = model.config;
    int n_layers = c.n_layers;

    /*
    cuda处理
    */


    // 把token_id变为[dim]维的一维向量
    Tensor x = EmbedToken(model, token);

    // transform的层级计算
    for (int layer = 0; layer < c.n_layers; ++layer) {
        x = ForwardLayer(ctx, model, x, layer, model.layers[layer], c);
    }

    Tensor logits = ComputeLogits(x, model, c);
    return logits;
}

}

