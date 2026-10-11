#include "mini_llama/ops.h"
#include "mini_llama/tensor.h"

#include <stdexcept>
#include <cmath>

namespace mini_llama {



Tensor Matmul(const Tensor& a, const Tensor& b) {
    
}

Tensor Linear(const Tensor& x, const Tensor& weight) {
    
}


// RmsNorm归一化: y = x / sqrt(mean(x^2) + eps) * weight
Tensor RmsNorm(const Tensor& x, const Tensor& weight, float eps) {
    // 输入向量与权重向量必须都是一维张量
    if (x.num_dims() != 1 || weight.num_dims() != 1) {
        throw std::runtime_error("RmsNorm: expected 1D tensors");
    }

    int dim = x.shape[0];
    // 输入向量与权重向量的维度必须匹配
    if (dim != weight.shape[0]) {
        throw std::runtime_error("RmsNorm: dimension mismatch");
    }

    float ss = 0.0f;
    for (int i = 0; i < dim; ++i) {
        ss += x.data[i] * x.data[i];
    }

    ss /= static_cast<size_t>(dim);
    float scale = 1.0f / std::sqrt(ss + eps);

    Tensor y({dim}, 0.0f);
    for (int i = 0; i < dim; ++i) {
        y.data[i] = x.data[i] * scale * weight.data[i];
    }
    return y;
}

// Softmax: y[i] = exp(x[i]) / sum(exp(x[j])) for j in 0..n-1                                       
Tensor Softmaxx(const Tensor& x) {
    if (x.num_dims() != 1) {
        throw std::runtime_error("Softmax: expected 1D tensor, got " + x.ShapeToStringShort());
    }
    if (x.num_elements() == 0) {
        throw std::runtime_error("Softmax: cannot compute Softmax of empty tensor");
    }

    int dim = x.shape[0];
    float max_val = x.data[0];
    // 找最大值
    for (int i = 1; i < dim; ++i) {
        if (max_val < x.data[i]) {
            max_val = x.data[i];
        }
    }

    float sum_val = 0.0f;
    Tensor y({dim}, 0.0f);
    for (int i = 0; i < dim; ++i) {
        y.data[i] = std::exp(x.data[i] - max_val);
        sum_val += y.data[i];
    }
    for (int i = 0; i < dim; ++i) {
        y.data[i] /= sum_val;
    }
    return y;
}


// SiLU激活函数: Silu(x) = x * sigmoid(x)
Tensor Silu(const Tensor& x) {
    Tensor y(x.shape, 0.0f);
    for (int i = 0; i < x.num_elements(); ++i) {
        float val = x.data[i];
        y.data[i] = val / (1.0f + std::exp(-val));
    }
    return y;
}



// ElementwiseMul  逐元素相乘
Tensor ElementwiseMul(const Tensor& a, const Tensor& b) {
    if (a.shape != b.shape) {
        throw std::runtime_error("ElementwiseMul: shape mismatch " + a.ShapeToStringShort() + " vs " + b.ShapeToStringShort());
    }

    Tensor y(a.shape, 0.0f);
    for (int i = 0; i < a.num_elements(); ++i) {
        y.data[i] = a.data[i] * b.data[i];
    }
    return y;
}


// SwiGLU激活函数: SwiGlu(gate, up) = Silu(gate) * up = gate * sigmoid(gate) * up
Tensor SwiGlu(const Tensor& gate, const Tensor& up) {
    if (gate.shape != up.shape) {
        throw std::runtime_error("SwiGlu: shape mismatch " + gate.ShapeToStringShort() + " vs " + up.ShapeToStringShort());
    }

    Tensor y(gate.shape, 0.0f);
    for (size_t i = 0; i < gate.num_elements(); ++i) {
        float val = gate[i];
        float sigmod = 1.0f / (1.0f + std::exp(-val));
        y.data[i] = val * sigmod * up.data[i];
    }
    return y;
}

// 辅助函数——Normal RoPE应用到多头张量
static void ApplyRopeNormalToHeads(Tensor& x, int pos, float theta) {
    int n_heads = x.shape[0];
    int head_dim = x.shape[1];
    for (int h = 0; h < n_heads; ++h) {
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / std::pow(theta, static_cast<float>(i) / static_cast<float>(head_dim));

            float cos_val = std::cos(pos * freq);
            float sin_val = std::sin(pos * freq);
            float x0 = x.data[h * head_dim + i];
            float x1 = x.data[h * head_dim + i + 1];

            x.data[h * head_dim + i] = x0 * cos_val - x1 * sin_val;
            x.data[h * head_dim + i + 1] = x0 * sin_val + x1 * cos_val;
        }
    }
}

// 赋值函数——NeoX RoPE应用到多头张量
static void ApplyRopeNeoXToHeads(Tensor& x, int pos, float theta) {
    int n_heads = x.shape[0];
    int head_dim = x.shape[1];
    int half_dim = head_dim / 2;
    for (int h = 0; h < n_heads; ++h) {
        for (int i = 0; i < half_dim; ++i) {
            float freq = 1.0f / std::pow(theta, static_cast<float>(2 * i) / static_cast<float>(head_dim));
            float cos_val = std::cos(pos * freq);
            float sin_val = std::sin(pos * freq);

            float x0 = x.data[h * head_dim + i];
            float x1 = x.data[h * head_dim + i + half_dim];
            x.data[h * head_dim + i] = x0 * cos_val - x1 * sin_val;
            x.data[h * head_dim + i + half_dim] = x0 * sin_val + x1 * cos_val;
        }
    }
}


// Rope: 对Attention中的Q、K添加位置信息  
void Rope(Tensor& q, Tensor& k, int pos, float theta, RopeType rope_type) {
    // q\k的维度必须是2维
    if (q.num_dims() != 2 || k.num_dims() != 2) {
        throw std::runtime_error("Rope: expected 2D tensors");
    }    

    if (pos < 0) {
        throw std::runtime_error("Rope: position must be positive");
    }

    // theta是有限值并且大于零
    if (!std::isfinite(theta) || theta <= 0.0f) {
        throw std::runtime_error("Rope: theta must be finite and positive");
    }

    int n_heads = q.shape[0];
    int head_dim = q.shape[1];
    int n_kv_heads = k.shape[0];
    int k_head_dim = k.shape[1];

    if (head_dim != k_head_dim) {
        throw std::runtime_error("Rope: q and k head_dim mismatch " + q.ShapeToStringShort() + " vs " + k.ShapeToStringShort());
    }
    if (head_dim <= 0 || head_dim % 2 != 0) {
        throw std::runtime_error("Rope: head_dim must be positive and even, got " + std::to_string(head_dim));
    }

    if (rope_type == RopeType::kNormal) {
        ApplyRopeNormalToHeads(q, pos, theta);
        ApplyRopeNormalToHeads(k, pos, theta);
    } else {
        ApplyRopeNeoXToHeads(q, pos, theta);
        ApplyRopeNeoXToHeads(k, pos, theta);
    }
}


// ArgMaX,返回tensor最大值的下标
int ArgMax(const Tensor& x) {
    if (x.num_elements() == 0) {
        throw std::runtime_error("ArgMax: cannot compute ArgMax of empty tensor");
    }
    int best = 0;
    float best_val = x.data[0];
    for (size_t i = 1; i < x.num_elements(); ++i) {
        if (x.data[i] > best_val) {
            best_val = x.data[i];
            best = static_cast<int>(i);
        }
    }
    return best;
}







}