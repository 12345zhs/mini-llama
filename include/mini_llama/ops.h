#pragma once

#include "mini_llama/tensor.h"
#include "mini_llama/model.h"
#include "mini_llama/quantized_tensor.h"

namespace mini_llama {

// 普通矩阵乘法
Tensor Matmul(const Tensor& a, const Tensor& b);

// 全连接层
Tensor Linear(const Tensor& x, const Tensor& weight);
Tensor Linear(const Tensor& x, const QuantizedTensor& weight);

// RmsNorm归一化
Tensor RmsNorm(const Tensor& x, const Tensor& weight, float eps);

// 注意力分数转变为分数
Tensor Softmax(const Tensor& x);

// SiLU激活函数
Tensor Silu(const Tensor& x);

// 两个张量逐元素相乘
Tensor ElementwiseMul(const Tensor& a, const Tensor& b);

// SwiGLU激活函数
Tensor SwiGlu(const Tensor& gate, const Tensor& up);

// 对Q\K应用RoPE旋转位置编码
void Rope(Tensor& q, Tensor& k, int pos, float theta, RopeType rope_type = RopeType::kNormal);

// 最大值所在的位置
int ArgMax(const Tensor& x);

}