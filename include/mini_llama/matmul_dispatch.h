#pragma once
#include "mini_llama/tensor.h"

namespace mini_llama{

// matmul采用那种模式实现
enum class MatmulMode {
    kNaive,                      // 普通循环
    kThreaded,                   // 多CPU线程执行
};


// 获取默认的计算模式
MatmulMode DefaultMatmulMode();

// 使用mode模式来计算矩阵乘法
Tensor MatmulDispatch(const Tensor& a, const Tensor& b, MatmulMode mode);

// 使用mode模式来计算线性变化
Tensor LinearDispatch(const Tensor& a, const Tensor& weight, MatmulMode mode);

}