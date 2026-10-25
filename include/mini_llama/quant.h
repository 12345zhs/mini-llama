#pragma once

#include "mini_llama/quantized_tensor.h"
#include "mini_llama/tensor.h"

#include <vector>

namespace mini_llama{

// 量化与解压
std::vector<BlockQ80> QuantizeToQ80(const Tensor& src);

Tensor DequantizeFromQ80(const std::vector<BlockQ80>& blocks, const std::vector<int>& shape);


std::vector<BlockQ40> QuantizeToQ40(const Tensor& src);

Tensor DequantizeFromQ40(const std::vector<BlockQ40>& blocks, const std::vector<int>& shape);



Tensor DequantizeFromQ41(const std::vector<BlockQ41>& blocks, const std::vector<int>& shape);



// 量化线性层Linear算子
Tensor LinearQ80(const Tensor& x, const std::vector<BlockQ80>& weight, const std::vector<int>& weight_shape);
Tensor LinearQ40(const Tensor& x, const std::vector<BlockQ40>& weight, const std::vector<int>& weight_shape);
Tensor LinearQ41(const Tensor& x, const std::vector<BlockQ41>& weight, const std::vector<int>& weight_shape);



Tensor MatmulQ80(const std::vector<BlockQ80>& weight, const Tensor& input, const std::vector<int>& weight_shape);
Tensor MatmulQ40(const std::vector<BlockQ80>& weight, const Tensor& input, const std::vector<int>& weight_shape);



float CompareMatmulError(const Tensor& weight, const Tensor& input);
float CompareQ40Error(const Tensor& weight, const Tensor& input);

}