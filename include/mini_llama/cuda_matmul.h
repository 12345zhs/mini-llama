#pragma once

#include "mini_llama/tensor.h"
#include "mini_llama/cuda_tensor.h"

#include <vector>

namespace mini_llama {

bool CudaMatmulBuilt();

Tensor CudaMatmul(const Tensor& a, const Tensor& b, int device_id = 0);

Tensor CudaLinear(const Tensor& x, const Tensor& weight, const Tensor* bias = nullptr, int device_id = 0);

Tensor CudaLinearDeviceWeight(const Tensor& x, const void* w_device, const std::vector<int>& w_shape, const Tensor* bias = nullptr, int device_id = 0);

CudaTensor CudaLinearDeviceInput(const CudaTensor& x, const void* w_device, const std::vector<int>& w_shape, int device_id = 0);
    
}