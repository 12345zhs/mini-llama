#pragma once

#include "mini_llama/quantized_tensor.h"
#include "mini_llama/tensor.h"
#include "mini_llama/cuda_tensor.h"

#include <vector>

namespace mini_llama {

bool CudaQuantBuilt();

Tensor CudaQ80Linear(const Tensor& x, const std::vector<BlockQ80>& weight, const std::vector<int>& weight_shape, const Tensor* bias = nullptr, int device_id);

Tensor CudaQ80LinearDeviceWeight(const Tensor& x, const void* q8_0_device, size_t q8_0_block_count, const std::vector<int>& weight_shape, const Tensor* bias = nullptr, int device_id = 0);


CudaTensor CudaQ80LinearDeviceInput(const CudaTensor& x, const void* q8_0_device, size_t q8_0_block_count, const std::vector<int>& weight_shape, int device_id = 0);

Tensor CudaQ40Linear(const Tensor& x, const std::vector<BlockQ40>& weight, const std::vector<int>& weight_shape, const Tensor* bias = nullptr, int device_id = 0);

Tensor CudaQ40LinearDeviceWeight(const Tensor& x, const void* q4_0_device, size_t q4_0_block_count, const std::vector<int>& weight_shape, const Tensor* bias = nullptr, int device_id = 0);

CudaTensor CudaQ40LinearDeviceInput(const CudaTensor& x, const void* q4_0_device, size_t q4_0_block_count, const std::vector<int>& weight_shape, int device_id = 0);

Tensor CudaQ41Linear(const Tensor& x, const std::vector<BlockQ41>& weight, const std::vector<int>& weight_shape, const Tensor* bias = nullptr, int device_id = 0);

Tensor CudaQ41LinearDeviceWeight(const Tensor& x, const void* q4_1_device, size_t q4_1_block_count, const std::vector<int>& weight_shape, const Tensor* bias = nullptr, int device_id = 0);

CudaTensor CudaQ41LinearDeviceInput(const CudaTensor& x, const void* q4_1_device, size_t q4_1_block_count, const std::vector<int>& weight_shape, int device_id = 0);

    
}