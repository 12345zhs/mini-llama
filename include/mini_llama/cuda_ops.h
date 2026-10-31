#pragma once

#include "mini_llama/tensor.h"
#include "mini_llama/cuda_tensor.h"
#include "mini_llama/model.h"

namespace mini_llama {

bool CudaOpsBuilt();

Tensor CudaRmsNorm(const Tensor& x, const Tensor& weight, float eps, int device_id = 0);

Tensor CudaSilu(const Tensor& x, int device_id = 0);

Tensor CudaElementwiseMul(const Tensor& a, const Tensor& b, int device_id = 0);

Tensor CudaElementwiseAdd(const Tensor& a, const Tensor& b, int device_id = 0);

Tensor CudaSoftmax(const Tensor& x, int device_id = 0);

CudaTensor CudaEmbeddingLookupDeviceWeight(const void* embedding_data, const std::vector<int>& embedding_shape, int token_id, int device_id = 0);

CudaTensor CudaRmsNormDeviceInput(const CudaTensor& x, const Tensor& weight, float eps, int device_id = 0);

CudaTensor CudaRmsNormDeviceWeight(const CudaTensor& x, const void* weight_data, const std::vector<int>& weight_shape, float eps, int device_id = 0);

CudaTensor CudaSiluDeviceInput(const CudaTensor& x, int device_id = 0);

CudaTensor CudaElementwiseMulDeviceInput(const CudaTensor& a, const CudaTensor& b, int device_id = 0);

CudaTensor CudaElementwiseAddDeviceInput(const CudaTensor& a, const CudaTensor& b, int device_id = 0);

CudaTensor CudaElementwiseAddDeviceWeight(const CudaTensor& a, const void* b_data, const std::vector<int>& b_shape, int device_id = 0);

void CudaRope(Tensor& q, Tensor& k, int pos, float theta, RopeType rope_type = RopeType::kNormal, int device_id = 0);

void CudaRopeDeviceInput(CudaTensor& q, CudaTensor& k, int n_heads, int n_kv_heads, int head_dim, int pos, float theta, RopeType rope_type = RopeType::kNormal, int device_id = 0);


}