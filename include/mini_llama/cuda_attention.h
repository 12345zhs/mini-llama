#pragma once

#include "mini_llama/tensor.h"
#include "mini_llama/cuda_kv_cache.h"
#include "mini_llama/cuda_tensor.h"

namespace mini_llama {

bool CudaAttentionBuilt();

Tensor CudaAttentionDecode(const Tensor& q, const CudaKvCache& kv_cache, int layer, int pos, int n_heads, int n_kv_heads, int head_dim, int device_id = 0);

CudaTensor CudaAttentionDecodeDeviceInput(const CudaTensor& q, const CudaKvCache& kv_cache, int layer, int pos, int n_heads, int n_kv_heads, int head_dim, int device_id = 0);

    
}