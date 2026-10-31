#include "mini_llama/cuda_ops.h"

#include <stdexcept>


namespace mini_llama {

static std::runtime_error CudaOpsNotBuiltError() {
    return std::runtime_error("CUDA ops were not built. Reconfigure with -DMINI_LLAMA_CUDA=ON on a " "NVIDIA CUDA machine.");
}

bool CudaOpsBuilt() {
#ifdef MINI_LLAMA_USE_CUDA
    return true;
#else 
    return false;
#endif
}

#ifndef MINI_LLAMA_USE_CUDA

Tensor CudaRmsNorm(const Tensor& x, const Tensor& weight, float eps, int device_id) {
    (void)x;
    (void)weight;
    (void)eps;
    (void)device_id;
    throw CudaOpsNotBuiltError();
}

Tensor CudaSilu(const Tensor& x, int device_id) {
    (void)x;
    (void)device_id;
    throw CudaOpsNotBuiltError();
}

Tensor CudaElementwiseMul(const Tensor& a, const Tensor& b, int device_id) {
    (void)a;
    (void)b;
    (void)device_id;
    throw CudaOpsNotBuiltError();
}

Tensor CudaElementwiseAdd(const Tensor& a, const Tensor& b, int device_id) {
    (void)a;
    (void)b;
    (void)device_id;
    throw CudaOpsNotBuiltError();
}

Tensor CudaSoftmax(const Tensor& x, int device_id) {
    (void)x;
    (void)device_id;
    throw CudaOpsNotBuiltError();
}

CudaTensor CudaEmbeddingLookupDeviceWeight(const void* embedding_data, const std::vector<int>& embedding_shape, int token_id, int device_id) {
    (void)embedding_data;
    (void)embedding_shape;
    (void)token_id;
    (void)device_id;
    throw CudaOpsNotBuiltError();
}

CudaTensor CudaRmsNormDeviceInput(const CudaTensor& x, const Tensor& weight, float eps, int device_id) {
    (void)x;
    (void)weight;
    (void)eps;
    (void)device_id;
    throw CudaOpsNotBuiltError();
}

CudaTensor CudaRmsNormDeviceWeight(const CudaTensor& x, const void* weight_data, const std::vector<int>& weight_shape, float eps, int device_id) {
    (void)x;
    (void)weight_data;
    (void)weight_shape;
    (void)eps;
    (void)device_id;
    throw CudaOpsNotBuiltError();
}

CudaTensor CudaSiluDeviceInput(const CudaTensor& x, int device_id) {
    (void)x;
    (void)device_id;
    throw CudaOpsNotBuiltError();
}


CudaTensor CudaElementwiseMulDeviceInput(const CudaTensor& a, const CudaTensor& b, int device_id) {
    (void)a;
    (void)b;
    (void)device_id;
    throw CudaOpsNotBuiltError();
}

CudaTensor CudaElementwiseAddDeviceInput(const CudaTensor& a, const CudaTensor& b, int device_id) {
    (void)a;
    (void)b;
    (void)device_id;
    throw CudaOpsNotBuiltError();
}

CudaTensor CudaElementwiseAddDeviceWeight(const CudaTensor& a, const void* b_data, const std::vector<int>& b_shape, int device_id) {
    (void)a;
    (void)b_data;
    (void)b_shape;
    (void)device_id;
    throw CudaOpsNotBuiltError();
}

void CudaRope(Tensor& q, Tensor& k, int pos, float theta, RopeType rope_type, int device_id) {
    (void)q;
    (void)k;
    (void)pos;
    (void)theta;
    (void)rope_type;
    (void)device_id;
    throw CudaOpsNotBuiltError();
}

void CudaRopeDeviceInput(CudaTensor& q, CudaTensor& k, int n_heads, int n_kv_heads, int head_dim, int pos, float theta, RopeType rope_type, int device_id) {
    (void)q;
    (void)k;
    (void)n_heads;
    (void)n_kv_heads;
    (void)head_dim;
    (void)pos;
    (void)theta;
    (void)rope_type;
    (void)device_id;
    throw CudaOpsNotBuiltError();
}


#endif

}