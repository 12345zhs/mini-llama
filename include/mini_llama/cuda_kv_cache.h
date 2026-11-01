#pragma once 

#include "mini_llama/cuda_runtime.h"
#include "mini_llama/cuda_tensor.h"


namespace mini_llama {

class CudaKvCache {
public:
    CudaKvCache() = default;
    CudaKvCache(int n_kayers, int max_seq_len, int n_kv_heads, int head_dim, int device_id = 0);

    CudaKvCache(const CudaKvCache&) = delete;
    CudaKvCache& operator=(const CudaKvCache&) = delete;

    CudaKvCache(CudaKvCache&&) noexcept = default;
    CudaKvCache& operator=(CudaKvCache&) noexcept = default;

    void Reset();
    void Reset(int n_layers, int max_seq_len, int n_kv_heads, int head_dim, int device_id = 0);
    void Clear();

    void Write(int layer, int pos, const Tensor& k, const Tensor& v);
    void WriteDevice(int layer, int pos, const CudaTensor& k, const CudaTensor& v);

    Tensor ReadKey(int layer, int pos) const;
    Tensor ReadValue(int layer, int pos) const;
    Tensor ReadKeyHead(int layer, int pos, int kv_head) const;
    Tensor ReadValueHead(int layer, int pos, int kv_head) const;

    const void* keys_data() const;
    const void* values_data() const;

    bool empty() const;
    size_t bytes() const;
    int n_layers() const;
    int max_seq_len() const;
    int n_kv_heads() const;
    int head_dim() const;
    int device_id() const;

private:
    void ValidateIndices(int layer, int pos) const;
    void ValidateHeadIndex(int kv_head) const;
    void ValidataWriteTensors(const Tensor& k, const Tensor& v) const;
    size_t SlotOffsetBytes(int layer, int pos) const;
    size_t HeadOffsetBytes(int layer, int pos, int kv_head) const;

    int n_layers_ = 0;
    int max_seq_len_ = 0;
    int n_kv_heads_ = 0;
    int head_dim_ = 0;
    int device_id_ = 0;
    CudaDeviceBuffer keys_;
    CudaDeviceBuffer values_;
};

bool CudaKvCacheBuilt();

    
}