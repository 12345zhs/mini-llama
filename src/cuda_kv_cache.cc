#include "mini_llama/cuda_kv_cache.h"


#include <stdexcept>
#include <limits>
#include <vector>

namespace mini_llama {

// 报错
static std::runtime_error CudaKvCacheNotBuiltError() {
    return std::runtime_error( "CUDA KV cache was not built. Reconfigure with -DMINI_LLAMA_CUDA=ON on a " "NVIDIA CUDA machine.");
}

// 检测必须为正数
static void RequirePositive(int value, const char* name) {
    if (value <= 0) {
        throw std::runtime_error(std::string("CudaKvCache: ") + name + " must be positive");
    }
}

// 整数乘法检测器
static size_t CheckedMul(size_t a, size_t b, const char* label) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        throw std::runtime_error(std::string("CudaKvCache: size overflow for ") + label);
    }
    return a * b;
}


// CudaKvCache是否建立成功
bool CudaKvCacheBuilt() {
#ifdef MINI_LLAMA_USE_CUDA
    return true;
#else 
    return false;
#endif
}


 
CudaKvCache::CudaKvCache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim, int device_id) {
    Reset(n_layers, max_seq_len, n_kv_heads, head_dim, device_id);
}


// kvcache的清理
void CudaKvCache::Reset() {
    keys_.Reset();
    values_.Reset();
    n_layers_ = 0;
    max_seq_len_ = 0;
    n_kv_heads_ = 0;
    head_dim_ = 0;
    device_id_ = 0;
}


// kvcache的清理与分配
void CudaKvCache::Reset(int n_layers, int max_seq_len, int n_kv_heads, int head_dim, int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
    RequirePositive(n_layers, "n_layers");
    RequirePositive(max_seq_len, "max_seq_len");
    RequirePositive(n_kv_heads, "n_kv_heads");
    RequirePositive(head_dim, "head_dim");


    // 总的元素个数以及总的字节数
    size_t elements = static_cast<size_t>(n_layers);
    elements = CheckedMul(elements, static_cast<size_t>(max_seq_len), "layers * seq");
    elements = CheckedMul(elements, static_cast<size_t>(n_kv_heads), "layers * seq * heads");
    elements = CheckedMul(elements, static_cast<size_t>(head_dim), "layers * seq * heads * dim");
    size_t bytes_per_cache = CheckedMul(elements, sizeof(float), "cache bytes");

    n_layers_ = n_layers;
    max_seq_len_ = max_seq_len;
    n_kv_heads_ = n_kv_heads;
    head_dim_ = head_dim;
    device_id_ = device_id;
    keys_.Reset(bytes_per_cache, device_id);
    values_.Reset(bytes_per_cache, device_id);

    Clear();
#else
    (void)n_layers;
    (void)max_seq_len;
    (void)n_kv_heads;
    (void)head_dim;
    (void)device_id;
    throw CudaKvCacheNotBuiltError();
#endif
}


// 把申请的CudaDeviceBuffer数据全部置0
void CudaKvCache::Clear() {
#ifdef MINI_LLAMA_USE_CUDA
    // kvcache没有分配过内存
    if (empty()) {
        return;
    }
    CudaSetDevice(device_id_);
    // 内存全部填0
    std::vector<float> zeros(keys_.bytes() / sizeof(float), 0.0f);
    keys_.Upload(zeros.data(), keys_.bytes());
    values_.Upload(zeros.data(), values_.bytes());
#else
    throw CudaKvCacheNotBuiltError();
#endif
}


// 写入kvcache
void CudaKvCache::Write(int layer, int pos, const Tensor& k, const Tensor& v) {
#ifdef MINI_LLAMA_USE_CUDA
    ValidateIndices(layer, pos);
    ValidateWriteTensors(k, v);
    size_t offset = SlotOffsetBytes(layer, pos);
    size_t bytes_to_copy = k.size() * sizeof(float);
    char* key_dst = static_cast<char*>(keys_.data()) + offset;
    char* value_dst = static_cast<char*>(values_.data()) + offset;

    CudaSetDevice(device_id_);
    CudaMemcpyBytes(key_dst, k.data.data(), bytes_to_copy, CudaMemcpyKind::kHostToDevice);
    CudaMemcpyBytes(value_dst, v.data.data(), bytes_to_copy, CudaMemcpyKind::kHostToDevice);
#else
    (void)layer;
    (void)pos;
    (void)k;
    (void)v;
    throw CudaKvCacheNotBuiltError();
#endif
}


// GPU之间写入kvcache
void CudaKvCache::WriteDevice(int layer, int pos, const CudaTensor& k, const CudaTensor& v) {
#ifdef MINI_LLAMA_USE_CUDA
    ValidateIndices(layer, pos);
    if (k.device_id() != device_id_ || v.device_id() != device_id_) {
        throw std::runtime_error("CudaKvCache::WriteDevice tensor is on a different CUDA device");
    }
    const size_t expected = static_cast<size_t>(n_kv_heads_) * static_cast<size_t>(head_dim_);
    if (k.size() != expected || v.size() != expected) {
        throw std::out_of_range("CudaKvCache::WriteDevice tensor shape does not match cache shape");
    }

    size_t offset = SlotOffsetBytes(layer, pos);
    size_t bytes_to_copy = expected * sizeof(float);
    char* key_dst = static_cast<char*>(keys_.data()) + offset;
    char* value_dst = static_cast<char*>(values_.data()) + offset;
    
    CudaSetDevice(device_id_);
    CudaMemcpyBytes(key_dst, k.data(), bytes_to_copy, CudaMemcpyKind::kDeviceToDevice);
    CudaMemcpyBytes(value_dst, v.data(), bytes_to_copy, CudaMemcpyKind::kDeviceToDevice);
#else
    (void)layer;
    (void)pos;
    (void)k;
    (void)v;
    throw CudaKvCacheNotBuiltError();
#endif
}


// 把某一层、某个位置的所有 KV 头一次性全部读出来
Tensor CudaKvCache::ReadKey(int layer, int pos) const {
#ifdef MINI_LLAMA_USE_CUDA
    ValidateIndices(layer, pos);
    Tensor out({n_kv_heads_, head_dim_}, 0.0f);
    const char* src = static_cast<const char*>(keys_.data()) + SlotOffsetBytes(layer, pos);
    CudaSetDevice(device_id_);
    CudaMemcpyBytes(out.data.data(), src, out.size() * sizeof(float), CudaMemcpyKind::kDeviceToHost);
    return out;
#else
    (void)layer;
    (void)pos;
    throw CudaKvCacheNotBuiltError();
#endif
}


Tensor CudaKvCache::ReadValue(int layer, int pos) const {
#ifdef MINI_LLAMA_USE_CUDA
    ValidateIndices(layer, pos);
    Tensor out({n_kv_heads_, head_dim_}, 0.0f);
    const char* src = static_cast<const char*>(values_.data()) + SlotOffsetBytes(layer, pos);
    CudaSetDevice(device_id_);
    CudaMemcpyBytes(out.data.data(), src, out.size() * sizeof(float), CudaMemcpyKind::kDeviceToHost);
    return out;
#else
    (void)layer;
    (void)pos;
    throw CudaKvCacheNotBuiltError();
#endif
}





// 只把某一层、某个位置、指定某一个 KV 头的数据读出来
Tensor CudaKvCache::ReadKeyHead(int layer, int pos, int kv_head) const {
#ifdef MINI_LLAMA_USE_CUDA
    ValidateIndices(layer, pos);
    ValidateHeadIndex(kv_head);
    Tensor out({head_dim_}, 0.0f);
    const char* src = static_cast<const char*>(keys_.data()) + HeadOffsetBytes(layer, pos, kv_head);
    CudaSetDevice(device_id_);
    CudaMemcpyBytes(out.data.data(), src, out.size() * sizeof(float), CudaMemcpyKind::kDeviceToHost);
    return out;
#else
    (void)layer;
    (void)pos;
    (void)kv_head;
    throw CudaKvCacheNotBuiltError();
#endif
}



Tensor CudaKvCache::ReadValueHead(int layer, int pos, int kv_head) const {
#ifdef MINI_LLAMA_USE_CUDA
    ValidateIndices(layer, pos);
    ValidateHeadIndex(kv_head);
    Tensor out({head_dim_}, 0.0f);
    const char* src = static_cast<const char*>(values_.data()) + HeadOffsetBytes(layer, pos, kv_head);
    CudaSetDevice(device_id_);
    CudaMemcpyBytes(out.data.data(), src, out.size() * sizeof(float), CudaMemcpyKind::kDeviceToHost);
    return out; 
#else
    (void)layer;
    (void)pos;
    (void)kv_head;
    throw CudaKvCacheNotBuiltError();
#endif
}


// 当前k_cache的起始地址
const void* CudaKvCache::keys_data() const {
    if (empty()) {
        throw std::runtime_error("CudaKvCache: cache is empty");
    }
    return keys_.data();
}

// 当前v_cache的起始地址
const void* CudaKvCache::values_data() const {
    if (empty()) {
        throw std::runtime_error("CudaKvCache: cache is empty");
    }
    return values_.data();
}


// cache是否有空
bool CudaKvCache::empty() const {
    return keys_.empty() || values_.empty();
}

// cache有多少字节
size_t CudaKvCache::bytes() const {
    return keys_.bytes() + values_.bytes();
}


// 返回层数
int CudaKvCache::n_layers() const {
    return n_layers_;
}

// 返回最长序列长度
int CudaKvCache::max_seq_len() const {
    return max_seq_len_;
}

// 返回kv头数
int CudaKvCache::n_kv_heads() const {
    return n_kv_heads_;
}

// 返回头的维度
int CudaKvCache::head_dim() const {
    return head_dim_;
}

// 返回显卡id
int CudaKvCache::device_id() const {
    return device_id_;
}


// 检查下标是否正确
void CudaKvCache::ValidateIndices(int layer, int pos) const {
    if (empty()) {
        throw std::runtime_error("CudaKvCache: cache is empty");
    }
    if (layer < 0 || layer >= n_layers_) {
        throw std::out_of_range("CudaKvCache layer out of range");
    }
    if (pos < 0 || pos >= max_seq_len_) {
        throw std::out_of_range("CudaKvCache position out of range");
    }
}


// 检查head索引是否正确
void CudaKvCache::ValidateHeadIndex(int kv_head) const {
    if (kv_head < 0 || kv_head >= n_kv_heads_) {
        throw std::out_of_range("CudaKvCache head out of range");
    }
}


// 检查写入的张量
void CudaKvCache::ValidataWriteTensors(const Tensor& k, const Tensor& v) const {
    if (k.num_dims() != 2 || v.num_dims() != 2 || k.shape != v.shape) {
        throw std::out_of_range("CudaKvCache::Write expected matching [n_kv_heads, head_dim] tensors");
    }
    if (k.shape[1] != head_dim_ || k.shape[0] != n_kv_heads_) {
        throw std::out_of_range("CudaKvCache::Write tensor shape does not match cache shape");
    }
}


// 计算字节偏移量
size_t CudaKvCache::SlotOffsetBytes(int layer, int pos) const {
    size_t slot = static_cast<size_t>(layer);
    slot = slot * static_cast<size_t>(max_seq_len_) + static_cast<size_t>(pos);
    slot = slot * static_cast<size_t>(n_kv_heads_) * static_cast<size_t>(head_dim_);
    return slot * sizeof(float);
}


// 计算具体的头偏移
size_t CudaKvCache::HeadOffsetBytes(int layer, int pos, int kv_head) const {
    size_t offset = SlotOffsetBytes(layer, pos);  // 这个token的基准地址
    offset += static_cast<size_t>(kv_head) * static_cast<size_t>(head_dim_) * sizeof(float);

    return offset;
}





}