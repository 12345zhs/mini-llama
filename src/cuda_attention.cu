#include "mini_llama/cuda_attention.h"
#include "mini_llama/cuda_runtime.h"


#include <cuda_runtime.h>
#include <vector>
#include <cmath>

namespace mini_llama {

constexpr int kAttentionBlockSize = 128;

static void CheckCudaAttention(cudaError_t err, const char* expr) {
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA attention error in " + std::string(expr) + ": " + cudaGetErrorString(err));
    }
}


static void CheckLastAttentionKernel(const char* kernel_name) {
    CheckCudaAttention(cudaGetLastError(), kernel_name);
    CheckCudaAttention(cudaDeviceSynchronize(), kernel_name);
}


// 防溢出乘法器
static size_t CheckedMul(size_t a, size_t b, const char* label) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        throw std::runtime_error(std::string("CudaAttentionDecode: size overflow for ") + label);
    }
    return a * b;
}


// 检验attention的输入参数
static void ValidateAttentionInputs(const Tensor& q, const CudaKvCache& kv_cache, int layer, int pos, int n_heads, int n_kv_heads, int head_dim) {
    if (q.shape != std::vector<int>{n_heads, head_dim}) {
        throw std::runtime_error("CudaAttentionDecode: expected q shape [" + std::to_string(n_heads) +", " + std::to_string(head_dim) + "], got " + q.ShapeToStringShort());
    }

    if (kv_cache.empty()) {
        throw std::runtime_error("CudaAttentionDecode: CUDA KV cache is empty");
    }

    if (layer < 0 || layer >= kv_cache.n_layers()) {
        throw std::out_of_range("CudaAttentionDecode: layer out of range");
    }

    if (pos < 0 || pos >= kv_cache.max_seq_len()) {
        throw std::out_of_range("CudaAttentionDecode: position out of range");
    }

    if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0) {
        throw std::runtime_error("CudaAttentionDecode: head counts and head_dim must be positive");
    }
    if (n_heads % n_kv_heads != 0) {
        throw std::runtime_error("CudaAttentionDecode: n_heads must be divisible by n_kv_heads");
    }

    if (kv_cache.n_kv_heads() != n_kv_heads || kv_cache.head_dim() != head_dim) {
        throw std::runtime_error("CudaAttentionDecode: KV cache shape does not match attention shape");
    }
}




// 检验attention的输入参数
static void ValidateAttentionDeviceInputs(const CudaTensor& q, const CudaKvCache& kv_cache, int layer, int pos, int n_heads, int n_kv_heads, int head_dim, int device_id) {
    if (q.device_id() != device_id) {
        throw std::runtime_error("CudaAttentionDecodeDeviceInput: q tensor is on a different CUDA ""device");
    }

    if (q.size() != static_cast<size_t>(n_heads) * static_cast<size_t>(head_dim)) {
        throw std::runtime_error("CudaAttentionDecodeDeviceInput: q shape mismatch");
    }

    if (kv_cache.empty()) {
        throw std::runtime_error("CudaAttentionDecodeDeviceInput: CUDA KV cache is empty");
    }    

    if (layer < 0 || layer >= kv_cache.n_layers()) {
        throw std::out_of_range("CudaAttentionDecodeDeviceInput: layer out of range");
    }
    if (pos < 0 || pos >= kv_cache.max_seq_len()) {
        throw std::out_of_range("CudaAttentionDecodeDeviceInput: position out of range");
    }
    if (n_heads < 0 || n_kv_heads < 0 || head_dim <= 0) {
        throw std::runtime_error("CudaAttentionDecodeDeviceInput: head counts and head_dim must be ""positive");
    }
    if (n_heads % n_kv_heads != 0) {
        throw std::runtime_error("CudaAttentionDecodeDeviceInput: n_heads must be divisible by ""n_kv_heads");
    }
    if (kv_cache.n_kv_heads() != n_kv_heads || kv_cache.head_dim() != head_dim || kv_cache.device_id() != device_id) {
        throw std::runtime_error("CudaAttentionDecodeDeviceInput: KV cache shape or device does not ""match attention shape");
    }
}




// 用当前的查询向量 Query，去和历史上每一个词对应的 Key 做点积（内积），算出一个分数，代表当前词对那个历史词的“关注度”有多高
static __global__ void AttentionScoresKernel(const float* q, const float* keys, float* scores, int layer, int max_seq_len, int n_heads, int n_kv_heads, int head_dim, int pos, float scale) {
    int h = blockIdx.x;      // 负责哪个注意力头
    int t = blockIdx.y;      // 历史序列中第几个词
    if (h >= n_heads || t > pos) {
        return;
    }


    int kv_group = n_heads / n_kv_heads;         // kv的组
    int kv_head = h / kv_group;                  // 当前h对应的kv头

    // key的显存地址
    size_t key_base = ((static_cast<size_t>(layer) * static_cast<size_t>(max_seq_len) + static_cast<size_t>(t)) * static_cast<size_t>(n_kv_heads) + static_cast<size_t>(kv_head) ) * static_cast<size_t>(head_dim);
    // q的显存地址
    size_t q_base =  static_cast<size_t>(h) * static_cast<size_t>(head_dim);


    __shared__ float partial[kAttentionBlockSize];
    float sum = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        sum += q[q_base + d] * keys[key_base + d];
    }
    partial[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        scores[static_cast<size_t>(h) * static_cast<size_t>(pos + 1) + static_cast<size_t>(t)] = partial[0] * scale;
    }

}


// Softmax 归一化 和 乘上 Value 加权求和
static __global__ void AttentionReduceKernel(const float* scores, const float* values, float* out, int layer, int max_seq_len, int n_heads, int n_kv_heads, int head_dim, int pos) {
    int h = blockIdx.x;
    if (h >= n_heads) {
        return;
    }    

    int kv_group = n_heads / n_kv_heads;
    int kv_head = h / kv_group;
    size_t scores_base = static_cast<size_t>(h) * static_cast<size_t>(pos + 1);

    // 找最大值
    __shared__ float partial[kAttentionBlockSize];
    float local_max = -INFINITY;

    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        local_max = fmaxf(local_max, scores[scores_base + static_cast<size_t>(t)]);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    float max_score = partial[0];
    

    // softmax分母
    float local_sum = 0.0f;
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        local_sum += expf(scores[scores_base + static_cast<size_t>(t)] - max_score);
    }
    partial[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float denom = partial[0];



    // 加权求和
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float value = 0.0f;
        for (int t = 0; t <= pos; ++t) {
            float prob = expf(scores[scores_base + static_cast<size_t>(t)] - max_score) / denom;

            size_t value_idx = ((static_cast<size_t>(layer) * static_cast<size_t>(max_seq_len) + static_cast<size_t>(t))
                               * static_cast<size_t>(n_kv_heads) + static_cast<size_t>(kv_head))
                               * static_cast<size_t>(head_dim) + static_cast<size_t>(d);
            value += prob * values[value_idx];
        }
        out[static_cast<size_t>(h) * static_cast<size_t>(head_dim) + static_cast<size_t>(d)] = value;
    }
}





Tensor CudaAttentionDecode(const Tensor& q, const CudaKvCache& kv_cache, int layer, int pos, int n_heads, int n_kv_heads, int head_dim, int device_id) {
    ValidateAttentionInputs(q, kv_cache, layer, pos, n_heads, n_kv_heads, head_dim);
    CudaSetDevice(device_id);

    const size_t q_bytes = CheckedMul(q.size(), sizeof(float), "q bytes");
    const size_t out_element = CheckedMul(static_cast<size_t>(n_heads), static_cast<size_t>(head_dim), "output elements");
    const size_t out_bytes = CheckedMul(static_cast<size_t>(out_element), sizeof(float), "output bytes");
    const size_t score_element = CheckedMul(static_cast<size_t>(n_heads), static_cast<size_t>(pos + 1), "score elements");
    const size_t score_bytes = CheckedMul(static_cast<size_t>(score_element), sizeof(float), "score bytes");

    CudaDeviceBuffer q_dev(q_bytes, device_id);
    CudaDeviceBuffer out_dev(out_bytes, device_id);
    CudaDeviceBuffer score_dev(score_bytes, device_id);
    q_dev.Upload(q.data.data(), q_bytes);


    // 缩放因子
    float scale = 1.0f / std::sqrt(static_cast<size_t>(head_dim));

    dim3 score_grid(static_cast<unsigned int>(n_heads), static_cast<unsigned int>(pos + 1));
    AttentionScoresKernel<<<score_grid, kAttentionBlockSize>>>(
        static_cast<const float*>(q_dev.data()),
        static_cast<const float*>(kv_cache.keys_data()),
        static_cast<float*>(score_dev.data()),
        layer,
        kv_cache.max_seq_len(),
        n_heads,
        n_kv_heads,
        head_dim,
        pos,
        scale
    );
    CheckLastAttentionKernel("AttentionScoresKernel");

    AttentionReduceKernel<<<n_heads, kAttentionBlockSize>>>(
        static_cast<const float*>(score_dev.data()),
        static_cast<const float*>(kv_cache.values_data()),
        static_cast<float*>(out_dev.data()),
        layer,
        kv_cache.max_seq_len(),
        n_heads,
        n_kv_heads,
        head_dim,
        pos
    );
    CheckLastAttentionKernel("AttentionReduceKernel");

    Tensor out({n_heads, head_dim}, 0.0f);
    out_dev.Download(out.data.data(), out_bytes);
    return out;
}


CudaTensor CudaAttentionDecodeDeviceInput(const CudaTensor& q, const CudaKvCache& kv_cache, int layer, int pos, int n_heads, int n_kv_heads, int head_dim, int device_id) {
    ValidateAttentionDeviceInputs(q, kv_cache, layer, pos, n_heads, n_kv_heads, head_dim, device_id);
    CudaSetDevice(device_id);


    const size_t out_element = CheckedMul(static_cast<size_t>(n_heads), static_cast<size_t>(head_dim), "output elements");
    const size_t score_element = CheckedMul(static_cast<size_t>(n_heads), static_cast<size_t>(pos + 1), "score elements");
    const size_t score_bytes = CheckedMul(static_cast<size_t>(score_element), sizeof(float), "score bytes");

    CudaDeviceBuffer score_dev(score_bytes, device_id);
    CudaTensor out({static_cast<int>(out_element)}, device_id);


    // 缩放因子
    float scale = 1.0f / std::sqrt(static_cast<size_t>(head_dim));

    dim3 score_grid(static_cast<unsigned int>(n_heads), static_cast<unsigned int>(pos + 1));
    AttentionScoresKernel<<<score_grid, kAttentionBlockSize>>>(
        static_cast<const float*>(q.data()),
        static_cast<const float*>(kv_cache.keys_data()),
        static_cast<float*>(score_dev.data()),
        layer,
        kv_cache.max_seq_len(),
        n_heads,
        n_kv_heads,
        head_dim,
        pos,
        scale
    );
    CheckLastAttentionKernel("AttentionScoresKernel");

    AttentionReduceKernel<<<n_heads, kAttentionBlockSize>>>(
        static_cast<const float*>(score_dev.data()),
        static_cast<const float*>(kv_cache.values_data()),
        static_cast<float*>(out.data()),
        layer,
        kv_cache.max_seq_len(),
        n_heads,
        n_kv_heads,
        head_dim,
        pos
    );
    CheckLastAttentionKernel("AttentionReduceKernel");


    return out;
}

}