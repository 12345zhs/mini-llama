#include "mini_llama/cuda_ops.h"
#include "mini_llama/cuda_runtime.h"


#include <cuda_runtime_api.h>
#include <cmath>
#include <limits>


namespace mini_llama {

constexpr int kBlockSize = 256;

static void CheckCudaOps(cudaError_t error, const char* expr) {
    if (error != cudaSuccess) {
        throw std::runtime_error("CUDA ops error in " + std::string(expr) + ": " + cudaGetErrorString(error));
    }
}


static void CheckLastKernel(const char* name) {
    return CheckCudaOps(cudaGetLastError(), name);
}



// 计算一个数组内所有元素的平方和
static __global__ void SumSquaresKernel(const float* x, float* sum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(sum, x[i] * x[i]);
    }
}



// 均方根归一化
static __global__ void RmsNormKernel(const float* x, const float* weight, float* y, const float* sum, int n, float eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float scale = rsqrtf(*sum / n + eps);
        y[i] = x[i] * scale * weight[i];
    }
}




// Silu激活函数
static __global__ void SiluKernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float v = x[i];
        y[i] = v / (1.0f + expf(-v));
    }
}



// 逐元素相乘
static __global__ void ElementwiseMulKernel(const float* a, const float* b, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a[i] * b[i];
    }
}



// 逐元素相加
static __global__ void ElementwiseAddKernel(const float* a, const float* b, float* y, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        y[i] = a[i] + b[i];
    }
}



// 词嵌入查表
static __global__ void EmbeddingLookupKernel(const float* embedding, float* y, int token_id, int dim) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < dim) {
        y[i] = embedding[token_id * dim + i];
    }

}



// Softmax中求max
static __global__ void SoftmaxMaxKernel(const float* x, float* max_out, int n) {
    __shared__ float shared[kBlockSize];
    int tid = threadIdx.x;
    float local_max = -3.402823466e+38F;
    for (int i = tid; i < n; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }

    shared[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        *max_out = shared[tid];
    }

}



// softmax中的指数求和
static __global__ void SoftmaxExpSumKernel(const float* x, float* y, const float* max_value, float* sum_out, int n) {
    __shared__ float shared[kBlockSize];
    int tid = threadIdx.x;
    float local_sum = 0.0f;
    float max_v = *max_value;
    for (int i = tid; i < n; i += blockDim.x) {
        float e = expf(x[i] - max_v);
        y[i] = e;
        local_sum += e;
    }
    shared[tid] = local_sum;
    __syncthreads();


    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *sum_out = shared[tid];
    }
}




// softmax
static __global__ void SoftmaxNormKernel(float* y, const float* sum, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        y[i] /= *sum;
    }
}



// Normal的Rope
static __global__ void RopeNormalKernal(float* x, int n_heads, int head_dim, int pos, float theta) {
    int pair_index = blockDim.x * blockIdx.x + threadIdx.x;
    int pairs_per_head = head_dim / 2;
    int total_pairs = n_heads * pairs_per_head;

    if (pair_index >= total_pairs) {
        return;
    }

    int head = pair_index / pairs_per_head;
    int pair = pair_index % pairs_per_head;
    int dim = pair* 2;
    int base = head * head_dim + dim;

    float freq = 1.0f / powf(theta, static_cast<float>(dim) / static_cast<float>(head_dim));

    float cos_val = cosf(static_cast<float>(pos) * freq);
    float sin_val = sinf(static_cast<float>(pos) * freq);
    float x1 = x[base];
    float x2 = x[base + 1];
    x[base] = x1 * cos_val - x2 * sin_val;
    x[base + 1] = x1 * sin_val + x2 * cos_val;
}



// NeoX的Rope
static __global__ void RopeNeoXKernel(float* x, int n_heads, int head_dim, int pos, float theta) {
    int pair_index = blockDim.x * blockIdx.x + threadIdx.x;

    int half_dim = head_dim / 2;
    int total_pairs = half_dim * n_heads;
    if (pair_index >= total_pairs) {
        return;
    }

    int head = pair_index / half_dim;
    int pair = pair_index % half_dim;
    int base = head * head_dim;
    float freq  = 1.0f / powf(theta, static_cast<float>(2 * pair) / static_cast<float>(head_dim));

    float cos_val = cosf(static_cast<float>(pos) * freq);
    float sin_val = sinf(static_cast<float>(pos) * freq);
    float x1 = x[base + pair];
    float x2 = x[base + pair + half_dim];
    x[base + pair] = x1 * cos_val - x2 * sin_val;
    x[base + pair + half_dim] = x1 * sin_val + x2 * cos_val;
}



// 需要分配多少个block
static int GridFor(int n) {
    return (n + kBlockSize - 1) / kBlockSize;
}


// 两个tensor形状是否相同
static void RequireSameShape(const Tensor& a, const Tensor& b, const char* caller) {
    if (a.shape != b.shape) {
        throw std::runtime_error(std::string(caller) + ": shape mismatch " + a.ShapeToStringShort() + " vs " + b.ShapeToStringShort());
    }
}


// 两个cudaTensor形状,device_id是否相同
static void RequireSameShape(const CudaTensor& a, const CudaTensor& b, const char* caller) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error(std::string(caller) + ": shape mismatch " + a.ShapeStringShort() + " vs " + b.ShapeStringShort());
    }
    if (a.device_id() != b.device_id()) {
        throw std::runtime_error(std::string(caller) + ": tensors are on different CUDA devices");
    }
}



// tensor的维度必须为1
static void Require1D(const Tensor& x, const char* caller) {
    if (x.num_dims() != 1) {
        throw std::runtime_error(std::string(caller) + ": expected 1D tensor, got " + x.ShapeToStringShort());
    }
    if (x.num_elements() == 0) {
        throw std::runtime_error(std::string(caller) + ": empty tensor");
    } 
}



// cudaTensor的维度必须为1
static void Require1D(const CudaTensor& x, const char* caller) {
    if (x.num_dims() != 1) {
        throw std::runtime_error(std::string(caller) + ": expected 1D tensor, got " + x.ShapeStringShort());
    }
    if (x.size() == 0) {
        throw std::runtime_error(std::string(caller) + ": empty tensor");
    }
}


// Rope的输入检测
static void ValidateRopeInputs(const Tensor& q, const Tensor& k, int pos, float theta) {
    if (q.shape.size() != 2 || k.shape.size() != 2) {
        throw std::runtime_error("CudaRope: expected 2D tensors");
    }

    if (pos < 0) {
        throw std::out_of_range("CudaRope: position must be non-negative");
    }

    if (!std::isfinite(theta) || theta <= 0.0f) {
        throw std::runtime_error("CudaRope: theta must be finite and positive");
    }

    if (q.shape[1] != k.shape[1]) {
        throw std::runtime_error("CudaRope: q and k head_dim mismatch " + q.ShapeToStringShort() + " vs " + k.ShapeToStringShort());
    }

    if (q.shape[1] <= 0 || q.shape[1] % 2 != 0) {
        throw std::runtime_error("CudaRope: head_dim must be positive and even");
    }

}


// Rope的cudaTensor输入检测
static void ValidateRopeDeviceInputs(const CudaTensor& q, const CudaTensor& k, int n_heads, int n_kv_heads, int head_dim, int pos, float theta, int device_id) {
    if (q.device_id() != device_id || k.device_id() != device_id) {
        throw std::runtime_error("CudaRopeDeviceInput: input tensor is on a different CUDA device");
    }
    if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || head_dim % 2 != 0) {
        throw std::runtime_error("CudaRopeDeviceInput: invalid head shape");
    }
    if (q.size() != static_cast<size_t>(n_heads) * static_cast<size_t>(head_dim)) {
        throw std::runtime_error("CudaRopeDeviceInput: q shape mismatch");
    }
    if (k.size() != static_cast<size_t>(n_kv_heads) * static_cast<size_t>(head_dim)) {
        throw std::runtime_error("CudaRopeDeviceInput: k shape mismatch");
    }

    if (pos < 0) {
        throw std::out_of_range("CudaRopeDeviceInput: position must be non-negative");
    }

    if (!std::isfinite(theta) || theta <= 0.0f) {
        throw std::runtime_error("CudaRopeDeviceInput: theta must be finite and positive");
    }
}



// embedding的检验
static void ValidateEmbeddingDeviceWeightInputs(const void* embedding_data, const std::vector<int>& embedding_shape, int token_id) {
    if (embedding_data == nullptr) {
        throw std::runtime_error("CudaEmbeddingLookupDeviceWeight: embedding data is null");
    }

    if (embedding_shape.size() != 2) {
        throw std::runtime_error("CudaEmbeddingLookupDeviceWeight: expected 2D embedding weight");
    }

    int vocab_size = embedding_shape[0];
    int dim = embedding_shape[1];
    if (vocab_size < 0 || dim < 0) {
        throw std::runtime_error("CudaEmbeddingLookupDeviceWeight: embedding shape must be positive");
    }
    if (token_id < 0 || token_id >= vocab_size) {
        throw std::out_of_range("CudaEmbeddingLookupDeviceWeight: token id out of range");
    }
}




// RmsNorm的检测
static void ValidateRmsNormDeviceWeightInputs(const CudaTensor& x, const void* weight_data, const std::vector<int>& weight_shape, float eps, int device_id) {
    Require1D(x, "CudaRmsNormDeviceWeight");

    if (weight_data == nullptr) {
        throw std::runtime_error("CudaRmsNormDeviceWeight: x and weight shape mismatch");
    }
    if (x.shape() != weight_shape) {
        throw std::runtime_error("CudaRmsNormDeviceWeight: x and weight shape mismatch");
    }
    if (x.device_id() != device_id) {
        throw std::runtime_error("CudaRmsNormDeviceWeight: input tensor is on a different CUDA device");
    }

    if (!std::isfinite(eps) || eps <= 0.0f) {
        throw std::runtime_error("CudaRmsNormDeviceWeight: eps must be finite and positive");
    }
}




// 逐元素相加检测
static void ValidateAddDeviceWeightInputs(const CudaTensor& a, const void* b_data, const std::vector<int>& b_shape, int device_id) {
    if (a.device_id() != device_id) {
        throw std::runtime_error("CudaElementwiseAddDeviceWeight: input tensor is on a different CUDA ""device");
    }

    if (b_data == nullptr) {
        throw std::runtime_error("CudaElementwiseAddDeviceWeight: weight data is null");
    }

    if (a.shape() != b_shape) {
        throw std::runtime_error("CudaElementwiseAddDeviceWeight: input and weight shape mismatch");
    }
}






//RmsNorm 先加载数据到GPU，计算完后再返回cpu
Tensor CudaRmsNorm(const Tensor& x, const Tensor& weight, float eps, int device_id) {
    Require1D(x, "CudaRmsNorm");
    Require1D(weight, "CudaRmsNorm");

    if (x.shape != weight.shape) {
        throw std::runtime_error("CudaRmsNorm: x and weight shape mismatch");
    }
    if (!std::isfinite(eps) || eps <= 0.0f) {
        throw std::runtime_error("CudaRmsNorm: eps must be finite and positive");
    }
    
    CudaSetDevice(device_id);
    Tensor y(x.shape, 0.0f);
    CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
    CudaDeviceBuffer w_dev(weight.size() * sizeof(float), device_id);
    CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);

    x_dev.Upload(x.data.data(), x.size() * sizeof(float));
    w_dev.Upload(weight.data.data(), weight.size() * sizeof(float));

    CudaDeviceBuffer sum_dev(sizeof(float), device_id);
    CheckCudaOps(cudaMemset(sum_dev.data(), 0.0f, sizeof(float)), "cudaMemset(RmsNorm sum)");

    int n = static_cast<int>(x.size());
    SumSquaresKernel<<<GridFor(n), kBlockSize>>>(
        static_cast<const float*>(x_dev.data()),
        static_cast<float*>(sum_dev.data()),
        n
    );
    CheckLastKernel("SumSquaresKernel");

    RmsNormKernel<<<GridFor(n), kBlockSize>>>(
        static_cast<const float*>(x_dev.data()),
        static_cast<const float*>(w_dev.data()),
        static_cast<float*>(y_dev.data()),
        static_cast<const float*>(sum_dev.data()),
        n, eps
    );
    CheckLastKernel("RmsNormKernel");

    y_dev.Download(y.data.data(), y.size() * sizeof(float));
    return y;
}


// weight不是cudaTensor， 整个流程在cuda上
CudaTensor CudaRmsNormDeviceInput(const CudaTensor& x, const Tensor& weight, float eps, int device_id) {
    Require1D(x, "CudaRmsNormDeviceInput");
    Require1D(weight, "CudaRmsNormDeviceInput");
    if (x.shape() != weight.shape) {
        throw std::runtime_error("CudaRmsNormDeviceInput: x and weight shape mismatch");
    }
    if (!std::isfinite(eps) || eps <= 0.0f) {
        throw std::runtime_error("CudaRmsNormDeviceInput: eps must be finite and positive");
    }
    if (x.device_id() != device_id) {
        throw std::runtime_error("CudaRmsNormDeviceInput: input tensor is on a different CUDA device");
    }


    CudaSetDevice(device_id);
    CudaDeviceBuffer w_dev(weight.size() * sizeof(float), device_id);
    w_dev.Upload(weight.data.data(), weight.size() * sizeof(float));
    return CudaRmsNormDeviceWeight(x, w_dev.data(), weight.shape, eps, device_id);

}


// RmsNorm整个在gpu上
CudaTensor CudaRmsNormDeviceWeight(const CudaTensor& x, const void* weight_data, const std::vector<int>& weight_shape, float eps, int device_id) {
    ValidateRmsNormDeviceWeightInputs(x, weight_data, weight_shape, eps, device_id);
    
    CudaSetDevice(device_id);
    CudaTensor y(x.shape(), device_id);

    CudaDeviceBuffer sum_dev(sizeof(float), device_id);
    CheckCudaOps(cudaMemset(sum_dev.data(), 0, sizeof(float)), "cudaMemset(RmsNorm sum)");

    int n = x.size();
    SumSquaresKernel<<<GridFor(n), kBlockSize>>>(
        static_cast<const float*>(x.data()),
        static_cast<float*>(sum_dev.data()),
        n
    );
    CheckLastKernel("SumSquaresKernel");

    RmsNormKernel<<<GridFor(n), kBlockSize>>>(
        static_cast<const float*>(x.data()),
        static_cast<const float*>(weight_data),
        static_cast<float*>(y.data()),
        static_cast<const float*>(sum_dev.data()),
        n, eps
    );
    CheckLastKernel("RmsNormKernel");
    
    return y;
}






// Silu 先加载数据到GPU，计算完后再返回cpu
Tensor CudaSilu(const Tensor& x, int device_id) {
    CudaSetDevice(device_id);
    Tensor y(x.shape, 0.0f);
    CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
    CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
    x_dev.Upload(x.data.data(), x.size() * sizeof(float));

    int n = static_cast<int>(x.size());
    SiluKernel<<<GridFor(n), kBlockSize>>>(
        static_cast<const float*>(x_dev.data()),
        static_cast<float*>(y_dev.data()),
        n
    );
    CheckLastKernel("SiluKernel");

    y_dev.Download(y.data.data(), sizeof(float) * y.size());
    return y;
}


// Silu整个在gpu上
CudaTensor CudaSiluDeviceInput(const CudaTensor& x, int device_id) {
    if (x.device_id() != device_id) {
        throw std::runtime_error("CudaSiluDeviceInput: input tensor is on a different CUDA device");
    }
    CudaSetDevice(device_id);
    CudaTensor y(x.shape(), device_id);

    int n = static_cast<int>(x.size());
    SiluKernel<<<GridFor(n), kBlockSize>>>(
        static_cast<const float*>(x.data()),
        static_cast<float*>(y.data()),
        n
    );
    return y;
}






// ElementwiseMul先加载数据到GPU，计算完后再返回cpu
Tensor CudaElementwiseMul(const Tensor& a, const Tensor& b, int device_id) {
    RequireSameShape(a, b, "CudaElementwiseMul");
    CudaSetDevice(device_id);

    Tensor y(a.shape, 0.0f);
    CudaDeviceBuffer a_dev(a.size() * sizeof(float), device_id);
    CudaDeviceBuffer b_dev(b.size() * sizeof(float), device_id);
    CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
    a_dev.Upload(a.data.data(), a.size() * sizeof(float));
    b_dev.Upload(b.data.data(), b.size() * sizeof(float));

    int n = static_cast<int>(a.size());
    ElementwiseMulKernel<<<GridFor(n), kBlockSize>>> (
        static_cast<const float*>(a_dev.data()),
        static_cast<const float*>(b_dev.data()),
        static_cast<float*>(y_dev.data()),
        n
    );
    y_dev.Download(y.data.data(), y.size() * sizeof(float));
    return y;
}


// ElementwiseMul整个在gpu上
CudaTensor CudaElementwiseMulDeviceInput(const CudaTensor& a, const CudaTensor& b, int device_id) {
    RequireSameShape(a, b, "CudaElementwiseMulDeviceInput");
    if (a.device_id() != device_id) {
        throw std::runtime_error("CudaElementwiseMulDeviceInput: input tensor is on a different CUDA ""device");
    }

    CudaSetDevice(device_id);
    CudaTensor y(a.shape(), device_id);

    int n = static_cast<int>(a.size());
    ElementwiseMulKernel<<<GridFor(n), kBlockSize>>> (
        static_cast<const float*>(a.data()),
        static_cast<const float*>(b.data()),
        static_cast<float*>(y.data()),
        n
    );
    CheckLastKernel("ElementwiseMulKernel");

    return y;
}






// ElementwiseAdd先加载数据到GPU，计算完后再返回cpu
Tensor CudaElementwiseAdd(const Tensor& a, const Tensor& b, int device_id) {
    RequireSameShape(a, b, "CudaElementwiseAdd");
    CudaSetDevice(device_id);

    Tensor y(a.shape, 0.0f);
    CudaDeviceBuffer a_dev(a.size() * sizeof(float), device_id);
    CudaDeviceBuffer b_dev(b.size() * sizeof(float), device_id);
    CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
    a_dev.Upload(a.data.data(), a.size() * sizeof(float));
    b_dev.Upload(b.data.data(), b.size() * sizeof(float));
    int n = static_cast<int>(a.size());

    ElementwiseAddKernel<<<GridFor(n), kBlockSize>>> (
        static_cast<const float*>(a_dev.data()),
        static_cast<const float*>(b_dev.data()),
        static_cast<float*>(y_dev.data()),
        n
    );
    CheckLastKernel("ElementwiseAddKernel");
    y_dev.Download(y.data.data(), sizeof(float) * y.size());
    return y;
}

// ElementwiseAdd整个在gpu上
CudaTensor CudaElementwiseAddDeviceInput(const CudaTensor& a, const CudaTensor& b, int device_id) {
    RequireSameShape(a, b, "CudaElementwiseAddDeviceInput");
    if (a.device_id() != device_id) {
        throw std::runtime_error("CudaElementwiseAddDeviceInput: input tensor is on a different CUDA ""device");
    }
    CudaTensor y(a.shape(), device_id);
    CudaSetDevice(device_id);
    int n = static_cast<int>(a.size());
    ElementwiseAddKernel<<<GridFor(n), kBlockSize>>> (
        static_cast<const float*>(a.data()),
        static_cast<const float*>(b.data()),
        static_cast<float*>(y.data()),
        n
    );
    CheckLastKernel("ElementwiseAddKernel");
    return y;
}

// 整个流程在cuda上,但b参数传的指针
CudaTensor CudaElementwiseAddDeviceWeight(const CudaTensor& a, const void* b_data, const std::vector<int>& b_shape, int device_id) {
    ValidateAddDeviceWeightInputs(a, b_data, b_shape, device_id);
    CudaSetDevice(device_id);
    CudaTensor y(a.shape(), device_id);
    int n = static_cast<int>(a.size());
    ElementwiseAddKernel<<<GridFor(n), kBlockSize>>> (
        static_cast<const float*>(a.data()),
        static_cast<const float*>(b_data),
        static_cast<float*>(y.data()),
        n
    );
    CheckLastKernel("ElementwiseAddKernel");
    return y;
}







Tensor CudaSoftmax(const Tensor& x, int device_id) {
    Require1D(x, "CudaSoftmax");
    CudaSetDevice(device_id);
    Tensor y(x.shape, 0.0f);
    CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
    CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
    CudaDeviceBuffer max_dev(sizeof(float), device_id);
    CudaDeviceBuffer sum_dev(sizeof(float), device_id);
    x_dev.Upload(x.data.data(), x.size() * sizeof(float));

    int n = static_cast<int>(x.size());
    SoftmaxMaxKernel<<<GridFor(n), kBlockSize>>>(
        static_cast<const float*>(x_dev.data()),
        static_cast<float*>(max_dev.data()),
        n
    );
    CheckLastKernel("SoftmaxMaxKernel");


    SoftmaxExpSumKernel<<<GridFor(n), kBlockSize>>>(
        static_cast<const float*>(x_dev.data()),
        static_cast<float*>(y_dev.data()),
        static_cast<const float*>(max_dev.data()),
        static_cast<float*>(sum_dev.data()),
        n
    );
    CheckLastKernel("SoftmaxExpSumKernel");


    SoftmaxNormKernel<<<GridFor(n), kBlockSize>>>(
        static_cast<float*>(y_dev.data()),
        static_cast<const float*>(sum_dev.data()),
        n
    );
    CheckLastKernel("SoftmaxNormKernel");

    y_dev.Download(y.data.data(), sizeof(float) * y.size());
    return y;
}







CudaTensor CudaEmbeddingLookupDeviceWeight(const void* embedding_data, const std::vector<int>& embedding_shape, int token_id, int device_id) {
    ValidateEmbeddingDeviceWeightInputs(embedding_data, embedding_shape, token_id);
    cudaSetDevice(device_id);

    int dim = embedding_shape[1];
    CudaTensor y({dim}, device_id);
    EmbeddingLookupKernel<<<GridFor(dim), kBlockSize>>>(
        static_cast<const float*>(embedding_data),
        static_cast<float*>(y.data()),
        token_id, dim
    );
    CheckLastKernel("EmbeddingLookupKernel");
    return y;
}





void CudaRope(Tensor& q, Tensor& k, int pos, float theta, RopeType rope_type, int device_id) {
    ValidateRopeInputs(q, k, pos, theta);
    CudaSetDevice(device_id);
    CudaDeviceBuffer q_dev(q.size() * sizeof(float), device_id);
    CudaDeviceBuffer k_dev(k.size() * sizeof(float), device_id);
    q_dev.Upload(q.data.data(), q.size() * sizeof(float));
    k_dev.Upload(k.data.data(), k.size() * sizeof(float));

    int q_pairs = q.shape[0] * (q.shape[1] / 2);
    int k_pairs = k.shape[0] * (k.shape[1] / 2);

    if (rope_type == RopeType::kNormal) {
        RopeNormalKernal<<<GridFor(q_pairs), kBlockSize>>>(
            static_cast<float*>(q_dev.data()),
            q.shape[0],
            q.shape[1],
            pos,
            theta
        );
        CheckLastKernel("RopeNormalKernel(q)");


        RopeNormalKernal<<<GridFor(k_pairs), kBlockSize>>>(
            static_cast<float*>(k_dev.data()),
            k.shape[0],
            k.shape[1],
            pos,
            theta
        );
        CheckLastKernel("RopeNormalKernel(k)");
    } else {
        RopeNeoXKernel<<<GridFor(q_pairs), kBlockSize>>>(
            static_cast<float*>(q_dev.data()),
            q.shape[0],
            q.shape[1],
            pos,
            theta
        );
        CheckLastKernel("RopeNeoXKernel(q)");


        RopeNeoXKernel<<<GridFor(k_pairs), kBlockSize>>>(
            static_cast<float*>(k_dev.data()),
            k.shape[0],
            k.shape[1],
            pos,
            theta
        );
        CheckLastKernel("RopeNeoXKernel(k)");
    }

    q_dev.Download(q.data.data(), q.size() * sizeof(float));
    k_dev.Download(k.data.data(), k.size() * sizeof(float));

}

void CudaRopeDeviceInput(CudaTensor& q, CudaTensor& k, int n_heads, int n_kv_heads, int head_dim, int pos, float theta, RopeType rope_type, int device_id) {
    ValidateRopeDeviceInputs(q, k, n_heads, n_kv_heads, head_dim, pos, theta, device_id);
    CudaSetDevice(device_id);

    int q_pairs = n_heads * (head_dim / 2);
    int k_pairs = n_kv_heads * (head_dim / 2);

    if (rope_type == RopeType::kNormal) {
        RopeNormalKernal<<<GridFor(q_pairs), kBlockSize>>>(
            static_cast<float*>(q.data()),
            n_heads,
            head_dim,
            pos,
            theta
        );
        CheckLastKernel("RopeNormalKernel(q)");


        RopeNormalKernal<<<GridFor(k_pairs), kBlockSize>>>(
            static_cast<float*>(k.data()),
            n_kv_heads,
            head_dim,
            pos,
            theta
        );
        CheckLastKernel("RopeNormalKernel(k)");
    } else {
        RopeNeoXKernel<<<GridFor(q_pairs), kBlockSize>>>(
            static_cast<float*>(q.data()),
            head_dim,
            pos,
            pos,
            theta
        );
        CheckLastKernel("RopeNeoXKernel(q)");


        RopeNeoXKernel<<<GridFor(k_pairs), kBlockSize>>>(
            static_cast<float*>(k.data()),
            head_dim,
            pos,
            pos,
            theta
        );
        CheckLastKernel("RopeNeoXKernel(k)");
    }
}



}