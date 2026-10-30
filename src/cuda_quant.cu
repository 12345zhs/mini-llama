#include "mini_llama/cuda_quant.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>



namespace mini_llama {

constexpr int kBlockSize = 128;

static void CheckCudaQuant(cudaError_t error, const char* expr) {
    if (error != cudaSuccess) {
        throw std::runtime_error("CUDA quant error in " + std::string(expr) + ": " + cudaGetErrorString(error));
    }
}


static void CheckLastKernel(const char* name) {
    CheckCudaQuant(cudaGetLastError(), name);
}



// 2个字节的Fp16的原始二进制数据，还原为标准的32位浮点数F32
static __device__ float HalfBitsToFloat(uint16_t bits) {
    __half_raw raw;
    raw.x = bits;
    return __half2float(raw);
}


// Q80量化矩阵乘法
//                                                                               输入矩阵行    输入特征矩阵        输出特征矩阵        权重矩阵一行有多少块
//                           [rows, in_features]  [out_features, in_features]  [rows, out_features]
static __global__ void Q80LinearKernel(const float* x, const BlockQ80* weight, float* y, int rows, int in_features, int out_features, int blocks_per_row) {
    int out = blockIdx.x * blockDim.x + threadIdx.x;  // 负责输出矩阵的哪一列
    int row = blockIdx.y;                             // 负责输出矩阵的哪一行
 
    if (out >= out_features || row >= rows) {
        return;
    }

    const float* x_row = x + static_cast<size_t>(row) *  in_features;   // 显存全局地址
    const BlockQ80* w_row = weight + static_cast<size_t>(out) * blocks_per_row;

    float sum = 0.0f;
    for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
        const BlockQ80& block = w_row[block_index];
        float d = HalfBitsToFloat(block.d);
        int base_k = block_index * kQ80BlockSize;
        int k_end = base_k + kQ80BlockSize < in_features ? base_k + kQ80BlockSize : in_features;

        for (int k = base_k; k < k_end; ++k) {
            sum += d * static_cast<float>(block.qs[k - base_k]) * x_row[k];
        }
    } 

    y[static_cast<size_t>(row) * out_features + out] = sum;
}


// 给定一个压缩块Q40和索引，还原成float
static __device__ __forceinline__ float Q40Value(const BlockQ40& block, int flat_index) {
    int q = flat_index < 16 ? static_cast<int>(block.qs[flat_index] & 0x0f) : static_cast<int>(block.qs[flat_index - 16] >> 4);
    return HalfBitsToFloat(block.d) * static_cast<float>(q - 8);
}


// 给定一个压缩块Q41和索引，还原成float
static __device__ __forceinline__ float Q41Value(const BlockQ41& block, int flat_index) {
    int q = flat_index < 16 ? static_cast<int>(block.qs[flat_index] & 0x0f) : static_cast<int>(block.qs[flat_index - 16] >> 4);
    return HalfBitsToFloat(block.d) * HalfBitsToFloat(block.m) * static_cast<float>(q); 
}



// Q40量化矩阵乘法
//                                                                               输入矩阵行    输入特征矩阵        输出特征矩阵        权重矩阵一行有多少块
//                           [rows, in_features]  [out_features, in_features]  [rows, out_features]
static __global__ void Q40LinearKernel(const float* x, const BlockQ40* weight, float* y, int rows, int in_features, int out_features, int blocks_per_row) {
    int out = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;

    if (out >= out_features || row >= rows) {
        return;
    }

    const float* x_row = x + static_cast<size_t>(row) * in_features;
    const BlockQ40* w_row = weight + static_cast<size_t>(out) * blocks_per_row;
    float sum = 0.0f;
    for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
        const BlockQ40& block = w_row[block_index];
        int base_k = block_index * kQ40BlockSize;
        int k_end = base_k + kQ40BlockSize < in_features ? base_k + kQ40BlockSize : in_features;
        
        for (int k = base_k; k < k_end; ++k) {
            sum += Q40Value(block, k - base_k) * x_row[k];
        }
    }
    y[static_cast<size_t>(row) * out_features + out] = sum;
}



// Q41量化矩阵乘法
//                                                                               输入矩阵行    输入特征矩阵        输出特征矩阵        权重矩阵一行有多少块
//                           [rows, in_features]  [out_features, in_features]  [rows, out_features]
static __global__ void Q41LinearKernel(const float* x, const BlockQ41* weight, float* y, int rows, int in_features, int out_features, int blocks_per_row) {
    int out = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;

    if (out >= out_features || row >= rows) {
        return;
    }

    const float* x_row = x + static_cast<size_t>(row) * in_features;
    const BlockQ41* w_row = weight + static_cast<size_t>(out) * blocks_per_row;
    float sum = 0.0f;
    for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
        const BlockQ41& block = w_row[block_index];
        int base_k = block_index * kQ41BlockSize;
        int k_end = base_k + kQ41BlockSize < in_features ? base_k + kQ41BlockSize : in_features;
        
        for (int k = base_k; k < k_end; ++k) {
            sum += Q41Value(block, k - base_k) * x_row[k];
        }
    }
    y[static_cast<size_t>(row) * out_features + out] = sum;
}



// 结果加上bias
static void AddBiasInPlace(Tensor& y, const Tensor& bias, int rows, int cols, const char* caller) {
    if (bias.num_dims() != 1 || bias.shape[0] != cols) {
        throw std::runtime_error(std::string(caller) + ": expected bias shape [out_features]");
    }
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; ++col) {
            y.data[static_cast<size_t>(row) * cols + col] += bias.data[col];
        }
    }
}



// 从一个张量（输入矩阵 X）的形状中，精准地提取出“输入特征维度
static int Q80InFeaturesFromShape(const std::vector<int>& x_shape, const std::string& x_shape_str) {
    if (x_shape.size() == 1) {
        return x_shape[0];
    }
    if (x_shape.size() == 2) {
        return x_shape[1];
    }

    throw std::runtime_error( "CudaQ80Linear: expected x shape [in_features] or [batch, in_features], " "got " + x_shape_str);
}



// 检查Q80权重的形状
static void ValidateQ80LinearShape(const std::vector<int>& x_shape, const std::string& x_shape_str, const std::vector<int>& weight_shape, size_t block_count) {
    if (weight_shape.size() != 2) {
        throw std::runtime_error("CudaQ80Linear: expected weight shape [out_features, in_features]");
    }

    int in_features = Q80InFeaturesFromShape(x_shape, x_shape_str);
    if (in_features <= 0 || weight_shape[0] <= 0 || weight_shape[1] <= 1) {
        throw std::runtime_error("CudaQ80Linear: empty tensors are not supported");
    } 

    if (in_features != weight_shape[1]) {
        throw std::runtime_error("CudaQ80Linear: dimension mismatch x=" + x_shape_str + " weight=[" + std::to_string(weight_shape[0]) + ", " + std::to_string(weight_shape[1]) + "]");
    }

    int blocks_per_row = (in_features + kQ80BlockSize - 1) / kQ80BlockSize;
    size_t expected_blocks = static_cast<size_t>(weight_shape[0]) * blocks_per_row;
    if (block_count != expected_blocks) {
        throw std::runtime_error("CudaQ80Linear: block count mismatch: expected " + std::to_string(expected_blocks) + ", got " + std::to_string(block_count));
    }
}


// 检查Linear的 Q80权重输入
static void ValidateQ80LinearInputs(const Tensor& x, const std::vector<int>& weight_shape, size_t block_count) {
    ValidateQ80LinearShape(x.shape, x.ShapeToStringShort(), weight_shape, block_count);
}



// 通用的维度提取器
static int QuantInFeaturesFromShape(const std::vector<int>& x_shape, const std::string& x_shape_str, const char* caller) {
    if (x_shape.size() == 1) {
        return x_shape[0];
    }
    if (x_shape.size() == 2) {
        return x_shape[1];
    }
    throw std::runtime_error( std::string(caller) + ": expected x shape [in_features] or [batch, in_features], got " + x_shape_str);
}



// 检查Linear的输入形状
static void ValidateQuantLinearShape(const std::vector<int>& x_shape, const std::string& x_shape_str, const std::vector<int>& weight_shape, size_t block_count, int block_size, const char* caller) {
    if (weight_shape.size() != 2) {
        throw std::runtime_error(std::string(caller) + ": expected weight shape [out_features, in_features]");
    } 
    int in_features = QuantInFeaturesFromShape(x_shape, x_shape_str, caller);
    if (in_features <= 0 || weight_shape[0] <= 0 || weight_shape[1] <= 0) {
        throw std::runtime_error(std::string(caller) + ": empty tensors are not supported");
    }
    if (weight_shape[1] != in_features) {
        throw std::runtime_error(std::string(caller) + ": dimension mismatch x=" + x_shape_str + " weight=[" + std::to_string(weight_shape[0]) + ", " + std::to_string(weight_shape[1]) + "]");
    }

    int blocks_per_row = (in_features + block_size - 1) / block_size;
    size_t expected_blocks = static_cast<size_t>(weight_shape[0]) * blocks_per_row;
    if (block_count != expected_blocks) {
        throw std::runtime_error(std::string(caller) + ": block count mismatch: expected " + std::to_string(expected_blocks) + ", got " + std::to_string(block_count));
    }
}



// linear的Q80权重在cpu端启动，Gpu执行，最后搬运到cpu
static Tensor CudaQ80LinearWithDevicePointer(const Tensor& x, const void* q8_0_device, size_t q8_0_block_count, const std::vector<int>& weight_shape, const Tensor* bias, int device_id) {
    if (q8_0_device == nullptr) {
        throw std::runtime_error("CudaQ80Linear: device weight pointer is null");
    }
    ValidateQ80LinearInputs(x, weight_shape, q8_0_block_count);

    bool input_is_1d = x.num_dims() == 1;
    int rows = input_is_1d ? 1 : x.shape[0];
    int in_features = input_is_1d ? x.shape[0] : x.shape[1];
    int out_features = weight_shape[0];
    int block_per_row = (in_features + kQ80BlockSize - 1) / kQ80BlockSize;

    Tensor y(input_is_1d ? std::vector<int>{out_features} : std::vector<int>{rows, out_features}, 0.0f);
    CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
    CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
    x_dev.Upload(x.data.data(), x.size() * sizeof(float));

    dim3 block(kBlockSize);
    dim3 grid((out_features + kBlockSize - 1) / kBlockSize, rows);
    Q80LinearKernel<<<grid, block>>>(static_cast<const float*>(x_dev.data()), 
                                     static_cast<const BlockQ80*>(q8_0_device),
                                     static_cast<float*>(y_dev.data()), rows,
                                     in_features, out_features, block_per_row
    );

    CheckLastKernel("Q80LinearKernel");
    y_dev.Download(y.data.data(), y.size() * sizeof(float));

    if (bias != nullptr) {
        AddBiasInPlace(y, *bias, rows, out_features, "CudaQ80Linear");
    }
    return y;
}



// linear的Q80权重在cpu端启动，Gpu执行,直接返回CudaTensor
static CudaTensor CudaQ80LinearWithDeviceInput(const CudaTensor& x, const void* q8_0_device, size_t q8_0_block_count, const std::vector<int>& weight_shape, int device_id) {
    if (q8_0_device == nullptr) {
        throw std::runtime_error("CudaQ80Linear: device weight pointer is null");
    }
    if (x.device_id() != device_id) {
        throw std::runtime_error("CudaQ80Linear: input tensor is on a different CUDA device");
    }

    ValidateQ80LinearShape(x.shape(), x.ShapeStringShort(), weight_shape, q8_0_block_count);

    bool input_is_1d = x.num_dims() == 1;
    int rows = input_is_1d ? 1 : x.shape()[0];
    int in_features = input_is_1d ? x.shape()[0] : x.shape()[1];
    int out_features = weight_shape[0];
    int block_per_row = (in_features + kQ80BlockSize - 1) / kQ80BlockSize;

    CudaTensor y(input_is_1d ? std::vector<int>{out_features} : std::vector<int>{rows, out_features}, device_id);

    dim3 block(kBlockSize);
    dim3 grid((out_features + kBlockSize - 1) / kBlockSize, rows);
    Q80LinearKernel<<<grid, block>>>(static_cast<const float*>(x.data()), 
                                     static_cast<const BlockQ80*>(q8_0_device),
                                     static_cast<float*>(y.data()), rows,
                                     in_features, out_features, block_per_row
    );

    CheckLastKernel("Q80LinearKernel");
    return y;
}



// linear的Q40权重在cpu端启动，Gpu执行，最后搬运到cpu
static Tensor CudaQ40LinearWithDevicePointer(const Tensor& x, const void* q4_0_device, size_t q4_0_block_count, const std::vector<int>& weight_shape, const Tensor* bias, int device_id) {
    static constexpr char* kCaller = "CudaQ40Linear";
    if (q4_0_device == nullptr) {
        throw std::runtime_error("CudaQ80Linear: device weight pointer is null");
    }
    ValidateQuantLinearShape(x.shape, x.ShapeToStringShort(),  weight_shape, q4_0_block_count, kQ40BlockSize, kCaller);

    bool input_is_1d = x.num_dims() == 1;
    int rows = input_is_1d ? 1 : x.shape[0];
    int in_features = input_is_1d ? x.shape[0] : x.shape[1];
    int out_features = weight_shape[0];
    int block_per_row = (in_features + kQ40BlockSize - 1) / kQ40BlockSize;

    Tensor y(input_is_1d ? std::vector<int>{out_features} : std::vector<int>{rows, out_features}, 0.0f);
    CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
    CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
    x_dev.Upload(x.data.data(), x.size() * sizeof(float));

    dim3 block(kBlockSize);
    dim3 grid((out_features + kBlockSize - 1) / kBlockSize, rows);
    Q40LinearKernel<<<grid, block>>>(static_cast<const float*>(x_dev.data()), 
                                     static_cast<const BlockQ40*>(q4_0_device),
                                     static_cast<float*>(y_dev.data()), rows,
                                     in_features, out_features, block_per_row
    );

    CheckLastKernel("Q40LinearKernel");
    y_dev.Download(y.data.data(), y.size() * sizeof(float));

    if (bias != nullptr) {
        AddBiasInPlace(y, *bias, rows, out_features, kCaller);
    }
    return y;
}



// linear的Q40权重在cpu端启动，Gpu执行,直接返回CudaTensor
static CudaTensor CudaQ40LinearWithDeviceInput(const CudaTensor& x, const void* q4_0_device, size_t q4_0_block_count, const std::vector<int>& weight_shape, int device_id) {
    
    static constexpr const char* kCaller = "CudaQ40Linear";
    if (q4_0_device == nullptr) {
        throw std::runtime_error(std::string(kCaller) + ": device weight pointer is null");
    }
    if (x.device_id() != device_id) {
        throw std::runtime_error(std::string(kCaller) + ": input tensor is on a different CUDA device");
    }

    ValidateQuantLinearShape(x.shape(), x.ShapeStringShort(), weight_shape, q4_0_block_count, kQ40BlockSize, kCaller);

    bool input_is_1d = x.num_dims() == 1;
    int rows = input_is_1d ? 1 : x.shape()[0];
    int in_features = input_is_1d ? x.shape()[0] : x.shape()[1];
    int out_features = weight_shape[0];
    int block_per_row = (in_features + kQ40BlockSize - 1) / kQ40BlockSize;

    CudaTensor y(input_is_1d ? std::vector<int>{out_features} : std::vector<int>{rows, out_features}, device_id);

    dim3 block(kBlockSize);
    dim3 grid((out_features + kBlockSize - 1) / kBlockSize, rows);
    Q40LinearKernel<<<grid, block>>>(static_cast<const float*>(x.data()), 
                                     static_cast<const BlockQ40*>(q4_0_device),
                                     static_cast<float*>(y.data()), rows,
                                     in_features, out_features, block_per_row
    );

    CheckLastKernel("Q80LinearKernel");
    return y;
}



// linear的Q41权重在cpu端启动，Gpu执行，最后搬运到cpu
static Tensor CudaQ41LinearWithDevicePointer(const Tensor& x, const void* q4_1_device, size_t q4_1_block_count, const std::vector<int>& weight_shape, const Tensor* bias, int device_id) {
    static constexpr char* kCaller = "CudaQ41Linear";
    if (q4_1_device == nullptr) {
        throw std::runtime_error(std::string(kCaller) + ": device weight pointer is null");
    }
    ValidateQuantLinearShape(x.shape, x.ShapeToStringShort(),  weight_shape, q4_1_block_count, kQ41BlockSize, kCaller);

    bool input_is_1d = x.num_dims() == 1;
    int rows = input_is_1d ? 1 : x.shape[0];
    int in_features = input_is_1d ? x.shape[0] : x.shape[1];
    int out_features = weight_shape[0];
    int block_per_row = (in_features + kQ41BlockSize - 1) / kQ41BlockSize;

    Tensor y(input_is_1d ? std::vector<int>{out_features} : std::vector<int>{rows, out_features}, 0.0f);
    CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
    CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
    x_dev.Upload(x.data.data(), x.size() * sizeof(float));

    dim3 block(kBlockSize);
    dim3 grid((out_features + kBlockSize - 1) / kBlockSize, rows);
    Q41LinearKernel<<<grid, block>>>(static_cast<const float*>(x_dev.data()), 
                                     static_cast<const BlockQ41*>(q4_1_device),
                                     static_cast<float*>(y_dev.data()), rows,
                                     in_features, out_features, block_per_row
    );

    CheckLastKernel("Q41LinearKernel");
    y_dev.Download(y.data.data(), y.size() * sizeof(float));

    if (bias != nullptr) {
        AddBiasInPlace(y, *bias, rows, out_features, kCaller);
    }
    return y;
}



// linear的Q41权重在cpu端启动，Gpu执行,直接返回CudaTensor
static CudaTensor CudaQ41LinearWithDeviceInput(const CudaTensor& x, const void* q4_1_device, size_t q4_1_block_count, const std::vector<int>& weight_shape, int device_id) {
    
    static constexpr const char* kCaller = "CudaQ41Linear";
    if (q4_1_device == nullptr) {
        throw std::runtime_error(std::string(kCaller) + ": device weight pointer is null");
    }
    if (x.device_id() != device_id) {
        throw std::runtime_error(std::string(kCaller) + ": input tensor is on a different CUDA device");
    }

    ValidateQuantLinearShape(x.shape(), x.ShapeStringShort(), weight_shape, q4_1_block_count, kQ41BlockSize, kCaller);

    bool input_is_1d = x.num_dims() == 1;
    int rows = input_is_1d ? 1 : x.shape()[0];
    int in_features = input_is_1d ? x.shape()[0] : x.shape()[1];
    int out_features = weight_shape[0];
    int block_per_row = (in_features + kQ41BlockSize - 1) / kQ41BlockSize;

    CudaTensor y(input_is_1d ? std::vector<int>{out_features} : std::vector<int>{rows, out_features}, device_id);

    dim3 block(kBlockSize);
    dim3 grid((out_features + kBlockSize - 1) / kBlockSize, rows);
    Q41LinearKernel<<<grid, block>>>(static_cast<const float*>(x.data()), 
                                     static_cast<const BlockQ41*>(q4_1_device),
                                     static_cast<float*>(y.data()), rows,
                                     in_features, out_features, block_per_row
    );

    CheckLastKernel("Q41LinearKernel");
    return y;
}






Tensor CudaQ80Linear(const Tensor& x, const std::vector<BlockQ80>& weight, const std::vector<int>& weight_shape, const Tensor* bias, int device_id) {
    ValidateQ80LinearInputs(x, weight_shape, weight.size());
    CudaDeviceBuffer w_dev(weight.size() * sizeof(BlockQ80), device_id);
    w_dev.Upload(weight.data(), weight.size() * sizeof(BlockQ80));
    return CudaQ80LinearWithDevicePointer(x, w_dev.data(), weight.size(), weight_shape, bias, device_id);
}


Tensor CudaQ80LinearDeviceWeight(const Tensor& x, const void* q8_0_device, size_t q8_0_block_count, const std::vector<int>& weight_shape, const Tensor* bias, int device_id) {
    return CudaQ80LinearWithDevicePointer(x, q8_0_device, q8_0_block_count, weight_shape, bias, device_id);
}


CudaTensor CudaQ80LinearDeviceInput(const CudaTensor& x, const void* q8_0_device, size_t q8_0_block_count, const std::vector<int>& weight_shape, int device_id) {
    return CudaQ80LinearWithDeviceInput(x, q8_0_device, q8_0_block_count, weight_shape, device_id);
}





Tensor CudaQ40Linear(const Tensor& x, const std::vector<BlockQ40>& weight, const std::vector<int>& weight_shape, const Tensor* bias, int device_id) {
    ValidateQuantLinearShape(x.shape, x.ShapeToStringShort(), weight_shape, weight.size(), kQ40BlockSize, "CudaQ40Linear");
    CudaDeviceBuffer w_dev(weight.size() * sizeof(BlockQ40), device_id);
    w_dev.Upload(weight.data(), weight.size() * sizeof(BlockQ40));
    return CudaQ40LinearWithDevicePointer(x, w_dev.data(), weight.size(), weight_shape, bias, device_id);
}

Tensor CudaQ40LinearDeviceWeight(const Tensor& x, const void* q4_0_device, size_t q4_0_block_count, const std::vector<int>& weight_shape, const Tensor* bias, int device_id) {
    return CudaQ40LinearWithDevicePointer(x, q4_0_device, q4_0_block_count, weight_shape, bias, device_id);
}

CudaTensor CudaQ40LinearDeviceInput(const CudaTensor& x, const void* q4_0_device, size_t q4_0_block_count, const std::vector<int>& weight_shape, int device_id) {
    return CudaQ40LinearWithDeviceInput(x, q4_0_device, q4_0_block_count, weight_shape, device_id);
}





Tensor CudaQ41Linear(const Tensor& x, const std::vector<BlockQ41>& weight, const std::vector<int>& weight_shape, const Tensor* bias, int device_id) {
    ValidateQuantLinearShape(x.shape, x.ShapeToStringShort(), weight_shape, weight.size(), kQ41BlockSize, "CudaQ41Linear");
    CudaDeviceBuffer w_dev(weight.size() * sizeof(BlockQ41), device_id);
    w_dev.Upload(weight.data(), weight.size() * sizeof(BlockQ41));
    return CudaQ41LinearWithDevicePointer(x, w_dev.data(), weight.size(), weight_shape, bias, device_id);
}

Tensor CudaQ41LinearDeviceWeight(const Tensor& x, const void* q4_1_device, size_t q4_1_block_count, const std::vector<int>& weight_shape, const Tensor* bias, int device_id) {
    return CudaQ41LinearWithDevicePointer(x, q4_1_device, q4_1_block_count, weight_shape, bias, device_id);
}

CudaTensor CudaQ41LinearDeviceInput(const CudaTensor& x, const void* q4_1_device, size_t q4_1_block_count, const std::vector<int>& weight_shape, int device_id) {
    return CudaQ41LinearWithDeviceInput(x, q4_1_device, q4_1_block_count, weight_shape, device_id);
}




}