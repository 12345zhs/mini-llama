#include "mini_llama/cuda_matmul.h"
#include "mini_llama/cuda_runtime.h"


#include <stdexcept>

#ifdef MINI_LLAMA_USE_CUDA
#include <cublas_v2.h>
#endif


namespace mini_llama {

static std::runtime_error CudaMatmulNotBuiltError() {
    return std::runtime_error("CUDA Matmul was not built. Reconfigure with -DMINI_LLAMA_CUDA=ON on a " "NVIDIA CUDA machine.");
}


#ifdef MINI_LLAMA_USE_CUDA
// 报错类型
static const char* CublasStatusName(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: {
            return "CUBLAS_STATUS_SUCCESS";
        }
        case CUBLAS_STATUS_NOT_INITIALIZED: {
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        }
        case CUBLAS_STATUS_ALLOC_FAILED: {
            return "CUBLAS_STATUS_ALLOC_FAILED";
        }

        case CUBLAS_STATUS_INVALID_VALUE: {
            return "CUBLAS_STATUS_INVALID_VALUE";
        }
        case CUBLAS_STATUS_ARCH_MISMATCH: {            
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        }
        case CUBLAS_STATUS_MAPPING_ERROR: {
            return "CUBLAS_STATUS_MAPPING_ERROR";
        }
        case CUBLAS_STATUS_EXECUTION_FAILED: {
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        }
        case CUBLAS_STATUS_INTERNAL_ERROR: {
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        }
        case CUBLAS_STATUS_NOT_SUPPORTED: {
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        }
        case CUBLAS_STATUS_LICENSE_ERROR : {
            return "CUBLAS_STATUS_LICENSE_ERROR";
        }
    }
    return "CUBLAS_STATUS_UNKNOWN";
}



static void CheckCublas(cublasStatus_t status, const char* expr) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS error in " + std::string(expr) + ": " + CublasStatusName(status));
    }
}


// cublas的工作台
class CublasHandle {
public:
    explicit CublasHandle(int device_id) {
        CudaSetDevice(device_id);
        CheckCublas(cublasCreate(&handle_), "cublasCreate");
    }

    ~CublasHandle() {
        if (handle_ != nullptr) {
            cublasDestroy(handle_);
        }
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    cublasHandle_t get() const {
        return handle_;
    }

private:
    cublasHandle_t handle_ = nullptr;
};



// gemm的两个矩阵形状是否正确
static void ValidateMatmulShape(const Tensor& a, const Tensor& b) {
    if (a.num_dims() != 2 || b.num_dims() != 2) {
        throw std::runtime_error("CudaMatmul: expected a and b to be 2D tensors");
    }
    if (a.shape[1] != b.shape[0]) {
        throw std::runtime_error("CudaMatmul: dimension mismatch " + a.ShapeToStringShort() + " vs " + b.ShapeToStringShort());
    }
}


// 偏置
// y:[rows, cols], bias[cols]
static void AddBiasInPlace(Tensor& y, const Tensor& bias, int rows, int cols) {
    if (bias.num_dims() != 1 || bias.shape[0] != cols) {
        throw std::runtime_error("CudaLinear: expected bias shape [out_features]");
    }
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols, ++col) {
            y[row * cols + col] += bias.data[col];
        }
    }
}



// 维度提取
static int LinearInFeaturesFromShape(const std::vector<int>& x_shape, const std::string& x_shape_str, const char* caller) {
    if (x_shape.size() == 1) {
        return x_shape[0];
    }
    if (x_shape.size() == 2) {
        return x_shape[1];
    }
    throw std::runtime_error(std::string(caller) + ": expected x shape [in_features] or [batch, in_features], got " + x_shape_str);
}



// 检查输入矩阵与权重矩阵的形状
static void ValidateLinearShape(const std::vector<int>& x_shape, const std::string& x_shape_str, const std::vector<int>& w_shape, const char* caller) {
    if (w_shape.size() != 2) {
        throw std::runtime_error(std::string(caller) + ": expected weight shape [out_features, in_features]");
    }
    int in_features = LinearInFeaturesFromShape(x_shape, x_shape_str, caller);
    if (w_shape[1] != in_features) {
        throw std::runtime_error(std::string(caller) + ": dimension mismatch x=" + x_shape_str + " weight=[" + std::to_string(w_shape[0]) + ", " + std::to_string(w_shape[1]) + "]");
    }
}



// weight[out_feature, in_feature]
// x[in_feature]   /   [batch, in_feature]
// y[out_feature]  /   [batch, out_feature]
static void ValidateLinearInputs(const Tensor& x, const std::vector<int>& w_shape) {
    ValidateLinearShape(x.shape, x.ShapeToStringShort(), w_shape, "CudaLinear");
}



// cuda的linear,涉及数据搬运
static Tensor CudaLinearWithDevicePointer(const Tensor& x, const void* w_device, const std::vector<int>& w_shape, const Tensor* bias, int device_id) {
    if (w_device == nullptr) {
        throw std::runtime_error("CudaLinear: device weight pointer is null");
    }
    ValidateLinearInputs(x, w_shape);

    bool input_is_1d = (x.num_dims() == 1);
    int rows = input_is_1d ? 1 : x.shape[0];
    int in_features = input_is_1d ? x.shape[0] : x.shape[1];
    int out_features = w_shape[0];

    // 在cpu上创建结果容器
    Tensor y(input_is_1d ? std::vector<int>{out_features} : std::vector<int>{in_features}, 0.0f);
    // gpu上申请内存
    CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
    // 把x传入gpu
    x_dev.Upload(x.data.data(), x.size() * sizeof(float));

    CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);

    CublasHandle handle(device_id);
    const float alpha = 1.0f;
    const float beta = 0.0f;


    // cublas的矩阵计算是列主序的
    // x × W^T ====  W × X^T
    //        Y^T  ==  [out_feature, in_feature]  * [in_feature, row]
    //      [out_feature, row]
    CheckCublas(
        cublasSgemm(handle.get(), CUBLAS_OP_T, CUBLAS_OP_N, out_features, rows, in_features,
                    &alpha, 
                    static_cast<const float*>(w_device), in_features,
                    static_cast<const float*>(x_dev.data()), in_features,
                    &beta,
                    static_cast<float*>(y_dev.data()), out_features),
        "cublasSgemm(CudaLinear)"  
    );

    // 把上面计算完的结果传给cpu的tensor
    y_dev.Download(y.data.data(), y.size() * sizeof(float));

    if (bias != nullptr) {
        AddBiasInPlace(y, *bias, rows, out_features);
    }

    return y;
}


// cuda的linear， 不涉及数据搬运，全部在gpu上运行
CudaTensor CudaLinearDeviceInputWithDevicePointer(const CudaTensor& x, const void* w_device, const std::vector<int>& w_shape, int device_id)  {
    if (w_device == nullptr) {
        throw std::runtime_error("CudaLinearDeviceInput: device weight pointer is null");
    }
    if (x.device_id() == device_id) {
        throw std::runtime_error("CudaLinearDeviceInput: input tensor is on a different CUDA device");
    }
    ValidateLinearShape(x.shape(), x.ShapeStringShort(), w_shape, "CudaLinearDeviceInput");


    bool input_is_1d = (x.num_dims() == 1);
    int rows = input_is_1d ? 1 : x.shape()[0];
    int in_features = input_is_1d ? x.shape()[0] : x.shape()[1];
    int out_features = w_shape[0];

    CudaTensor y(input_is_1d ? std::vector<int>({out_features}) : std::vector<int>({rows, out_features}), device_id);

    CublasHandle handle(device_id);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cublas的矩阵计算是列主序的
    // x × W^T ====  W × X^T
    //        Y^T  ==  [out_feature, in_feature]  * [in_feature, row]
    //      [out_feature, row]
    CheckCublas(
        cublasSgemm(handle.get(), CUBLAS_OP_T, CUBLAS_OP_N, out_features, rows, in_features,
                    &alpha,
                    static_cast<const float*>(w_device), in_features,
                    static_cast<const float*>(x.data()), in_features,
                    &beta,
                    static_cast<float*>(y.data()), out_features),
        "cublasSgemm(CudaLinearDeviceInput)"
    );

    return y;
}
#endif

bool CudaMatmulBuilt() {
#ifdef MINI_LLAMA_USE_CUDA
    return true;
#else   
    return false;
#endif
}


// GEMM
Tensor CudaMatmul(const Tensor& a, const Tensor& b, int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
    ValidateMatmulShape(a, b);

    int m = a.shape[0];
    int n = b.shape[1];
    int k = a.shape[1];
    Tensor c({m, n}, 0.0f);

    CudaDeviceBuffer a_dev(a.size() * sizeof(float), device_id);
    CudaDeviceBuffer b_dev(b.size() * sizeof(float), device_id);
    CudaDeviceBuffer c_dev(c.size() * sizeof(float), device_id);

    a_dev.upload(a.data.data(), a.size() * sizeof(float));
    b_dev.upload(b.data.data(), b.size() * sizeof(float));

    CublasHandle handle(device_id);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // C^T = B^T * A ^ T
    // [N, M]   =  [N, K]  [K, M]
    CheckCublas(
        cublasSgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k
                    &alpha,
                    static_cast<const float*>(b_dev.data()), n,
                    static_cast<const float*>(a_dev.data()), k,
                    &beta,
                    static_cast<float*>(c_dev), n);
        "cublasSgemm(CudaMatmul)"
    );
    c_dev.Download(c.data.data(), c.size() * sizeof(float));
    return c;

#else
    (void)a;
    (void)b;
    (void)device_id;
    throw CudaMatmulNotBuiltError();
#endif
}


// linear  不涉及数据搬运
Tensor CudaLinear(const Tensor& x, const Tensor& weight, const Tensor* bias, int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
    if (weight.num_dims() != 2) {
        throw std::runtime_error("CudaLinear: expected weight shape [out_features, in_features]");
    }
    CudaDeviceBuffer w_dev(weight.size() * sizeof(float), device_id);
    w_dev.upload(weight.data.data(), weight.size() * sizeof(float));
    return CudaLinearWithDevicePointer(x, w_dev.data(), weight.shape, bias, device_id);
#else
    (void)x;
    (void)weight;
    (void)bias;
    (void)device_id;
    throw CudaMatmulNotBuiltError();
#endif
}

Tensor CudaLinearrDeviceWeight(const Tensor& x, const void* w_device, const std::vector<int>& w_shape,  const Tensor* bias, int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
    return CudaLinearWithDevicePointer(x, w_device, w_shape, bias, device_id);
#else
    (void)x;
    (void)w_device;
    (void)w_shape;
    (void)bias;
    (void)device_id;
    throw CudaMatmulNotBuiltError();
#endif
}



// linear 不涉及数据搬运
CudaTensor CudaLinearrDeviceInput(const CudaTensor& x, const void* w_device, const std::vector<int>& w_shape, int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
    return CudaLinearDeviceInputWithDevicePointer(x, w_device, w_shape, device_id);
#else
    (void)x;
    (void)w_device;
    (void)w_shape;
    (void)device_id;
    throw CudaMatmulNotBuiltError();
#endif
}



}