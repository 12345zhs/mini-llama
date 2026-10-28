#include "mini_llama/cuda_runtime.h"

#include <stdexcept>
#include <sstream>

#ifdef MINI_LLAMA_USE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace mini_llama {

// 没有开启MINI_LLAMA_USE_CUDA,都抛出这个错误
static std::runtime_error CudaNotBuildError() {
    return std::runtime_error( "CUDA backend was not built. Reconfigure with -DMINI_LLAMA_CUDA=ON on a " "NVIDIA CUDA machine.");
}

#ifdef MINI_LLAMA_USE_CUDA
static void CheckCuda(cudaError_t err, const char* expr) {
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA runtime error in " + std::string(expr) + ": " + cudaGetErrorString(err))
    }
}

// 将CudaMemcpyKind转化为CUDA官方枚举
static cudaMemcpyKind ToCudaMemcpyKind(CudaMemcpyKind kind) {
    switch(kind) {
        case CudaMemcpyKind::kHostToDevice : {
            return cudaMemcpyHostToDevice;
        }
        case CudaMemcpyKind::kDeviceToHost : {
            return cudaMemcpyDevideToHost;
        }
        case CudaMemcpyKind::KDeviceToDevice : {
            return cudaMemcpyDeviceToDevice;
        }
    }
    throw std::runtime_error("unkown CUDA memcpy kind");
}
#endif

// 检测环境，是否开启了MINI_LLAMA_USE_CUDA
bool CudaRuntimeBuilt() {
#ifdef MINI_LLAMA_USE_CUDA 
    return true;
#endif 
    return false;
}


// 检查显卡数量
int CudaDeviceCount() {
#ifdef MINI_LLAMA_USE_CUDA
    int count = 0;
    CheckCuda(CudaGetDeviceCount(&count), "cudaGetDeviceCount");
    return count;
#else 
    throw CudaNotBuildError();
#endif
}


// 切换显卡
void CudaSetDevice(int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
    CheckCuda(cudaSetDevice(device_id), "cudaSetDEvice");
#else 
    (void) device_id;        // 存在device_id, 但不使用
    throw CudaNotBuildError();
#endif
}


// 显卡信息
CudaDeviceInfo CudaGetDeviceInfo(int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
    int count = cudaGetDeviceCount();
    if (device_id < 0 || device_id >= count) {
        throw std::runtime_error("CUDA device " + std::to_string(device_id) + " is not available; detected " + std::to_string(count) + " device(s)");
    }

    // 获得底层设备属性
    cudaDeviceProp prop{};
    CheckCuda(cudaGetDeviceProperties(&prop, device_id), "cudaGetDeviceProperties");

    CudaDeviceInfo info;
    info.id = device_id;                               // 显卡编号        
    info.name = prop.name;                             // 显卡名字
    info.compute_major = prop.major;                   // 计算能力
    info.compute_minor = prop.minor;
    info.total_memory_bytes = prop.totalGlobalMem;     // 显存大小
    CheckCuda(cudaDeviceGetVersion(&info.driver_version), "cudaDeviceGetVersion");       // 显卡驱动版本
    CheckCuda(cudaRuntimeGetVersion(&info.runtime_version), "cudaRuntimeGetVersion");     // cuda ToolKit版本
    return info;
#else 
    (void) device_id;
    throw CudaNotBuildError();
#endif
}


// 打印显卡信息
std::string CudaFormatDeviceInfo(const CudaDeviceInfo& info) {
    double total_gb = static_cast<double>(info.total_memory_bytes) / (1024.0 * 1024.0 * 1024.0);
    std::ostringstream out;
    out << "device " << info.id << ": " << info.name << ", compute capability "
        << info.compute_major << "." << info.compute_minor << ", total memory "
        << total_gb << " GB"
        << ", cuda runtime " << info.runtime_version << ", driver "
        << info.driver_version;
    return out.str();
}


// 申请内存
void* CudaMallocBytes(size_t bytes) {
#ifdef MINI_LLAMA_USE_CUDA
    if (bytes == 0) {
        return nullptr;
    }
    void* ptr = nullptr;
    CheckCuda(cudaMalloc(&ptr, bytes), "cudaMalloc");
    return ptr;
#else 
    (void)bytes;
    throw CudaNotBuildError();
#endif
}


// 释放内存
void CudaFreeBytes(void* ptr) {
#ifdef MINI_LLAMA_USE_CUDA
    if (ptr == nullptr) {
        return;
    }
    CheckCuda(cudaFree(ptr), "cudaFree");
#else   
    (void)ptr;
    throw CudaNotBuildError();
#endif
}


// 数据搬运
void CudaMmecpyBytes(void* dst, const void* src, size_t bytes, CudaMemcpyKind kind) {
#ifdef MINi_LLAMA_USE_CUDA
    if (bytes == 0) {
        return;
    }
    if (dst == nullptr || src == nullptr) {
        throw std::runtime_error("cudaMemcpy requires non-null src and dst for non-empty copies")
    }
    CheckCuda(cudaMemcpy(dst, src, bytes, ToCudaMemcpyKind(kind)), "cudaMemcpy");
#else 
    (void)dst;
    (void)src;
    (void)bytes;
    (void)kind;
    throw CudaNotBuildError();
#endif
}



// cuda的内存管理
CudaDeviceBuffer::~CudaDeviceBuffer() {
    ReleaseNoexcept();
}


CudaDeviceBuffer::CudaDeviceBuffer(size_t type, int device_id = 0) {
    Reset(type, device_id);
}


CudaDeviceBuffer::CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept : data_(other.data_), bytes_(other.bytes_), device_id_(other.device_id_) {
    other.data_ = nullptr;
    other.bytes_ = 0;
    other.device_id_ = 0;
}
    

CudaDeviceBuffer& CudaDeviceBuffer::operator=(CudaDeviceBuffer&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    ReleaseNoexcept();

    data_ = other.data_;
    bytes_ = other.bytes_;
    device_id_ = other.device_id_;

    other.data_ = nullptr;
    other.bytes_ = 0;
    other.device_id_ = 0;
    return *this;
}


// 清空并且释放
void CudaDeviceBuffer::Reset() {
    if (data_ == nullptr) {
        return;
    }
    CudaSetDevice(device_id_);
    CudaFreeBytes(data_);

    data_ = nullptr;
    bytes_ = 0;
    device_id_ = 0;

}


void CudaDeviceBuffer::Reset(size_t bytes, int device_id) {
    Reset();
    if (bytes == 0) {
        device_id_ = device_id;
        return;
    }

    CudaSetDevice(device_id);
    data_ = CudaMallocBytes(bytes);
    bytes_ = bytes;
    device_id_ = device_id; 
}


// 把cpu的内容加载到GPU上
void CudaDeviceBuffer::Upload(const void* src, size_t bytes) {
    if (bytes > bytes_) {
        throw std::runtime_error("CUDA upload exceeds device buffer size");
    }
    CudaSetDevice(device_id_);
    CudaMmecpyBytes(data_, src, bytes, CudaMemcpyKind::kHostToDevice);

}


// 把gpu上的内容加载到cpu上
void CudaDeviceBuffer::Download(void* dst, size_t bytes) const {
    if (bytes > bytes_) {
        throw std::runtime_error("CUDA download exceeds device buffer size");
    }
    CudaSetDevice(device_id_);
    CudaMmecpyBytes(dst ,data_, bytes, CudaMemcpyKind::kDeviceToHost);
}

void* CudaDeviceBuffer::data() {
    return data_;
}

const void* CudaDeviceBuffer::data() const {
    return  data_;
}

size_t CudaDeviceBuffer::bytes() const  {
    return bytes_;
}
int CudaDeviceBuffer::device_id() const {
    return device_id_;
}

bool CudaDeviceBuffer::empty() const {
    return data_ == nullptr;
}


void CudaDeviceBuffer::ReleaseNoexcept() noexcept {
    try {
        Reset();
    } catch (...) {
    }
}

}