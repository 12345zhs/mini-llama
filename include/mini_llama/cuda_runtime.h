#pragma once

#include <string>

namespace mini_llama{

// cuda硬件信息
struct CudaDeviceInfo {
    int id = 0;
    std::string name;
    int compute_major = 0;
    int compute_minor = 0;
    size_t total_memory_bytes = 0;
    int driver_version = 0;
    int runtime_version = 0; 
};

enum class CudaMemcpyKind {
    kHostToDevice,
    kDeviceToHost,
    kDeviceToDevice,
};

bool CudaRuntimeBuilt();
int CudaDeviceCount();
void CudaSetDevice(int device_id);
CudaDeviceInfo CudaGetDeviceInfo(int device_id);
std::string CudaFormatDeviceInfo(const CudaDeviceInfo& info);


void* CudaMallocBytes(size_t bytes);
void CudaFreeBytes(void* ptr);
void CudaMmecpyBytes(void* dst, const void* src, size_t bytes, CudaMemcpyKind kind);



// cuda的内存管理
class CudaDeviceBuffer {
public:
    CudaDeviceBuffer() = default;
    ~CudaDeviceBuffer();
    CudaDeviceBuffer(size_t type, int device_id = 0);

    // 禁用拷贝构造和拷贝赋值
    CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
    CudaDeviceBuffer& operator= (const CudaDeviceBuffer&) = delete;

    CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept;
    CudaDeviceBuffer& operator=(CudaDeviceBuffer&& other) noexcept;

    void Reset();
    void Reset(size_t bytes, int device_id = 0);
    void Upload(const void* src, size_t bytes);
    void Download(void* dst, size_t bytes) const;

    void* data();
    const void* data() const;
    size_t bytes() const;
    int device_id() const;
    bool empty() const;

private:
    void ReleaseNoexcept() noexcept;

    void* data_ = nullptr;
    size_t bytes_ = 0;
    int device_id_ = 0;

};

}