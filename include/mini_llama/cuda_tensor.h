#pragma once

#include "mini_llama/cuda_runtime.h"
#include "mini_llama/tensor.h"

#include <vector>

namespace mini_llama {

class CudaTensor {

public:
    CudaTensor() = default;
    explicit CudaTensor(const std::vector<int> shape, int devide_id = 0);

    CudaTensor(const CudaTensor&) = delete;
    CudaTensor& operator=(const CudaTensor&) = delete;

    CudaTensor(CudaTensor&&) noexcept = default;
    CudaTensor& operator=(CudaTensor&&) noexcept = default;

    void Reset();
    void Reset(std::vector<int> shape, int device_id = 0);
    void UploadFrom(const Tensor& src);
    Tensor Download() const;
    void DownloadTo(Tensor& dst) const;
    void* data();
    const void* data() const;

    const std::vector<int>& shape() const;
    int num_dims() const;
    size_t size() const;
    size_t bytes() const;
    int device_id() const;
    bool empty() const;
    std::string ShapeStringShort() const; 

private:
    std::vector<int> shape_;
    size_t numel_ = 0;
    int device_id_ = 0;
    CudaDeviceBuffer buffer_;
};

CudaTensor CudaTensorFromHost(const Tensor& src, int device_id = 0);

}