#include "mini_llama/cuda_tensor.h"
#include "mini_llama/tensor.h"

#include <limits>
#include <string>

namespace mini_llama {


// 获取数量
static size_t CheckNumel(const std::vector<int>& shape, const char* caller) {
    if (shape.empty()) {
        return 0;
    }
    size_t total = 1;
    for (size_t axis = 0; axis < shape.size(); ++axis) {
        int dim = shape[axis];
        if (dim <= 0) {
            throw std::runtime_error(std::string(caller) + ": dimension at axis " + std::to_string(axis) + " must be positive, got " + std::to_string(dim));
        }
        size_t dim_size = static_cast<size_t>(dim);
        if (dim_size * total > std::numeric_limits<size_t>::max()) {
            throw std::runtime_error(std::string(caller) + ": shape element count overflow");
        }
        total *= dim_size;
    }
    return total;
}


// shape用string表示
static std::string ShapeToString(const std::vector<int>& shape) {
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            out += ",";
        }
        out += std::to_string(shape[i]);
    }
    out += "]";
    return out;
}



CudaTensor::CudaTensor(const std::vector<int> shape, int devide_id) {
    Reset(shape, devide_id);
}


// 释放成员变量
void CudaTensor::Reset() {
    buffer_.Reset();
    shape_.clear();
    numel_ = 0;
    device_id_ = 0;
}


// 释放 + 赋值
void CudaTensor::Reset(std::vector<int> shape, int device_id) {
    Reset();
    size_t num_elements = CheckNumel(shape, "cudaTensor");
    shape_ = shape;
    device_id_ = device_id;
    numel_ = num_elements;
    buffer_.Reset(num_elements * sizeof(float), device_id);
}


// 将tensor上传到gpu中cuda_tensor
void CudaTensor::UploadFrom(const Tensor& src) {
    if (src.shape != shape_) {
        throw std::runtime_error("CudaTensor::UploadFrom shape mismatch: device=" + ShapeStringShort() + " host=" + src.ShapeToStringShort());
    }

    buffer_.Upload(src.data.data(), src.num_elements() * sizeof(float));
}


// 将gpu的cuda_tensor上传到cpu的tensor
Tensor CudaTensor::Download() const {
    if (shape_.empty()) {
        return Tensor();
    }
    Tensor dst(shape_, 0.0f);
    DownloadTo(dst);
    return dst;
}


void CudaTensor::DownloadTo(Tensor& dst) const {
    if (dst.shape != shape_) {
        throw std::runtime_error("CudaTensor::DownloadTo shape mismatch: device=" + ShapeStringShort() + " host=" + dst.ShapeToStringShort());
    }
    buffer_.Download(dst.data.data(), dst.size() * sizeof(float));
}


void* CudaTensor::data() {
    return buffer_.data();
}


const void* CudaTensor::data() const {
    return buffer_.data();
}


const std::vector<int>& CudaTensor::shape() const {
    return shape_;
}


int CudaTensor::num_dims() const {
    return static_cast<int>(shape_.size());
}

size_t CudaTensor::size() const {
    return numel_;
}


size_t CudaTensor::bytes() const {
    return buffer_.bytes();
}


int CudaTensor::device_id() const {
    return device_id_;
}


bool CudaTensor::empty() const {
    return buffer_.empty();
}


std::string CudaTensor::ShapeStringShort() const {
    return ShapeToString(shape_);
}


// 从cpu的tensor转换到gpu的cuda_tensor
CudaTensor CudaTensorFromHost(const Tensor& src, int device_id = 0) {
    CudaTensor dst(src.shape, device_id);
    dst.UploadFrom(src);
    return dst;
}


}