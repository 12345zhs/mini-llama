#include "mini_llama/quantized_tensor.h"


namespace mini_llama {


// 计算元素总个数
size_t QuantizedTensor::num_elements() const {
    size_t n = 1;
    for (int d : shape) {
        n *= static_cast<size_t>(d);
    }
    return n;
}


// shape 转化为 string形式
std::string QuantizedTensor::ShapeStringShort() const {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        s += std::to_string(shape[i]);
        if (i + 1 < shape.size()) {
            s += ", ";
        }
    }
    s += "]";
    return s;
}

// 校验shape
void QuantizedTensor::AssertShape(const std::vector<int>& expected, const char* caller) const {
    if (shape != expected) {
        throw std::runtime_error(std::string(caller) + ": shape mismatch, expected " + QuantizedTensor{QuantType::kF32, expected}.ShapeStringShort() + ", got " + ShapeStringShort());
    }

}

// 计算维数
int QuantizedTensor::num_dims() const {
    return static_cast<int>(shape.size());
}


// QuantizedTensor 转变为 tensor
Tensor ToTensor(const QuantizedTensor& q) {
    // tensor的type为float
    if (q.type != QuantType::kF32) {
        throw std::runtime_error("ToTensor: expected F32 QuantizedTensor, got quantized type");
    }

    Tensor t(q.shape, 0.0f);
    if (q.f32_data.size() == t.data.size()) {
        t.data = q.f32_data;
    }
    return t;
}


// tensor 转变为 quantizedtensor
QuantizedTensor toQuantizedTensor(const Tensor& t) {
    QuantizedTensor q;
    q.type = QuantType::kF32;
    q.shape = t.shape;
    q.f32_data = t.data;
    return q;
}



}