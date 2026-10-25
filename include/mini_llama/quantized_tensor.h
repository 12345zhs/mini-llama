#pragma once

#include "mini_llama/tensor.h"

#include <cstdint>
#include <vector>
#include <string>

namespace mini_llama {

// 32个一组
constexpr int kQ80BlockSize = 32;
constexpr int kQ40BlockSize = 32;
constexpr int kQ41BlockSize = 32;

// 量化枚举类型
enum class QuantType {
    kF32,
    kQ80,
    kQ40,
    kQ41
};


// 8位量化
struct BlockQ80 {
    uint16_t d;          // 缩放因子
    int8_t qs[32];       // 32个int8整数
};




struct BlockQ40 {
    uint16_t d;
    uint8_t qs[16];
};


struct BlockQ41 {
    uint16_t d;
    uint16_t m;               // 最小偏移量
    uint8_t qs[16];
};


// 根据type的类型，选择对应的容器
struct QuantizedTensor {
    QuantType type = QuantType::kF32;
    std::vector<int> shape;

    std::vector<float> f32_data; 
    std::vector<BlockQ80> q8_0_data;
    std::vector<BlockQ40> q4_0_data;
    std::vector<BlockQ41> q4_1_data;

    size_t num_elements() const;
    std::string ShapeStringShort() const;
    void AssertShape(const std::vector<int>& expected, const char* caller) const;
    int num_dims() const;

};

Tensor ToTensor(const QuantizedTensor& q);
QuantizedTensor toQuantizedTensor(const Tensor& t);


}