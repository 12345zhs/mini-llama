#pragma once
#include "mini_llama/tensor.h"

#include <vector>
#include <string>
#include <stdexcept>
#include <numeric>
#include <limits>
#include <iostream>
#include <iomanip>


namespace mini_llama {

// 辅助函数——返回调用者
const char* CallerName(const char* caller) {
    return caller == nullptr ? "Tensor" : caller;
}

// 辅助函数——把张量形状转换为字符串
std::string ShapeToStirng(const std::vector<int>& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            s += ", ";
        }
        s += std::to_string(shape[i]);
    }
    s += "]";
    return s;
}


// 辅助函数——计算元素总数，同时检查各个维度必须大于零，各个维度相乘时不能超过size_t表示的最大值
size_t CheckedNumel(const std::vector<int>& shape, const char* caller) {
    size_t total = 1;
    for (size_t axis = 0; axis < shape.size(); ++axis) {
        const int dim = shape[axis];

        // 各个维度必须大于零
        if (dim <= 0) {
            throw std::runtime_error(std::string(CallerName(caller)) + ": dimension at axis " + std::to_string(axis)
                                     + "must be positive, got" + std::to_string(dim) + "in shape" + ShapeToStirng(shape));
        }
        const size_t dim_size = static_cast<size_t>(dim);

        // 防止张量的数据数量超过size_t的最大值
        if (total > std::numeric_limits<size_t>::max() / dim_size) {
            throw std::runtime_error(std::string(CallerName(caller)) + ": shape element count overflow for " + ShapeToStirng(shape));
        }
        total *= dim_size;
    }
    return total;
}



// 辅助函数——检查张量维度的数量
void CheckRank(const Tensor& t, int expected, const char* caller) {
    if (t.num_dims() != expected) {
        throw std::runtime_error(std::string(caller) + " expected " + std::to_string(expected) + "D tensor, got" + t.ShapeToStringShort());
    }
}


// 辅助函数——检查张量某一维度上的索引是否有效
void CheckAxisIndex(const Tensor& t, int axis, int flat_index, const char* caller) {
    const int dim = t.shape[axis];
    if (flat_index < 0 || flat_index >= dim) {
        throw std::out_of_range(std::string(caller) + ": FlatIndex " + std::to_string(flat_index) + " out of range for axis"
                                + std::to_string(axis) + "with size" + std::to_string(dim) + " in tensor" + t.ShapeToStringShort());
    }
}


// 构造函数
Tensor::Tensor(const std::vector<int>& input_shape, float fill) : shape(input_shape) {
    const size_t total = CheckedNumel(input_shape, "Tensor constructor");
    data.resize(total, fill);
}


// 将多维坐标转化为一维数组下标
size_t Tensor::FlatIndex(const std::vector<int>& indices) const {
    // 索引个数必须等于张量维度
    if (indices.size() != shape.size()) {
        throw std::runtime_error("Tensor::FlatIndex expected" + std::to_string(shape.size()) + " indices for tensor " + this->ShapeToStringShort() 
                                 + ", got " + std::to_string(indices.size()));
    }

    // 索引必须在张量的每个维度范围内
    for (size_t axis = 0; axis < indices.size(); ++axis) {
        CheckAxisIndex(*this, axis, indices[axis], "Tensor::FlatIndex");
    }

    size_t flat = 0;
    size_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        flat += static_cast<size_t>(indices[i]) * stride;
        stride *= static_cast<size_t>(shape[i]); 
    }
    return flat;
}


// 把张量的形状转化为字符串
std::string Tensor::ShapeToStringShort() const {
    return ShapeToStirng(shape);
}
std::string Tensor::ShapeString() const {
    return ShapeToStringShort();
}



// 通用的多维元素访问接口
// 非const版本返回引用，可以修改
float& Tensor::At(const std::vector<int>& indices) {
    return data[FlatIndex(indices)];
}
// const版本，用于读取
float Tensor::At(const std::vector<int>& indices) const {
    return data[FlatIndex(indices)];
}


// 访问一维张量
float& Tensor::At1(int i) {
    CheckRank(*this, 1, "At1");
    CheckAxisIndex(*this, 0, i, "At1");
    return data[i];
}
float Tensor::At1(int i) const {
    CheckRank(*this, 1, "At1");
    CheckAxisIndex(*this, 0, i, "At1");
    return data[i];
}


// 访问二维张量的第i行，第j列
float& Tensor::At2(int i, int j) {
    CheckRank(*this, 2, "At2");
    CheckAxisIndex(*this, 0, i, "At2");
    CheckAxisIndex(*this, 1, j, "At2");
    return data[i * shape[1] + j];
}
float Tensor::At2(int i, int j) const {
    CheckRank(*this, 2, "At2");
    CheckAxisIndex(*this, 0, i, "At2");
    CheckAxisIndex(*this, 1, j, "At2");
    return data[i * shape[1] + j];
}

// 访问三维张量元素
float& Tensor::At3(int i, int j, int k) {
    CheckRank(*this, 3, "At3");
    CheckAxisIndex(*this, 0, i, "At3");
    CheckAxisIndex(*this, 1, j, "At3");
    CheckAxisIndex(*this, 2, k, "At3");
    return data[i * shape[1]  * shape[2] + j * shape[2] + k];
}
float Tensor::At3(int i, int j, int k) const {
    CheckRank(*this, 3, "At3");
    CheckAxisIndex(*this, 0, i, "At3");
    CheckAxisIndex(*this, 1, j, "At3");
    CheckAxisIndex(*this, 2, k, "At3");
    return data[i * shape[1]  * shape[2] + j * shape[2] + k];
}

//访问四维张量
float& Tensor::At4(int i, int j, int k, int l) {
    CheckRank(*this, 4, "At4");
    CheckAxisIndex(*this, 0, i, "At4");
    CheckAxisIndex(*this, 1, j, "At4");
    CheckAxisIndex(*this, 2, k, "At4");
    CheckAxisIndex(*this, 3, l, "At4");
    return data[i * shape[1] * shape[2] * shape[3] + j * shape[2] * shape[3] + k * shape[3] + l];
}
float Tensor::At4(int i, int j, int k, int l) const {
    CheckRank(*this, 4, "At4");
    CheckAxisIndex(*this, 0, i, "At4");
    CheckAxisIndex(*this, 1, j, "At4");
    CheckAxisIndex(*this, 2, k, "At4");
    CheckAxisIndex(*this, 3, l, "At4");
    return data[i * shape[1] * shape[2] * shape[3] + j * shape[2] * shape[3] + k * shape[3] + l];
}

// 重载方括号标识符
float& Tensor::operator[](size_t i) {
    return data[i];
}
float Tensor::operator[](size_t i) const {
    return data[i];
}


// 取出一整行的起始地址(只适用于二维张量)
float* Tensor::RowPtr(int row) {
    CheckRank(*this, 2, "RowPtr");
    CheckAxisIndex(*this, 0, row, "RowPtr");
    return data.data() + row * shape[1];
}
const float* Tensor::RowPtr(int row) const {
    CheckRank(*this, 2, "RowPtr");
    CheckAxisIndex(*this, 0, row, "RowPtr");
    return data.data() + row * shape[1];
}


// 形状检查
void Tensor::AssertShape(const std::vector<int>& expected, const char* caller) const {
    if (shape != expected) {
        throw std::runtime_error(std::string(CallerName(caller)) + ": shape mismatch. expected " + ShapeToStirng(expected) + ", got " + ShapeToStringShort());
    }
}

// 形状转换
Tensor Tensor::ReshapeChecked(const std::vector<int>& new_shape, const char* caller) const {
    const size_t new_total = CheckedNumel(new_shape, CallerName(caller));
    if (new_total != data.size()) {
        throw std::runtime_error(std::string(CallerName(caller)) + ": cannot reshape with tensor with " + std::to_string(data.size()) + "elements into shape with " 
                                             + std::to_string(new_total) + "elements");
    }
    Tensor r = *this;
    r.shape = new_shape;
    return r;
}

// 打印张量信息
void Tensor::Print(const std::string& name, bool print_data) const {
    if (!name.empty()) {
        std::cout << name << " ";
    }
    std::cout << "shape = " << ShapeToStringShort() << " size = " << size() << std::endl;
    if (print_data) {
        for (int i = 0; i < data.size(); ++i) {
            std::cout << std::fixed << std::setprecision(6) << data[i];
            if (i + 1 < data.size()) {
                std::cout << " ";
            }
        }
        std::cout << std::endl;
    }
}


Tensor MakeTensor1D(int d0, float fill) {
    return Tensor({d0}, fill);
}

Tensor MakeTensor2D(int d0, int d1, float fill) {
    return Tensor({d0, d1}, fill);
}

Tensor MakeTensor3D(int d0, int d1, int d2, float fill) {
    return Tensor({d0, d1, d2}, fill);
}

Tensor MakeTensor4D(int d0, int d1, int d2, int d3, float fill) {
    return Tensor({d0, d1, d2, d3}, fill);
}

}