#pragma once
#include <vector>
#include <string>
#include <stdexcept>

namespace mini_llama {

struct Tensor { 
    std::vector<float> data;         // 保存张量的实际数据
    std::vector<int> shape;          // 保存张量的形状信息

    Tensor() = default;              // 默认构造函数
    explicit Tensor(const std::vector<int>& input_shape, float fill = 0.0f);   // 构造函数，初始化张量形状和数据


    // 返回张量数据的数量
    size_t size() const {
        return data.size();
    }
    size_t num_elements() const {
        return data.size();
    }


    // 返回张量的维度
    size_t num_dims() const {
        return shape.size();
    } 


    // 把多维向量下标转化为一维数组下标
    size_t FlatIndex(const std::vector<int>& indices) const;


    // 把张量形状转化为字符串
    std::string ShapeToStringShort() const;
    std::string ShapeString() const;

    
    // 通用的多维元素访问接口
    float& At(const std::vector<int>& indices);
    float At(const std::vector<int>& indices) const;



    // 访问一维张量的第i个元素
    float& At1(int i);
    float At1(int i) const;

    // 访问二维张量的第i行，第j列
    float& At2(int i, int j);
    float At2(int i, int j) const;

    // 访问三维张量元素
    float& At3(int i, int j, int k);
    float At3(int i, int j, int k) const;

    //访问四维张量
    float& At4(int i, int j, int k, int l);
    float At4(int i, int j, int k, int l) const;


    // 重载方括号标识符(i只能为一维下标)
    float& operator[](size_t i);
    float operator[](size_t i) const;


    // 取出一整行的起始地址(只适用于二维张量)
    float* RowPtr(int row);
    const float* RowPtr(int row) const;


    // 形状检查
    void AssertShape(const std::vector<int>& expected, const char* caller) const;

    // 形状转换
    Tensor ReshapeChecked(const std::vector<int>& new_shape, const char* caller) const;


    // 打印张量信息
    void Print(const std::string& name = "", bool print_data = false) const;
};


// 创建张量的辅助函数
Tensor MakeTensor1D(int d0, float fill = 0.0f);
Tensor MakeTensor2D(int d0, int d1, float fill = 0.0f);
Tensor MakeTensor3D(int d0, int d1, int d2, float fill = 0.0f);
Tensor MakeTensor4D(int d0, int d1, int d2, int d3, float fill = 0.0f);


}
