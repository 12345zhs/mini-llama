#pragma once

#include <vector>
#include <cstdint>

namespace mini_llama {

// 张量数据类型
enum GgmlType : uint32_t {
    kGgmlTypeF32 = 0,
    kGgmlTypeF16 = 1,
    kGgmlTypeQ40 = 2,
    kGgmlTypeQ41 = 3,
    kGgmlTypeQ80 = 8,
};


// 元数据的类型
enum GgufValueType : uint32_t {
    kGgufTypeUint8 = 0,
    kGgufTypeInt8 = 1,
    kGgufTypeUint16 = 2,
    kGgufTypeInt16 = 3,
    kGgufTypeUint32 = 4,
    kGgufTypeInt32 = 5,
    kGgufTypeFloat32 = 6,
    kGgufTypeBool = 7,
    kGgufTypeString = 8,
    kGgufTypeArray = 9,
    kGgufTypeUint64 = 10,
    kGgufTypeInt64 = 11,
    kGgufTypeFloat64 = 12,    
};


// gguf的Tensor信息
struct GgufTensorInfo {
    std::string name;
    std::vector<uint64_t> shape;
    uint32_t type = 0;
    uint64_t offset = 0;
};



// gguf的元数据键值对
struct GgufMetadataKv {
    std::string key;
    uint32_t type = 0;   // 对应GgufValueType
    std::string value_str;
};


// Gguf文件解析器
struct GgufReader {
    uint32_t version = 0;
    uint64_t n_tensors = 0;
    uint64_t n_metadata = 0;
    std::vector<GgufMetadataKv> metadata;    // 存储所有配置项
    std::vector<GgufTensorInfo> tensors;     // 存储张量的信息
    uint64_t data_offset = 0;                // 二进制数据开始的地址

    bool Load(const std::string& path);      // 执行解析
    std::string load_error;                  // 解析失败 

};

// 辅助函数
const char* GgmlTypename(uint32_t type);
const char* GgufValueTypeName(uint32_t type);

// 解析查看
void InspectGguf(const GgufReader& reader);


// 读取张量数据
std::vector<uint8_t> ReadGgufTensorRaw(const std::string& path, const GgufTensorInfo& info, uint64_t data_offset);

// 张量占多少个字节
bool GgufTensorDataSize(const GgufTensorInfo& info, uint64_t& out_size);


// 提取元数据
bool GgufGetMetadataInt(const std::string& path, const std::string& key, int64_t& out);
bool GgufGetMetadataFloat(const std::string& path, const std::string& key, double& out);
bool GgufGetMetadataString(const std::string& path, const std::string& key, std::string& out);

bool GgufGetMetadataStringArray(const std::string& path, const std::string& key, std::vector<std::string>& out);
bool GgufGetMetadataIntArray(const std::string& path, const std::string& key, std::vector<int32_t>& out);




 
}