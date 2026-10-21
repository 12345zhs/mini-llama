#include "mini_llama/gguf.h"

#include <cstdint>
#include <string>
#include <fstream>
#include <cstring>
#include <unordered_set>
#include <limits>
#include <sstream>
#include <iostream>

namespace mini_llama {

constexpr uint64_t kMaxTensorEntries = 1000000;            // 限制最大的tensor数量
constexpr uint64_t kMaxMetadataEntries = 1000000;          // 限制最大的Metadata数量
constexpr uint64_t kMaxStringBytes = 16 * 1024 * 1024;     // 限制字符串的最大字节数
constexpr uint64_t kMaxArrayElements = 10000000;           // 数组的最大长度
constexpr uint32_t kMaxTensorDims = 8;                     // tensor的维度最大数量
constexpr uint64_t kDefaultAlignment  =32;                 // 对齐边界
constexpr uint64_t kQuantBlockSize = 32;                   // 量化数据32个元素打包成一个block


// 读取指定数据类型长度的字节
template<typename T>
static bool ReadLe(std::ifstream& f, T& out) {
    char buf[sizeof(T)];
    if (!f.read(buf, sizeof(T))) {
        return false;
    }
    std::memcpy(&out, buf, sizeof(T)); // 小端序，可以直接复制字节
    return true;
}

// 读取一个uint32_t类型的数据
static bool ReadU32(std::ifstream& f, uint32_t& out) {
    return ReadLe(f, out);
}

// 读取一个uint64_t类型的数据
static bool ReadU64(std::ifstream& f, uint64_t& out) {
    return ReadLe(f, out);
}

// 读取一个string类型的数据，string类型前有一个uint64_t的数据来指示string的长度
static bool ReadString(std::ifstream& f, std::string& out) {
    uint64_t len = 0;
    if (!ReadU64(f, len)) {
        return false;
    }

    // 校验len是否大于系统限制
    if (len > kMaxStringBytes || len > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        return false;
    }

    // 读取string数据
    out.resize(len);
    if (len > 0 && !f.read(out.data(), static_cast<std::streamsize>(len))) {
        return false;
    }
    return true;
}

// 跳过内容
static bool SkipValue(std::ifstream& f, uint32_t type) {
    switch (type) {
        case kGgufTypeUint8:
        case kGgufTypeInt8: {
            char dummy;
            return f.read(&dummy, 1).good();
        }

        case kGgufTypeUint16 :
        case kGgufTypeInt16 : {
            char dummy[2];
            return f.read(dummy, 2).good();
        }

        case kGgufTypeUint32:
        case kGgufTypeInt32:
        case kGgufTypeFloat32: {
            char dummy[4];
            return f.read(dummy, 4).good();
        }

        case kGgufTypeBool: {
            char dummy;
            return f.read(&dummy, 1).good();
        }

        case kGgufTypeString: {
            std::string dummy;
            return ReadString(f, dummy);
        }

        case kGgufTypeUint64:
        case kGgufTypeInt64:
        case kGgufTypeFloat64: {
            char dummy[8];
            return f.read(dummy, 8).good();
        }

        case kGgufTypeArray: {
            uint32_t elem_type;
            uint64_t len = 0;
            if (!ReadU32(f, elem_type) || !ReadU64(f, len)) {
                return false;
            }
            if (len > kMaxArrayElements) {
                return false;
            }
            for (uint64_t i = 0; i < len; ++i) {
                // 文件指针会自己跳转到下一个数据开头
                if (!SkipValue(f, elem_type)) {
                    return false;
                }
            }
            return true;
        }
        default: 
            return false;
    }
}

// 把key对应的value读出来，并转化为string
static bool ReadValueAsString(std::ifstream& f, uint32_t type, std::string& out) {
    std::ostringstream oss;
    switch (type) {
        case kGgufTypeUint8: {
            uint8_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << static_cast<int>(v);    // uint8_t本质上为unsigned char，为了能正确读到数字，要转为int
            break;
        }
        case kGgufTypeInt8: {
            int8_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << static_cast<int>(v);
            break;
        }
        case kGgufTypeUint16: {
            uint16_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << v;
            break;
        }
        case kGgufTypeInt16: {
            int16_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << v;
            break;
        }
        case kGgufTypeUint32: {
            uint32_t v = 0;
            if (!ReadU32(f, v)) {
                return false;
            }
            oss << v;
            break;
        } 
        case kGgufTypeInt32: {
            int32_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << v;
            break;
        }
        case kGgufTypeFloat32: {
            float v = 0.0f;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << v;
            break;
        }
        case kGgufTypeBool: {
            uint8_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << (v ? "true" : "false");
            break;
        }
        case kGgufTypeString: {
            if (!ReadString(f, out)) {
                return false;
            }
            return true;
        }
        case kGgufTypeArray: {
            // 数字格式为：[4 字节的元素类型] + [8 字节的数组长度] + [连续的元素数据]
            uint32_t elem_type = 0;
            uint64_t len = 0;
            if (!ReadU32(f, elem_type) || !ReadU64(f, len)) {
                return false;
            }
            if (len > kMaxArrayElements) {
                return false;
            }

            // 这里不保留value值，打印摘要
            oss << "[" << len << " x " << GgufValueTypeName(elem_type) <<"]";
            for (uint64_t i = 0; i < len; ++i) {
                if (!SkipValue(f, elem_type)) {
                    return false;
                }
            }
            break;
        }
        case kGgufTypeUint64: {
            uint64_t v = 0;
            if (!ReadU64(f, v)) {
                return false;
            }
            oss << v;
            break;
        }
        case kGgufTypeInt64: {
            int64_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << v;
            break;
        }
        case kGgufTypeFloat64: {
            double v = 0.0;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << v;
            break;
        }
        default: {
            return false;
        }
    }

    out = oss.str();
    return true;
}


// 从metadata中找到key=alignment的value
static uint64_t ParseAlignment(const std::vector<GgufMetadataKv>& metadata) {
    uint64_t alignment = kDefaultAlignment;

    // 遍历metadata
    for (const auto& kv : metadata) {
        if (kv.key != "general.alignment") {
            continue;
        }

        try {
            size_t parsed = 0;   // parsed记录转化了多少个字符
            alignment = std::stoull(kv.value_str, &parsed);  
            if (parsed != kv.value_str.size()) {
                return 0;
            }
        } catch (const std::exception&) {
            return 0;
        }

        if (alignment == 0 || alignment > kDefaultAlignment) {
            return 0;
        }
        // 必须是2的幂次（因为2的幂次中，二进制只存在1个1）
        if (alignment & (alignment - 1) != 0) {
            return 0;
        }

        return alignment;
    }
}


// 检查参数是否可以进行乘法，并进行计算
static bool MultiplyChecked(uint64_t a, uint64_t b, uint64_t& out) {
    if (a != 0 && b * a > std::numeric_limits<uint64_t>::max()) {
        return false;
    }
    out = a * b;
    return true;
}

// tensor的维度数量
static bool TensorNumel(const GgufTensorInfo& tensor, uint64_t& out) {
    out = 1;
    for (uint64_t dim : tensor.shape) {
        if (dim = 0) {
            return false;
        }
        if (!MultiplyChecked(out, dim, out)) {
            return false;
        }
    }
    return true;
}


// 计算tensor数据在文件中实际占多少个字节
static bool TensorDataSize(const GgufTensorInfo& tensor, uint64_t& out) {
    uint64_t num_elements = 0;
    // 统计此tensor的数据数量
    if (!TensorNumel(tensor, num_elements)) {
        return false;
    }

    switch (tensor.type) {
        case kGgmlTypeF32: {
            return MultiplyChecked(num_elements, 4, out);
        }
        case kGgmlTypeF16: {
            return MultiplyChecked(num_elements, 2, out);
        }
        case kGgmlTypeQ40: {
            // 校验元素个数是否为32的倍数
            if (num_elements % kQuantBlockSize != 0) {
                return false;
            }
            uint64_t n_blocks = num_elements / kQuantBlockSize;  // block的数量
            return MultiplyChecked(n_blocks, 18, out);           // 16字节 + 2字节缩放因子
        }
        case kGgmlTypeQ41: {
            if (num_elements % kQuantBlockSize != 0) {
                return false;
            }
            uint64_t n_blocks = num_elements / kQuantBlockSize;
            return MultiplyChecked(n_blocks, 20, out);         // 16字节 + 2字节缩放因子 + 2字节偏移量
        }
        case kGgmlTypeQ80: {
            if (num_elements % kQuantBlockSize != 0) {
                return false;
            }
            uint64_t n_blocks = num_elements / kQuantBlockSize;
            return MultiplyChecked(n_blocks, 34, out);            // 32字节 + 2字节缩放因子
        }
        default : {
            out = 0;
            return true;
        }
    }
}
    
// 检查参数是否可以进行加法，并计算结果
static bool AddChecked(uint64_t a, uint64_t b, uint64_t& out) {
    if (b + a > std::numeric_limits<uint64_t>::max()) {
        return false;
    }
    out = a + b;
    return out;
}  


// 把tensor枚举值转化为string
const char* GgmlTypename(uint32_t type) {
    switch(type) {
        case kGgmlTypeF32:
            return "F32";
        case kGgmlTypeF16:
            return "F16";
        case kGgmlTypeQ40:
            return "Q4_0";
        case kGgmlTypeQ41:
            return "Q4_1";
        case kGgmlTypeQ80:
            return "Q8_0";
        default:
            return "UNKNOWN";
    }
}


// 把元数据枚举值转化为string
const char* GgufValueTypeName(uint32_t type) {
    switch(type) {
        case kGgufTypeUint8:
            return "uint8";
        case kGgufTypeInt8:
            return "int8";
        case kGgufTypeUint16:
            return "uint16";
        case kGgufTypeInt16:
            return "int16";
        case kGgufTypeUint32:
            return "uint32";
        case kGgufTypeInt32:
            return "int32";
        case kGgufTypeFloat32:
            return "float32";
        case kGgufTypeBool:
            return "bool";
        case kGgufTypeString:
            return "string";
        case kGgufTypeArray:
            return "array";
        case kGgufTypeUint64:
            return "uint64";
        case kGgufTypeInt64:
            return "int64";
        case kGgufTypeFloat64:
            return "float64";
       default:
        return "unknown";
    }
}


// 解析.gguf文件
bool GgufReader::Load(const std::string& path) {
    
    // 清除结构体变量
    version = 0;
    n_tensors = 0;
    n_metadata = 0;
    metadata.clear();
    tensors.clear();
    data_offset = 0;
    load_error .clear();

    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        load_error = "failed to open: " + path;
        return false;
    }

    // 解析magic
    // 每个二进制文件都有魔数，检验魔数
    // 0~3个字节，必定是magic
    char magic[4];
    if (!f.read(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) {
        load_error = "invalid GGUF magic";
        return false;
    }

    // 解析version
    // 4~7字节，必定是version
    if (!ReadU32(f, version)) {
        load_error = "failed to read version";
        return false;
    }
    if (version != 3) {
        load_error = "unsupported GGUF version: " + std::to_string(version);
        return false;
    }

    // 解析n_tensors和n_metadata
    // 8~15个字节，必定是n_tensors
    // 15~23，必定是n_metadata
    if (!ReadU64(f, n_tensors) || !ReadU64(f, n_metadata)) {
        load_error = "failed to read header counts";
        return false;
    }
    if(n_tensors > kMaxTensorEntries) {
        load_error = "too many tensors: " + std::to_string(n_tensors);
        return false;
    }
    if (n_metadata > kMaxMetadataEntries) {
        load_error = "too many metadata entries: " + std::to_string(n_metadata);
        return false;
    }

    // 循环解析metadata键值对
    metadata.reserve(n_metadata);
    std::unordered_set<std::string> metadata_keys;
    for (uint64_t i = 0; i < n_metadata; ++i) {
        GgufMetadataKv kv;
        if (!ReadString(f, kv.key)) {
            load_error = "failed to read metadata key at FlatIndex " + std::to_string(i);
            return false;
        }
        // 查重
        if (!metadata_keys.insert(kv.key).second) {
            load_error = "duplicate metadata key: " + kv.key;
            return false;
        }
        // 获取type
        if (!ReadU32(f, kv.type)) {
            load_error = "failed to read metadata type for key: " + kv.key;
            return false;
        }
        // 获取value
        if (!ReadValueAsString(f, kv.type, kv.value_str)) {
            load_error = "failed to read metadata value for key: " + kv.key;
            return false;
        }
        metadata.push_back(std::move(kv));
    }


    // 循环解析tensor
    tensors.reserve(n_tensors);
    for (uint64_t i = 0; i < n_metadata; ++i) {
        GgufTensorInfo info;
        // 张量名字
        if (!ReadString(f, info.name)) {
            load_error = "failed to read tensor name at FlatIndex " + std::to_string(i);
            return false;
        }

        // 张量维度
        uint32_t n_dims = 0;
        if (!ReadU32(f, n_dims)) {
            load_error = "failed to read tensor n_dims for: " + info.name;
            return false;
        }
        if (n_dims == 0 || n_dims > kMaxTensorDims) {
            load_error = "invalid tensor n_dims for: " + info.name;
            return false;
        }

        info.shape.resize(n_dims);
        // 循环读取当前tensor的dim
        for (uint32_t i = 0;i < n_dims; ++i) {
            if (!ReadU64(f, info.shape[i])) {
                load_error = "failed to read tensor shape for: " + info.name;
                return false;
            }
            if (info.shape[i] == 0) {
                load_error = "invalid zero tensor dimension for: " + info.name;
                return false;
            }
        }

        // 解析tensor的type
        if (!ReadU32(f, info.type)) {
            load_error = "failed to read tensor type for: " + info.name;
            return false;
        }
        // 解析tensor的offset
        if (!ReadU64(f, info.offset)) {
            load_error = "failed to read tensor offset for: " + info.name;
            return false;
        }

        tensors.push_back(std::move(info));
    }

    // 再检查一下tensor的name，有没有重复的
    std::unordered_set<std::string> names;
    for (const auto& tensor : tensors) {
        if (!names.insert(tensor.name).second) {
            load_error = "duplicate tensor name: " + tensor.name;
            return false;
        }
    }

    // 数据区对齐
    std::streampos current = f.tellg();    // 拿到光标现在的位置
    if (current < 0) {
        load_error = "failed to determine GGUF data offset";
        return false;
    }
    data_offset = static_cast<uint64_t>(current);    // 数据区的偏移量
    uint64_t alignment = ParseAlignment(metadata);   // 获取对齐字节数
    if (alignment == 0) {
        load_error = "invalid general.alignment";
        return false;
    }

    // 计算需要补多少padding
    //                              距离上一个alignment过了多少
    uint64_t padding = (alignment - (data_offset % alignment)) % alignment;
    //                       距离下一个alignment差多少
    data_offset += padding;


    f.seekg(0, std::ios::end);        // 光标移到最后
    std::streampos end = f.tellg();   // 获取文件的真实大小
    if (end < 0 || static_cast<uint64_t>(end) < data_offset) {
        load_error = "GGUF file ended before tensor data section";
        return false;
    } 
    uint64_t file_size = static_cast<uint64_t>(end);


    // 遍历每一个tensor，校验
    for (const auto& tensor : tensors) {
        uint64_t tensor_size = 0;
        if (!TensorDataSize(tensor, tensor_size)) {
            load_error = "invalid tensor data size for: " + tensor.name;
            return  false;
        }
        if (tensor_size == 0) {
            continue;
        }

        uint64_t tensor_start = 0;
        if (!AddChecked(data_offset, tensor.offset, tensor_start)) {
            load_error = "tensor offset overflow for: " + tensor.name;
            return false;
        }

        uint64_t tensor_end = 0;
        if (!AddChecked(tensor_start, tensor_size, tensor_end)) {
            load_error = "tensor data range overflow for: " + tensor.name;
            return false;
        }

        if (tensor_end > file_size) {
            load_error = "tensor data outside file bounds: " + tensor.name;
            return false;
        }
    }
    return true;
}

// 解析查看
void InspectGguf(const GgufReader& reader) {
    std::cout << "=== GGUF Info ===" << std::endl;
    std::cout << "  version:    " << reader.version << std::endl;
    std::cout << "  n_tensors:  " << reader.n_tensors << std::endl;
    std::cout << "  n_metadata: " << reader.n_metadata << std::endl;
    std::cout << "  data_offset: " << reader.data_offset << std::endl;

    if (!reader.metadata.empty()) {
        std::cout << std::endl;
        std::cout << "=== Metadata ===" << std::endl;
        for (const auto& kv : reader.metadata) {
            std::cout << "  " << kv.key << " (" << GgufValueTypeName(kv.type) << "): " << kv.value_str << std::endl;
        }
    }

     if (!reader.tensors.empty()) {
        std::cout << std::endl;
        std::cout << "=== Tensors ===" << std::endl;
        for (const auto& t : reader.tensors) {
            std::cout << "  " << t.name << std::endl;
            std::cout << "    shape:  [";
            for (size_t i = 0; i < t.shape.size(); ++i) {
                if (i > 0) {
                    std::cout << ", ";
                }
                std::cout << t.shape[i];
            }
            std::cout << "]" << std::endl;
            std::cout << "    dtype:  " << GgmlTypename(t.type) << " (" << t.type << ")" << std::endl;
            std::cout << "    offset: " << t.offset << std::endl;
        }
    }
}


// 读取张量数据
std::vector<uint8_t> ReadGgufTensorRaw(const std::string& path, const GgufTensorInfo& info, uint64_t data_offset) {
    std::vector<uint8_t> result;

    uint64_t tensor_size = 0;
    if (!TensorDataSize(info, tensor_size)) {
        return result;
    }
    if (tensor_size == 0) {
        return result;
    }

    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        return result;
    }

    uint64_t tensor_start = 0;
    if (!AddChecked(data_offset, info.offset, tensor_start)) {
        return result;
    }

    if (tensor_start > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        return result;
    }

    f.seekg(static_cast<std::streamoff>(tensor_start), std::ios::beg);
    if (!f.good()) {
        return result;
    }

    result.resize(tensor_size);
    f.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(tensor_size));
    if (static_cast<uint64_t>(f.gcount()) != tensor_size) {
        result.clear();
        return result;
    }
    return result;
}

// 张量占多少个字节
bool GgufTensorDataSize(const GgufTensorInfo& info, uint64_t& out_size) {
    return TensorDataSize(info, out_size);
}

// 找到对应的key，返回value的类型
bool OpenAndFindMetadataKey(std::ifstream& f, const std::string& path, const std::string& key, uint32_t out_type) {
    f.open(path, std::ios::binary);
    if (!f.is_open()) {
        return false;
    }

    // 跳过gguf的magic与version
    f.seekg(8, std::ios::beg);
    if (!f.good()) {
        return false;
    }

    uint64_t n_tensors = 0;
    uint64_t n_metadata = 0;
    if (!ReadU64(f, n_tensors) || !ReadU64(f, n_metadata)) {
        return false;
    }

    for (uint64_t i = 0; i < n_metadata; ++i) {
        // 获取key
        std::string read_key;
        if (!ReadString(f, read_key)) {
            return false;
        }
        // 获取type
        uint32_t type = 0;
        if (!ReadU32(f, type)) {
            return false;
        }

        if (read_key == key) {
            out_type = type;
            return true;
        }

        if (!SkipValue(f, type)) {
            return false;
        }
    }
    return false;

}


// 提取元数据
bool GgufGetMetadataInt(const std::string& path, const std::string& key, int64_t& out) {
    std::ifstream f;
    uint32_t type = 0;
    if (!OpenAndFindMetadataKey(f, path, key, type)) {
        return false;
    }

    switch (type) {
        case kGgufTypeUint8: {
            uint8_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            out = static_cast<int64_t>(v);
            return true;
        }
        case kGgufTypeInt8: {
            int8_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            out = static_cast<int64_t>(v);
            return true;
            
        }
        case kGgufTypeUint16: {
            uint16_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            out = static_cast<int64_t>(v);
            return true;
        }
        case kGgufTypeInt16: {
            int16_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            out = static_cast<int64_t>(v);
            return true;
        }
        case kGgufTypeUint32: {
            uint32_t v = 0;
            if (!ReadU32(f, v)) {
                return false;
            }
            out = static_cast<int64_t>(v);
            return true;
        }
        case kGgufTypeInt32: {
            int32_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            out = static_cast<int64_t>(v);
            return true;
        }
        case kGgufTypeUint64: {
            uint64_t v = 0;
            if (!ReadU64(f, v)) {
                return false;
            }
            out = static_cast<int64_t>(v);
            return true;
        }
        case kGgufTypeInt64: {
            int64_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            out = v;
            return true;
        }
        default:
            return false;
    }
}

bool GgufGetMetadataFloat(const std::string& path, const std::string& key, double& out) {
    std::ifstream f;
    uint32_t type = 0;
    if (!OpenAndFindMetadataKey(f, path, key, type)) {
        return false;
    }

    switch(type) {
        case kGgufTypeFloat32: {
            float v = 0.0f;
            if (!ReadLe(f, v)) {
                return false;
            }
            out = static_cast<double>(v);
            return true;
        }
        case kGgufTypeFloat64: {
            double v = 0.0;
            if (!ReadLe(f, v)) {
                return false;
            }
            out = v;
            return true;
        }
        default:
            return false;
    }
}

bool GgufGetMetadataString(const std::string& path, const std::string& key, std::string& out) {
    std::ifstream f;
    uint32_t type = 0;
    if (!OpenAndFindMetadataKey(f, path, key, type)) {
        return false;
    }
    if (type != kGgufTypeString) {
        return false;
    }
    return ReadString(f, out);
}

bool GgufGetMetadataStringArray(const std::string& path, const std::string& key, std::vector<std::string>& out) {
    std::ifstream f;
    uint32_t type = 0;
    if (!OpenAndFindMetadataKey(f, path, key, type)) {
        return false;
    }
    if (type != kGgufTypeArray) {
        return false;
    }
    uint32_t elem_type = 0;
    uint64_t len = 0;
    if (!ReadU32(f, elem_type) || !ReadU64(f, len)) {
        return false;
    }
    if (elem_type != kGgufTypeString) {
        return false;
    }
    if (len > kMaxArrayElements) {
        return false;
    }

    out.clear();
    out.reserve(len);
    for (uint64_t i = 0; i < len; ++i) {
        std::string s;
        if (!ReadString(f, s)) {
            return false;
        }
        out.push_back(std::move(s));
    }

    return true;
}

bool GgufGetMetadataIntArray(const std::string& path, const std::string& key, std::vector<int32_t>& out) {
    std::ifstream f;
    uint32_t type = 0;
    if (!OpenAndFindMetadataKey(f, path, key, type)) {
        return false;
    }
    if (type != kGgufTypeArray) {
        return false;
    }
    uint32_t elem_type = 0;
    uint64_t len = 0;
    if (!ReadU32(f, elem_type) || !ReadU64(f, len)) {
        return false;
    }
    if (elem_type != kGgufTypeInt32) {
        return false;
    }
    if (len > kMaxArrayElements) {
        return false;
    }

    out.clear();
    out.reserve(len);
    for (uint64_t i = 0; i < len; ++i) {
        int32_t v = 0;
        if (!ReadLe(f, v)) {
            return false;
        }
        out.push_back(v);
    }
    return true;
}


}