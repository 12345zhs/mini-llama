#include "mini_llama/loader.h"
#include "mini_llama/model.h"

#include <sstream>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <map>
#include <numeric>
#include <algorithm>
#include <cerrno>


namespace mini_llama {


// 从json中查找某个key对应的value的位置
static size_t FindConfigKey(const std::string& content, const std::string& key) {

    // 拼接要查找的字符串
    const std::string marker = "\"" + key + "\"";
    size_t key_pos = content.find(marker);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("missing config key: " + key);
    }

    size_t colon_pos = content.find(":", key_pos + marker.size());
    if (colon_pos == std::string::npos) {
        throw std::runtime_error("miss ':' after config key: " + key);
    }

    size_t value_pos = colon_pos + 1;
    while (value_pos < content.size() && std::isspace(static_cast<unsigned char>(content[value_pos]))) {
        ++value_pos;
    }
    
    if (value_pos >= content.size()) {
        throw std::runtime_error("missing value for config key: " + key);
    }

    return value_pos;
}


// 判断当前字符是否是Json的合法JSON结束符
static bool startIsJsonValueTerminator(char c) {
    return std::isspace(static_cast<unsigned char>(c)) || c == ',' || c == '}' | c == ']';
}


// 检查数字后面的字符是否合法
static void ValidateNumberTerminator(const std::string& content, size_t value_pos, size_t parsed, const std::string& key) {
    size_t pos = value_pos + parsed;
    while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) {
        ++pos;
    }

    if (pos < content.size() && !startIsJsonValueTerminator(content[pos])) {
        throw std::runtime_error("invalid numeric value for config key: " + key);
    }

}



// 解析key对应的整形数字
static int ParseRequiredInt(const std::string& content, const std::string& key) {
    size_t value_pos = FindConfigKey(content, key);
    size_t parsed = 0;

    int value = 0;
    try {
        value = std::stoi(content.substr(value_pos), &parsed);
    } catch(const std::exception&) {
        throw std::runtime_error("invalid integer for config key: " + key);
    }

    // 检查是否真的解析到了数字
    if (parsed == 0) {
        throw std::runtime_error("invalid integer for config key: " + key);
    }

    // 检查数字后面的字符是否合法
    ValidateNumberTerminator(content, value_pos, parsed, key);
    return value;
}


// 解析key对应的float数字
static float ParseRequiredFloat(const std::string& content, const std::string& key) {
    size_t value_pos = FindConfigKey(content, key);
    size_t parsed = 0;
    float value;
    try {
        value = std::stof(content.substr(value_pos), &parsed);
    } catch(const std::exception&) {
        throw std::runtime_error("invalid float for config key: " + key);
    }

    if (parsed == 0) {
        throw std::runtime_error("invalid float for config key: " + key);
    }
    ValidateNumberTerminator(content, value_pos, parsed, key);
    return  value;
}

// 确保从json中解析出来的config是合法的
static void validateConfig(const ModelConfig& config) {
    if (config.vocab_size <= 0 || config.dim <= 0 || config.hidden_dim <= 0 ||
        config.n_layers <= 0 || config.n_heads <= 0 || config.n_kv_heads <= 0 ||
        config.max_seq_len <= 0) {
        throw std::runtime_error("all model dimensions must be positive");
    }

    if (config.dim % config.head_dim != 0) {
        throw std::runtime_error("dim must be divisible by n_heads");
    }
    if (config.n_heads % config.n_kv_heads != 0) {
        throw std::runtime_error("n_heads must be divisible by n_kv_heads");
    } 
    if (config.head_dim <= 0 || config.head_dim % 2 != 0) {
        throw std::runtime_error("head_dim must be a positive even number");
    }

    if (config.rope_theta <= 0.0f || std::isfinite(config.rope_theta)) {
        throw std::runtime_error("rope_theta must be positive");
    }

    if (config.rms_norm_eps <= 0.0f || std::isfinite(config.rms_norm_eps)) {
        throw std::runtime_error("rms_norm_eps must be positive");
    }
}

// 从字符串中解析出Config
static ModelConfig ParseConfigJson(const std::string content) {
    ModelConfig config;
    config.vocab_size = ParseRequiredInt(content, "vocab_size");
    config.dim = ParseRequiredInt(content, "dim");
    config.hidden_dim = ParseRequiredInt(content, "hidden_dim");
    config.n_layers = ParseRequiredInt(content, "n_layers");
    config.n_heads = ParseRequiredInt(content, "n_heads");
    config.n_kv_heads = ParseRequiredInt(content, "n_kv_heads");
    config.max_seq_len = ParseRequiredInt(content, "max_seq_len");
    config.rope_theta = ParseRequiredFloat(content, "rope_theta");
    config.rms_norm_eps = ParseRequiredFloat(content, "rms_norm_eps");
    config.head_dim = config.dim / config.n_heads;

    validateConfig(config);
    return config;
}


// 从字符串中解析tokenizer的那一部分   
static std::string ParseObjectForKey(const std::string& content, const std::string& key) {
    const std::string marker = "\"" + key + "\"";
    size_t key_pos = content.find(marker);
    if (key_pos == std::string::npos) {
        return "";
    }

    size_t obj_start = content.find("{", key_pos + marker.size());
    if (obj_start == std::string::npos) {
        throw std::runtime_error("missing object for key: " + key);
    }

    int depth = 0;
    for (size_t pos = obj_start; pos < content.size(); ++pos) {
        if (content[pos] == '{') {
            ++depth;
        } else if (content[pos] == '}') {
            --depth;
            if (depth == 0) {
                return content.substr(obj_start, pos - obj_start);
            }
        }
    }
    throw std::runtime_error("unclosed object for key: " + key);
}









// 找出key所对应的value所在的位置
static size_t FindOptionalKey(const std::string& content, const std::string& key) {
    const std::string marker = "\"" + key + "\"";
    size_t key_pos = content.find(marker);

    if (key_pos == std::string::npos) {
        return std::string::npos;
    }

    size_t colon_pos = content.find(":", key_pos + marker.size());
    if (colon_pos == std::string::npos) {
        throw std::runtime_error("missing ':' after key: " + key);
    }
    size_t value_pos = colon_pos + 1;
    while (value_pos < content.size() && std::isspace(static_cast<unsigned char>(content[value_pos]))) {
        value_pos++;
    }
    if (value_pos >= content.size()) {
        throw std::runtime_error("missing value for key: " + key);
    }
    return value_pos;
}

// 从tokenizer字段中提取string类型的配置项
static std::string ParseOptionalStringField(const std::string& content, const std::string& key, const std::string& fallback) {
    size_t value_pos = FindOptionalKey(content, key);
    // 找不到直接返回默认值
    if (value_pos == std::string::npos) {
        return fallback;
    }

    if (content[value_pos] != '"') {
        throw std::runtime_error("expected string value for key: " + key);
    }
    size_t end = content.find('"', value_pos + 1);
    if (end == std::string::npos) {
        throw std::runtime_error("unterminated string value for key: " + key);
    }
    return content.substr(value_pos + 1, end - value_pos - 1);
}


// 从tokenizer字段中提取int类型的配置项
static int ParseOptionalIntField(const std::string& content, const std::string& key, const int fallback) {
    size_t value_pos = FindOptionalKey(content, key);
    if (value_pos == std::string::npos) {
        return fallback;
    }

    size_t parsed = 0;
    int value = 0;
    try {
        value = std::stoi(content.substr(value_pos), &parsed);
    } catch(const std::exception&) {
        throw std::runtime_error("invalid integer for key: " + key);
    }

    if (parsed == 0) {
        throw std::runtime_error("invalid integer for key: " + key);
    }

    ValidateNumberTerminator(content, value_pos, parsed, key);
    return value;
}

// 从json中解析Tokenizer分词器
static TokenizerInfo ParseTokenizerJson(const std::string& content) {
    TokenizerInfo tokenizer;
    std::string obj = ParseObjectForKey(content, "tokenizer");

    if (obj.empty()) {
        return tokenizer;
    }

    tokenizer.type = ParseOptionalStringField(obj, "type", tokenizer.type);
    tokenizer.path = ParseOptionalStringField(obj, "path", tokenizer.path);
    tokenizer.bos_id = ParseOptionalIntField(obj, "bos_id", tokenizer.bos_id);
    tokenizer.eos_id = ParseOptionalIntField(obj, "eos_id", tokenizer.eos_id);
    tokenizer.unk_id = ParseOptionalIntField(obj, "unk_id", tokenizer.unk_id);

    if (tokenizer.type != "ascii" && tokenizer.type != "json_vacab") {
        throw std::runtime_error("unsupported tokenizer type: " + tokenizer.type);
    }
    if (tokenizer.bos_id < 0 || tokenizer.eos_id < 0 || tokenizer.unk_id < 0) {
        throw std::runtime_error("tokenizer special ids must be non-negative");
    }
    if (tokenizer.bos_id == tokenizer.eos_id || tokenizer.bos_id == tokenizer.unk_id || tokenizer.eos_id == tokenizer.unk_id) {
        throw std::runtime_error("tokenizer special ids must be distinct");
    }
    if (tokenizer.type == "json_vacab" && tokenizer.path.empty()) {
        throw std::runtime_error("json_vocab tokenizer requires a path");
    }
    return tokenizer;
}





// 解析ternsor中的各种信息
static std::vector<TensorInfo> ParseTensorMetadata(const std::string& content) {
    std::vector<TensorInfo> result;

    // 定位大数组
    const std::string tensor_marker = "\"tensors\"";
    size_t tensor_pos = content.find(tensor_marker);
    if (tensor_pos == std::string::npos) {
        throw std::runtime_error("missing 'tensors' field in model.json");
    }

    size_t bracket_open = content.find("[", tensor_pos);
    if (bracket_open == std::string::npos) {
        throw std::runtime_error("missing '[' after 'tensors'");
    }


    // 遍历每一个张量
    size_t i = bracket_open + 1;
    int depth = 1;
    while (i < content.size() && depth > 0) {


        // 查找下一个'{'
        while (i < content.size() && content[i] != '{') {
            if (content[i] == '[') {
                ++depth;
            } else if (content[i] ==']') {
                --depth;
            }
            ++i;
        }

        if (depth <= 0 || i >= content.size()) {
            break;
        }


        // 解析出来一个tensor
        TensorInfo info;
        size_t obj_start = i;
        size_t obj_end = content.find("}", obj_start);
        if (obj_end == std::string::npos) {
            throw std::runtime_error("unclosed tensor object");
        }
        std::string obj = content.substr(obj_start, obj_end - obj_start + 1);
    
    
    
        bool has_name = false;
        bool has_dtype = false;
        bool has_shape = false;
        bool has_offset = false;
        bool has_byte_size = false;

        // 解析name
        size_t name_pos = obj.find("\"name\"");
        if (name_pos != std::string::npos) {
            size_t name_colon = obj.find(":", name_pos);
            size_t name_quote = obj.find("\"", name_colon + 1);
            size_t name_end = obj.find("\"", name_quote + 1);

            if (name_quote != std::string::npos && name_end != std::string::npos) {
                info.name = obj.substr(name_quote + 1, name_end - name_quote - 1);
                has_name = true;
            }
        }



        // 解析dtype
        size_t dtype_pos = obj.find("\"dtype\"");
        if (dtype_pos != std::string::npos) {
            size_t dtype_colon = obj.find(":", name_pos);
            size_t dtype_quote = obj.find("\"", dtype_colon + 1);
            size_t dtype_end = obj.find("\"", dtype_quote + 1);

            if (dtype_quote != std::string::npos && dtype_end != std::string::npos) {
                info.dtype = obj.substr(dtype_quote + 1, dtype_end - dtype_quote - 1);
                has_dtype = true;
            }
        }




        // 解析 shape
        size_t shape_pos = obj.find("\"shape\"");
        if (shape_pos != std::string::npos) {
            size_t shape_bracket = obj.find("[", shape_pos);
            size_t shape_end = obj.find("]", shape_bracket);
            if (shape_bracket != std::string::npos && shape_end != std::string::npos) {
                // 提取出来维度string
                std::string shape_string_short = obj.substr(shape_bracket + 1, shape_end - shape_bracket - 1);

                std::stringstream ss(shape_string_short);
                std::string token;

                while (std::getline(ss, token, ',')) {
                    size_t start = token.find_first_not_of(" \t\n\r");
                    size_t end = token.find_last_not_of(" \t\n\r");

                    if (start != std::string::npos) {
                        int dim = std::stoi(token.substr(start, end - start + 1));
                        info.shape.push_back(dim);
                    }
                }
            }
            has_shape = true;
        }



        // 解析offset
        size_t offset_pos = obj.find("\"offset\"");
        if (offset_pos != std::string::npos) {
            size_t offset_colon = obj.find(":", offset_pos);
            if (offset_colon == std::string::npos) {
                throw std::runtime_error("tensor offset field is malformed");
            }

            info.offset = static_cast<size_t>(std::stoull(obj.substr(offset_colon  +1)));

            has_offset = true;
        }


        // 解析byte_size
        size_t byte_size_pos = obj.find("\"byte_size\"");
        if (byte_size_pos != std::string::npos) {
            size_t byte_size_colon = obj.find(":", byte_size_pos);
            if (byte_size_colon == std::string::npos) {
                throw std::runtime_error("tensor offset field is malformed");
            }

            info.byte_size = static_cast<size_t>(std::stoull(obj.substr(byte_size_colon  +1)));

            has_byte_size = true;
        }


        if (!has_name || info.name.empty()) {
            throw std::runtime_error("tensor missing name field");
        }
        if (!has_dtype) {
            throw std::runtime_error("tensor missing dtype field: " + info.name);
        }
        if (!has_shape) {
            throw std::runtime_error("tensor missing shape field: " + info.name);
        }
        if (!has_offset) {
            throw std::runtime_error("tensor missing offset field: " + info.name);
        }
        if (!has_byte_size) {
            throw std::runtime_error("tensor missing byte_size field: " + info.name);
        }

        result.push_back(info);
        i  = obj_end + 1; 
    }


    return result;
}


// 计算tensor的字节数
static size_t CheckedTensorByteSize(const TensorInfo& info) {
    size_t num_elements = 1;
    for (int dim : info.shape) {
        if (dim < 0) {
            throw std::runtime_error("invalid tensor shape for " + info.name);
        }

        const size_t d = static_cast<size_t>(dim);
        if (num_elements * d > std::numeric_limits<size_t>::max()) {
            throw std::runtime_error("tensor size overflow for " + info.name);
        }
        num_elements *= d;
    }

    if (num_elements * sizeof(float) > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("tensor byte size overflow for " + info.name);
    }

    return num_elements * sizeof(float);
}


// 根据ModelConfig，生成权重清单
static std::vector<std::pair<std::string, std::vector<int>>> ExpectedTensorShapes(const ModelConfig& config) {
    
    // 不随层数变化的全局组件：token——embedding、final_norm、lm_head
    std::vector<std::pair<std::string, std::vector<int>>> expected = {
        {"token_embedding", {config.vocab_size, config.dim}},
        {"final_norm", {config.dim}},
        {"lm_head", {config.vocab_size, config.dim}}
    };

    for (int layer = 0; layer < config.n_layers; layer++) {
        const std::string prefix = "layers." + std::to_string(layer) + ".";
        expected.push_back({prefix + "attention_norm", {config.dim}});
        expected.push_back({prefix + "wq", {config.n_heads * config.head_dim, config.dim}});
        expected.push_back({prefix + "wk", {config.n_kv_heads * config.head_dim, config.dim}});
        expected.push_back({prefix + "wv", {config.n_kv_heads * config.head_dim, config.dim}});

        expected.push_back({prefix + "wo", {config.dim, config.n_heads * config.head_dim}});

        expected.push_back({prefix + "ffn_norm", {config.dim}});
        expected.push_back({prefix + "w_gate", {config.hidden_dim, config.dim}});
        expected.push_back({prefix + "w_up", {config.hidden_dim, config.dim}});
        expected.push_back({prefix + "w_down", {config.dim, config.hidden_dim}});
    }
    return expected;
}


static std::string ShapeToString(const std::vector<int>& shape) {
    std::string result = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            result += ", ";
        }
        result += std::to_string(shape[i]);
    }
    result += "]";
    return result;
} 

// 校验manifest
static void ValidateManifest(const ModelManifest& manifest) {
    if (manifest.tensors.empty()) {
        throw std::runtime_error("model.json contains no tensors");
    }

    std::map<std::string, TensorInfo> tensor_map;
    std::vector<TensorInfo> by_offset;
    by_offset.reserve(manifest.tensors.size());

    for (const auto& info : manifest.tensors) {
        if (info.dtype != "float32") {
            throw std::runtime_error("unsupported dtype '" + info.dtype + "' for tensor: " + info.name);
        }

        if (info.shape.empty()) {
            throw std::runtime_error("tensor shape must not be empty: " + info.name);
        }

        // 字节数是否正确
        size_t expected_bytes = CheckedTensorByteSize(info);
        if (expected_bytes != info.byte_size) {
            throw std::runtime_error("tensor byte_size mismatch for " + info.name + ": expected " + std::to_string(expected_bytes) + ", manifest says " + std::to_string(info.byte_size));
        }

        // 内存偏移量是否溢出校验
        if (info.offset + info.byte_size > std::numeric_limits<size_t>::max()) {
            throw std::runtime_error("tensor offset overflow for " + info.name);
        }

        // 重名检测
        auto inserted = tensor_map.emplace(info.name, info); // 返回std::pair<iterator, bool>, bool代表是否插入成功
        if (!inserted.second) {
            throw std::runtime_error("duplicate tensor name: " + info.name);
        }

        by_offset.push_back(info);

        // 按照张量在json中的起始位置排序
        std::sort(by_offset.begin(), by_offset.end(), [](const TensorInfo& a, const TensorInfo& b) {
            return a.offset < b.offset;
        });


        // 检查当前tensor的起点是否小于前一个tensor的终点
        for (size_t i = 1; i < by_offset.size(); ++i) {
            size_t previous_end = by_offset[i - 1].offset + by_offset[i - 1].byte_size;   // 前一个tensor的终点

            if (by_offset[i].offset < previous_end) {
                throw std::runtime_error("tensor ranges overlap: " + by_offset[i - 1].name + " and " + by_offset[i].name);
            }
        }

        for (const auto& expected : ExpectedTensorShapes(manifest.config)) {
            auto it = tensor_map.find(expected.first);
            if (it == tensor_map.end()) {
                throw std::runtime_error("missing required tensor: " + expected.first);
            }
            if (it->second.shape != expected.second) {
                throw std::runtime_error("tensor shape mismatch for " + expected.first + ": expected " + ShapeToString(expected.second) + ", manifest says " + ShapeToString(it->second.shape));
            }
        }
    }

}

// 解析model.json文件，解析出来ModelManifest结构体
ModelManifest ParseManifest(const std::string& config_path) {
    ModelManifest manifest;

    // 打开文件
    std::ifstream cf(config_path);
    if (!cf.is_open()) {
        throw std::runtime_error("failed to open config: " + config_path);
    }

    // 提取全部内容到内存
    std::stringstream buffer;
    buffer << cf.rdbuf();
    if (cf.bad()) {
        throw std::runtime_error("failed to read config: " + config_path);
    }
    std::string content = buffer.str();

    manifest.config = ParseConfigJson(content);
    manifest.tokenizer = ParseTokenizerJson(content);
    manifest.tensors = ParseTensorMetadata(content);
    ValidateManifest(manifest);

    return manifest;
}

// 模型加载失败
static MiniLlamaModel LoadFailure(const std::string& message) {
    MiniLlamaModel model;
    model.loaded = false;
    model.load_error = message;
    std::cerr <<  "Failed to load model: " << message << std::endl;
    return model;
}


// 之前都是从json中读配置，现在从weight中获得权重
static Tensor LoadTensorByName(std::ifstream& wf, const std::map<std::string, TensorInfo>& tensor_map, const std::string name) {
    auto it = tensor_map.find(name);
    if (it == tensor_map.end()) {
        throw std::runtime_error("missing required tensor: " + name);
    }

    const TensorInfo& info = it->second;

    if (info.byte_size > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("tensor byte size exceeds stream limit for " + name);
    }

    if (info.offset > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("tensor offset exceeds stream limit for " + name);
    }

    // 移动文件指针,直接移动到offset位置
    wf.seekg(static_cast<std::streamoff>(info.offset), std::ios::beg);
    if (!wf.good()) {
        throw std::runtime_error("failed to seek to offset for tensor: " + name);
    }

    // 分配内润并读取
    Tensor t(info.shape, 0.0f);
    wf.read(reinterpret_cast<char*>(t.data.data()), static_cast<std::streamsize>(info.byte_size));
    //检查有没有读完
    if (static_cast<size_t>(wf.gcount()) != info.byte_size) {
        throw std::runtime_error("weights file ended while reading tensor: " + name);
    }

    return t;
}


// 加载.json与.bin文件
MiniLlamaModel LoadModel(const std::string& config_path, const std::string& weights_path) {
    // 建立config
    ModelManifest manifest;
    try {
        manifest = ParseManifest(config_path);
    } catch (const std::exception& e) {
        return LoadFailure(e.what());
    }


    std::cout << "Loaded config: vacab=" << manifest.config.vocab_size
              << " dim=" << manifest.config.dim
              << " layers=" << manifest.config.n_layers
              << " heads=" << manifest.config.n_heads
              << " kv_heads=" << manifest.config.n_kv_heads
              << " head_dim=" << manifest.config.head_dim
              << " hidden_dim=" << manifest.config.hidden_dim << std::endl;


    // 将张量存入map字典
    std::map<std::string, TensorInfo> tensor_map;
    for (const auto& info : manifest.tensors) {
        tensor_map[info.name] = info;
    }

    std::ifstream wf(weights_path, std::ios::binary);
    if (!wf.is_open()) {
        return LoadFailure("failed to open weights: " + weights_path);
    }


    // 把配置好的config传给MiniLlaModel
    MiniLlamaModel model;
    model.config = manifest.config;

    try {
        model.token_embedding = LoadTensorByName(wf, tensor_map, "token_embedding");
        model.layers.reserve(manifest.config.n_layers);

        // 给每一层layer传权重
        for (int layer = 0; layer < manifest.config.n_layers; ++layer) {
            LayerWeights& lw = model.layers[layer];
            const std::string prefix = "layers." + std::to_string(layer) + ".";

            lw.attention_norm = LoadTensorByName(wf, tensor_map, prefix + "attention_norm");
            lw.wq = ToQuantizedTensor(LoadTensorByName(wf, tensor_map, prefix + "wq"));
            lw.wk = ToQuantizedTensor(LoadTensorByName(wf, tensor_map, prefix + "wk"));
            lw.wv = ToQuantizedTensor(LoadTensorByName(wf, tensor_map, prefix + "wv"));
            lw.wo = ToQuantizedTensor(LoadTensorByName(wf, tensor_map, prefix + "wo"));

            lw.ffn_norm = LoadTensorByName(wf, tensor_map, prefix + "ffn_norm");
            lw.w_gate = ToQuantizedTensor(LoadTensorByName(wf, tensor_map, prefix + "w_gate"));
            lw.w_up = ToQuantizedTensor(LoadTensorByName(wf, tensor_map, prefix + "w_up"));
            lw.w_down = ToQuantizedTensor(LoadTensorByName(wf, tensor_map, prefix + "w_down"));

        }

        model.final_norm = LoadTensorByName(wf, tensor_map, "final_norm");
        model.lm_head = LoadTensorByName(wf, tensor_map, "lm_head");


    } catch (const std::exception& e) {
        return LoadFailure(e.what());
    }



    // 移动文件指针到末尾，获取文件大小
    wf.seekg(0, std::ios::end);
    std::streamoff file_size = wf.tellg();
    if (file_size < 0) {
        return LoadFailure("failed to determine weights file size");
    }

    // 遍历json中的所有tensor，算出文件最大应该延申到哪个字节(最大的那个tensor)
    size_t expected_end = 0;
    for (const auto& info : manifest.tensors) {
        size_t tensor_end = info.byte_size + info.offset;
        if (tensor_end > expected_end) {
            expected_end = tensor_end;
        }
    }

    if (expected_end != static_cast<size_t>(file_size)) {
        if (static_cast<size_t>(file_size) > expected_end) {
            return LoadFailure("weights file has trailing bytes: file size=" + std::to_string(file_size) + ", expected=" + std::to_string(expected_end));
        } else {
            return LoadFailure("weights file is too short: file size=" + std::to_string(file_size) + ", expected=" + std::to_string(expected_end));
        }
    } 

    model.loaded = true;
    std::cout << "Model loaded successfully from " << weights_path << std::endl;
    return model;
}


// 打印metadata信息
bool InspectModel(const std::string& config_path) {
    ModelManifest manifest;
    try {
        manifest = ParseManifest(config_path);
    } catch (const std::exception& e) {
        std::cerr << "Failed to inspect model: " << e.what() << std::endl;
        return false;
    }

    const ModelConfig& c = manifest.config;
    std::cout << "=== Model Config ===" << std::endl;
    std::cout << "  vocab_size:    " << c.vocab_size << std::endl;
    std::cout << "  dim:           " << c.dim << std::endl;
    std::cout << "  hidden_dim:    " << c.hidden_dim << std::endl;
    std::cout << "  n_layers:      " << c.n_layers << std::endl;
    std::cout << "  n_heads:       " << c.n_heads << std::endl;
    std::cout << "  n_kv_heads:    " << c.n_kv_heads << std::endl;
    std::cout << "  head_dim:      " << c.head_dim << std::endl;
    std::cout << "  max_seq_len:   " << c.max_seq_len << std::endl;
    std::cout << "  rope_theta:    " << c.rope_theta << std::endl;
    std::cout << "  rms_norm_eps:  " << c.rms_norm_eps << std::endl;

    std::cout << std::endl;
    std::cout << "=== Tensors (" << manifest.tensors.size() << ") ===" << std::endl;

    size_t total_bytes = 0;
    for (const auto& t : manifest.tensors) {
        std::cout << "  " << t.name << std::endl;
        std::cout << "    shape:     [";
        for (size_t i = 0; i < t.shape.size(); ++i) {
            if (i > 0) {
                std::cout << ", ";
            }
            std::cout << t.shape[i];
        }
        std::cout << "]" << std::endl;
        std::cout << "    dtype:     " << t.dtype << std::endl;
        std::cout << "    offset:    " << t.offset << std::endl;
        std::cout << "    byte_size: " << t.byte_size << std::endl;
        total_bytes += t.byte_size;
    }

    std::cout << std::endl;
    std::cout << "=== Tokenizer ===" << std::endl;
    std::cout << "  type: " << manifest.tokenizer.type << std::endl;
    std::cout << "  path: " << (manifest.tokenizer.path.empty() ? "<none>" : manifest.tokenizer.path) << std::endl;
    std::cout << "  bos_id: " << manifest.tokenizer.bos_id << std::endl;
    std::cout << "  eos_id: " << manifest.tokenizer.eos_id << std::endl;
    std::cout << "  unk_id: " << manifest.tokenizer.unk_id << std::endl;
    std::cout << std::endl;
    std::cout << "Total tensor bytes: " << total_bytes << std::endl;
    return true;
}

}