#include "mini_llama/gguf_loader.h"
#include "mini_llama/gguf.h"


#include <string>
#include <iostream>
#include <map>
#include <cstring>

namespace mini_llama {


// 从gguf文件中配置model.config
static ModelConfig BuildConfig(const std::string& path) {
    ModelConfig config;

    // 匿名函数，从gguf中找指定整数key
    auto get_u32 = [&](const std::string& key, int fallback) -> int {
        int64_t v = 0;
        if (GgufGetMetadataInt(path, key, v)) {
            return static_cast<int>(v);
        }
        return fallback;
    };

    // 匿名函数，从gguf中找指定key对应的float
    auto get_f32 = [&](const std::string& key, float fallback) -> float {
        double v = 0.0;
        if (GgufGetMetadataFloat(path, key, v)) {
            return static_cast<float>(v);
        }
        return fallback;
    };

    // 判断结构，是否是qwen2.并采用NeoX旋转位置编码
    std::string arch;
    if (GgufGetMetadataString(path, "general.architecture", arch) && arch == "qwen2") {
        config.rope_type = RopeType::kNeoX;
    }

    // 匹配qwen2 --->  匹配 llama -----> 默认值
    // 配置层数
    config.n_layers = get_u32("qwen2.block_count", get_u32("llama.block_count", 0));
    // 配置隐藏层维数
    config.dim = get_u32("qwen2.emdedding_length", get_u32("llama.embedding_length", 0));
    // FFN中间层维度
    config.hidden_dim = get_u32("qwen2.feed_forward_length", get_u32("llama.feed_forward_length", 0));
    // Q头数
    config.n_heads = get_u32("qwen2.attention.head_count", get_u32("llama.attention.head_count", 0));
    // KV头数
    config.n_kv_heads = get_u32("qwen2.attention.head_count_kv", get_u32("llama.attention.head_count_kv", 0));
    // 最大上下文长度
    config.max_seq_len = get_u32("qwen2.context_length", get_u32("llama.context_length", 0));
    // rope的参数
    config.rope_theta = get_u32("qwen2.rope.freq_base", get_u32("llama.rope.freq_base", 0));
    // RMSNorm的参数
    config.rms_norm_eps = get_u32("qwen2.attention.layer_norm_rms_epsilon", get_u32("llama.attention.layer_norm_rms_epsilon", 0));


    // 获取此表大小
    // gguf不存”词表大小“的这个数字，gguf存的是整个词表数组
    std::vector<std::string> tokens;
    if (GgufGetMetadataStringArray(path, "tokenizer.ggml.tokens", tokens)) {
        config.vocab_size = static_cast<int>(tokens.size());
    }

    // 合法性验证
    if (config.n_layers <= 0 || config.dim <= 0 || config.hidden_dim <= 0 || config.n_heads <= 0 || config.n_kv_heads <= 0 || config.max_seq_len <= 0 || config.vocab_size <= 0) {
        throw std::runtime_error("Invalid model config from GGUF: dim=" + std::to_string(config.dim) + " layers=" + std::to_string(config.n_layers) + " heads=" + std::to_string(config.n_heads) + " vocab=" + std::to_string(config.vocab_size));
    }
    if (config.dim % config.n_heads != 0) {
        throw std::runtime_error("dim not divisible by n_heads");
    }
    if (config.n_heads % config.n_kv_heads != 0) {
        throw std::runtime_error("n_heads not divisible by n_kv_heads");
    }

    config.head_dim = config.dim / config.n_heads;        // 头的维度

    std::cout << "GGUF config: vocab=" << config.vocab_size
            << " dim=" << config.dim << " layers=" << config.n_layers
            << " heads=" << config.n_heads << " kv_heads=" << config.n_kv_heads
            << " head_dim=" << config.head_dim
            << " hidden_dim=" << config.hidden_dim
            << " max_seq_len=" << config.max_seq_len
            << " rope_theta=" << config.rope_theta
            << " rms_norm_eps=" << config.rms_norm_eps << std::endl;

    return config;
}

// 把gguf的向量名字映射为代码内部的tensor 名字
static std::string MapTensorName(const std::string& gguf_name) {
    // embedding
    if (gguf_name == "token_embd.weight") {
        return "token_embedding";
    }
    // output_norm
    if (gguf_name == "output_norm.weight") {
        return "final_norm";
    }   
    // lm_head
    if (gguf_name == "output.weight") {
        return "lm_head";
    }

    // blk
    //  blk  .  0  .  attn_q  .  weight
    if (gguf_name.rfind("blk.", 0) == 0) {
        size_t dot1 = gguf_name.find('.', 4);   // 找blk.后面的第一个.
        if (dot1 == std::string::npos) {
            return "";
        }

        // 层号
        std::string layer_str = gguf_name.substr(4, dot1 - 4);
        size_t dot2 = gguf_name.find('.', dot1 + 1);         // 找第二个.
        if (dot2 == std::string::npos) {
            return "";
        }
        // 提取操作名
        std::string op = gguf_name.substr(dot1 + 1, dot2 - dot1 -1);        
        std::string suffix = gguf_name.substr(dot2 + 1);
        std::string mapped_op;
        
        // 处理bias
        if (suffix == "bias") {
            if (op == "attn_q") {
                mapped_op = "bq";
            } else if (op == "attn_k") {
                mapped_op = "bk";
            } else if (op == "attn_v") {
                mapped_op = "bv";
            } else {
                return "";
            }
        } else if (suffix != "weight") {
            return "";
        } else if (op == "attn_norm") {
            mapped_op = "attention_norm";
        } else if (op == "attn_q") {
            mapped_op = "wq";
        } else if (op == "attn_k") {
            mapped_op = "wk";
        } else if (op == "attn_v") {
            mapped_op = "wv";
        } else if (op == "attn_output") {
            mapped_op = "wo"; 
        } else if (op == "ffn_norm") {
            mapped_op = "ffn_norm";
        } else if (op == "ffn_gate") {
            mapped_op = "w_gate";
        } else if (op == "ffn_up") {
            mapped_op = "w_up";
        } else if (op == "ffn_down") {
            mapped_op = "w_down";
        } else {
            return "";
        }

        return "layers." + layer_str + "." + mapped_op;
    }
    return "";
}

// tensor形状行列反转
static std::vector<int> MapShape(const std::vector<uint64_t>& gguf_shape) {
    std::vector<int> shape;
    shape.reserve(gguf_shape.size());

    if (gguf_shape.size() == 2) {
        shape.push_back(static_cast<int>(gguf_shape[1]));
        shape.push_back(static_cast<int>(gguf_shape[0]));
        return shape;
    }
    for (uint64_t dim : gguf_shape) {
        shape.push_back(static_cast<int>(dim));
    }
    return shape;
}


static QuantizedTensor LoadQuantizedTensor(const std::vector<uint8_t>& raw, const std::vector<int>& shape, int ggml_type) {
    // 计算tensor的总数量
    size_t num_elements = 1;
    for (int d : shape) {
        num_elements *= shape[d];
    }

    QuantizedTensor q;
    q.shape = shape;

    switch (ggml_type) {
        case kGgmlTypeF32: {
            q.type = QuantType::kF32;
            size_t expected_bytes = num_elements * sizeof(float);
            if (expected_bytes != raw.size()) {
                throw std::runtime_error("F32 tensor size mismatch: expected " + std::to_string(expected_bytes) + " bytes, got " + std::to_string(raw.size()));
            }

            q.f32_data.resize(num_elements);
            std::memcpy(q.f32_data.data(), raw.data(), expected_bytes);
            break;
        }
        case kGgmlTypeQ80: {
            q.type = QuantType::kQ80;
            size_t n_blocks = (num_elements + kQ80BlockSize - 1) / kQ80BlockSize;
            size_t expected_bytes = n_blocks * sizeof(BlockQ80);
            if (expected_bytes != raw.size()) {
                throw std::runtime_error("Q8_0 tensor size mismatch: expected " + std::to_string(expected_bytes) + " bytes, got " + std::to_string(raw.size()));
            }
            q.q8_0_data.resize(n_blocks);
            std::memcpy(q.q8_0_data.data(), raw.data(), expected_bytes);
            break;
        }
        case kGgmlTypeQ40: {
            q.type = QuantType::kQ40;
            size_t n_blocks = (num_elements + kQ40BlockSize - 1) / kQ40BlockSize;
            size_t expected_bytes = n_blocks * sizeof(BlockQ40);
            if (expected_bytes != raw.size()) {
                throw std::runtime_error("Q4_0 tensor size mismatch: expected " + std::to_string(expected_bytes) + " bytes, got " + std::to_string(raw.size()));
            }
            q.q4_0_data.resize(n_blocks);
            std::memcpy(q.q4_0_data.data(), raw.data(), expected_bytes);
            break;
        }
        case kGgmlTypeQ41: {
            q.type = QuantType::kQ41;
            size_t n_blocks = (num_elements + kQ41BlockSize - 1) / kQ41BlockSize;
            size_t expected_bytes = n_blocks * sizeof(BlockQ41);
            if (expected_bytes != raw.size()) {
                throw std::runtime_error("Q4_1 tensor size mismatch: expected " + std::to_string(expected_bytes) + " bytes, got " + std::to_string(raw.size()));
            }
            q.q4_1_data.resize(n_blocks);
            std::memcpy(q.q4_1_data.data(), raw.data(), expected_bytes);
            break;
        }
        default : {
            throw std::runtime_error("Unsupported GGUF tensor type: " + std::to_string(ggml_type));
        }
    }
    return q;
}



static void RequireShape(const Tensor& tensor, const std::vector<int>& expected, const std::string& name) {
    tensor.AssertShape(expected, name.c_str());
}


static void RequireShape(const QuantizedTensor& tensor, const std::vector<int>& expected, const std::string& name) {
    tensor.AssertShape(expected, name.c_str());
}


static void RequireOptionalShape(const QuantizedTensor& tensor, const std::vector<int>& expected, const std::string& name) {
    if (tensor.data.empty()) {
        return;
    }
    RequireShape(tensor, expected, name);
}

static void ValidateLoadedShapes(const MiniLlamaModel& model) {
    const ModelConfig& c = model.config;
    const int q_dim = c.n_heads * c.head_dim;
    const int kv_dim = c.n_kv_heads * c.head_dim;

    RequireShape(model.token_embedding, {c.vocab_size, c.dim}, "GGUF token_embedding");
    RequireShape(model.final_norm, {c.vocab_size, c.dim}, "GGUF final_norm");
    RequireShape(model.lm_head, {c.vocab_size, c.dim}, "GGUF lm_head");

    if (model.layers.size() != static_cast<int>(c.n_layers)) {
        throw std::runtime_error("GGUF layer count mismatch: expected " + std::to_string(c.n_layers) + ", got " + std::to_string(model.layers.size()));
    }

    for (int layer = 0; layer < c.n_layers; ++layer) {
        const LayerWeights& lw = model.layers[layer];
        const std::string prefix = "GGUF layers." + std::to_string(layer) + ".";
        RequireShape(lw.attention_norm, {c.dim}, prefix + "attention_norm");
        RequireShape(lw.wq, {q_dim, c.dim}, prefix + "wq");
        RequireShape(lw.wk, {kv_dim, c.dim}, prefix + "wk");
        RequireShape(lw.wv, {kv_dim ,c.dim}, prefix + "wv");
        RequireOptionalShape(lw.bq, {q_dim}, prefix + "bq");
        RequireOptionalShape(lw.bk, {kv_dim}, prefix + "bk");
        RequireOptionalShape(lw.bv, {kv_dim}, prefix + "bv");
        RequireShape(lw.wo, {c.dim, q_dim}, prefix + "wo");
        RequireShape(lw.ffn_norm, {c.dim}, prefix + "ffn_norm");
        RequireShape(lw.w_gate, {c.hidden_dim, c.dim}, prefix + "w_gate");
        RequireShape(lw.w_up, {c.hidden_dim, c.dim}, prefix + "w_up");
        RequireShape(lw.w_down, {c.dim, c.hidden_dim}, prefix + "w_down");
    }
}


MiniLlamaModel LoadGgufModel(const std::string& gguf_path) {
    MiniLlamaModel model;
    GgufReader reader;
    if (!reader.Load(gguf_path)) {
        model.load_error = "GGUF parse error: " + reader.load_error;
        return model;
    }

    // 建立config
    try {
        model.config = BuildConfig(gguf_path);
    } catch(const std::exception& e) {
        model.load_error = std::string("Failed to build config: ") + e.what();
        return model;
    }

    const ModelConfig& c = model.config;
    std::cout << "GGUF config: vocab=" << c.vocab_size << " dim=" << c.dim
            << " layers=" << c.n_layers << " heads=" << c.n_heads
            << " kv_heads=" << c.n_kv_heads << " head_dim=" << c.head_dim
            << " hidden_dim=" << c.hidden_dim
            << " max_seq_len=" << c.max_seq_len << std::endl;


    // tensor字典
    std::map<std::string, const GgufTensorInfo*> tensor_map;
    for (const auto& t : reader.tensors) {
        std::string mapped = MapTensorName(t.name);
        if (mapped.empty()) {
            continue;
        }
        tensor_map[mapped] = &t;
    }

    // 匿名函数，用来加载tensor的权重
    auto load_weight = [&](const std::string& mapped_name) -> QuantizedTensor {
        auto it = tensor_map.find(mapped_name);
        if (it == tensor_map.end()) {
            throw std::runtime_error("Missing tensor: " + mapped_name);
        }
        const GgufTensorInfo* info = it->second;
        std::vector<int> shape = MapShape(info->shape);  // ”转置“

        // 读取info的原始张量内容
        std::vector<uint8_t> raw = ReadGgufTensorRaw(gguf_path, *info, reader.data_offset);
        // 数据不能空，且数据类型可以量化(可能存在元素个数为 0 的 F32 张量)
        if (raw.empty() && info->type != kGgmlTypeF32) {
            throw std::runtime_error("Failed to read tensor: " + mapped_name);
        }
        return LoadQuantizedTensor(raw, shape, info->type);
    };


    // 加载可选的权重
    auto load_optional_weight = [&](const std::string& mapped_name) -> QuantizedTensor {
        if (tensor_map.find(mapped_name) == tensor_map.end()) {
            return QuantizedTensor();    // 返回空
        }
        return load_weight(mapped_name);
    };

    // 加载浮点tensor
    auto load_f32 = [&](const std::string& mapped_name) -> Tensor {
        QuantizedTensor q = load_weight(mapped_name);

        switch (q.type) {
            case QuantType::kF32: {
                return ToTensor(q);
            }
            case QuantType::kQ80: {
                return DequantizeFromQ80(q.q8_0_data, q.shape);
            }

            case QuantType::kQ40: {
                return DequantizeFromQ40(q.q4_0_data, q.shape);
            }
            case QuantType::kQ41: {
                return DequantizeFromQ41(q.q4_1_data, q.shape);
            }
        }
        throw std::runtime_error("Cannot convert tensor to F32: " + mapped_name);
    };


    auto load_optional_f32 = [&](const std::string& mapped_name) -> Tensor {
        if (tensor_map.find(mapped_name) == tensor_map.end()) {
            return Tensor();
        }
        return load_f32(mapped_name);
    };


    try {
        model.token_embedding = load_f32("token_emdedding");
        model.final_norm = load_f32("final_norm");
        model.layers.resize(c.n_layers);
        for (int layer = 0; layer < c.n_layers; ++layer) {
            LayerWeights& lw = model.layers[layer];
            const std::string prefix = "layers." + std::to_string(layer) + ".";
            lw.attention_norm = load_f32(prefix + "attention_norm");
            lw.wq = load_weight(prefix + "wq");
            lw.wk = load_weight(prefix + "wk");
            lw.wv = load_weight(prefix + "wv");

            lw.bq = load_optional_f32(prefix + "bq");
            lw.bk = load_optional_f32(prefix + "bk");
            lw.bv = load_optional_f32(prefix + "bv");
            lw.wo = load_weight(prefix + "wo");
            lw.ffn_norm = load_f32(prefix + "ffn_norm");
            lw.w_gate = load_weight(prefix + "w_up");
            lw.w_down = load_weight(prefix + "w_dowm");
        }

        // 有的模型没有lm_head,就用token_embedding
        auto it = tensor_map.find("lm_head");
        if (it != tensor_map.end()) {
            model.lm_haed = load_weight("lm_weight");
        } else {
            model.lm_haed = ToQuantizedTensor(model.token_embedding);
        }
        
        ValidateLoadedShapes(model);
    } catch (const std::exception& e) {
        model.load_error = std::string("Failed to load GGUF weights: ") + e.what();
        return model;
    }

    model.loaded = true;
    std::cout << "GGUF model loaded successfully from " << gguf_path << std::endl;
    return model;
}

}