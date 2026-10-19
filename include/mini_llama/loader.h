#pragma once

#include "mini_llama/model.h"

#include <string>
#include <vector>

namespace mini_llama {


// 告诉加载器，每个张量长什么样
struct TensorInfo {
    std::string name;
    std::vector<int> shape;
    std::string dtype;
    size_t offset = 0;
    size_t byte_size = 0;
};


// 分词器配置
struct TokenizerInfo {
    std::string type = "ascii";
    std::string path;
    int bos_id = 1;
    int eos_id = 2;
    int unk_id = 0;
};


// model反序列化后的产物
struct ModelManifest {
    ModelConfig config;
    TokenizerInfo tokenizer;
    std::vector<TensorInfo> tensors;
};

// 解析配置文件
ModelManifest ParseManifest(const std::string& config_path);


// 加载真正的模型
MiniLlamaModel LoadModel(const std::string& config_path, const std::string& weights_path);


// 打印metadata信息
bool InspectModel(const std::string& config_path);

}