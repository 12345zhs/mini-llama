#include "mini_llama/model.h"
#include "mini_llama/quantized_tensor.h"
#include "mini_llama/quant.h"

#include <utility>

namespace mini_llama {


// cpu的tensor的字节数
static size_t TensorByts(const Tensor& t) {
    return t.num_elements() * sizeof(float);
}


// QuantTensor的字节数
static size_t QuantizedTensorBytes(const QuantizedTensor& qt) {
    switch (qt.type) {
        case QuantType::kF32 : {
            return qt.f32_data.size() * sizeof(float);
        }
        case QuantType::kQ40 : {
            return qt.q4_0_data.size() * sizeof(BlockQ40);
        }
        case QuantType::kQ41 : {
            return qt.q4_1_data.size() * sizeof(BlockQ41);
        }
        case QuantType::kQ80 : {
            return qt.q8_0_data.size() * sizeof(BlockQ80);
        }
    }
    return 0;
}

// 量化还原为F32后的字节数
static size_t QuantizedTensorBytesF32(const QuantizedTensor& qt) {
    return qt.num_elements() * sizeof(float);
}



// 把未量化的权重上传给cuda（linear）
static void UploadF32LinearWeight(CudaModelWeights& storage, const QuantizedTensor& weight, const std::string& name, int device_id) {
    if (weight.type != QuantType::kF32) {
        return;
    }
    if (weight.f32_data.size() == 0) {
        throw std::runtime_error("CUDA weight upload: empty F32 data for " + name);
    }
    const size_t expected_numel = weight.num_elements();
    if (expected_numel != weight.f32_data.size()) {
        throw std::runtime_error("CUDA weight upload: data size mismatch for " + name + ", expected " + std::to_string(expected_numel) + ", got " + std::to_string(weight.f32_data.size()));
    }

    const size_t bytes = weight.f32_data.size() * sizeof(float);
    CudaUploadedWeight uploaded;
    uploaded.name = name;
    uploaded.type = QuantType::kF32;
    uploaded.linear = true;
    uploaded.shape = weight.shape;
    uploaded.block_count = 0;
    uploaded.buffer.Reset(bytes, device_id);
    uploaded.buffer.Upload(weight.f32_data.data(), bytes);

    storage.uploaded_weight_count += 1;
    storage.uploaded_bytes += bytes;
    storage.weights.push_back(std::move(uploaded));
}


// 给非线性层权重传给cuda（归一层）
static void UploadF32TensorWeight(CudaModelWeights& storage, const Tensor& weight, const std::string& name, int device_id) {
    if (weight.data.empty()) {
        return;
    }
    const size_t expected_numel = weight.size();
    if (weight.data.size() != expected_numel) {
        throw std::runtime_error("CUDA weight upload: tensor data size mismatch for " + name + ", expected " + std::to_string(expected_numel) + ", got " + std::to_string(weight.data.size()));
    }

    const size_t bytes = weight.data.size() * sizeof(float);
    CudaUploadedWeight uploaded;
    uploaded.name = name;
    uploaded.type = QuantType::kF32;
    uploaded.linear = false;
    uploaded.block_count = 0;
    uploaded.shape = weight.shape;
    uploaded.buffer.Reset(bytes, device_id);
    uploaded.buffer.Upload(weight.data.data(), bytes);

    storage.uploaded_weight_count += 1;
    storage.uploaded_bytes += bytes;
    storage.weights.push_back(std::move(uploaded));

}


// 把Q80量化的权重传给cuda
static void UploadQ80LinearWeight(CudaModelWeights& storage, const QuantizedTensor& weight, const std::string& name, int device_id) {
    if (weight.type != QuantType::kQ80) {
        return;
    }
    if (weight.shape.size() != 2) {
        throw std::runtime_error("CUDA weight upload: expected 2D Q8_0 weight for " + name);
    }
    if (weight.q8_0_data.empty()) {
        throw std::runtime_error("CUDA weight upload: empty Q8_0 data for " + name);
    }

    const int out_features = weight.shape[0];
    const int in_features = weight.shape[1];
    const int blocks_per_row = (in_features + kQ80BlockSize - 1) / kQ80BlockSize;
    const size_t expected_blocks = static_cast<size_t>(out_features) * blocks_per_row;
    if (weight.q8_0_data.size() != expected_blocks) {
        throw std::runtime_error("CUDA weight upload: Q8_0 block count mismatch for " + name + ", expected " + std::to_string(expected_blocks) + ", got " + std::to_string(weight.q8_0_data.size()));
    }

    const size_t bytes = weight.q8_0_data.size() * sizeof(BlockQ80);
    CudaUploadedWeight uploaded;
    uploaded.name = name;
    uploaded.type = QuantType::kQ80;
    uploaded.linear = true;
    uploaded.shape = weight.shape;
    uploaded.block_count = weight.q8_0_data.size();
    uploaded.buffer.Reset(bytes, device_id);
    uploaded.buffer.Upload(weight.q8_0_data.data(), bytes);

    storage.uploaded_weight_count += 1;
    storage.uploaded_bytes += bytes;
    storage.weights.push_back(std::move(uploaded));
}



// 把Q40量化的权重传给cuda
static void UploadQ40LinearWeight(CudaModelWeights& storage, const QuantizedTensor& weight, const std::string& name, int device_id) {
    if (weight.type != QuantType::kQ40) {
        return;
    }
    if (weight.shape.size() != 2) {
        throw std::runtime_error("CUDA weight upload: expected 2D Q4_0 weight for " + name);
    }
    if (weight.q4_0_data.empty()) {
        throw std::runtime_error("CUDA weight upload: empty Q4_0 data for " + name);
    }

    const int out_features = weight.shape[0];
    const int in_features = weight.shape[1];
    const int blocks_per_row = (in_features + kQ40BlockSize - 1) / kQ40BlockSize;
    const size_t expected_blocks = static_cast<size_t>(out_features) * blocks_per_row;
    if (weight.q4_0_data.size() != expected_blocks) {
        throw std::runtime_error("CUDA weight upload: Q4_0 block count mismatch for " + name + ", expected " + std::to_string(expected_blocks) + ", got " + std::to_string(weight.q4_0_data.size()));
    }

    const size_t bytes = weight.q4_0_data.size() * sizeof(BlockQ40);
    CudaUploadedWeight uploaded;
    uploaded.name = name;
    uploaded.type = QuantType::kQ40;
    uploaded.linear = true;
    uploaded.shape = weight.shape;
    uploaded.block_count = weight.q4_0_data.size();
    uploaded.buffer.Reset(bytes, device_id);
    uploaded.buffer.Upload(weight.q4_0_data.data(), bytes);

    storage.uploaded_weight_count += 1;
    storage.uploaded_bytes += bytes;
    storage.weights.push_back(std::move(uploaded));
}




// 把Q41量化的权重传给cuda
static void UploadQ41LinearWeight(CudaModelWeights& storage, const QuantizedTensor& weight, const std::string& name, int device_id) {
    if (weight.type != QuantType::kQ41) {
        return;
    }
    if (weight.shape.size() != 2) {
        throw std::runtime_error("CUDA weight upload: expected 2D Q4_1 weight for " + name);
    }
    if (weight.q4_1_data.empty()) {
        throw std::runtime_error("CUDA weight upload: empty Q4_1 data for " + name);
    }

    const int out_features = weight.shape[0];
    const int in_features = weight.shape[1];
    const int blocks_per_row = (in_features + kQ41BlockSize - 1) / kQ41BlockSize;
    const size_t expected_blocks = static_cast<size_t>(out_features) * blocks_per_row;
    if (weight.q4_1_data.size() != expected_blocks) {
        throw std::runtime_error("CUDA weight upload: Q4_1 block count mismatch for " + name + ", expected " + std::to_string(expected_blocks) + ", got " + std::to_string(weight.q4_1_data.size()));
    }

    const size_t bytes = weight.q4_1_data.size() * sizeof(BlockQ41);
    CudaUploadedWeight uploaded;
    uploaded.name = name;
    uploaded.type = QuantType::kQ41;
    uploaded.linear = true;
    uploaded.shape = weight.shape;
    uploaded.block_count = weight.q4_1_data.size();
    uploaded.buffer.Reset(bytes, device_id);
    uploaded.buffer.Upload(weight.q4_1_data.data(), bytes);

    storage.uploaded_weight_count += 1;
    storage.uploaded_bytes += bytes;
    storage.weights.push_back(std::move(uploaded));
}



// 上传linear量化权重
static void UploadLinearWeight(CudaModelWeights& storage, const QuantizedTensor& weight, const std::string& name, int device_id) {
    UploadF32LinearWeight(storage, weight, name, device_id);
    UploadQ80LinearWeight(storage, weight, name, device_id);
    UploadQ40LinearWeight(storage, weight, name, device_id);
    UploadQ41LinearWeight(storage, weight, name, device_id);
}


// 有多少linear有多少特定量化的权重
static size_t ModelCudaUploadedWeightCountByType(const MiniLlamaModel& model, QuantType type) {
    if (!model.cuda_weights) {
        return 0;
    }
    size_t count = 0;
    for (const auto& weight : model.cuda_weights->weights) {
        if (weight.linear && weight.type == type) {
            count += 1;
        }
    }
    return count;
}


// QuantTensor转为Tensor
static Tensor QuantizedTensorToF32(const QuantizedTensor& qt) {
    switch(qt.type) {
        case QuantType::kF32 : {
            return ToTensor(qt);
        }
        case QuantType::kQ40 : {
            return DequantizeFromQ40(qt.q4_0_data, qt.shape);
        }
        case QuantType::kQ41 : {
            return DequantizeFromQ41(qt.q4_1_data, qt.shape);
        }
        case QuantType::kQ80 : {
            return DequantizeFromQ80(qt.q8_0_data, qt.shape);
        }
    }
    throw std::runtime_error("quantized_tensor_to_f32: unknown quant type");
}


// 强行把类型修改为Q80量化
static void QuantizeQtToQ80(QuantizedTensor& qt) {
    if (qt.type == QuantType::kQ80) {
        return;
    }
    Tensor t = QuantizedTensorToF32(qt);   // 先转化为fp32
    qt.q8_0_data = QuantizeToQ80(t);
    qt.type = QuantType::kQ80;

    qt.f32_data.clear();                // 清理元素
    qt.f32_data.shrink_to_fit();        // 释放内存
    qt.q4_1_data.clear();
    qt.q4_1_data.shrink_to_fit();
    qt.q4_0_data.clear();
    qt.q4_0_data.shrink_to_fit();
}



// 强行把类型修改为Q40量化
static void QuantizeQtToQ40(QuantizedTensor& qt) {
    if (qt.type == QuantType::kQ40) {
        return;
    }
    Tensor t = QuantizedTensorToF32(qt);   // 先转化为fp32
    qt.q4_0_data = QuantizeToQ40(t);
    qt.type = QuantType::kQ40;

    qt.f32_data.clear();                // 清理元素
    qt.f32_data.shrink_to_fit();        // 释放内存
    qt.q4_1_data.clear();
    qt.q4_1_data.shrink_to_fit();
    qt.q8_0_data.clear();
    qt.q8_0_data.shrink_to_fit();
}





// 整个模型权重的字节大小
size_t ModelWeightBytes(const MiniLlamaModel& model) {
    size_t bytes = 0;
    bytes += TensorByts(model.token_embedding);
    bytes += TensorByts(model.final_norm);
    bytes += QuantizedTensorBytes(model.lm_head);

    for (const auto& lw : model.layers) {
        bytes += TensorByts(lw.attention_norm);
        bytes += QuantizedTensorBytes(lw.wq);
        bytes += QuantizedTensorBytes(lw.wk);
        bytes += QuantizedTensorBytes(lw.wv);
        bytes += TensorByts(lw.bq);
        bytes += TensorByts(lw.bk);
        bytes += TensorByts(lw.bv);
        bytes += QuantizedTensorBytes(lw.wo);
        bytes += TensorByts(lw.ffn_norm);
        bytes += QuantizedTensorBytes(lw.w_gate);
        bytes += QuantizedTensorBytes(lw.w_up);
        bytes += QuantizedTensorBytes(lw.w_dowm);
    }
    return bytes;
}


// 将量化权重全部解压为F32会占多少字节
size_t ModelWeightBytesF32(const MiniLlamaModel& model) {
    size_t bytes = 0;
    bytes += TensorByts(model.token_embedding);
    bytes += TensorByts(model.final_norm);
    bytes += QuantizedTensorBytes(model.lm_head);

    for (const auto& lw : model.layers) {
        bytes += TensorByts(lw.attention_norm);
        bytes += QuantizedTensorBytesF32(lw.wq);
        bytes += QuantizedTensorBytesF32(lw.wk);
        bytes += QuantizedTensorBytesF32(lw.wv);
        bytes += TensorByts(lw.bq);
        bytes += TensorByts(lw.bk);
        bytes += TensorByts(lw.bv);
        bytes += QuantizedTensorBytesF32(lw.wo);
        bytes += TensorByts(lw.ffn_norm);
        bytes += QuantizedTensorBytesF32(lw.w_gate);
        bytes += QuantizedTensorBytesF32(lw.w_up);
        bytes += QuantizedTensorBytesF32(lw.w_dowm);
    }
    return bytes;
}


// 上传模型权重给cuda
void UploadModelWeightsToCuda(MiniLlamaModel& model, int device_id) {
    if (!model.loaded) {
        throw std::runtime_error("CUDA weight upload requires a loaded model");
    }
    CudaSetDevice(device_id);
    auto storage = std::make_shared<CudaModelWeights>();
    storage->device_id = device_id;

    UploadF32TensorWeight(*storage, model.token_embedding, "token_embedding", device_id);
    UploadF32TensorWeight(*storage, model.final_norm, "final_norm", device_id);
    UploadLinearWeight(*storage, model.lm_head, "lm_head", device_id);

    for (size_t layer = 0; layer < model.layers.size(); ++layer) {
        const LayerWeights& lw = model.layers[layer];
        const std::string prefix = "layers" + std::to_string(layer) + ".";
        UploadF32TensorWeight(*storage, lw.attention_norm,  prefix + "attention_norm", device_id);
        UploadLinearWeight(*storage, lw.wq, prefix + "wq", device_id);
        UploadLinearWeight(*storage, lw.wk, prefix + "wk", device_id);
        UploadLinearWeight(*storage, lw.wv, prefix + "wv", device_id);
        UploadF32TensorWeight(*storage, lw.bq,  prefix + "bq", device_id);
        UploadF32TensorWeight(*storage, lw.bk,  prefix + "bk", device_id);
        UploadF32TensorWeight(*storage, lw.bv,  prefix + "bv", device_id);
        UploadLinearWeight(*storage, lw.wo, prefix + "wo", device_id);
        UploadF32TensorWeight(*storage, lw.ffn_norm,  prefix + "ffn_norm", device_id);
        UploadLinearWeight(*storage, lw.w_gate, prefix + "w_gate", device_id);
        UploadLinearWeight(*storage, lw.w_up, prefix + "w_up", device_id);
        UploadLinearWeight(*storage, lw.w_dowm, prefix + "w_dowm", device_id);
    }
}


// 清空cuda权重
void ClearModelCudaWeights(MiniLlamaModel& model) {
    model.cuda_weights.reset();
}


// 查看是否有cuda权重
bool ModelHasCudaWeights(const MiniLlamaModel& model) {
    return model.cuda_weights != nullptr;
}


// 有多少cude权重
size_t ModelCudaUploadedWeightCount(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }

    return model.cuda_weights->uploaded_weight_count;
}


// 有多少个f32量化权重linear
size_t ModelCudaUploadedF32WeightCount(const MiniLlamaModel& model) {
    return ModelCudaUploadedWeightCountByType(model, QuantType::kF32);
}


// 有多少个Q80量化权重linear
size_t ModelCudaUploadedQ80WeightCount(const MiniLlamaModel& model) {
    return ModelCudaUploadedWeightCountByType(model, QuantType::kQ80);
}


// 有多少个Q40量化权重linear
size_t ModelCudaUploadedQ40WeightCount(const MiniLlamaModel& model) {
    return ModelCudaUploadedWeightCountByType(model, QuantType::kQ40);
}


// 有多少个Q41量化权重linear
size_t ModelCudaUploadedQ41WeightCount(const MiniLlamaModel& model) {
    return ModelCudaUploadedWeightCountByType(model, QuantType::kQ41);
}


// cuda权重多少个字节
size_t ModelCudaMemoryBytes(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->uploaded_bytes;
}


// 运行数据清零
void ResetModelCudaRuntimeStats(MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return;
    }
    model.cuda_weights->linear_calls = 0;
    model.cuda_weights->attention_calls = 0;
    model.cuda_weights->activation_calls = 0;
    model.cuda_weights->kv_cache_read_bytes = 0;
    model.cuda_weights->kv_cache_write_bytes = 0;
    model.cuda_weights->device_to_host_copies = 0;
    model.cuda_weights->devide_to_host_bytes = 0;
    model.cuda_weights->host_to_device_copies = 0;
    model.cuda_weights->host_to_devide_bytes = 0;
}


// cuda权重得linear调用了多少次
size_t ModelCudaLinearCalls(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->linear_calls;
}


// cuda权重的激活函数调用了多少次
size_t ModelCudaActivationCalls(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->activation_calls;
}


// cuda权重的记忆力计算调用了多少次
size_t ModelCudaAttentionCalls(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->attention_calls;
}


// cuda权重的注意力回退cpu了多少次
size_t ModelCudaAttentionCpuFallbacks(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->attention_cpu_fallbacks;
}


// kvCache写了多少字节
size_t ModelCudaKvCacheWriteBytes(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->kv_cache_write_bytes;
}


// kvCache读了多少字节
size_t ModelCudaKvCacheReadBytes(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->kv_cache_read_bytes;
}


// cpu向gpu传递了多少次数据
size_t ModelCudaHostToDeviceCopies(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->host_to_device_copies;
}

// gpu向cpu传递了多少次数据
size_t ModelCudaDeviceToHostCopies(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->device_to_host_copies;
}


// cpu向gpu传递了多少字节
size_t ModelCudaHostToDeviceBytes(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->host_to_devide_bytes;
}


// gpu向cpu传递了多少字节
size_t ModelCudaDeviceToHostBytes(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->devide_to_host_bytes;
}


// 把所有linear的权重压缩到Q80
void QuantizeModelToQ80(MiniLlamaModel& model) {
    QuantizeQtToQ80(model.lm_head);
    for (auto& lw : model.layers) {
        QuantizeQtToQ80(lw.wq);
        QuantizeQtToQ80(lw.wk);
        QuantizeQtToQ80(lw.wv);
        QuantizeQtToQ80(lw.wo);
        QuantizeQtToQ80(lw.w_gate);
        QuantizeQtToQ80(lw.w_up);
        QuantizeQtToQ80(lw.w_dowm);
    }
}


// 把所有linear的权重压缩到Q40
void QuantizeModelToQ40(MiniLlamaModel& model) {
    QuantizeQtToQ40(model.lm_head);
    for (auto& lw : model.layers) {
        QuantizeQtToQ40(lw.wq);
        QuantizeQtToQ40(lw.wk);
        QuantizeQtToQ40(lw.wv);
        QuantizeQtToQ40(lw.wo);
        QuantizeQtToQ40(lw.w_gate);
        QuantizeQtToQ40(lw.w_up);
        QuantizeQtToQ40(lw.w_dowm);
    }
}

}