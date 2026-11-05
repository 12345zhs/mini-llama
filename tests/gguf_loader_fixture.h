#ifndef TESTS_GGUF_LOADER_FIXTURE_H_
#define TESTS_GGUF_LOADER_FIXTURE_H_

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/gguf.h"
#include "mini_llama/loader.h"
#include "mini_llama/model.h"
#include "tests/test_names.h"

namespace mini_llama::test {

struct GgufTensorFixture {
  std::string gguf_name;
  std::vector<int> shape;
  const float* data = nullptr;
  size_t data_size = 0;
  uint64_t offset = 0;
};

inline std::filesystem::path GgufLoaderTempPath(const std::string& name) {
  return std::filesystem::temp_directory_path() / ("mini_llama_" + name);
}

inline void AppendU32Le(std::vector<char>& data, uint32_t value) {
  data.push_back(static_cast<char>(value & 0xff));
  data.push_back(static_cast<char>((value >> 8) & 0xff));
  data.push_back(static_cast<char>((value >> 16) & 0xff));
  data.push_back(static_cast<char>((value >> 24) & 0xff));
}

inline void AppendU64Le(std::vector<char>& data, uint64_t value) {
  for (size_t i = 0; i < 8; ++i) {
    data.push_back(static_cast<char>((value >> (i * 8)) & 0xff));
  }
}

inline void AppendF32Le(std::vector<char>& data, float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  AppendU32Le(data, bits);
}

inline void AppendString(std::vector<char>& data, const std::string& value) {
  AppendU64Le(data, value.size());
  data.insert(data.end(), value.begin(), value.end());
}

inline void AppendMetadataU32(std::vector<char>& data, const std::string& key,
                              uint32_t value) {
  AppendString(data, key);
  AppendU32Le(data, kGgufTypeUint32);
  AppendU32Le(data, value);
}

inline void AppendMetadataF32(std::vector<char>& data, const std::string& key,
                              float value) {
  AppendString(data, key);
  AppendU32Le(data, kGgufTypeFloat32);
  AppendF32Le(data, value);
}

inline void AppendMetadataString(std::vector<char>& data,
                                 const std::string& key,
                                 const std::string& value) {
  AppendString(data, key);
  AppendU32Le(data, kGgufTypeString);
  AppendString(data, value);
}

inline void AppendMetadataTokens(std::vector<char>& data, int vocab_size) {
  AppendString(data, "tokenizer.ggml.tokens");
  AppendU32Le(data, kGgufTypeArray);
  AppendU32Le(data, kGgufTypeString);
  AppendU64Le(data, static_cast<uint64_t>(vocab_size));
  for (int i = 0; i < vocab_size; ++i) {
    AppendString(data, "tok_" + std::to_string(i));
  }
}

inline std::vector<uint64_t> GgufShapeForFixture(
    const GgufTensorFixture& fixture) {
  if (fixture.shape.size() == 2) {
    return {
        static_cast<uint64_t>(fixture.shape[1]),
        static_cast<uint64_t>(fixture.shape[0]),
    };
  }

  std::vector<uint64_t> shape;
  shape.reserve(fixture.shape.size());
  for (int dim : fixture.shape) {
    shape.push_back(static_cast<uint64_t>(dim));
  }
  return shape;
}

inline void AppendTensorInfo(std::vector<char>& data,
                             const GgufTensorFixture& fixture) {
  AppendString(data, fixture.gguf_name);
  std::vector<uint64_t> shape = GgufShapeForFixture(fixture);
  AppendU32Le(data, static_cast<uint32_t>(shape.size()));
  for (uint64_t dim : shape) {
    AppendU64Le(data, dim);
  }
  AppendU32Le(data, kGgmlTypeF32);
  AppendU64Le(data, fixture.offset);
}

inline void AppendTensorBytes(std::vector<char>& data,
                              const GgufTensorFixture& fixture) {
  const char* bytes = reinterpret_cast<const char*>(fixture.data);
  size_t n_bytes = fixture.data_size * sizeof(float);
  data.insert(data.end(), bytes, bytes + n_bytes);
}

inline void AddLayerTensors(std::vector<GgufTensorFixture>& tensors,
                            const MiniLlamaModel& model, int layer) {
  const LayerWeights& lw = model.layers[layer];
  const std::string prefix = "blk." + std::to_string(layer) + ".";
  tensors.push_back({prefix + "attn_norm.weight", lw.attention_norm.shape,
                     lw.attention_norm.data.data(),
                     lw.attention_norm.data.size(), 0});
  tensors.push_back({prefix + "attn_q.weight", lw.wq.shape,
                     lw.wq.f32_data.data(), lw.wq.f32_data.size(), 0});
  tensors.push_back({prefix + "attn_k.weight", lw.wk.shape,
                     lw.wk.f32_data.data(), lw.wk.f32_data.size(), 0});
  tensors.push_back({prefix + "attn_v.weight", lw.wv.shape,
                     lw.wv.f32_data.data(), lw.wv.f32_data.size(), 0});
  tensors.push_back({prefix + "attn_output.weight", lw.wo.shape,
                     lw.wo.f32_data.data(), lw.wo.f32_data.size(), 0});
  tensors.push_back({prefix + "ffn_norm.weight", lw.ffn_norm.shape,
                     lw.ffn_norm.data.data(), lw.ffn_norm.data.size(), 0});
  tensors.push_back({prefix + "ffn_gate.weight", lw.w_gate.shape,
                     lw.w_gate.f32_data.data(), lw.w_gate.f32_data.size(), 0});
  tensors.push_back({prefix + "ffn_up.weight", lw.w_up.shape,
                     lw.w_up.f32_data.data(), lw.w_up.f32_data.size(), 0});
  tensors.push_back({prefix + "ffn_down.weight", lw.w_down.shape,
                     lw.w_down.f32_data.data(), lw.w_down.f32_data.size(), 0});
}

inline std::filesystem::path WriteTinyLlamaGgufFixture(
    const std::string& name) {
  MiniLlamaModel source =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (!source.loaded) {
    throw std::runtime_error("failed to load tiny source model: " +
                             source.load_error);
  }

  std::vector<GgufTensorFixture> tensors;
  tensors.push_back({"token_embd.weight", source.token_embedding.shape,
                     source.token_embedding.data.data(),
                     source.token_embedding.data.size(), 0});
  for (int layer = 0; layer < source.config.n_layers; ++layer) {
    AddLayerTensors(tensors, source, layer);
  }
  tensors.push_back({"output_norm.weight", source.final_norm.shape,
                     source.final_norm.data.data(),
                     source.final_norm.data.size(), 0});
  tensors.push_back({"output.weight", source.lm_head.shape,
                     source.lm_head.f32_data.data(),
                     source.lm_head.f32_data.size(), 0});

  uint64_t offset = 0;
  for (GgufTensorFixture& tensor : tensors) {
    tensor.offset = offset;
    offset += static_cast<uint64_t>(tensor.data_size * sizeof(float));
  }

  constexpr uint64_t kAlignment = 32;
  constexpr uint64_t kMetadataCount = 12;

  std::vector<char> data;
  data.insert(data.end(), {'G', 'G', 'U', 'F'});
  AppendU32Le(data, 3);
  AppendU64Le(data, static_cast<uint64_t>(tensors.size()));
  AppendU64Le(data, kMetadataCount);

  AppendMetadataString(data, "general.architecture", "llama");
  AppendMetadataString(data, "general.name", "mini-llama-tiny-loader-test");
  AppendMetadataU32(data, "general.alignment",
                    static_cast<uint32_t>(kAlignment));
  AppendMetadataU32(data, "llama.block_count",
                    static_cast<uint32_t>(source.config.n_layers));
  AppendMetadataU32(data, "llama.context_length",
                    static_cast<uint32_t>(source.config.max_seq_len));
  AppendMetadataU32(data, "llama.embedding_length",
                    static_cast<uint32_t>(source.config.dim));
  AppendMetadataU32(data, "llama.feed_forward_length",
                    static_cast<uint32_t>(source.config.hidden_dim));
  AppendMetadataU32(data, "llama.attention.head_count",
                    static_cast<uint32_t>(source.config.n_heads));
  AppendMetadataU32(data, "llama.attention.head_count_kv",
                    static_cast<uint32_t>(source.config.n_kv_heads));
  AppendMetadataF32(data, "llama.rope.freq_base", source.config.rope_theta);
  AppendMetadataF32(data, "llama.attention.layer_norm_rms_epsilon",
                    source.config.rms_norm_eps);
  AppendMetadataTokens(data, source.config.vocab_size);

  for (const GgufTensorFixture& tensor : tensors) {
    AppendTensorInfo(data, tensor);
  }

  size_t padding = static_cast<size_t>(
      (kAlignment - (data.size() % kAlignment)) % kAlignment);
  data.insert(data.end(), padding, '\0');
  for (const GgufTensorFixture& tensor : tensors) {
    AppendTensorBytes(data, tensor);
  }

  std::filesystem::path path = GgufLoaderTempPath(name);
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    throw std::runtime_error("failed to create GGUF fixture: " + path.string());
  }
  out.write(data.data(), static_cast<std::streamsize>(data.size()));
  return path;
}

}  // namespace mini_llama::test

#endif  // TESTS_GGUF_LOADER_FIXTURE_H_
