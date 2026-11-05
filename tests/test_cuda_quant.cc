#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/batch.h"
#include "mini_llama/context.h"
#include "mini_llama/cuda_quant.h"
#include "mini_llama/cuda_runtime.h"
#include "mini_llama/forward.h"
#include "mini_llama/loader.h"
#include "mini_llama/model.h"
#include "mini_llama/quant.h"
#include "tests/test_names.h"
#include "tests/test_main.h"

namespace {

constexpr float kAbsTol = 1e-3f;
constexpr float kRelTol = 1e-3f;

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool CloseEnough(float actual, float expected) {
  float abs_err = std::abs(actual - expected);
  float scale = std::max(1.0f, std::abs(expected));
  return abs_err <= kAbsTol || abs_err / scale <= kRelTol;
}

void RequireCloseTensor(const Tensor& actual, const Tensor& expected,
                        const std::string& label) {
  Require(actual.shape == expected.shape, label + ": shape mismatch");
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!CloseEnough(actual.data[i], expected.data[i])) {
      throw std::runtime_error(
          label + ": value mismatch at " + std::to_string(i) +
          ", actual=" + std::to_string(actual.data[i]) +
          ", expected=" + std::to_string(expected.data[i]));
    }
  }
}

Tensor MakePatternTensor(const std::vector<int>& shape, float scale,
                         float shift) {
  Tensor t(shape, 0.0f);
  for (size_t i = 0; i < t.size(); ++i) {
    int bucket = static_cast<int>((i * 19 + 7) % 31);
    t.data[i] = static_cast<float>(bucket - 15) * scale + shift;
  }
  return t;
}

uint16_t FloatToFp16ForTest(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));

  uint32_t sign = (bits >> 16) & 0x8000u;
  int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
  uint32_t mantissa = bits & 0x7fffffu;

  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<uint16_t>(sign);
    }
    mantissa |= 0x800000u;
    uint32_t shifted = mantissa >> (1 - exponent);
    if ((shifted & 0x00001000u) != 0) {
      shifted += 0x00002000u;
    }
    return static_cast<uint16_t>(sign | (shifted >> 13));
  }

  if (exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u);
  }

  if ((mantissa & 0x00001000u) != 0) {
    mantissa += 0x00002000u;
    if ((mantissa & 0x00800000u) != 0) {
      mantissa = 0;
      ++exponent;
      if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
      }
    }
  }

  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
                               (mantissa >> 13));
}

float Fp16ToFloatForTest(uint16_t value) {
  uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
  uint32_t exponent = (value >> 10) & 0x1fu;
  uint32_t mantissa = value & 0x03ffu;
  uint32_t bits = 0;

  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      exponent = 1;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x03ffu;
      uint32_t exp32 = exponent + (127 - 15);
      bits = sign | (exp32 << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    uint32_t exp32 = exponent + (127 - 15);
    bits = sign | (exp32 << 23) | (mantissa << 13);
  }

  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

std::vector<BlockQ41> QuantizeToQ41ForTest(const Tensor& src) {
  int row_size =
      src.num_dims() >= 2 ? src.shape.back() : static_cast<int>(src.size());
  int n_rows =
      src.num_dims() >= 2 ? static_cast<int>(src.size()) / row_size : 1;
  int row_blocks = (row_size + kQ41BlockSize - 1) / kQ41BlockSize;

  std::vector<BlockQ41> blocks;
  blocks.reserve(static_cast<size_t>(n_rows) * row_blocks);
  for (int row = 0; row < n_rows; ++row) {
    int row_offset = row * row_size;
    for (int rb = 0; rb < row_blocks; ++rb) {
      int base = row_offset + rb * kQ41BlockSize;
      int k_end = std::min(base + kQ41BlockSize, row_offset + row_size);
      float min_value = std::numeric_limits<float>::infinity();
      float max_value = -std::numeric_limits<float>::infinity();
      for (int k = base; k < k_end; ++k) {
        min_value = std::min(min_value, src.data[k]);
        max_value = std::max(max_value, src.data[k]);
      }

      BlockQ41 block{};
      std::memset(&block, 0, sizeof(block));
      float range = max_value - min_value;
      if (range > 0.0f) {
        block.d = FloatToFp16ForTest(range / 15.0f);
        block.m = FloatToFp16ForTest(min_value);
        float stored_d = Fp16ToFloatForTest(block.d);
        float stored_m = Fp16ToFloatForTest(block.m);
        float inv_d = stored_d == 0.0f ? 0.0f : 1.0f / stored_d;
        for (int j = 0; j < kQ41BlockSize / 2; ++j) {
          int idx0 = base + j;
          int idx1 = base + j + kQ41BlockSize / 2;
          int q0 = 0;
          int q1 = 0;
          if (idx0 < k_end) {
            q0 = static_cast<int>(
                std::round((src.data[idx0] - stored_m) * inv_d));
          }
          if (idx1 < k_end) {
            q1 = static_cast<int>(
                std::round((src.data[idx1] - stored_m) * inv_d));
          }
          q0 = std::clamp(q0, 0, 15);
          q1 = std::clamp(q1, 0, 15);
          block.qs[j] = static_cast<uint8_t>(q0 | (q1 << 4));
        }
      } else {
        block.d = 0;
        block.m = FloatToFp16ForTest(
            min_value == std::numeric_limits<float>::infinity() ? 0.0f
                                                                : min_value);
      }
      blocks.push_back(block);
    }
  }
  return blocks;
}

void QuantizeQtToQ41ForTest(QuantizedTensor& qt) {
  Tensor t = ToTensor(qt);
  qt.q4_1_data = QuantizeToQ41ForTest(t);
  qt.type = QuantType::kQ41;
  qt.f32_data.clear();
  qt.f32_data.shrink_to_fit();
  qt.q8_0_data.clear();
  qt.q8_0_data.shrink_to_fit();
  qt.q4_0_data.clear();
  qt.q4_0_data.shrink_to_fit();
}

void QuantizeModelToQ41ForTest(MiniLlamaModel& model) {
  QuantizeQtToQ41ForTest(model.lm_head);
  for (auto& lw : model.layers) {
    QuantizeQtToQ41ForTest(lw.wq);
    QuantizeQtToQ41ForTest(lw.wk);
    QuantizeQtToQ41ForTest(lw.wv);
    QuantizeQtToQ41ForTest(lw.wo);
    QuantizeQtToQ41ForTest(lw.w_gate);
    QuantizeQtToQ41ForTest(lw.w_up);
    QuantizeQtToQ41ForTest(lw.w_down);
  }
}

size_t QuantLinearWeightBytes(const MiniLlamaModel& model, QuantType type) {
  auto bytes_for = [type](const QuantizedTensor& weight) -> size_t {
    if (type == QuantType::kQ80) {
      return weight.q8_0_data.size() * sizeof(BlockQ80);
    }
    if (type == QuantType::kQ40) {
      return weight.q4_0_data.size() * sizeof(BlockQ40);
    }
    if (type == QuantType::kQ41) {
      return weight.q4_1_data.size() * sizeof(BlockQ41);
    }
    return 0;
  };
  size_t bytes = bytes_for(model.lm_head);
  for (const auto& lw : model.layers) {
    bytes += bytes_for(lw.wq);
    bytes += bytes_for(lw.wk);
    bytes += bytes_for(lw.wv);
    bytes += bytes_for(lw.wo);
    bytes += bytes_for(lw.w_gate);
    bytes += bytes_for(lw.w_up);
    bytes += bytes_for(lw.w_down);
  }
  return bytes;
}

size_t ResidentF32TensorBytes(const MiniLlamaModel& model) {
  auto tensor_bytes = [](const Tensor& tensor) -> size_t {
    return tensor.data.size() * sizeof(float);
  };
  size_t bytes =
      tensor_bytes(model.token_embedding) + tensor_bytes(model.final_norm);
  for (const auto& lw : model.layers) {
    bytes += tensor_bytes(lw.attention_norm);
    bytes += tensor_bytes(lw.bq);
    bytes += tensor_bytes(lw.bk);
    bytes += tensor_bytes(lw.bv);
    bytes += tensor_bytes(lw.ffn_norm);
  }
  return bytes;
}

void RequireCudaQuantNotBuilt() {
  Require(!CudaQuantBuilt(), "CudaQuantBuilt should be false in CPU build");
  Tensor x({4}, 1.0f);
  Tensor w = MakePatternTensor({3, 4}, 0.05f, 0.0f);
  std::vector<BlockQ80> q8 = QuantizeToQ80(w);
  std::vector<BlockQ40> q4_0 = QuantizeToQ40(w);
  std::vector<BlockQ41> q4_1 = QuantizeToQ41ForTest(w);

  bool threw = false;
  try {
    (void)CudaQ80Linear(x, q8, w.shape);
  } catch (const std::exception& e) {
    threw = true;
    Require(std::string(e.what()).find("CUDA quant kernels were not built") !=
                std::string::npos,
            "CPU build should report missing CUDA quant kernels");
  }
  Require(threw, "CudaQ80Linear should throw in CPU build");

  threw = false;
  try {
    (void)CudaQ40Linear(x, q4_0, w.shape);
  } catch (const std::exception& e) {
    threw = true;
    Require(std::string(e.what()).find("CUDA quant kernels were not built") !=
                std::string::npos,
            "CPU build should report missing CUDA quant kernels for Q4_0");
  }
  Require(threw, "CudaQ40Linear should throw in CPU build");

  threw = false;
  try {
    (void)CudaQ41Linear(x, q4_1, w.shape);
  } catch (const std::exception& e) {
    threw = true;
    Require(std::string(e.what()).find("CUDA quant kernels were not built") !=
                std::string::npos,
            "CPU build should report missing CUDA quant kernels for Q4_1");
  }
  Require(threw, "CudaQ41Linear should throw in CPU build");
}

void TestCudaQ80LinearCases() {
  Require(CudaQuantBuilt(), "CudaQuantBuilt should be true in CUDA build");
  Require(CudaDeviceCount() > 0, "CUDA build should see at least one device");

  Tensor w1 = MakePatternTensor({5, 13}, 0.037f, -0.11f);
  Tensor x1 = MakePatternTensor({13}, 0.071f, 0.03f);
  Tensor b1 = MakePatternTensor({5}, 0.013f, -0.02f);
  std::vector<BlockQ80> q8_1 = QuantizeToQ80(w1);
  Tensor expected1 = LinearQ80(x1, q8_1, w1.shape);
  Tensor actual1 = CudaQ80Linear(x1, q8_1, w1.shape);
  RequireCloseTensor(actual1, expected1, "CudaQ80Linear 1D no bias");

  Tensor expected1_bias = expected1;
  for (int i = 0; i < b1.shape[0]; ++i) {
    expected1_bias.data[i] += b1.data[i];
  }
  Tensor actual1_bias = CudaQ80Linear(x1, q8_1, w1.shape, &b1);
  RequireCloseTensor(actual1_bias, expected1_bias, "CudaQ80Linear 1D bias");

  Tensor w2 = MakePatternTensor({32, 64}, 0.019f, 0.04f);
  Tensor x2 = MakePatternTensor({3, 64}, 0.029f, -0.06f);
  std::vector<BlockQ80> q8_2 = QuantizeToQ80(w2);
  Tensor expected2 = LinearQ80(x2, q8_2, w2.shape);
  Tensor actual2 = CudaQ80Linear(x2, q8_2, w2.shape);
  RequireCloseTensor(actual2, expected2, "CudaQ80Linear 2D batch");

  CudaDeviceBuffer w_dev(q8_2.size() * sizeof(BlockQ80), 0);
  w_dev.Upload(q8_2.data(), q8_2.size() * sizeof(BlockQ80));
  Tensor actual_device =
      CudaQ80LinearDeviceWeight(x2, w_dev.data(), q8_2.size(), w2.shape);
  RequireCloseTensor(actual_device, expected2, "CudaQ80Linear device weight");
}

void TestCudaQ40LinearCases() {
  Require(CudaQuantBuilt(), "CudaQuantBuilt should be true in CUDA build");
  Require(CudaDeviceCount() > 0, "CUDA build should see at least one device");

  Tensor w1 = MakePatternTensor({5, 13}, 0.037f, -0.11f);
  Tensor x1 = MakePatternTensor({13}, 0.071f, 0.03f);
  Tensor b1 = MakePatternTensor({5}, 0.013f, -0.02f);
  std::vector<BlockQ40> q4_1 = QuantizeToQ40(w1);
  Tensor expected1 = LinearQ40(x1, q4_1, w1.shape);
  Tensor actual1 = CudaQ40Linear(x1, q4_1, w1.shape);
  RequireCloseTensor(actual1, expected1, "CudaQ40Linear 1D no bias");

  Tensor expected1_bias = expected1;
  for (int i = 0; i < b1.shape[0]; ++i) {
    expected1_bias.data[i] += b1.data[i];
  }
  Tensor actual1_bias = CudaQ40Linear(x1, q4_1, w1.shape, &b1);
  RequireCloseTensor(actual1_bias, expected1_bias, "CudaQ40Linear 1D bias");

  Tensor w2 = MakePatternTensor({32, 64}, 0.019f, 0.04f);
  Tensor x2 = MakePatternTensor({3, 64}, 0.029f, -0.06f);
  std::vector<BlockQ40> q4_2 = QuantizeToQ40(w2);
  Tensor expected2 = LinearQ40(x2, q4_2, w2.shape);
  Tensor actual2 = CudaQ40Linear(x2, q4_2, w2.shape);
  RequireCloseTensor(actual2, expected2, "CudaQ40Linear 2D batch");

  CudaDeviceBuffer w_dev(q4_2.size() * sizeof(BlockQ40), 0);
  w_dev.Upload(q4_2.data(), q4_2.size() * sizeof(BlockQ40));
  Tensor actual_device =
      CudaQ40LinearDeviceWeight(x2, w_dev.data(), q4_2.size(), w2.shape);
  RequireCloseTensor(actual_device, expected2, "CudaQ40Linear device weight");

  CudaTensor x_dev = CudaTensorFromHost(x2, 0);
  CudaTensor y_dev =
      CudaQ40LinearDeviceInput(x_dev, w_dev.data(), q4_2.size(), w2.shape);
  RequireCloseTensor(y_dev.Download(), expected2, "CudaQ40Linear device input");
}

void TestCudaQ41LinearCases() {
  Require(CudaQuantBuilt(), "CudaQuantBuilt should be true in CUDA build");
  Require(CudaDeviceCount() > 0, "CUDA build should see at least one device");

  Tensor w1 = MakePatternTensor({5, 13}, 0.037f, -0.11f);
  Tensor x1 = MakePatternTensor({13}, 0.071f, 0.03f);
  Tensor b1 = MakePatternTensor({5}, 0.013f, -0.02f);
  std::vector<BlockQ41> q4_1 = QuantizeToQ41ForTest(w1);
  Tensor expected1 = LinearQ41(x1, q4_1, w1.shape);
  Tensor actual1 = CudaQ41Linear(x1, q4_1, w1.shape);
  RequireCloseTensor(actual1, expected1, "CudaQ41Linear 1D no bias");

  Tensor expected1_bias = expected1;
  for (int i = 0; i < b1.shape[0]; ++i) {
    expected1_bias.data[i] += b1.data[i];
  }
  Tensor actual1_bias = CudaQ41Linear(x1, q4_1, w1.shape, &b1);
  RequireCloseTensor(actual1_bias, expected1_bias, "CudaQ41Linear 1D bias");

  Tensor w2 = MakePatternTensor({17, 65}, 0.019f, 0.04f);
  Tensor x2 = MakePatternTensor({3, 65}, 0.029f, -0.06f);
  std::vector<BlockQ41> q4_2 = QuantizeToQ41ForTest(w2);
  Tensor expected2 = LinearQ41(x2, q4_2, w2.shape);
  Tensor actual2 = CudaQ41Linear(x2, q4_2, w2.shape);
  RequireCloseTensor(actual2, expected2, "CudaQ41Linear 2D batch");

  CudaDeviceBuffer w_dev(q4_2.size() * sizeof(BlockQ41), 0);
  w_dev.Upload(q4_2.data(), q4_2.size() * sizeof(BlockQ41));
  Tensor actual_device =
      CudaQ41LinearDeviceWeight(x2, w_dev.data(), q4_2.size(), w2.shape);
  RequireCloseTensor(actual_device, expected2, "CudaQ41Linear device weight");

  CudaTensor x_dev = CudaTensorFromHost(x2, 0);
  CudaTensor y_dev =
      CudaQ41LinearDeviceInput(x_dev, w_dev.data(), q4_2.size(), w2.shape);
  RequireCloseTensor(y_dev.Download(), expected2, "CudaQ41Linear device input");
}

void TestCudaForwardQuant(QuantType quant_type) {
  Require(CudaQuantBuilt(), "CudaQuantBuilt should be true in CUDA build");
  Require(CudaDeviceCount() > 0, "CUDA build should see at least one device");

  MiniLlamaModel cpu_model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  MiniLlamaModel cuda_model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  Require(cpu_model.loaded, "CPU tiny model should Load");
  Require(cuda_model.loaded, "CUDA tiny model should Load");
  if (quant_type == QuantType::kQ80) {
    QuantizeModelToQ80(cpu_model);
    QuantizeModelToQ80(cuda_model);
  } else if (quant_type == QuantType::kQ40) {
    QuantizeModelToQ40(cpu_model);
    QuantizeModelToQ40(cuda_model);
  } else if (quant_type == QuantType::kQ41) {
    QuantizeModelToQ41ForTest(cpu_model);
    QuantizeModelToQ41ForTest(cuda_model);
  } else {
    throw std::runtime_error("TestCudaForwardQuant: unsupported quant type");
  }

  MiniLlamaContext cpu_ctx(&cpu_model);
  MiniLlamaContext cuda_ctx(&cuda_model);
  MiniBatch batch = MiniBatch::Single(1, 0);
  Tensor cpu_logits = ForwardBatch(cpu_ctx, cpu_model, batch);

  UploadModelWeightsToCuda(cuda_model, 0);
  Require(ModelCudaUploadedWeightCount(cuda_model) == 21,
          "tiny quant model should upload 21 CUDA resident weights");
  Require(ModelCudaUploadedF32WeightCount(cuda_model) == 0,
          "tiny quant model should upload no F32 Linear weights");
  size_t uploaded_quant = 0;
  if (quant_type == QuantType::kQ80) {
    uploaded_quant = ModelCudaUploadedQ80WeightCount(cuda_model);
  } else if (quant_type == QuantType::kQ40) {
    uploaded_quant = ModelCudaUploadedQ40WeightCount(cuda_model);
  } else {
    uploaded_quant = ModelCudaUploadedQ41WeightCount(cuda_model);
  }
  Require(uploaded_quant == 15,
          "tiny quant model should upload 15 quantized Linear weights");
  Require(ModelCudaMemoryBytes(cuda_model) ==
              QuantLinearWeightBytes(cuda_model, quant_type) +
                  ResidentF32TensorBytes(cuda_model),
          "quant uploaded byte count should match resident weights");

  Tensor cuda_logits = ForwardBatch(cuda_ctx, cuda_model, batch);
  RequireCloseTensor(cuda_logits, cpu_logits, "cuda_forward_quant logits");
  Require(ModelCudaLinearCalls(cuda_model) == 15,
          "one tiny token should run 15 CUDA quant Linear calls");
  Require(ModelCudaAttentionCalls(cuda_model) == 2,
          "one tiny token should run 2 CUDA attention calls");
  Require(ModelCudaHostToDeviceCopies(cuda_model) == 0,
          "one tiny token should keep host->device copies at zero");
  Require(ModelCudaDeviceToHostCopies(cuda_model) == 1,
          "one tiny token should download only logits");
}

}  // namespace

static bool TestCudaQuantRunner() {
  try {
#ifdef MINI_LLAMA_USE_CUDA
    TestCudaQ80LinearCases();
    TestCudaQ40LinearCases();
    TestCudaQ41LinearCases();
    TestCudaForwardQuant(QuantType::kQ80);
    TestCudaForwardQuant(QuantType::kQ40);
    TestCudaForwardQuant(QuantType::kQ41);
#else
    RequireCudaQuantNotBuilt();
#endif
    return true;
  } catch (const std::exception& e) {
    MINI_LLAMA_ASSERT_FAIL(std::string("cuda_quant failed: ") + e.what());
    return false;
  }
}

static struct CudaQuantRegistrar {
  CudaQuantRegistrar() {
    RegisterTest("cuda_quant_all", TestCudaQuantRunner);
  }
} cuda_quant_registrar;
