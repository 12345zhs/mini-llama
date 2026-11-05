#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/batch.h"
#include "mini_llama/context.h"
#include "mini_llama/cuda_attention.h"
#include "mini_llama/cuda_kv_cache.h"
#include "mini_llama/cuda_runtime.h"
#include "mini_llama/forward.h"
#include "mini_llama/kv_cache.h"
#include "mini_llama/loader.h"
#include "mini_llama/model.h"
#include "mini_llama/ops.h"
#include "tests/test_names.h"
#include "tests/test_main.h"

namespace {

constexpr float kAbsTol = 2e-3f;
constexpr float kRelTol = 2e-3f;

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

int MapQHeadToKvHead(int q_head, int n_heads, int n_kv_heads) {
  return q_head / (n_heads / n_kv_heads);
}

Tensor CpuAttentionReference(const Tensor& q, const KvCache& kv_cache,
                             int layer, int pos, int n_heads, int n_kv_heads,
                             int head_dim) {
  Tensor out({n_heads, head_dim}, 0.0f);
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  for (int h = 0; h < n_heads; ++h) {
    int kv_head = MapQHeadToKvHead(h, n_heads, n_kv_heads);
    Tensor scores({pos + 1}, 0.0f);
    for (int t = 0; t <= pos; ++t) {
      const float* k_ptr = kv_cache.KeyPtr(layer, t, kv_head);
      float dot = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        dot += q.data[h * head_dim + d] * k_ptr[d];
      }
      scores.data[t] = dot * scale;
    }
    Tensor probs = Softmax(scores);
    for (int d = 0; d < head_dim; ++d) {
      float value = 0.0f;
      for (int t = 0; t <= pos; ++t) {
        const float* v_ptr = kv_cache.ValuePtr(layer, t, kv_head);
        value += probs.data[t] * v_ptr[d];
      }
      out.data[h * head_dim + d] = value;
    }
  }
  return out;
}

void RequireCpuBuildPath() {
  Require(!CudaRuntimeBuilt(), "CudaRuntimeBuilt should be false in CPU build");
  Require(!CudaAttentionBuilt(),
          "CudaAttentionBuilt should be false in CPU build");
  try {
    Tensor q({1, 2}, 0.0f);
    CudaKvCache cache;
    CudaAttentionDecode(q, cache, 0, 0, 1, 1, 2);
    throw std::runtime_error(
        "expected CudaAttentionDecode to fail in CPU build");
  } catch (const std::exception& e) {
    Require(std::string(e.what()).find("CUDA attention was not built") !=
                std::string::npos,
            "CPU build should explain that CUDA attention was not built");
  }
}

void TestCudaAttentionDirectGqa() {
  Require(CudaRuntimeBuilt(), "CudaRuntimeBuilt should be true in CUDA build");
  Require(CudaAttentionBuilt(),
          "CudaAttentionBuilt should be true in CUDA build");
  Require(CudaDeviceCount() > 0, "CUDA build should see at least one device");

  const int n_heads = 2;
  const int n_kv_heads = 1;
  const int head_dim = 3;
  const int layer = 0;
  const int pos = 1;

  Tensor q({n_heads, head_dim}, 0.0f);
  q.data = {
      0.2f, -0.1f, 0.4f, -0.3f, 0.5f, 0.1f,
  };

  Tensor k0({n_kv_heads, head_dim}, 0.0f);
  Tensor v0({n_kv_heads, head_dim}, 0.0f);
  Tensor k1({n_kv_heads, head_dim}, 0.0f);
  Tensor v1({n_kv_heads, head_dim}, 0.0f);
  k0.data = {0.1f, 0.2f, -0.1f};
  v0.data = {1.0f, -2.0f, 0.5f};
  k1.data = {-0.4f, 0.3f, 0.2f};
  v1.data = {-1.0f, 0.25f, 2.0f};

  KvCache cpu_cache(1, 4, n_kv_heads, head_dim);
  CudaKvCache cuda_cache(1, 4, n_kv_heads, head_dim);
  cpu_cache.Write(layer, 0, k0, v0);
  cpu_cache.Write(layer, 1, k1, v1);
  cuda_cache.Write(layer, 0, k0, v0);
  cuda_cache.Write(layer, 1, k1, v1);

  Tensor expected = CpuAttentionReference(q, cpu_cache, layer, pos, n_heads,
                                          n_kv_heads, head_dim);
  Tensor actual = CudaAttentionDecode(q, cuda_cache, layer, pos, n_heads,
                                      n_kv_heads, head_dim);
  RequireCloseTensor(actual, expected, "CudaAttentionDecode gqa");
}

void TestCudaForwardAttentionTiny() {
  MiniLlamaModel cpu_model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  MiniLlamaModel cuda_model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  Require(cpu_model.loaded, "CPU tiny model should load");
  Require(cuda_model.loaded, "CUDA tiny model should load");

  MiniLlamaContext cpu_ctx(&cpu_model);
  MiniLlamaContext cuda_ctx(&cuda_model);
  MiniBatch batch = MiniBatch::Single(1, 0);
  Tensor cpu_logits = ForwardBatch(cpu_ctx, cpu_model, batch);

  UploadModelWeightsToCuda(cuda_model, 0);
  ResetModelCudaRuntimeStats(cuda_model);
  Tensor cuda_logits = ForwardBatch(cuda_ctx, cuda_model, batch);

  RequireCloseTensor(cuda_logits, cpu_logits, "cuda_forward_attention logits");
  Require(ModelCudaAttentionCalls(cuda_model) == 2,
          "one tiny token should run 2 CUDA attention calls");
  Require(ModelCudaAttentionCpuFallbacks(cuda_model) == 0,
          "CUDA attention should avoid CPU attention fallback");
  Require(ModelCudaKvCacheWriteBytes(cuda_model) == 512,
          "tiny KV write bytes should match two layers");
  Require(ModelCudaKvCacheReadBytes(cuda_model) == 512,
          "tiny KV read bytes should match two layers at pos 0");
}

}  // namespace

static bool TestCudaAttentionRunner() {
  try {
#ifdef MINI_LLAMA_USE_CUDA
    TestCudaAttentionDirectGqa();
    TestCudaForwardAttentionTiny();
#else
    RequireCpuBuildPath();
#endif
    return true;
  } catch (const std::exception& e) {
    MINI_LLAMA_ASSERT_FAIL(std::string("cuda_attention failed: ") + e.what());
    return false;
  }
}

static struct CudaAttentionRegistrar {
  CudaAttentionRegistrar() {
    RegisterTest("cuda_attention", TestCudaAttentionRunner);
  }
} cuda_attention_registrar;
