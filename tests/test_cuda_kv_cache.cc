#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/batch.h"
#include "mini_llama/context.h"
#include "mini_llama/cuda_kv_cache.h"
#include "mini_llama/cuda_runtime.h"
#include "mini_llama/forward.h"
#include "mini_llama/kv_cache.h"
#include "mini_llama/loader.h"
#include "mini_llama/model.h"
#include "tests/test_names.h"
#include "tests/test_main.h"

namespace {

constexpr float kAbsTol = 1e-6f;

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void RequireClose(float actual, float expected, const std::string& label) {
  if (std::fabs(actual - expected) > kAbsTol) {
    throw std::runtime_error(label + ": actual=" + std::to_string(actual) +
                             ", expected=" + std::to_string(expected));
  }
}

void RequireCloseTensor(const Tensor& actual, const Tensor& expected,
                        const std::string& label) {
  Require(actual.shape == expected.shape, label + ": shape mismatch");
  for (size_t i = 0; i < actual.size(); ++i) {
    RequireClose(actual.data[i], expected.data[i],
                 label + " at " + std::to_string(i));
  }
}

void RequireThrowsWith(const std::function<void()>& fn,
                       const std::string& needle, const std::string& label) {
  try {
    fn();
  } catch (const std::exception& e) {
    Require(std::string(e.what()).find(needle) != std::string::npos,
            label + ": unexpected error message: " + e.what());
    return;
  }
  throw std::runtime_error(label + ": expected exception");
}

Tensor MakePatternTensor(const std::vector<int>& shape, float base) {
  Tensor t(shape, 0.0f);
  for (size_t i = 0; i < t.size(); ++i) {
    t.data[i] = base + static_cast<float>(i) * 0.25f;
  }
  return t;
}

Tensor ReadCpuSlot(const KvCache& cache, bool key, int layer, int pos) {
  int n_kv_heads = cache.keys.shape[2];
  int head_dim = cache.keys.shape[3];
  Tensor out({n_kv_heads, head_dim}, 0.0f);
  for (int h = 0; h < n_kv_heads; ++h) {
    const float* src =
        key ? cache.KeyPtr(layer, pos, h) : cache.ValuePtr(layer, pos, h);
    for (int d = 0; d < head_dim; ++d) {
      out.data[static_cast<size_t>(h * head_dim + d)] = src[d];
    }
  }
  return out;
}

void RequireCudaNotBuilt() {
  Require(!CudaKvCacheBuilt(), "CudaKvCacheBuilt should be false in CPU build");
  RequireThrowsWith(
      []() {
        CudaKvCache cache(1, 2, 1, 2);
        (void)cache;
      },
      "CUDA KV cache was not built", "CPU build constructor");
  RequireThrowsWith(
      []() {
        CudaKvCache cache;
        cache.Clear();
      },
      "CUDA KV cache was not built", "CPU build Clear");
}

void TestConstructWriteReadClear() {
  CudaKvCache cache(2, 4, 2, 3);
  Require(!cache.empty(), "cache should allocate device buffers");
  Require(cache.bytes() == 2 * 2 * 4 * 2 * 3 * sizeof(float),
          "cache byte count mismatch");
  Require(cache.n_layers() == 2, "n_layers mismatch");
  Require(cache.max_seq_len() == 4, "max_seq_len mismatch");
  Require(cache.n_kv_heads() == 2, "n_kv_heads mismatch");
  Require(cache.head_dim() == 3, "head_dim mismatch");

  Tensor k = MakePatternTensor({2, 3}, 1.0f);
  Tensor v = MakePatternTensor({2, 3}, 10.0f);
  cache.Write(1, 2, k, v);

  RequireCloseTensor(cache.ReadKey(1, 2), k, "ReadKey");
  RequireCloseTensor(cache.ReadValue(1, 2), v, "ReadValue");

  Tensor expected_key_head_0({3}, 0.0f);
  Tensor expected_key_head_1({3}, 0.0f);
  Tensor expected_value_head_0({3}, 0.0f);
  Tensor expected_value_head_1({3}, 0.0f);
  for (int d = 0; d < 3; ++d) {
    expected_key_head_0.data[d] = k.data[d];
    expected_key_head_1.data[d] = k.data[3 + d];
    expected_value_head_0.data[d] = v.data[d];
    expected_value_head_1.data[d] = v.data[3 + d];
  }
  RequireCloseTensor(cache.ReadKeyHead(1, 2, 0), expected_key_head_0,
                     "ReadKeyHead 0");
  RequireCloseTensor(cache.ReadKeyHead(1, 2, 1), expected_key_head_1,
                     "ReadKeyHead 1");
  RequireCloseTensor(cache.ReadValueHead(1, 2, 0), expected_value_head_0,
                     "ReadValueHead 0");
  RequireCloseTensor(cache.ReadValueHead(1, 2, 1), expected_value_head_1,
                     "ReadValueHead 1");

  cache.Clear();
  Tensor zeros({2, 3}, 0.0f);
  RequireCloseTensor(cache.ReadKey(1, 2), zeros, "Clear key");
  RequireCloseTensor(cache.ReadValue(1, 2), zeros, "Clear value");
}

void TestRejectsBadInputs() {
  CudaKvCache cache(1, 2, 1, 2);
  Tensor good({1, 2}, 0.0f);
  Tensor rank_3({1, 1, 2}, 0.0f);
  Tensor wrong_heads({2, 2}, 0.0f);
  Tensor wrong_dim({1, 3}, 0.0f);

  RequireThrowsWith([&]() { cache.Write(-1, 0, good, good); },
                    "layer out of range", "layer underflow");
  RequireThrowsWith([&]() { cache.Write(0, 2, good, good); },
                    "position out of range", "position overflow");
  RequireThrowsWith([&]() { (void)cache.ReadKeyHead(0, 0, 1); },
                    "head out of range", "head overflow");
  RequireThrowsWith([&]() { cache.Write(0, 0, rank_3, good); },
                    "expected matching", "rank mismatch");
  RequireThrowsWith([&]() { cache.Write(0, 0, wrong_heads, wrong_heads); },
                    "does not match", "head mismatch");
  RequireThrowsWith([&]() { cache.Write(0, 0, wrong_dim, wrong_dim); },
                    "does not match", "dim mismatch");
}

void TestCpuGpuCacheAlignment() {
  KvCache cpu_cache(2, 4, 2, 3);
  CudaKvCache cuda_cache(2, 4, 2, 3);

  Tensor k = MakePatternTensor({2, 3}, -2.0f);
  Tensor v = MakePatternTensor({2, 3}, 5.0f);
  cpu_cache.Write(1, 3, k, v);
  cuda_cache.Write(1, 3, k, v);

  RequireCloseTensor(cuda_cache.ReadKey(1, 3),
                     ReadCpuSlot(cpu_cache, true, 1, 3), "CPU/GPU key slot");
  RequireCloseTensor(cuda_cache.ReadValue(1, 3),
                     ReadCpuSlot(cpu_cache, false, 1, 3), "CPU/GPU value slot");
}

void TestForwardSyncsCudaKvCache() {
  MiniLlamaModel cpu_model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  MiniLlamaModel cuda_model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  Require(cpu_model.loaded, "CPU tiny model should load");
  Require(cuda_model.loaded, "CUDA tiny model should load");

  MiniLlamaContext cpu_ctx(&cpu_model);
  MiniLlamaContext cuda_ctx(&cuda_model);
  Require(cuda_ctx.cuda_kv_cache.empty(),
          "context should start without CUDA cache before upload");

  MiniBatch batch = MiniBatch::FromTokens({1, 2}, 0);
  (void)ForwardBatch(cpu_ctx, cpu_model, batch);

  UploadModelWeightsToCuda(cuda_model, 0);
  (void)ForwardBatch(cuda_ctx, cuda_model, batch);

  Require(!cuda_ctx.cuda_kv_cache.empty(),
          "forward should allocate CUDA KV cache lazily");
  Require(cuda_ctx.cuda_kv_cache.bytes() > 0,
          "CUDA KV cache should report bytes");

  RequireCloseTensor(cuda_ctx.cuda_kv_cache.ReadKey(0, 0),
                     ReadCpuSlot(cpu_ctx.kv_cache, true, 0, 0),
                     "forward key layer0 pos0");
  RequireCloseTensor(cuda_ctx.cuda_kv_cache.ReadValue(0, 0),
                     ReadCpuSlot(cpu_ctx.kv_cache, false, 0, 0),
                     "forward value layer0 pos0");
  RequireCloseTensor(cuda_ctx.cuda_kv_cache.ReadKey(1, 1),
                     ReadCpuSlot(cpu_ctx.kv_cache, true, 1, 1),
                     "forward key layer1 pos1");
  RequireCloseTensor(cuda_ctx.cuda_kv_cache.ReadValue(1, 1),
                     ReadCpuSlot(cpu_ctx.kv_cache, false, 1, 1),
                     "forward value layer1 pos1");
}

void TestCudaKvCacheCases() {
  Require(CudaKvCacheBuilt(), "CudaKvCacheBuilt should be true in CUDA build");
  Require(CudaDeviceCount() > 0, "CUDA build should see at least one device");
  TestConstructWriteReadClear();
  TestRejectsBadInputs();
  TestCpuGpuCacheAlignment();
  TestForwardSyncsCudaKvCache();
}

}  // namespace


static bool TestCudaKvCacheRunner() {
  try {
#ifdef MINI_LLAMA_USE_CUDA
    TestCudaKvCacheCases();
#else
    RequireCudaNotBuilt();
#endif
    return true;
  } catch (const std::exception& e) {
    MINI_LLAMA_ASSERT_FAIL(std::string("cuda_kv_cache failed: ") + e.what());
    return false;
  }
}

static struct CudaKvCacheRegistrar {
  CudaKvCacheRegistrar() {
    RegisterTest("cuda_kv_cache", TestCudaKvCacheRunner);
  }
} cuda_kv_cache_registrar;