#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/batch.h"
#include "mini_llama/context.h"
#include "mini_llama/forward.h"
#include "mini_llama/loader.h"
#include "mini_llama/model.h"
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

void RequireCudaNotBuilt() {
  Require(!CudaRuntimeBuilt(), "CudaRuntimeBuilt should be false in CPU build");
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  Require(model.loaded, "tiny model should load");

  bool threw = false;
  try {
    UploadModelWeightsToCuda(model, 0);
  } catch (const std::exception& e) {
    threw = true;
    Require(std::string(e.what()).find("CUDA backend was not built") !=
                std::string::npos,
            "CPU build should report missing CUDA backend");
  }
  Require(threw, "CUDA weight upload should throw in CPU build");
}

void TestCudaForwardLinear() {
  Require(CudaRuntimeBuilt(), "CudaRuntimeBuilt should be true in CUDA build");
  Require(CudaDeviceCount() > 0, "CUDA build should see at least one device");

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
  Require(ModelCudaUploadedWeightCount(cuda_model) == 21,
          "tiny model should upload 21 CUDA resident weights");
  Require(ModelCudaUploadedF32WeightCount(cuda_model) == 15,
          "tiny model should upload 15 F32 Linear weights");
  Require(ModelCudaMemoryBytes(cuda_model) == 132224,
          "tiny model uploaded byte count should match resident weights");

  Tensor cuda_logits = ForwardBatch(cuda_ctx, cuda_model, batch);
  RequireCloseTensor(cuda_logits, cpu_logits, "cuda_forward_linear logits");

  Require(ModelCudaLinearCalls(cuda_model) == 15,
          "one tiny token should run 15 CUDA Linear calls");
  Require(ModelCudaAttentionCalls(cuda_model) == 2,
          "one tiny token should run 2 CUDA attention calls");
  Require(ModelCudaHostToDeviceCopies(cuda_model) == 0,
          "one tiny token should keep host->device copies at zero");
  Require(ModelCudaDeviceToHostCopies(cuda_model) == 1,
          "one tiny token should download only logits");
  Require(ModelCudaHostToDeviceBytes(cuda_model) == 0,
          "one tiny token should keep host->device bytes at zero");
  Require(ModelCudaDeviceToHostBytes(cuda_model) > 0,
          "device->host byte count should be recorded");
}

}  // namespace

static bool TestCudaForwardLinearRunner() {
  try {
#ifdef MINI_LLAMA_USE_CUDA
    TestCudaForwardLinear();
#else
    RequireCudaNotBuilt();
#endif
    return true;
  } catch (const std::exception& e) {
    MINI_LLAMA_ASSERT_FAIL(std::string("cuda_forward_linear: ") + e.what());
    return false;
  }
}

static struct CudaForwardLinearRegistrar {
  CudaForwardLinearRegistrar() {
    RegisterTest("cuda_forward_linear", TestCudaForwardLinearRunner);
  }
} cuda_forward_linear_registrar;
