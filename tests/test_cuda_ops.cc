#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/cuda_ops.h"
#include "mini_llama/cuda_runtime.h"
#include "mini_llama/ops.h"
#include "tests/test_names.h"
#include "tests/test_main.h"

namespace {

constexpr float kAbsTol = 1e-4f;
constexpr float kRelTol = 1e-4f;

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
    int bucket = static_cast<int>((i * 17 + 5) % 29);
    t.data[i] = static_cast<float>(bucket - 14) * scale + shift;
  }
  return t;
}

Tensor CpuAddReference(const Tensor& a, const Tensor& b) {
  Require(a.shape == b.shape, "cpu_add_reference shape mismatch");
  Tensor y(a.shape, 0.0f);
  for (size_t i = 0; i < a.size(); ++i) {
    y.data[i] = a.data[i] + b.data[i];
  }
  return y;
}

void RequireCudaNotBuilt() {
  Require(!CudaOpsBuilt(), "CudaOpsBuilt should be false in CPU build");
  Tensor x({4}, 1.0f);
  bool threw = false;
  try {
    (void)CudaSilu(x);
  } catch (const std::exception& e) {
    threw = true;
    Require(std::string(e.what()).find("CUDA ops were not built") !=
                std::string::npos,
            "CPU build should report missing CUDA ops");
  }
  Require(threw, "CudaSilu should throw in CPU build");
}

void TestCudaRmsNorm() {
  Tensor x = MakePatternTensor({32}, 0.071f, -0.13f);
  Tensor weight = MakePatternTensor({32}, 0.019f, 1.0f);
  RequireCloseTensor(CudaRmsNorm(x, weight, 1e-5f), RmsNorm(x, weight, 1e-5f),
                     "CudaRmsNorm");
}

void TestCudaEmbeddingLookupDeviceWeight() {
  Tensor embedding = MakePatternTensor({5, 7}, 0.031f, -0.2f);
  CudaDeviceBuffer embedding_dev(embedding.size() * sizeof(float), 0);
  embedding_dev.Upload(embedding.data.data(), embedding.size() * sizeof(float));

  Tensor expected({7}, 0.0f);
  int token_id = 3;
  for (int i = 0; i < 7; ++i) {
    expected.data[i] =
        embedding
            .data[static_cast<size_t>(token_id) * 7 + static_cast<size_t>(i)];
  }

  CudaTensor actual_dev = CudaEmbeddingLookupDeviceWeight(
      embedding_dev.data(), embedding.shape, token_id);
  RequireCloseTensor(actual_dev.Download(), expected,
                     "CudaEmbeddingLookupDeviceWeight");
}

void TestCudaRmsNormDeviceWeight() {
  Tensor x = MakePatternTensor({32}, 0.071f, -0.13f);
  Tensor weight = MakePatternTensor({32}, 0.019f, 1.0f);
  CudaTensor x_dev = CudaTensorFromHost(x);
  CudaDeviceBuffer weight_dev(weight.size() * sizeof(float), 0);
  weight_dev.Upload(weight.data.data(), weight.size() * sizeof(float));

  CudaTensor actual_dev =
      CudaRmsNormDeviceWeight(x_dev, weight_dev.data(), weight.shape, 1e-5f);
  RequireCloseTensor(actual_dev.Download(), RmsNorm(x, weight, 1e-5f),
                     "CudaRmsNormDeviceWeight");
}

void TestCudaSilu() {
  Tensor x = MakePatternTensor({65}, 0.083f, -0.2f);
  RequireCloseTensor(CudaSilu(x), Silu(x), "CudaSilu");
}

void TestCudaElementwise() {
  Tensor a = MakePatternTensor({3, 7}, 0.041f, -0.2f);
  Tensor b = MakePatternTensor({3, 7}, 0.023f, 0.13f);
  RequireCloseTensor(CudaElementwiseMul(a, b), ElementwiseMul(a, b),
                     "CudaElementwiseMul");
  RequireCloseTensor(CudaElementwiseAdd(a, b), CpuAddReference(a, b),
                     "CudaElementwiseAdd");

  CudaTensor a_dev = CudaTensorFromHost(a);
  CudaDeviceBuffer b_dev(b.size() * sizeof(float), 0);
  b_dev.Upload(b.data.data(), b.size() * sizeof(float));
  CudaTensor actual_dev =
      CudaElementwiseAddDeviceWeight(a_dev, b_dev.data(), b.shape);
  RequireCloseTensor(actual_dev.Download(), CpuAddReference(a, b),
                     "CudaElementwiseAddDeviceWeight");
}

void TestCudaSoftmax() {
  Tensor x = MakePatternTensor({37}, 0.11f, -0.3f);
  RequireCloseTensor(CudaSoftmax(x), Softmax(x), "CudaSoftmax");
}

void TestCudaRopeNormal() {
  Tensor q = MakePatternTensor({4, 8}, 0.037f, -0.1f);
  Tensor k = MakePatternTensor({2, 8}, 0.029f, 0.07f);
  Tensor expected_q = q;
  Tensor expected_k = k;

  Rope(expected_q, expected_k, 5, 10000.0f, RopeType::kNormal);
  CudaRope(q, k, 5, 10000.0f, RopeType::kNormal);

  RequireCloseTensor(q, expected_q, "CudaRope normal q");
  RequireCloseTensor(k, expected_k, "CudaRope normal k");
}

void TestCudaRopeNeoX() {
  Tensor q = MakePatternTensor({3, 8}, 0.031f, -0.04f);
  Tensor k = MakePatternTensor({1, 8}, 0.027f, 0.09f);
  Tensor expected_q = q;
  Tensor expected_k = k;

  Rope(expected_q, expected_k, 7, 10000.0f, RopeType::kNeoX);
  CudaRope(q, k, 7, 10000.0f, RopeType::kNeoX);

  RequireCloseTensor(q, expected_q, "CudaRope neox q");
  RequireCloseTensor(k, expected_k, "CudaRope neox k");
}

void TestCudaOpsCases() {
  Require(CudaOpsBuilt(), "CudaOpsBuilt should be true in CUDA build");
  Require(CudaDeviceCount() > 0, "CUDA build should see at least one device");
  TestCudaRmsNorm();
  TestCudaEmbeddingLookupDeviceWeight();
  TestCudaRmsNormDeviceWeight();
  TestCudaSilu();
  TestCudaElementwise();
  TestCudaSoftmax();
  TestCudaRopeNormal();
  TestCudaRopeNeoX();
}

}  // namespace

static bool TestCudaOpsRunner() {
  try {
#ifdef MINI_LLAMA_USE_CUDA
    TestCudaOpsCases();
#else
    RequireCudaNotBuilt();
#endif
    return true;
  } catch (const std::exception& e) {
    MINI_LLAMA_ASSERT_FAIL(std::string("cuda_ops failed: ") + e.what());
    return false;
  }
}

static struct CudaOpsRegistrar {
  CudaOpsRegistrar() {
    RegisterTest("cuda_ops", TestCudaOpsRunner);
  }
} cuda_ops_registrar;
