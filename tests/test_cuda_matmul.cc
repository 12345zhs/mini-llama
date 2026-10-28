#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/cuda_matmul.h"
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
    int bucket = static_cast<int>((i * 17 + 11) % 23);
    t.data[i] = static_cast<float>(bucket - 9) * scale + shift;
  }
  return t;
}

Tensor CpuMatmulReference(const Tensor& a, const Tensor& b) {
  int m = a.shape[0];
  int inner_dim = a.shape[1];
  int n = b.shape[1];
  Tensor c({m, n}, 0.0f);
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (int k = 0; k < inner_dim; ++k) {
        sum += a.data[row * inner_dim + k] * b.data[k * n + col];
      }
      c.data[row * n + col] = sum;
    }
  }
  return c;
}

Tensor CpuLinearReference(const Tensor& x, const Tensor& weight,
                          const Tensor* bias) {
  bool input_is_1d = x.num_dims() == 1;
  int rows = input_is_1d ? 1 : x.shape[0];
  int in_features = input_is_1d ? x.shape[0] : x.shape[1];
  int out_features = weight.shape[0];
  Tensor y(input_is_1d ? std::vector<int>{out_features}
                       : std::vector<int>{rows, out_features},
           0.0f);
  for (int row = 0; row < rows; ++row) {
    for (int out = 0; out < out_features; ++out) {
      float sum = bias == nullptr ? 0.0f : bias->data[out];
      for (int k = 0; k < in_features; ++k) {
        sum +=
            x.data[row * in_features + k] * weight.data[out * in_features + k];
      }
      y.data[row * out_features + out] = sum;
    }
  }
  return y;
}

void RequireCudaNotBuilt() {
  Require(!CudaMatmulBuilt(), "CudaMatmulBuilt should be false in CPU build");
  Tensor a({1, 4}, 1.0f);
  Tensor b({4, 3}, 1.0f);
  bool threw = false;
  try {
    (void)CudaMatmul(a, b);
  } catch (const std::exception& e) {
    threw = true;
    Require(std::string(e.what()).find("CUDA Matmul was not built") !=
                std::string::npos,
            "CPU build should report missing CUDA Matmul");
  }
  Require(threw, "CudaMatmul should throw in CPU build");
}

void TestCudaMatmulCases() {
  Require(CudaMatmulBuilt(), "CudaMatmulBuilt should be true in CUDA build");
  const std::vector<std::pair<int, int>> shapes = {
      {1, 4},
      {2, 4},
      {8, 16},
  };
  const std::vector<int> out_cols = {3, 3, 32};
  for (size_t i = 0; i < shapes.size(); ++i) {
    int rows = shapes[i].first;
    int inner = shapes[i].second;
    int cols = out_cols[i];
    Tensor a = MakePatternTensor({rows, inner}, 0.071f, -0.13f);
    Tensor b = MakePatternTensor({inner, cols}, 0.037f, 0.08f);
    Tensor expected = CpuMatmulReference(a, b);
    Tensor actual = CudaMatmul(a, b);
    RequireCloseTensor(
        actual, expected,
        "CudaMatmul " + a.ShapeToStringShort() + " x " + b.ShapeToStringShort());
  }
}

void TestCudaLinearCases() {
  Require(CudaMatmulBuilt(), "CudaMatmulBuilt should be true in CUDA build");

  Tensor x1 = MakePatternTensor({4}, 0.11f, -0.2f);
  Tensor w1 = MakePatternTensor({3, 4}, 0.043f, 0.03f);
  Tensor b1 = MakePatternTensor({3}, 0.019f, -0.01f);
  RequireCloseTensor(CudaLinear(x1, w1), CpuLinearReference(x1, w1, nullptr),
                     "CudaLinear 1D no bias");
  RequireCloseTensor(CudaLinear(x1, w1, &b1), CpuLinearReference(x1, w1, &b1),
                     "CudaLinear 1D bias");

  Tensor x2 = MakePatternTensor({2, 4}, 0.083f, 0.17f);
  Tensor w2 = MakePatternTensor({3, 4}, 0.029f, -0.04f);
  Tensor b2 = MakePatternTensor({3}, 0.017f, 0.05f);
  RequireCloseTensor(CudaLinear(x2, w2), CpuLinearReference(x2, w2, nullptr),
                     "CudaLinear 2D no bias");
  RequireCloseTensor(CudaLinear(x2, w2, &b2), CpuLinearReference(x2, w2, &b2),
                     "CudaLinear 2D bias");

  Tensor x3 = MakePatternTensor({8, 16}, 0.031f, -0.09f);
  Tensor w3 = MakePatternTensor({32, 16}, 0.023f, 0.07f);
  Tensor b3 = MakePatternTensor({32}, 0.013f, -0.03f);
  RequireCloseTensor(CudaLinear(x3, w3), CpuLinearReference(x3, w3, nullptr),
                     "CudaLinear 8x16 no bias");
  RequireCloseTensor(CudaLinear(x3, w3, &b3), CpuLinearReference(x3, w3, &b3),
                     "CudaLinear 8x16 bias");
}

}  // namespace


static bool TestCudaMatmulRunner() {
  try {
#ifdef MINI_LLAMA_USE_CUDA
    TestCudaMatmulCases();
#else
    RequireCudaNotBuilt();
#endif
    return true;
  } catch (const std::exception& e) {
    MINI_LLAMA_ASSERT_FAIL(std::string("cuda_matmul failed: ") + e.what());
    return false;
  }
}

static bool TestCudaLinearRunner() {
  try {
#ifdef MINI_LLAMA_USE_CUDA
    TestCudaLinearCases();
#else
    RequireCudaNotBuilt(); // 没 CUDA 环境时，验证一下它能否正确抛错即可
#endif
    return true;
  } catch (const std::exception& e) {
    MINI_LLAMA_ASSERT_FAIL(std::string("cuda_linear failed: ") + e.what());
    return false;
  }
}

static struct CudaMatmulRegistrar {
  CudaMatmulRegistrar() {
    RegisterTest("cuda_matmul_op", TestCudaMatmulRunner);
    RegisterTest("cuda_linear_op", TestCudaLinearRunner);
  }
} cuda_matmul_registrar;