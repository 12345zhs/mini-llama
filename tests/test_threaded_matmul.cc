#include <cmath>
#include <stdexcept>
#include <string>

#include "mini_llama/matmul_dispatch.h"
#include "mini_llama/tensor.h"
#include "mini_llama/thread_pool.h"
#include "tests/test_main.h"
#include "tests/test_names.h"

static bool TestMatmulNaiveVsThreaded() {
  Tensor a({4, 8}, 0.0f);
  Tensor b({8, 4}, 0.0f);
  for (size_t i = 0; i < a.size(); ++i) {
    a.data[i] = static_cast<float>(i) * 0.1f - 0.5f;
  }
  for (size_t i = 0; i < b.size(); ++i) {
    b.data[i] = static_cast<float>(i) * 0.05f - 0.3f;
  }

  Tensor naive = MatmulDispatch(a, b, MatmulMode::kNaive);
  Tensor threaded = MatmulDispatch(a, b, MatmulMode::kThreaded);

  MINI_LLAMA_ASSERT_EQ(naive.shape.size(), threaded.shape.size());
  for (size_t i = 0; i < naive.shape.size(); ++i) {
    MINI_LLAMA_ASSERT_EQ(naive.shape[i], threaded.shape[i]);
  }
  for (size_t i = 0; i < naive.size(); ++i) {
    MINI_LLAMA_ASSERT_NEAR(naive.data[i], threaded.data[i], 1e-6f);
  }
  return true;
}

static bool TestLinearAllModesMatch() {
  Tensor weight({8, 16}, 0.0f);
  Tensor x({16}, 0.0f);
  for (size_t i = 0; i < weight.size(); ++i) {
    weight.data[i] = static_cast<float>(i) * 0.05f - 0.4f;
  }
  for (size_t i = 0; i < x.size(); ++i) {
    x.data[i] = static_cast<float>(i) * 0.1f - 0.5f;
  }

  Tensor naive = LinearDispatch(x, weight, MatmulMode::kNaive);
  Tensor threaded = LinearDispatch(x, weight, MatmulMode::kThreaded);


  MINI_LLAMA_ASSERT_EQ(naive.shape.size(), threaded.shape.size());
  for (size_t i = 0; i < naive.shape.size(); ++i) {
    MINI_LLAMA_ASSERT_EQ(naive.shape[i], threaded.shape[i]);
  }
  for (size_t i = 0; i < naive.size(); ++i) {
    MINI_LLAMA_ASSERT_NEAR(naive.data[i], threaded.data[i], 1e-5f);
  }
  return true;
}

static bool TestLinear2dInputAllModesMatch() {
  Tensor weight({6, 12}, 0.0f);
  Tensor x({1, 12}, 0.0f);
  for (size_t i = 0; i < weight.size(); ++i) {
    weight.data[i] = static_cast<float>(i) * 0.03f - 0.2f;
  }
  for (size_t i = 0; i < x.size(); ++i) {
    x.data[i] = static_cast<float>(i) * 0.07f - 0.3f;
  }

  Tensor naive = LinearDispatch(x, weight, MatmulMode::kNaive);
  Tensor threaded_simd = LinearDispatch(x, weight, MatmulMode::kThreaded);

  MINI_LLAMA_ASSERT_EQ(naive.shape.size(), threaded_simd.shape.size());
  for (size_t i = 0; i < naive.shape.size(); ++i) {
    MINI_LLAMA_ASSERT_EQ(naive.shape[i], threaded_simd.shape[i]);
  }
  for (size_t i = 0; i < naive.size(); ++i) {
    MINI_LLAMA_ASSERT_NEAR(naive.data[i], threaded_simd.data[i], 1e-5f);
  }
  return true;
}

static bool TestDifferentThreadCountsSameOutput() {
  Tensor weight({10, 32}, 0.0f);
  Tensor x({32}, 0.0f);
  for (size_t i = 0; i < weight.size(); ++i) {
    weight.data[i] = static_cast<float>(i) * 0.02f - 0.3f;
  }
  for (size_t i = 0; i < x.size(); ++i) {
    x.data[i] = static_cast<float>(i) * 0.05f - 0.4f;
  }

  Tensor baseline = LinearDispatch(x, weight, MatmulMode::kNaive);

  int saved = GetThreadCount();
  int thread_counts[] = {1, 2, 4};
  for (int tc : thread_counts) {
    SetThreadCount(tc);
    Tensor result = LinearDispatch(x, weight, MatmulMode::kThreaded);
    MINI_LLAMA_ASSERT_EQ(baseline.shape.size(), result.shape.size());
    for (size_t i = 0; i < baseline.shape.size(); ++i) {
      MINI_LLAMA_ASSERT_EQ(baseline.shape[i], result.shape[i]);
    }
    for (size_t i = 0; i < baseline.size(); ++i) {
      MINI_LLAMA_ASSERT_NEAR(baseline.data[i], result.data[i], 1e-5f);
    }
  }

  SetThreadCount(saved);
  return true;
}

static bool TestThreadCountApi() {
  SetThreadCount(0);
  int hw = static_cast<int>(std::thread::hardware_concurrency());
  if (hw > 0) {
    MINI_LLAMA_ASSERT_EQ(GetThreadCount(), hw);
  } else {
    MINI_LLAMA_ASSERT_EQ(GetThreadCount(), 4);
  }

  SetThreadCount(2);
  MINI_LLAMA_ASSERT_EQ(GetThreadCount(), 2);
  SetThreadCount(8);
  MINI_LLAMA_ASSERT_EQ(GetThreadCount(), 8);
  SetThreadCount(0);
  return true;
}

static bool TestParallelForPropagatesException() {
  int saved = GetThreadCount();
  SetThreadCount(4);
  try {
    ParallelFor(128, [](int begin, int end) {
      if (begin <= 64 && 64 < end) {
        throw std::runtime_error("parallel worker failed");
      }
    });
    MINI_LLAMA_ASSERT_FAIL("expected worker exception to propagate");
  } catch (const std::runtime_error& e) {
    MINI_LLAMA_ASSERT_TRUE(std::string(e.what()).find(
                               "parallel worker failed") != std::string::npos);
  }
  SetThreadCount(saved);
  return true;
}

// ---------------------------------------------------------------------------
// Auto-register
// ---------------------------------------------------------------------------
static struct ThreadedMatmulTestRegistrar {
  ThreadedMatmulTestRegistrar() {
    RegisterTest("matmul_naive_vs_threaded", TestMatmulNaiveVsThreaded);
    RegisterTest("linear_all_modes_match", TestLinearAllModesMatch);
    RegisterTest("linear_2d_input_all_modes_match",
                 TestLinear2dInputAllModesMatch);
    RegisterTest("different_thread_counts_same_output",
                 TestDifferentThreadCountsSameOutput);
    RegisterTest("thread_count_api", TestThreadCountApi);
    RegisterTest("parallel_for_propagates_exception",
                 TestParallelForPropagatesException);
  }
} threaded_matmul_test_registrar;
