#include "mini_llama/ops.h"
#include "mini_llama/tensor.h"
#include "tests/test_main.h"
#include "tests/test_names.h"

// ==========================================================================
// Matmul
// ==========================================================================
static bool TestMatmulIdentity() {
  Tensor a({2, 2}, 0.0f);
  a[0] = 1.0f;
  a[1] = 0.0f;
  a[2] = 0.0f;
  a[3] = 1.0f;
  Tensor b({2, 2}, 0.0f);
  b[0] = 1.0f;
  b[1] = 2.0f;
  b[2] = 3.0f;
  b[3] = 4.0f;
  Tensor c = Matmul(a, b);
  MINI_LLAMA_ASSERT_EQ(c.shape[0], 2);
  MINI_LLAMA_ASSERT_EQ(c.shape[1], 2);
  MINI_LLAMA_ASSERT_NEAR(c[0], 1.0f, 1e-5f);
  MINI_LLAMA_ASSERT_NEAR(c[1], 2.0f, 1e-5f);
  MINI_LLAMA_ASSERT_NEAR(c[2], 3.0f, 1e-5f);
  MINI_LLAMA_ASSERT_NEAR(c[3], 4.0f, 1e-5f);
  return true;
}

static bool TestMatmulSimple() {
  Tensor a({2, 2}, 0.0f);
  a[0] = 1.0f;
  a[1] = 2.0f;
  a[2] = 3.0f;
  a[3] = 4.0f;
  Tensor b({2, 2}, 0.0f);
  b[0] = 5.0f;
  b[1] = 6.0f;
  b[2] = 7.0f;
  b[3] = 8.0f;
  Tensor c = Matmul(a, b);
  MINI_LLAMA_ASSERT_NEAR(c[0], 19.0f, 1e-5f);
  MINI_LLAMA_ASSERT_NEAR(c[1], 22.0f, 1e-5f);
  MINI_LLAMA_ASSERT_NEAR(c[2], 43.0f, 1e-5f);
  MINI_LLAMA_ASSERT_NEAR(c[3], 50.0f, 1e-5f);
  return true;
}

static bool TestMatmulRectangular() {
  // a: [2, 3], b: [3, 4] -> c: [2, 4]
  Tensor a({2, 3}, 0.0f);
  a[0] = 1.0f;
  a[1] = 2.0f;
  a[2] = 3.0f;
  a[3] = 4.0f;
  a[4] = 5.0f;
  a[5] = 6.0f;
  Tensor b({3, 4}, 0.0f);
  for (int i = 0; i < 12; ++i) {
    b[i] = static_cast<float>(i);
  }
  Tensor c = Matmul(a, b);
  MINI_LLAMA_ASSERT_EQ(c.shape[0], 2);
  MINI_LLAMA_ASSERT_EQ(c.shape[1], 4);
  // c[0,0] = 1*0 + 2*4 + 3*8 = 0 + 8 + 24 = 32
  MINI_LLAMA_ASSERT_NEAR(c.At2(0, 0), 32.0f, 1e-5f);
  // c[1,0] = 4*0 + 5*4 + 6*8 = 0 + 20 + 48 = 68
  MINI_LLAMA_ASSERT_NEAR(c.At2(1, 0), 68.0f, 1e-5f);
  return true;
}

static bool TestMatmulShapeMismatch() {
  Tensor a({2, 3}, 0.0f);
  Tensor b({2, 3}, 0.0f);
  try {
    Matmul(a, b);
    MINI_LLAMA_ASSERT_FAIL("expected exception for Matmul shape mismatch");
  } catch (const std::runtime_error&) {
    // expected
  }
  return true;
}

// ==========================================================================
// Linear
// ==========================================================================
static bool TestLinear1d() {
  Tensor x({3}, 0.0f);
  x[0] = 1.0f;
  x[1] = 2.0f;
  x[2] = 3.0f;
  Tensor weight({3, 3}, 0.0f);
  weight[0] = 1.0f;
  weight[4] = 1.0f;
  weight[8] = 1.0f;
  Tensor y = Linear(x, weight);
  MINI_LLAMA_ASSERT_EQ(y.num_dims(), 1);
  MINI_LLAMA_ASSERT_EQ(y.shape[0], 3);
  MINI_LLAMA_ASSERT_NEAR(y[0], 1.0f, 1e-5f);
  MINI_LLAMA_ASSERT_NEAR(y[1], 2.0f, 1e-5f);
  MINI_LLAMA_ASSERT_NEAR(y[2], 3.0f, 1e-5f);
  return true;
}

static bool TestLinear2d() {
  Tensor x({1, 3}, 0.0f);
  x[0] = 1.0f;
  x[1] = 2.0f;
  x[2] = 3.0f;
  Tensor weight({2, 3}, 0.0f);
  weight[0] = 1.0f;
  weight[1] = 0.0f;
  weight[2] = 0.0f;
  weight[3] = 0.0f;
  weight[4] = 1.0f;
  weight[5] = 0.0f;
  Tensor y = Linear(x, weight);
  MINI_LLAMA_ASSERT_EQ(y.num_dims(), 2);
  MINI_LLAMA_ASSERT_EQ(y.shape[0], 1);
  MINI_LLAMA_ASSERT_EQ(y.shape[1], 2);
  MINI_LLAMA_ASSERT_NEAR(y[0], 1.0f, 1e-5f);
  MINI_LLAMA_ASSERT_NEAR(y[1], 2.0f, 1e-5f);
  return true;
}

static bool TestLinearShapeMismatch() {
  Tensor x({3}, 0.0f);
  Tensor weight({2, 4}, 0.0f);
  try {
    Linear(x, weight);
    MINI_LLAMA_ASSERT_FAIL("expected exception for Linear shape mismatch");
  } catch (const std::runtime_error&) {
    // expected
  }
  return true;
}

// ==========================================================================
// RmsNorm
// ==========================================================================
static bool TestRmsNormUnit() {
  Tensor x({4}, 1.0f);
  Tensor w({4}, 1.0f);
  Tensor y = RmsNorm(x, w, 1e-5f);
  MINI_LLAMA_ASSERT_EQ(y.shape[0], 4);
  for (int i = 0; i < 4; ++i) {
    MINI_LLAMA_ASSERT_NEAR(y[i], 1.0f, 1e-4f);
  }
  return true;
}

static bool TestRmsNormScaled() {
  Tensor x({4}, 0.0f);
  x[0] = 2.0f;
  Tensor w({4}, 1.0f);
  Tensor y = RmsNorm(x, w, 1e-5f);
  MINI_LLAMA_ASSERT_NEAR(y[0], 2.0f, 1e-4f);
  MINI_LLAMA_ASSERT_NEAR(y[1], 0.0f, 1e-4f);
  return true;
}

static bool TestRmsNormWithWeight() {
  // x = [1, 1, 1, 1], weight = [2, 2, 2, 2]
  // rms = 1, scale = 1, y = x * scale * weight = [2, 2, 2, 2]
  Tensor x({4}, 1.0f);
  Tensor w({4}, 2.0f);
  Tensor y = RmsNorm(x, w, 1e-5f);
  for (int i = 0; i < 4; ++i) {
    MINI_LLAMA_ASSERT_NEAR(y[i], 2.0f, 1e-4f);
  }
  return true;
}

static bool TestRmsNormShapeMismatch() {
  Tensor x({4}, 0.0f);
  Tensor w({3}, 0.0f);
  try {
    RmsNorm(x, w, 1e-5f);
    MINI_LLAMA_ASSERT_FAIL("expected exception for RmsNorm shape mismatch");
  } catch (const std::runtime_error&) {
    // expected
  }
  return true;
}

// ==========================================================================
// Softmax
// ==========================================================================
static bool TestSoftmax() {
  Tensor x({3}, 0.0f);
  x[0] = 1.0f;
  x[1] = 2.0f;
  x[2] = 3.0f;
  Tensor y = Softmax(x);
  MINI_LLAMA_ASSERT_EQ(y.shape[0], 3);
  MINI_LLAMA_ASSERT_NEAR(y[0], 0.09003057f, 1e-5f);
  MINI_LLAMA_ASSERT_NEAR(y[1], 0.24472847f, 1e-5f);
  MINI_LLAMA_ASSERT_NEAR(y[2], 0.66524096f, 1e-5f);
  float sum = y[0] + y[1] + y[2];
  MINI_LLAMA_ASSERT_NEAR(sum, 1.0f, 1e-5f);
  return true;
}

static bool TestSoftmaxUniform() {
  // Softmax([0, 0, 0]) = [1/3, 1/3, 1/3]
  Tensor x({3}, 0.0f);
  Tensor y = Softmax(x);
  for (int i = 0; i < 3; ++i) {
    MINI_LLAMA_ASSERT_NEAR(y[i], 1.0f / 3.0f, 1e-5f);
  }
  return true;
}

static bool TestSoftmaxWrongDim() {
  Tensor x({2, 3}, 0.0f);
  try {
    Softmax(x);
    MINI_LLAMA_ASSERT_FAIL("expected exception for Softmax on 2D tensor");
  } catch (const std::runtime_error&) {
    // expected
  }
  return true;
}

static bool TestSoftmaxEmpty() {
  Tensor x;
  x.shape = {0};
  try {
    Softmax(x);
    MINI_LLAMA_ASSERT_FAIL("expected exception for Softmax on empty tensor");
  } catch (const std::runtime_error&) {
    // expected
  }
  return true;
}

// ==========================================================================
// Silu
// ==========================================================================
static bool TestSilu() {
  Tensor x({1}, 0.0f);
  Tensor y = Silu(x);
  MINI_LLAMA_ASSERT_NEAR(y[0], 0.0f, 1e-5f);

  x[0] = 1.0f;
  y = Silu(x);
  MINI_LLAMA_ASSERT_NEAR(y[0], 0.7310586f, 1e-5f);
  return true;
}

// ==========================================================================
// SwiGlu
// ==========================================================================
static bool TestSwiglu() {
  // SwiGlu(gate, up) = Silu(gate) * up
  Tensor gate({3}, 0.0f);
  gate[0] = 0.0f;
  gate[1] = 1.0f;
  gate[2] = 2.0f;
  Tensor up({3}, 0.0f);
  up[0] = 1.0f;
  up[1] = 2.0f;
  up[2] = 3.0f;
  Tensor y = SwiGlu(gate, up);

  // Silu(0) = 0, so SwiGlu(0, 1) = 0
  MINI_LLAMA_ASSERT_NEAR(y[0], 0.0f, 1e-5f);
  // Silu(1) ≈ 0.731, so SwiGlu(1, 2) ≈ 0.731 * 2 ≈ 1.462
  MINI_LLAMA_ASSERT_NEAR(y[1], 0.7310586f * 2.0f, 1e-5f);
  // Silu(2) ≈ 1.761, so SwiGlu(2, 3) ≈ 1.761 * 3 ≈ 5.284
  MINI_LLAMA_ASSERT_NEAR(y[2], 1.7615942f * 3.0f, 1e-4f);
  return true;
}

static bool TestSwigluShapeMismatch() {
  Tensor a({3}, 0.0f);
  Tensor b({4}, 0.0f);
  try {
    SwiGlu(a, b);
    MINI_LLAMA_ASSERT_FAIL("expected exception for SwiGlu shape mismatch");
  } catch (const std::runtime_error&) {
    // expected
  }
  return true;
}

// ==========================================================================
// ElementwiseMul
// ==========================================================================
static bool TestElementwiseMul() {
  Tensor a({3}, 0.0f);
  a[0] = 1.0f;
  a[1] = 2.0f;
  a[2] = 3.0f;
  Tensor b({3}, 0.0f);
  b[0] = 4.0f;
  b[1] = 5.0f;
  b[2] = 6.0f;
  Tensor y = ElementwiseMul(a, b);
  MINI_LLAMA_ASSERT_NEAR(y[0], 4.0f, 1e-5f);
  MINI_LLAMA_ASSERT_NEAR(y[1], 10.0f, 1e-5f);
  MINI_LLAMA_ASSERT_NEAR(y[2], 18.0f, 1e-5f);
  return true;
}

static bool TestElementwiseMulShapeMismatch() {
  Tensor a({3}, 0.0f);
  Tensor b({4}, 0.0f);
  try {
    ElementwiseMul(a, b);
    MINI_LLAMA_ASSERT_FAIL(
        "expected exception for ElementwiseMul shape mismatch");
  } catch (const std::runtime_error&) {
    // expected
  }
  return true;
}

// ==========================================================================
// ArgMax
// ==========================================================================
static bool TestArgMax() {
  Tensor x({5}, 0.0f);
  x[0] = 1.0f;
  x[1] = 5.0f;
  x[2] = 3.0f;
  x[3] = 5.0f;
  x[4] = 2.0f;
  int idx = ArgMax(x);
  MINI_LLAMA_ASSERT_EQ(idx, 1);
  return true;
}

static bool TestArgMaxEmpty() {
  Tensor x;
  try {
    ArgMax(x);
    MINI_LLAMA_ASSERT_FAIL("expected exception for ArgMax on empty tensor");
  } catch (const std::runtime_error&) {
    // expected
  }
  return true;
}

// ==========================================================================
// Rope
// ==========================================================================
static bool TestRopePreservesNorm() {
  int n_heads = 2;
  int head_dim = 4;
  Tensor q({n_heads, head_dim}, 1.0f);
  Tensor k({n_heads, head_dim}, 1.0f);
  Rope(q, k, 5, 10000.0f);
  for (int h = 0; h < n_heads; ++h) {
    float q_norm = 0.0f;
    float k_norm = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      MINI_LLAMA_ASSERT_TRUE(std::isfinite(q.At2(h, d)));
      MINI_LLAMA_ASSERT_TRUE(std::isfinite(k.At2(h, d)));
      q_norm += q.At2(h, d) * q.At2(h, d);
      k_norm += k.At2(h, d) * k.At2(h, d);
    }
    MINI_LLAMA_ASSERT_NEAR(q_norm, 4.0f, 1e-5f);
    MINI_LLAMA_ASSERT_NEAR(k_norm, 4.0f, 1e-5f);
  }
  return true;
}

static bool TestRopeNeoXMatchesSplitHalfPairs() {
  Tensor q({1, 4}, 0.0f);
  Tensor k({1, 4}, 0.0f);
  q[0] = 1.0f;
  q[1] = 2.0f;
  q[2] = 3.0f;
  q[3] = 4.0f;
  k = q;

  Rope(q, k, 1, 10000.0f, RopeType::kNeoX);

  float c0 = std::cos(1.0f);
  float s0 = std::sin(1.0f);
  float c1 = std::cos(0.01f);
  float s1 = std::sin(0.01f);

  MINI_LLAMA_ASSERT_NEAR(q[0], 1.0f * c0 - 3.0f * s0, 1e-5f);
  MINI_LLAMA_ASSERT_NEAR(q[2], 1.0f * s0 + 3.0f * c0, 1e-5f);
  MINI_LLAMA_ASSERT_NEAR(q[1], 2.0f * c1 - 4.0f * s1, 1e-5f);
  MINI_LLAMA_ASSERT_NEAR(q[3], 2.0f * s1 + 4.0f * c1, 1e-5f);
  for (int i = 0; i < 4; ++i) {
    MINI_LLAMA_ASSERT_NEAR(k[i], q[i], 1e-5f);
  }
  return true;
}

static bool TestRopeWrongDim() {
  Tensor q({4}, 0.0f);
  Tensor k({2, 4}, 0.0f);
  try {
    Rope(q, k, 0, 10000.0f);
    MINI_LLAMA_ASSERT_FAIL("expected exception for Rope with 1D q");
  } catch (const std::runtime_error&) {
    // expected
  }
  return true;
}

static bool TestRopeRejectsMismatchedHeadDim() {
  Tensor q({2, 4}, 0.0f);
  Tensor k({2, 6}, 0.0f);
  try {
    Rope(q, k, 0, 10000.0f);
    MINI_LLAMA_ASSERT_FAIL("expected exception for mismatched RoPE head_dim");
  } catch (const std::runtime_error&) {
    // expected
  }
  return true;
}

static bool TestRopeRejectsOddHeadDim() {
  Tensor q({2, 3}, 0.0f);
  Tensor k({2, 3}, 0.0f);
  try {
    Rope(q, k, 0, 10000.0f);
    MINI_LLAMA_ASSERT_FAIL("expected exception for odd RoPE head_dim");
  } catch (const std::runtime_error&) {
    // expected
  }
  return true;
}

static bool TestRopeRejectsNegativePosition() {
  Tensor q({2, 4}, 0.0f);
  Tensor k({2, 4}, 0.0f);
  try {
    Rope(q, k, -1, 10000.0f);
    MINI_LLAMA_ASSERT_FAIL("expected exception for negative RoPE position");
  } catch (const std::out_of_range&) {
    // expected
  }
  return true;
}

static bool TestRopeRejectsNonpositiveTheta() {
  Tensor q({2, 4}, 0.0f);
  Tensor k({2, 4}, 0.0f);
  try {
    Rope(q, k, 0, 0.0f);
    MINI_LLAMA_ASSERT_FAIL("expected exception for nonpositive RoPE theta");
  } catch (const std::runtime_error&) {
    // expected
  }
  return true;
}

static struct OpsTestRegistrar {
  OpsTestRegistrar() {
    RegisterTest("matmul_identity", TestMatmulIdentity);
    RegisterTest("matmul_simple", TestMatmulSimple);
    RegisterTest("matmul_rectangular", TestMatmulRectangular);
    RegisterTest("matmul_shape_mismatch", TestMatmulShapeMismatch);
    RegisterTest("linear_1d", TestLinear1d);
    RegisterTest("linear_2d", TestLinear2d);
    RegisterTest("linear_shape_mismatch", TestLinearShapeMismatch);
    RegisterTest("rmsnorm_unit", TestRmsNormUnit);
    RegisterTest("rmsnorm_scaled", TestRmsNormScaled);
    RegisterTest("rmsnorm_with_weight", TestRmsNormWithWeight);
    RegisterTest("rmsnorm_shape_mismatch", TestRmsNormShapeMismatch);
    RegisterTest("Softmax", TestSoftmax);
    RegisterTest("softmax_uniform", TestSoftmaxUniform);
    RegisterTest("softmax_wrong_dim", TestSoftmaxWrongDim);
    RegisterTest("softmax_empty", TestSoftmaxEmpty);
    RegisterTest("Silu", TestSilu);
    RegisterTest("SwiGlu", TestSwiglu);
    RegisterTest("swiglu_shape_mismatch", TestSwigluShapeMismatch);
    RegisterTest("ElementwiseMul", TestElementwiseMul);
    RegisterTest("elementwise_mul_shape_mismatch",
                 TestElementwiseMulShapeMismatch);
    RegisterTest("ArgMax", TestArgMax);
    RegisterTest("argmax_empty", TestArgMaxEmpty);
    RegisterTest("rope_preserves_norm", TestRopePreservesNorm);
    RegisterTest("rope_neox_matches_split_half_pairs",
                 TestRopeNeoXMatchesSplitHalfPairs);
    RegisterTest("rope_wrong_dim", TestRopeWrongDim);
    RegisterTest("rope_rejects_mismatched_head_dim",
                 TestRopeRejectsMismatchedHeadDim);
    RegisterTest("rope_rejects_odd_head_dim", TestRopeRejectsOddHeadDim);
    RegisterTest("rope_rejects_negative_position",
                 TestRopeRejectsNegativePosition);
    RegisterTest("rope_rejects_nonpositive_theta",
                 TestRopeRejectsNonpositiveTheta);
  }
} ops_test_registrar;
