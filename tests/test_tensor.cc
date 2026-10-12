#include "include/mini_llama/tensor.h"
#include "tests/test_main.h"
#include "tests/test_names.h"

// ---------------------------------------------------------------------------
// Basic construction and properties
// ---------------------------------------------------------------------------
static bool TestTensorShapeAndSize() {
  Tensor t1({3, 4, 5}, 0.0f);
  MINI_LLAMA_ASSERT_EQ(t1.num_dims(), 3);
  MINI_LLAMA_ASSERT_EQ(t1.size(), 60);
  MINI_LLAMA_ASSERT_EQ(t1.num_elements(), 60);
  MINI_LLAMA_ASSERT_EQ(t1.shape[0], 3);
  MINI_LLAMA_ASSERT_EQ(t1.shape[1], 4);
  MINI_LLAMA_ASSERT_EQ(t1.shape[2], 5);
  return true;
}

static bool TestTensorFill() {
  Tensor t({10}, 3.14f);
  MINI_LLAMA_ASSERT_EQ(t.size(), 10);
  for (size_t i = 0; i < t.size(); ++i) {
    MINI_LLAMA_ASSERT_NEAR(t.data[i], 3.14f, 1e-6f);
  }
  return true;
}

static bool TestTensorRejectsZeroDimension() {
  try {
    Tensor t({2, 0}, 0.0f);
    MINI_LLAMA_ASSERT_FAIL("expected exception for zero tensor dimension");
  } catch (const std::runtime_error&) {
    // expected
  }
  return true;
}

static bool TestTensorRejectsNegativeDimension() {
  try {
    Tensor t({2, -3}, 0.0f);
    MINI_LLAMA_ASSERT_FAIL("expected exception for negative tensor dimension");
  } catch (const std::runtime_error&) {
    // expected
  }
  return true;
}

static bool TestTensorIndexing() {
  Tensor t({2, 3}, 0.0f);
  t.At({0, 0}) = 1.0f;
  t.At({0, 1}) = 2.0f;
  t.At({1, 2}) = 6.0f;
  MINI_LLAMA_ASSERT_NEAR(t.At({0, 0}), 1.0f, 1e-6f);
  MINI_LLAMA_ASSERT_NEAR(t.At({0, 1}), 2.0f, 1e-6f);
  MINI_LLAMA_ASSERT_NEAR(t.At({1, 2}), 6.0f, 1e-6f);
  return true;
}

static bool TestTensorIndexWrongRank() {
  Tensor t({2, 3}, 0.0f);
  try {
    t.At({1});
    MINI_LLAMA_ASSERT_FAIL("expected exception for wrong FlatIndex rank");
  } catch (const std::runtime_error&) {
    // expected
  }
  return true;
}

static bool TestTensorIndexNegative() {
  Tensor t({2, 3}, 0.0f);
  try {
    t.At({-1, 0});
    MINI_LLAMA_ASSERT_FAIL("expected exception for negative FlatIndex");
  } catch (const std::out_of_range&) {
    // expected
  }
  return true;
}

static bool TestTensorIndexTooLarge() {
  Tensor t({2, 3}, 0.0f);
  try {
    t.At({0, 3});
    MINI_LLAMA_ASSERT_FAIL(
        "expected exception for FlatIndex above shape bound");
  } catch (const std::out_of_range&) {
    // expected
  }
  return true;
}

static bool TestTensor1dAccess() {
  Tensor t({5}, 0.0f);
  t[2] = 42.0f;
  MINI_LLAMA_ASSERT_NEAR(t[2], 42.0f, 1e-6f);
  return true;
}

static bool TestMakeTensorHelpers() {
  auto t1 = MakeTensor1D(10, 1.0f);
  MINI_LLAMA_ASSERT_EQ(t1.num_dims(), 1);
  MINI_LLAMA_ASSERT_EQ(t1.shape[0], 10);

  auto t2 = MakeTensor2D(3, 4, 1.0f);
  MINI_LLAMA_ASSERT_EQ(t2.num_dims(), 2);
  MINI_LLAMA_ASSERT_EQ(t2.shape[0], 3);
  MINI_LLAMA_ASSERT_EQ(t2.shape[1], 4);

  auto t3 = MakeTensor3D(2, 3, 4, 1.0f);
  MINI_LLAMA_ASSERT_EQ(t3.num_dims(), 3);
  MINI_LLAMA_ASSERT_EQ(t3.size(), 24);

  auto t4 = MakeTensor4D(2, 3, 4, 5, 1.0f);
  MINI_LLAMA_ASSERT_EQ(t4.num_dims(), 4);
  MINI_LLAMA_ASSERT_EQ(t4.size(), 120);
  return true;
}

// ---------------------------------------------------------------------------
// At1 / At2 / At3 / At4 convenience accessors
// ---------------------------------------------------------------------------
static bool TestTensorAt1() {
  Tensor t({5}, 0.0f);
  t.At1(2) = 7.0f;
  MINI_LLAMA_ASSERT_NEAR(t.At1(2), 7.0f, 1e-6f);
  return true;
}

static bool TestTensorAt1WrongDim() {
  Tensor t({2, 3}, 0.0f);
  try {
    t.At1(0);
    MINI_LLAMA_ASSERT_FAIL("expected exception for At1 on 2D tensor");
  } catch (const std::runtime_error&) {
    // expected
  }
  return true;
}

static bool TestTensorAt2() {
  Tensor t({2, 3}, 0.0f);
  t.At2(0, 1) = 5.0f;
  t.At2(1, 2) = 9.0f;
  MINI_LLAMA_ASSERT_NEAR(t.At2(0, 1), 5.0f, 1e-6f);
  MINI_LLAMA_ASSERT_NEAR(t.At2(1, 2), 9.0f, 1e-6f);
  return true;
}

static bool TestTensorAt2OutOfRange() {
  Tensor t({2, 3}, 0.0f);
  try {
    t.At2(2, 0);
    MINI_LLAMA_ASSERT_FAIL("expected exception for At2 row out of range");
  } catch (const std::out_of_range&) {
    // expected
  }
  return true;
}

static bool TestTensorAt3() {
  Tensor t({2, 2, 2}, 0.0f);
  t.At3(1, 0, 1) = 3.0f;
  MINI_LLAMA_ASSERT_NEAR(t.At3(1, 0, 1), 3.0f, 1e-6f);
  return true;
}

static bool TestTensorAt4() {
  Tensor t({2, 2, 2, 2}, 0.0f);
  t.At4(1, 1, 0, 0) = 4.0f;
  MINI_LLAMA_ASSERT_NEAR(t.At4(1, 1, 0, 0), 4.0f, 1e-6f);
  return true;
}

// ---------------------------------------------------------------------------
// row_ptr
// ---------------------------------------------------------------------------
static bool TestTensorRowPtr() {
  Tensor t({3, 4}, 0.0f);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      t.At2(i, j) = static_cast<float>(i * 10 + j);
    }
  }
  const float* row1 = t.RowPtr(1);
  MINI_LLAMA_ASSERT_NEAR(row1[0], 10.0f, 1e-6f);
  MINI_LLAMA_ASSERT_NEAR(row1[3], 13.0f, 1e-6f);
  return true;
}

static bool TestTensorRowPtrWrongDim() {
  Tensor t({5}, 0.0f);
  try {
    t.RowPtr(0);
    MINI_LLAMA_ASSERT_FAIL("expected exception for row_ptr on 1D tensor");
  } catch (const std::runtime_error&) {
    // expected
  }
  return true;
}

static bool TestTensorRowPtrOutOfRange() {
  Tensor t({3, 4}, 0.0f);
  try {
    t.RowPtr(3);
    MINI_LLAMA_ASSERT_FAIL("expected exception for row_ptr row out of range");
  } catch (const std::out_of_range&) {
    // expected
  }
  return true;
}

// ---------------------------------------------------------------------------
// AssertShape
// ---------------------------------------------------------------------------
static bool TestTensorAssertShapePass() {
  Tensor t({2, 3, 4}, 0.0f);
  t.AssertShape({2, 3, 4}, "TestPass");
  return true;
}

static bool TestTensorAssertShapeFail() {
  Tensor t({2, 3, 4}, 0.0f);
  try {
    t.AssertShape({2, 3, 5}, "TestFail");
    MINI_LLAMA_ASSERT_FAIL("expected exception for shape mismatch");
  } catch (const std::runtime_error&) {
    // expected
  }
  return true;
}

// ---------------------------------------------------------------------------
// ReshapeChecked
// ---------------------------------------------------------------------------
static bool TestTensorReshapeCheckedPass() {
  Tensor t({2, 3}, 0.0f);
  for (int i = 0; i < 6; ++i) {
    t[i] = static_cast<float>(i);
  }
  Tensor r = t.ReshapeChecked({3, 2}, "TestReshape");
  MINI_LLAMA_ASSERT_EQ(r.num_dims(), 2);
  MINI_LLAMA_ASSERT_EQ(r.shape[0], 3);
  MINI_LLAMA_ASSERT_EQ(r.shape[1], 2);
  MINI_LLAMA_ASSERT_NEAR(r.At2(0, 0), 0.0f, 1e-6f);
  MINI_LLAMA_ASSERT_NEAR(r.At2(2, 1), 5.0f, 1e-6f);
  return true;
}

static bool TestTensorReshapeCheckedFail() {
  Tensor t({2, 3}, 0.0f);
  try {
    t.ReshapeChecked({4, 2}, "TestReshapeFail");
    MINI_LLAMA_ASSERT_FAIL("expected exception for incompatible reshape");
  } catch (const std::runtime_error&) {
    // expected
  }
  return true;
}

static bool TestTensorReshapeCheckedNegativeDim() {
  Tensor t({2, 3}, 0.0f);
  try {
    t.ReshapeChecked({-1, 6}, "TestReshapeNeg");
    MINI_LLAMA_ASSERT_FAIL("expected exception for negative dimension");
  } catch (const std::runtime_error&) {
    // expected
  }
  return true;
}

// ---------------------------------------------------------------------------
// ShapeString alias
// ---------------------------------------------------------------------------
static bool TestTensorShapeStringAlias() {
  Tensor t({2, 3, 4}, 0.0f);
  MINI_LLAMA_ASSERT_TRUE(t.ShapeString() == t.ShapeString());
  return true;
}

// ---------------------------------------------------------------------------
// Auto-register
// ---------------------------------------------------------------------------
static struct TensorTestRegistrar {
  TensorTestRegistrar() {
    RegisterTest("tensor_shape_and_size", TestTensorShapeAndSize);
    RegisterTest("tensor_fill", TestTensorFill);
    RegisterTest("tensor_rejects_zero_dimension",
                 TestTensorRejectsZeroDimension);
    RegisterTest("tensor_rejects_negative_dimension",
                 TestTensorRejectsNegativeDimension);
    RegisterTest("tensor_indexing", TestTensorIndexing);
    RegisterTest("tensor_index_wrong_rank", TestTensorIndexWrongRank);
    RegisterTest("tensor_index_negative", TestTensorIndexNegative);
    RegisterTest("tensor_index_too_large", TestTensorIndexTooLarge);
    RegisterTest("tensor_1d_access", TestTensor1dAccess);
    RegisterTest("make_tensor_helpers", TestMakeTensorHelpers);
    RegisterTest("tensor_at1", TestTensorAt1);
    RegisterTest("tensor_at1_wrong_dim", TestTensorAt1WrongDim);
    RegisterTest("tensor_at2", TestTensorAt2);
    RegisterTest("tensor_at2_out_of_range", TestTensorAt2OutOfRange);
    RegisterTest("tensor_at3", TestTensorAt3);
    RegisterTest("tensor_at4", TestTensorAt4);
    RegisterTest("tensor_row_ptr", TestTensorRowPtr);
    RegisterTest("tensor_row_ptr_wrong_dim", TestTensorRowPtrWrongDim);
    RegisterTest("tensor_row_ptr_out_of_range", TestTensorRowPtrOutOfRange);
    RegisterTest("tensor_assert_shape_pass", TestTensorAssertShapePass);
    RegisterTest("tensor_assert_shape_fail", TestTensorAssertShapeFail);
    RegisterTest("tensor_reshape_checked_pass", TestTensorReshapeCheckedPass);
    RegisterTest("tensor_reshape_checked_fail", TestTensorReshapeCheckedFail);
    RegisterTest("tensor_reshape_checked_negative_dim",
                 TestTensorReshapeCheckedNegativeDim);
    RegisterTest("tensor_shape_string_alias", TestTensorShapeStringAlias);
  }
} tensor_test_registrar;
