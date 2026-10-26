#include <vector>

#include "mini_llama/context.h"
#include "mini_llama/kv_cache.h"
#include "mini_llama/model.h"
#include "tests/test_main.h"
#include "tests/test_names.h"

static bool TestKvCacheConstructsExpectedStorageShape() {
  KvCache cache(2, 8, 3, 4);

  MINI_LLAMA_ASSERT_TRUE(cache.keys.shape == std::vector<int>({2, 8, 3, 4}));
  MINI_LLAMA_ASSERT_TRUE(cache.values.shape == std::vector<int>({2, 8, 3, 4}));
  MINI_LLAMA_ASSERT_EQ(cache.keys.size(), 192);
  MINI_LLAMA_ASSERT_EQ(cache.values.size(), 192);
  for (size_t i = 0; i < cache.keys.size(); ++i) {
    MINI_LLAMA_ASSERT_NEAR(cache.keys.data[i], 0.0f, 1e-6f);
    MINI_LLAMA_ASSERT_NEAR(cache.values.data[i], 0.0f, 1e-6f);
  }
  return true;
}

static bool TestKvCacheWriteAndPtrReadback() {
  KvCache cache(2, 4, 2, 3);
  Tensor k({2, 3}, 0.0f);
  Tensor v({2, 3}, 0.0f);

  for (int i = 0; i < 6; ++i) {
    k[i] = static_cast<float>(i + 1);
    v[i] = static_cast<float>(100 + i);
  }

  cache.Write(1, 2, k, v);

  const float* key_head_0 = cache.KeyPtr(1, 2, 0);
  const float* key_head_1 = cache.KeyPtr(1, 2, 1);
  const float* value_head_0 = cache.ValuePtr(1, 2, 0);
  const float* value_head_1 = cache.ValuePtr(1, 2, 1);

  MINI_LLAMA_ASSERT_NEAR(key_head_0[0], 1.0f, 1e-6f);
  MINI_LLAMA_ASSERT_NEAR(key_head_0[1], 2.0f, 1e-6f);
  MINI_LLAMA_ASSERT_NEAR(key_head_0[2], 3.0f, 1e-6f);
  MINI_LLAMA_ASSERT_NEAR(key_head_1[0], 4.0f, 1e-6f);
  MINI_LLAMA_ASSERT_NEAR(key_head_1[1], 5.0f, 1e-6f);
  MINI_LLAMA_ASSERT_NEAR(key_head_1[2], 6.0f, 1e-6f);

  MINI_LLAMA_ASSERT_NEAR(value_head_0[0], 100.0f, 1e-6f);
  MINI_LLAMA_ASSERT_NEAR(value_head_0[1], 101.0f, 1e-6f);
  MINI_LLAMA_ASSERT_NEAR(value_head_0[2], 102.0f, 1e-6f);
  MINI_LLAMA_ASSERT_NEAR(value_head_1[0], 103.0f, 1e-6f);
  MINI_LLAMA_ASSERT_NEAR(value_head_1[1], 104.0f, 1e-6f);
  MINI_LLAMA_ASSERT_NEAR(value_head_1[2], 105.0f, 1e-6f);
  return true;
}

static bool TestKvCacheWritePreservesOtherPositions() {
  KvCache cache(1, 3, 1, 2);
  Tensor k({1, 2}, 0.0f);
  Tensor v({1, 2}, 0.0f);
  k[0] = 7.0f;
  k[1] = 8.0f;
  v[0] = 9.0f;
  v[1] = 10.0f;

  cache.Write(0, 1, k, v);

  const float* untouched_key = cache.KeyPtr(0, 0, 0);
  const float* untouched_value = cache.ValuePtr(0, 2, 0);
  MINI_LLAMA_ASSERT_NEAR(untouched_key[0], 0.0f, 1e-6f);
  MINI_LLAMA_ASSERT_NEAR(untouched_key[1], 0.0f, 1e-6f);
  MINI_LLAMA_ASSERT_NEAR(untouched_value[0], 0.0f, 1e-6f);
  MINI_LLAMA_ASSERT_NEAR(untouched_value[1], 0.0f, 1e-6f);
  return true;
}

static bool TestKvCacheRejectsBadIndices() {
  KvCache cache(1, 2, 1, 2);
  Tensor k({1, 2}, 0.0f);
  Tensor v({1, 2}, 0.0f);

  try {
    cache.Write(-1, 0, k, v);
    MINI_LLAMA_ASSERT_FAIL("expected layer underflow exception");
  } catch (const std::out_of_range&) {
  }
  try {
    cache.Write(0, 2, k, v);
    MINI_LLAMA_ASSERT_FAIL("expected position overflow exception");
  } catch (const std::out_of_range&) {
  }
  try {
    cache.KeyPtr(0, 0, 1);
    MINI_LLAMA_ASSERT_FAIL("expected head overflow exception");
  } catch (const std::out_of_range&) {
  }
  try {
    cache.ValuePtr(0, -1, 0);
    MINI_LLAMA_ASSERT_FAIL("expected negative position exception");
  } catch (const std::out_of_range&) {
  }
  return true;
}

static bool TestKvCacheRejectsBadTensorShapes() {
  KvCache cache(1, 2, 1, 2);
  Tensor good({1, 2}, 0.0f);
  Tensor rank_3({1, 1, 2}, 0.0f);
  Tensor wrong_heads({2, 2}, 0.0f);
  Tensor wrong_dim({1, 3}, 0.0f);

  try {
    cache.Write(0, 0, rank_3, good);
    MINI_LLAMA_ASSERT_FAIL("expected rank mismatch exception");
  } catch (const std::out_of_range&) {
  }
  try {
    cache.Write(0, 0, wrong_heads, wrong_heads);
    MINI_LLAMA_ASSERT_FAIL("expected head count mismatch exception");
  } catch (const std::out_of_range&) {
  }
  try {
    cache.Write(0, 0, wrong_dim, wrong_dim);
    MINI_LLAMA_ASSERT_FAIL("expected head dim mismatch exception");
  } catch (const std::out_of_range&) {
  }
  return true;
}

static bool TestContextInitializesCacheFromModelConfig() {
  MiniLlamaModel model;
  model.config.n_layers = 3;
  model.config.max_seq_len = 16;
  model.config.n_kv_heads = 2;
  model.config.head_dim = 5;

  MiniLlamaContext ctx(&model);

  MINI_LLAMA_ASSERT_TRUE(ctx.model == &model);
  MINI_LLAMA_ASSERT_EQ(ctx.pos, 0);
  MINI_LLAMA_ASSERT_TRUE(ctx.token_history.empty());
  MINI_LLAMA_ASSERT_EQ(ctx.n_prefill_tokens, 0);
  MINI_LLAMA_ASSERT_EQ(ctx.n_decode_tokens, 0);
  MINI_LLAMA_ASSERT_TRUE(ctx.kv_cache.keys.shape ==
                         std::vector<int>({3, 16, 2, 5}));
  MINI_LLAMA_ASSERT_TRUE(ctx.kv_cache.values.shape ==
                         std::vector<int>({3, 16, 2, 5}));
  return true;
}

static struct KvCacheTestRegistrar {
  KvCacheTestRegistrar() {
    RegisterTest("kv_cache_constructs_expected_storage_shape",
                 TestKvCacheConstructsExpectedStorageShape);
    RegisterTest("kv_cache_write_and_ptr_readback",
                 TestKvCacheWriteAndPtrReadback);
    RegisterTest("kv_cache_write_preserves_other_positions",
                 TestKvCacheWritePreservesOtherPositions);
    RegisterTest("kv_cache_rejects_bad_indices", TestKvCacheRejectsBadIndices);
    RegisterTest("kv_cache_rejects_bad_tensor_shapes",
                 TestKvCacheRejectsBadTensorShapes);
    RegisterTest("context_initializes_cache_from_model_config",
                 TestContextInitializesCacheFromModelConfig);
  }
} kv_cache_test_registrar;
