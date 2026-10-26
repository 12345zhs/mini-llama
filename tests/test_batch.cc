#include <vector>

#include "mini_llama/batch.h"
#include "mini_llama/context.h"
#include "mini_llama/forward.h"
#include "mini_llama/loader.h"
#include "mini_llama/model.h"
#include "tests/test_main.h"
#include "tests/test_names.h"

// ---------------------------------------------------------------------------
// MiniBatch construction
// ---------------------------------------------------------------------------
static bool TestBatchSingle() {
  MiniBatch b = MiniBatch::Single(42, 5);
  MINI_LLAMA_ASSERT_EQ(b.num_tokens(), 1);
  MINI_LLAMA_ASSERT_EQ(b.tokens[0], 42);
  MINI_LLAMA_ASSERT_EQ(b.positions[0], 5);
  return true;
}

static bool TestBatchFromTokens() {
  std::vector<int> toks = {10, 20, 30};
  MiniBatch b = MiniBatch::FromTokens(toks, 0);
  MINI_LLAMA_ASSERT_EQ(b.num_tokens(), 3);
  MINI_LLAMA_ASSERT_EQ(b.tokens[0], 10);
  MINI_LLAMA_ASSERT_EQ(b.tokens[1], 20);
  MINI_LLAMA_ASSERT_EQ(b.tokens[2], 30);
  MINI_LLAMA_ASSERT_EQ(b.positions[0], 0);
  MINI_LLAMA_ASSERT_EQ(b.positions[1], 1);
  MINI_LLAMA_ASSERT_EQ(b.positions[2], 2);
  return true;
}

static bool TestBatchFromTokensWithOffset() {
  std::vector<int> toks = {5, 6};
  MiniBatch b = MiniBatch::FromTokens(toks, 10);
  MINI_LLAMA_ASSERT_EQ(b.positions[0], 10);
  MINI_LLAMA_ASSERT_EQ(b.positions[1], 11);
  return true;
}

static bool TestBatchEmpty() {
  MiniBatch b;
  MINI_LLAMA_ASSERT_EQ(b.num_tokens(), 0);
  return true;
}

// ---------------------------------------------------------------------------
// ForwardBatch
// ---------------------------------------------------------------------------
static bool TestForwardBatchPrefillMatchesIndividual() {
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (!model.loaded) {
    MINI_LLAMA_ASSERT_FAIL("failed to load model");
  }

  std::vector<int> tokens = {1, 104, 101, 108, 108, 111};

  // Individual forward
  MiniLlamaContext ctx1(&model);
  Tensor logits1;
  for (size_t i = 0; i < tokens.size(); ++i) {
    ctx1.pos = static_cast<int>(i);
    logits1 = ForwardToken(ctx1, model, tokens[i]);
  }

  // Batch forward
  MiniLlamaContext ctx2(&model);
  MiniBatch batch = MiniBatch::FromTokens(tokens, 0);
  Tensor logits2 = ForwardBatch(ctx2, model, batch);

  MINI_LLAMA_ASSERT_EQ(logits1.num_dims(), 1);
  MINI_LLAMA_ASSERT_EQ(logits2.num_dims(), 1);
  MINI_LLAMA_ASSERT_EQ(logits1.shape[0], logits2.shape[0]);
  for (size_t i = 0; i < logits1.size(); ++i) {
    MINI_LLAMA_ASSERT_NEAR(logits1.data[i], logits2.data[i], 1e-6f);
  }
  return true;
}

static bool TestForwardBatchSingleMatchesIndividual() {
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (!model.loaded) {
    MINI_LLAMA_ASSERT_FAIL("failed to load model");
  }

  // Individual forward
  MiniLlamaContext ctx1(&model);
  ctx1.pos = 3;
  Tensor logits1 = ForwardToken(ctx1, model, 42);

  // Batch forward with single token
  MiniLlamaContext ctx2(&model);
  MiniBatch batch = MiniBatch::Single(42, 3);
  Tensor logits2 = ForwardBatch(ctx2, model, batch);

  MINI_LLAMA_ASSERT_EQ(logits1.num_dims(), 1);
  MINI_LLAMA_ASSERT_EQ(logits2.num_dims(), 1);
  MINI_LLAMA_ASSERT_EQ(logits1.shape[0], logits2.shape[0]);
  for (size_t i = 0; i < logits1.size(); ++i) {
    MINI_LLAMA_ASSERT_NEAR(logits1.data[i], logits2.data[i], 1e-6f);
  }
  return true;
}

static bool TestForwardBatchUpdatesContext() {
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (!model.loaded) {
    MINI_LLAMA_ASSERT_FAIL("failed to load model");
  }

  std::vector<int> tokens = {1, 104, 101};
  MiniLlamaContext ctx(&model);
  MiniBatch batch = MiniBatch::FromTokens(tokens, 0);
  ForwardBatch(ctx, model, batch);

  MINI_LLAMA_ASSERT_EQ(ctx.token_history.size(), tokens.size());
  for (size_t i = 0; i < tokens.size(); ++i) {
    MINI_LLAMA_ASSERT_EQ(ctx.token_history[i], tokens[i]);
  }
  MINI_LLAMA_ASSERT_EQ(ctx.pos, static_cast<int>(tokens.size()) - 1);
  return true;
}

static bool TestForwardBatchAppendsDecodeHistory() {
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (!model.loaded) {
    MINI_LLAMA_ASSERT_FAIL("failed to load model");
  }

  MiniLlamaContext ctx(&model);
  MiniBatch prefill = MiniBatch::FromTokens({1, 104}, 0);
  ForwardBatch(ctx, model, prefill);

  MiniBatch decode = MiniBatch::Single(101, 2);
  ForwardBatch(ctx, model, decode);

  MINI_LLAMA_ASSERT_EQ(ctx.token_history.size(), 3);
  MINI_LLAMA_ASSERT_EQ(ctx.token_history[0], 1);
  MINI_LLAMA_ASSERT_EQ(ctx.token_history[1], 104);
  MINI_LLAMA_ASSERT_EQ(ctx.token_history[2], 101);
  MINI_LLAMA_ASSERT_EQ(ctx.pos, 2);
  return true;
}

static bool TestForwardBatchEmptyRejected() {
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (!model.loaded) {
    MINI_LLAMA_ASSERT_FAIL("failed to load model");
  }

  MiniLlamaContext ctx(&model);
  MiniBatch batch;
  try {
    ForwardBatch(ctx, model, batch);
    MINI_LLAMA_ASSERT_FAIL("expected exception for empty batch");
  } catch (const std::runtime_error&) {
    // expected
  }
  return true;
}

static bool TestForwardBatchMismatchedSizesRejected() {
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (!model.loaded) {
    MINI_LLAMA_ASSERT_FAIL("failed to load model");
  }

  MiniLlamaContext ctx(&model);
  MiniBatch batch;
  batch.tokens = {1, 2};
  batch.positions = {0};
  try {
    ForwardBatch(ctx, model, batch);
    MINI_LLAMA_ASSERT_FAIL("expected exception for size mismatch");
  } catch (const std::runtime_error&) {
    // expected
  }
  return true;
}

// ---------------------------------------------------------------------------
// Auto-register
// ---------------------------------------------------------------------------
static struct BatchTestRegistrar {
  BatchTestRegistrar() {
    RegisterTest("batch_single", TestBatchSingle);
    RegisterTest("batch_from_tokens", TestBatchFromTokens);
    RegisterTest("batch_from_tokens_with_offset",
                 TestBatchFromTokensWithOffset);
    RegisterTest("batch_empty", TestBatchEmpty);
    RegisterTest("forward_batch_prefill_matches_individual",
                 TestForwardBatchPrefillMatchesIndividual);
    RegisterTest("forward_batch_single_matches_individual",
                 TestForwardBatchSingleMatchesIndividual);
    RegisterTest("forward_batch_updates_context",
                 TestForwardBatchUpdatesContext);
    RegisterTest("forward_batch_appends_decode_history",
                 TestForwardBatchAppendsDecodeHistory);
    RegisterTest("forward_batch_empty_rejected", TestForwardBatchEmptyRejected);
    RegisterTest("forward_batch_mismatched_sizes_rejected",
                 TestForwardBatchMismatchedSizesRejected);
  }
} batch_test_registrar;
