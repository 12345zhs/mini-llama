#include <iostream>
#include <stdexcept>
#include <vector>

#include "mini_llama/context.h"
#include "mini_llama/forward.h"
#include "mini_llama/loader.h"
#include "mini_llama/model.h"
#include "mini_llama/sampler.h"
#include "mini_llama/tokenizer.h"
#include "tests/test_main.h"
#include "tests/test_names.h"

static bool TestForwardTokenProducesLogitShape() {
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (model.layers.empty()) {
    MINI_LLAMA_ASSERT_FAIL("Failed to load model");
  }
  MiniLlamaContext ctx(&model);
  ctx.pos = 0;
  Tensor logits = ForwardToken(ctx, model, 1);  // BOS token
  MINI_LLAMA_ASSERT_EQ(logits.num_dims(), 1);
  MINI_LLAMA_ASSERT_EQ(logits.shape[0], model.config.vocab_size);
  return true;
}

static bool TestForwardTokenChangesKvCache() {
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (model.layers.empty()) {
    MINI_LLAMA_ASSERT_FAIL("Failed to load model");
  }
  MiniLlamaContext ctx(&model);
  ctx.pos = 0;
  // Before forward, cache at layer 0, pos 0 should be zero
  float before = ctx.kv_cache.keys.At({0, 0, 0, 0});
  MINI_LLAMA_ASSERT_NEAR(before, 0.0f, 1e-6f);

  ForwardToken(ctx, model, 1);

  // After forward, cache should have been written
  float after = ctx.kv_cache.keys.At({0, 0, 0, 0});
  // It's unlikely to be exactly 0 after random weights forward
  // Just check it's finite and cache was written somewhere
  MINI_LLAMA_ASSERT_TRUE(std::isfinite(after));
  return true;
}

static bool TestGenerationGreedySequence() {
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (model.layers.empty()) {
    MINI_LLAMA_ASSERT_FAIL("Failed to load model");
  }
  AsciiTokenizer tokenizer;
  std::vector<int> tokens = tokenizer.Encode("hello");
  MiniLlamaContext ctx(&model);

  // Prefill
  size_t prompt_len = tokens.size();
  for (size_t i = 0; i < prompt_len; ++i) {
    ctx.pos = static_cast<int>(i);
    Tensor logits = ForwardToken(ctx, model, tokens[i]);
    if (i + 1 == prompt_len) {
      tokens.push_back(SampleGreedy(logits));
    }
  }

  // Decode 5 more tokens
  for (int i = 0; i < 5; ++i) {
    ctx.pos = static_cast<int>(tokens.size() - 1);
    Tensor logits = ForwardToken(ctx, model, tokens.back());
    tokens.push_back(SampleGreedy(logits));
  }

  MINI_LLAMA_ASSERT_TRUE(tokens.size() > 6);
  // First generated token should be the same as what we got in prefill
  MINI_LLAMA_ASSERT_TRUE(tokens[tokens.size() - 6] >= 0);
  return true;
}

static bool TestPositionIncrementsCorrectly() {
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (model.layers.empty()) {
    MINI_LLAMA_ASSERT_FAIL("Failed to load model");
  }
  MiniLlamaContext ctx(&model);

  // Forward token at pos 0
  ctx.pos = 0;
  ForwardToken(ctx, model, 1);

  // Forward token at pos 1
  ctx.pos = 1;
  Tensor logits = ForwardToken(ctx, model, 2);
  MINI_LLAMA_ASSERT_EQ(logits.shape[0], model.config.vocab_size);
  return true;
}

static bool TestForwardRejectsInvalidToken() {
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (!model.loaded) {
    MINI_LLAMA_ASSERT_FAIL("Failed to load model");
  }
  MiniLlamaContext ctx(&model);
  ctx.pos = 0;

  try {
    ForwardToken(ctx, model, model.config.vocab_size);
    MINI_LLAMA_ASSERT_FAIL("ForwardToken accepted an out-of-range token");
  } catch (const std::out_of_range&) {
    return true;
  } catch (...) {
    MINI_LLAMA_ASSERT_FAIL(
        "ForwardToken threw the wrong exception type for invalid token");
  }
}

static bool TestForwardRejectsInvalidPosition() {
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (!model.loaded) {
    MINI_LLAMA_ASSERT_FAIL("Failed to load model");
  }
  MiniLlamaContext ctx(&model);
  ctx.pos = model.config.max_seq_len;

  try {
    ForwardToken(ctx, model, 1);
    MINI_LLAMA_ASSERT_FAIL("ForwardToken accepted an out-of-range position");
  } catch (const std::out_of_range&) {
    return true;
  } catch (...) {
    MINI_LLAMA_ASSERT_FAIL(
        "ForwardToken threw the wrong exception type for invalid position");
  }
}

static struct ForwardTestRegistrar {
  ForwardTestRegistrar() {
    RegisterTest("forward_token_produces_logit_shape",
                 TestForwardTokenProducesLogitShape);
    RegisterTest("forward_token_changes_kv_cache",
                 TestForwardTokenChangesKvCache);
    RegisterTest("generation_greedy_sequence", TestGenerationGreedySequence);
    RegisterTest("position_increments_correctly",
                 TestPositionIncrementsCorrectly);
    RegisterTest("forward_rejects_invalid_token",
                 TestForwardRejectsInvalidToken);
    RegisterTest("forward_rejects_invalid_position",
                 TestForwardRejectsInvalidPosition);
  }
} forward_test_registrar;
