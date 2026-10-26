#include <vector>

#include "mini_llama/debug.h"
#include "mini_llama/kv_cache.h"
#include "mini_llama/loader.h"
#include "mini_llama/model.h"
#include "mini_llama/tensor.h"
#include "mini_llama/tokenizer.h"
#include "tests/test_main.h"
#include "tests/test_names.h"

// ---------------------------------------------------------------------------
// BenchmarkResult
// ---------------------------------------------------------------------------
static bool TestBenchmarkResultZero() {
  BenchmarkResult r;
  MINI_LLAMA_ASSERT_EQ(r.n_prompt_tokens, 0);
  MINI_LLAMA_ASSERT_EQ(r.n_generated_tokens, 0);
  MINI_LLAMA_ASSERT_EQ(r.n_decode_tokens, 0);
  MINI_LLAMA_ASSERT_NEAR(r.tokens_per_sec(), 0.0, 1e-6);
  MINI_LLAMA_ASSERT_NEAR(r.decode_tokens_per_sec(), 0.0, 1e-6);
  return true;
}

static bool TestBenchmarkResultTokensPerSec() {
  BenchmarkResult r;
  r.n_prompt_tokens = 10;
  r.n_generated_tokens = 100;
  r.n_decode_tokens = 100;
  r.prefill_ms = 50.0;
  r.decode_ms = 450.0;
  MINI_LLAMA_ASSERT_NEAR(r.tokens_per_sec(), 200.0, 1e-6);  // 100 / 0.5
  MINI_LLAMA_ASSERT_NEAR(r.decode_tokens_per_sec(), 1000.0 / 4.5, 1e-6);
  return true;
}

static bool TestBenchmarkResultNoDecode() {
  BenchmarkResult r;
  r.n_generated_tokens = 0;
  r.n_decode_tokens = 0;
  r.prefill_ms = 100.0;
  r.decode_ms = 0.0;
  MINI_LLAMA_ASSERT_NEAR(r.tokens_per_sec(), 0.0, 1e-6);
  MINI_LLAMA_ASSERT_NEAR(r.decode_tokens_per_sec(), 0.0, 1e-6);
  return true;
}

// ---------------------------------------------------------------------------
// DumpTensorShape (smoke test: just verify no crash)
// ---------------------------------------------------------------------------
static bool TestDumpTensorShape() {
  Tensor t({2, 3, 4}, 1.0f);
  DumpTensorShape(t, "TestTensor");
  return true;
}

// ---------------------------------------------------------------------------
// DumpLogitsTopK (smoke test)
// ---------------------------------------------------------------------------
static bool TestDumpLogitsTopK() {
  Tensor logits({10}, 0.0f);
  for (int i = 0; i < 10; ++i) {
    logits.data[i] = static_cast<float>(i);
  }
  DumpLogitsTopK(logits, 3);
  return true;
}

static bool TestDumpLogitsTopKLargerThanVocab() {
  Tensor logits({5}, 0.0f);
  for (int i = 0; i < 5; ++i) {
    logits.data[i] = static_cast<float>(i);
  }
  DumpLogitsTopK(logits, 10);  // k > vocab_size
  return true;
}

// ---------------------------------------------------------------------------
// DumpKvCacheInfo (smoke test)
// ---------------------------------------------------------------------------
static bool TestDumpKvCacheInfo() {
  KvCache cache(2, 16, 4, 8);
  DumpKvCacheInfo(cache);
  return true;
}

// ---------------------------------------------------------------------------
// RunBenchmark (integration smoke test)
// ---------------------------------------------------------------------------
static bool TestRunBenchmarkSmoke() {
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (!model.loaded) {
    MINI_LLAMA_ASSERT_FAIL("failed to load model");
  }

  AsciiTokenizer tokenizer;
  std::vector<int> tokens = tokenizer.Encode("hi");
  BenchmarkResult result = RunBenchmark(model, tokens, 4, 42, false);

  MINI_LLAMA_ASSERT_EQ(result.n_prompt_tokens, static_cast<int>(tokens.size()));
  MINI_LLAMA_ASSERT_EQ(result.n_generated_tokens, 4);
  MINI_LLAMA_ASSERT_EQ(result.n_decode_tokens, 3);
  MINI_LLAMA_ASSERT_TRUE(result.prefill_ms >= 0.0);
  MINI_LLAMA_ASSERT_TRUE(result.decode_ms >= 0.0);
  MINI_LLAMA_ASSERT_TRUE(result.tokens_per_sec() >= 0.0);
  return true;
}

static bool TestRunBenchmarkZeroPredict() {
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (!model.loaded) {
    MINI_LLAMA_ASSERT_FAIL("failed to load model");
  }

  AsciiTokenizer tokenizer;
  std::vector<int> tokens = tokenizer.Encode("hi");
  BenchmarkResult result = RunBenchmark(model, tokens, 0, 42, false);

  MINI_LLAMA_ASSERT_EQ(result.n_prompt_tokens, static_cast<int>(tokens.size()));
  MINI_LLAMA_ASSERT_EQ(result.n_generated_tokens, 0);
  MINI_LLAMA_ASSERT_EQ(result.n_decode_tokens, 0);
  MINI_LLAMA_ASSERT_TRUE(result.prefill_ms >= 0.0);
  MINI_LLAMA_ASSERT_TRUE(result.decode_ms == 0.0);
  return true;
}

// ---------------------------------------------------------------------------
// Auto-register
// ---------------------------------------------------------------------------
static struct DebugTestRegistrar {
  DebugTestRegistrar() {
    RegisterTest("benchmark_result_zero", TestBenchmarkResultZero);
    RegisterTest("benchmark_result_tokens_per_sec",
                 TestBenchmarkResultTokensPerSec);
    RegisterTest("benchmark_result_no_decode", TestBenchmarkResultNoDecode);
    RegisterTest("DumpTensorShape", TestDumpTensorShape);
    RegisterTest("DumpLogitsTopK", TestDumpLogitsTopK);
    RegisterTest("dump_logits_topk_k_larger_than_vocab",
                 TestDumpLogitsTopKLargerThanVocab);
    RegisterTest("DumpKvCacheInfo", TestDumpKvCacheInfo);
    RegisterTest("run_benchmark_smoke", TestRunBenchmarkSmoke);
    RegisterTest("run_benchmark_zero_predict", TestRunBenchmarkZeroPredict);
  }
} debug_test_registrar;
