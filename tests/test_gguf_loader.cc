#include <filesystem>
#include <string>
#include <vector>

#include "mini_llama/context.h"
#include "mini_llama/forward.h"
#include "mini_llama/gguf_loader.h"
#include "tests/gguf_loader_fixture.h"
#include "tests/test_main.h"
#include "tests/test_names.h"
#include "tests/test_main.h"

// ---------------------------------------------------------------------------
// Load a generated tiny GGUF model and verify config + tensor shapes.
// ---------------------------------------------------------------------------
static bool TestGgufLoaderLoadsTinyFixture() {
  std::filesystem::path path =
      mini_llama::test::WriteTinyLlamaGgufFixture("tiny_loader_fixture.gguf");
  MiniLlamaModel model = LoadGgufModel(path.string());
  std::filesystem::remove(path);

  MINI_LLAMA_ASSERT_TRUE(model.loaded);
  MINI_LLAMA_ASSERT_TRUE(model.load_error.empty());

  MINI_LLAMA_ASSERT_EQ(model.config.vocab_size, 128);
  MINI_LLAMA_ASSERT_EQ(model.config.dim, 32);
  MINI_LLAMA_ASSERT_EQ(model.config.n_layers, 2);
  MINI_LLAMA_ASSERT_EQ(model.config.n_heads, 4);
  MINI_LLAMA_ASSERT_EQ(model.config.n_kv_heads, 4);
  MINI_LLAMA_ASSERT_EQ(model.config.head_dim, 8);
  MINI_LLAMA_ASSERT_EQ(model.config.hidden_dim, 86);
  MINI_LLAMA_ASSERT_EQ(model.config.max_seq_len, 128);
  MINI_LLAMA_ASSERT_NEAR(model.config.rms_norm_eps, 1e-5f, 1e-7f);

  MINI_LLAMA_ASSERT_TRUE(model.token_embedding.data.size() > 0);
  MINI_LLAMA_ASSERT_EQ(model.token_embedding.shape.size(), 2);
  MINI_LLAMA_ASSERT_EQ(model.token_embedding.shape[0], 128);
  MINI_LLAMA_ASSERT_EQ(model.token_embedding.shape[1], 32);

  MINI_LLAMA_ASSERT_EQ(static_cast<int>(model.layers.size()), 2);
  for (int i = 0; i < 2; ++i) {
    const LayerWeights& lw = model.layers[i];
    MINI_LLAMA_ASSERT_TRUE(lw.attention_norm.data.size() > 0);
    MINI_LLAMA_ASSERT_EQ(lw.attention_norm.shape[0], 32);

    MINI_LLAMA_ASSERT_TRUE(lw.wq.f32_data.size() > 0);
    MINI_LLAMA_ASSERT_EQ(lw.wq.shape, std::vector<int>({32, 32}));

    MINI_LLAMA_ASSERT_TRUE(lw.wk.f32_data.size() > 0);
    MINI_LLAMA_ASSERT_EQ(lw.wk.shape, std::vector<int>({32, 32}));
    MINI_LLAMA_ASSERT_TRUE(lw.wv.f32_data.size() > 0);
    MINI_LLAMA_ASSERT_EQ(lw.wv.shape, std::vector<int>({32, 32}));
    MINI_LLAMA_ASSERT_TRUE(lw.wo.f32_data.size() > 0);
    MINI_LLAMA_ASSERT_EQ(lw.wo.shape, std::vector<int>({32, 32}));

    MINI_LLAMA_ASSERT_TRUE(lw.ffn_norm.data.size() > 0);
    MINI_LLAMA_ASSERT_EQ(lw.ffn_norm.shape[0], 32);

    MINI_LLAMA_ASSERT_TRUE(lw.w_gate.f32_data.size() > 0);
    MINI_LLAMA_ASSERT_EQ(lw.w_gate.shape, std::vector<int>({86, 32}));
    MINI_LLAMA_ASSERT_TRUE(lw.w_up.f32_data.size() > 0);
    MINI_LLAMA_ASSERT_EQ(lw.w_up.shape, std::vector<int>({86, 32}));
    MINI_LLAMA_ASSERT_TRUE(lw.w_down.f32_data.size() > 0);
    MINI_LLAMA_ASSERT_EQ(lw.w_down.shape, std::vector<int>({32, 86}));
  }

  MINI_LLAMA_ASSERT_TRUE(model.final_norm.data.size() > 0);
  MINI_LLAMA_ASSERT_EQ(model.final_norm.shape[0], 32);

  MINI_LLAMA_ASSERT_TRUE(model.lm_head.f32_data.size() > 0);
  MINI_LLAMA_ASSERT_EQ(model.lm_head.shape.size(), 2);
  MINI_LLAMA_ASSERT_EQ(model.lm_head.shape[0], 128);
  MINI_LLAMA_ASSERT_EQ(model.lm_head.shape[1], 32);

  return true;
}

// ---------------------------------------------------------------------------
// The generated GGUF model must enter the normal forward path.
// ---------------------------------------------------------------------------
static bool TestGgufLoaderFixtureEntersForwardPath() {
  std::filesystem::path path =
      mini_llama::test::WriteTinyLlamaGgufFixture("tiny_loader_forward.gguf");
  MiniLlamaModel model = LoadGgufModel(path.string());
  std::filesystem::remove(path);

  MINI_LLAMA_ASSERT_TRUE(model.loaded);
  MiniLlamaContext ctx(&model);
  ctx.pos = 0;
  Tensor logits = ForwardToken(ctx, model, 1);
  MINI_LLAMA_ASSERT_EQ(logits.shape, std::vector<int>({128}));
  return true;
}

// ---------------------------------------------------------------------------
// Load the local demo GGUF when it exists. This keeps the large-file check
// useful on the development machine while the tiny fixture covers clean clones.
// ---------------------------------------------------------------------------
static bool TestGgufLoaderLoadsRealModelWhenAvailable() {
  const std::string path = "models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf";
  if (!std::filesystem::exists(path)) {
    return true;
  }

  MiniLlamaModel model = LoadGgufModel(path);
  MINI_LLAMA_ASSERT_TRUE(model.loaded);
  MINI_LLAMA_ASSERT_TRUE(model.load_error.empty());
  MINI_LLAMA_ASSERT_EQ(model.config.vocab_size, 151936);
  MINI_LLAMA_ASSERT_EQ(model.config.dim, 896);
  MINI_LLAMA_ASSERT_EQ(model.config.n_layers, 24);
  MINI_LLAMA_ASSERT_EQ(model.config.n_heads, 14);
  MINI_LLAMA_ASSERT_EQ(model.config.n_kv_heads, 2);
  MINI_LLAMA_ASSERT_EQ(model.config.head_dim, 64);
  MINI_LLAMA_ASSERT_EQ(model.config.hidden_dim, 4864);
  MINI_LLAMA_ASSERT_EQ(model.config.max_seq_len, 32768);
  MINI_LLAMA_ASSERT_NEAR(model.config.rms_norm_eps, 1e-6f, 1e-7f);
  return true;
}

// ---------------------------------------------------------------------------
// Error handling: missing file.
// ---------------------------------------------------------------------------
static bool TestGgufLoaderMissingFile() {
  MiniLlamaModel model = LoadGgufModel("models/chat/nonexistent.gguf");
  MINI_LLAMA_ASSERT_TRUE(!model.loaded);
  MINI_LLAMA_ASSERT_TRUE(!model.load_error.empty());
  return true;
}

// ---------------------------------------------------------------------------
// Error handling: non-GGUF file.
// ---------------------------------------------------------------------------
static bool TestGgufLoaderInvalidFile() {
  MiniLlamaModel model = LoadGgufModel("models/tiny/model.json");
  MINI_LLAMA_ASSERT_TRUE(!model.loaded);
  MINI_LLAMA_ASSERT_TRUE(!model.load_error.empty());
  return true;
}

// ---------------------------------------------------------------------------
// The tiny test.gguf lacks config metadata, so LoadGgufModel should fail
// gracefully during config validation.
// ---------------------------------------------------------------------------
static bool TestGgufLoaderRejectsIncompleteConfig() {
  MiniLlamaModel model = LoadGgufModel("models/tiny/test.gguf");
  MINI_LLAMA_ASSERT_TRUE(!model.loaded);
  MINI_LLAMA_ASSERT_TRUE(!model.load_error.empty());
  return true;
}

static struct GgufLoaderTestRegistrar {
  GgufLoaderTestRegistrar() {
    RegisterTest("gguf_loader_loads_tiny_fixture",
                 TestGgufLoaderLoadsTinyFixture);
    RegisterTest("gguf_loader_fixture_enters_forward_path",
                 TestGgufLoaderFixtureEntersForwardPath);
    RegisterTest("gguf_loader_loads_real_model_when_available",
                 TestGgufLoaderLoadsRealModelWhenAvailable);
    RegisterTest("gguf_loader_missing_file", TestGgufLoaderMissingFile);
    RegisterTest("gguf_loader_invalid_file", TestGgufLoaderInvalidFile);
    RegisterTest("gguf_loader_rejects_incomplete_config",
                 TestGgufLoaderRejectsIncompleteConfig);
  }
} gguf_loader_test_registrar;
