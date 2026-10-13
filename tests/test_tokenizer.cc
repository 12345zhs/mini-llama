
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "mini_llama/tokenizer.h"
#include "tests/test_main.h"
#include "tests/test_names.h"

namespace {

struct TempBpeFiles {
  std::filesystem::path dir;
  std::filesystem::path vocab;
  std::filesystem::path merges;
  std::filesystem::path special;
};

static TempBpeFiles WriteTempBpeFiles() {
  const auto stamp =
      std::chrono::steady_clock::now().time_since_epoch().count();
  TempBpeFiles files;
  files.dir = std::filesystem::temp_directory_path() /
              ("mini_llama_bpe_tokenizer_" + std::to_string(stamp));
  std::filesystem::create_directories(files.dir);
  files.vocab = files.dir / "vocab.json";
  files.merges = files.dir / "merges.txt";
  files.special = files.dir / "special_tokens.json";

  {
    std::ofstream out(files.vocab);
    out << "{\n"
        << "  \"h\": 0,\n"
        << "  \"e\": 1,\n"
        << "  \"l\": 2,\n"
        << "  \"o\": 3,\n"
        << "  \"he\": 4,\n"
        << "  \"hel\": 5,\n"
        << "  \"hell\": 6,\n"
        << "  \"hello\": 7,\n"
        << "  \"<|im_start|>\": 8,\n"
        << "  \"<unk>\": 9,\n"
        << "  \"<bos>\": 10,\n"
        << "  \"<eos>\": 11\n"
        << "}\n";
  }
  {
    std::ofstream out(files.merges);
    out << "h e\n"
        << "he l\n"
        << "hel l\n"
        << "hell o\n";
  }
  {
    std::ofstream out(files.special);
    out << "{ \"bos_id\": 10, \"eos_id\": 11, \"unk_id\": 9 }\n";
  }
  return files;
}

static void RemoveTempBpeFiles(const TempBpeFiles& files) {
  std::error_code ec;
  std::filesystem::remove_all(files.dir, ec);
}

}  // namespace

// ---------------------------------------------------------------------------
// AsciiTokenizer
// ---------------------------------------------------------------------------
static bool TestAsciiEncodeBasic() {
  AsciiTokenizer tok;
  auto tokens = tok.Encode("hi");
  MINI_LLAMA_ASSERT_EQ(tokens.size(), 3);
  MINI_LLAMA_ASSERT_EQ(tokens[0], tok.bos_id());  // 1
  MINI_LLAMA_ASSERT_EQ(tokens[1], static_cast<int>('h'));
  MINI_LLAMA_ASSERT_EQ(tokens[2], static_cast<int>('i'));
  return true;
}

static bool TestAsciiEncodeEmpty() {
  AsciiTokenizer tok;
  auto tokens = tok.Encode("");
  MINI_LLAMA_ASSERT_EQ(tokens.size(), 1);
  MINI_LLAMA_ASSERT_EQ(tokens[0], tok.bos_id());
  return true;
}

static bool TestAsciiDecodeToken() {
  AsciiTokenizer tok;
  MINI_LLAMA_ASSERT_TRUE(tok.DecodeToken(tok.bos_id()) == "<bos>");
  MINI_LLAMA_ASSERT_TRUE(tok.DecodeToken(tok.eos_id()) == "<eos>");
  MINI_LLAMA_ASSERT_TRUE(tok.DecodeToken(tok.unk_id()) == "<unk>");
  MINI_LLAMA_ASSERT_TRUE(tok.DecodeToken(static_cast<int>('a')) == "a");
  MINI_LLAMA_ASSERT_TRUE(tok.DecodeToken(32) == " ");
  MINI_LLAMA_ASSERT_TRUE(tok.DecodeToken(10) == "");        // control char
  MINI_LLAMA_ASSERT_TRUE(tok.DecodeToken(127) == "");       // DEL
  MINI_LLAMA_ASSERT_TRUE(tok.DecodeToken(200) == "<unk>");  // out of range
  return true;
}

static bool TestAsciiDecodeSequence() {
  AsciiTokenizer tok;
  std::vector<int> tokens = {tok.bos_id(), static_cast<int>('h'),
                             static_cast<int>('i')};
  std::string text = tok.Decode(tokens);
  MINI_LLAMA_ASSERT_TRUE(text == "<bos>hi");
  return true;
}

static bool TestAsciiVocabSize() {
  AsciiTokenizer tok;
  MINI_LLAMA_ASSERT_EQ(tok.vocab_size(), 128);
  MINI_LLAMA_ASSERT_EQ(tok.bos_id(), 1);
  MINI_LLAMA_ASSERT_EQ(tok.eos_id(), 2);
  MINI_LLAMA_ASSERT_EQ(tok.unk_id(), 0);
  return true;
}

// ---------------------------------------------------------------------------
// JsonVocabTokenizer
// ---------------------------------------------------------------------------
static bool TestJsonVocabLoads() {
  try {
    JsonVocabTokenizer tok("models/tiny/vocab.json");
    MINI_LLAMA_ASSERT_EQ(tok.vocab_size(), 128);
    MINI_LLAMA_ASSERT_EQ(tok.bos_id(), 1);
    MINI_LLAMA_ASSERT_EQ(tok.eos_id(), 2);
    MINI_LLAMA_ASSERT_EQ(tok.unk_id(), 0);
  } catch (const std::runtime_error& e) {
    MINI_LLAMA_ASSERT_FAIL(std::string("failed to load vocab.json: ") +
                           e.what());
  }
  return true;
}

static bool TestJsonVocabEncodeBasic() {
  try {
    JsonVocabTokenizer tok("models/tiny/vocab.json");
    auto tokens = tok.Encode("hi");
    MINI_LLAMA_ASSERT_EQ(tokens.size(), 3);
    MINI_LLAMA_ASSERT_EQ(tokens[0], tok.bos_id());
    MINI_LLAMA_ASSERT_EQ(tokens[1], static_cast<int>('h'));
    MINI_LLAMA_ASSERT_EQ(tokens[2], static_cast<int>('i'));
  } catch (const std::runtime_error& e) {
    MINI_LLAMA_ASSERT_FAIL(std::string("failed to load vocab.json: ") +
                           e.what());
  }
  return true;
}

static bool TestJsonVocabDecodeToken() {
  try {
    JsonVocabTokenizer tok("models/tiny/vocab.json");
    MINI_LLAMA_ASSERT_TRUE(tok.DecodeToken(0) == "<unk>");
    MINI_LLAMA_ASSERT_TRUE(tok.DecodeToken(1) == "<bos>");
    MINI_LLAMA_ASSERT_TRUE(tok.DecodeToken(2) == "<eos>");
    MINI_LLAMA_ASSERT_TRUE(tok.DecodeToken(static_cast<int>('a')) == "a");
    MINI_LLAMA_ASSERT_TRUE(tok.DecodeToken(32) == " ");
    MINI_LLAMA_ASSERT_TRUE(tok.DecodeToken(10) ==
                           "");  // control char maps to empty
    MINI_LLAMA_ASSERT_TRUE(tok.DecodeToken(127) == "");       // DEL
    MINI_LLAMA_ASSERT_TRUE(tok.DecodeToken(200) == "<unk>");  // out of range
  } catch (const std::runtime_error& e) {
    MINI_LLAMA_ASSERT_FAIL(std::string("failed to load vocab.json: ") +
                           e.what());
  }
  return true;
}

static bool TestJsonVocabDecodeSequence() {
  try {
    JsonVocabTokenizer tok("models/tiny/vocab.json");
    std::vector<int> tokens = {1, static_cast<int>('h'), static_cast<int>('i')};
    std::string text = tok.Decode(tokens);
    MINI_LLAMA_ASSERT_TRUE(text == "<bos>hi");
  } catch (const std::runtime_error& e) {
    MINI_LLAMA_ASSERT_FAIL(std::string("failed to load vocab.json: ") +
                           e.what());
  }
  return true;
}

static bool TestJsonVocabMatchesAscii() {
  try {
    AsciiTokenizer ascii_tok;
    JsonVocabTokenizer json_tok("models/tiny/vocab.json");

    std::string prompt = "hello!";
    auto ascii_tokens = ascii_tok.Encode(prompt);
    auto json_tokens = json_tok.Encode(prompt);

    MINI_LLAMA_ASSERT_EQ(ascii_tokens.size(), json_tokens.size());
    for (size_t i = 0; i < ascii_tokens.size(); ++i) {
      MINI_LLAMA_ASSERT_EQ(ascii_tokens[i], json_tokens[i]);
    }

    std::string ascii_decoded = ascii_tok.Decode(ascii_tokens);
    std::string json_decoded = json_tok.Decode(json_tokens);
    MINI_LLAMA_ASSERT_TRUE(ascii_decoded == json_decoded);
  } catch (const std::runtime_error& e) {
    MINI_LLAMA_ASSERT_FAIL(std::string("failed to load vocab.json: ") +
                           e.what());
  }
  return true;
}

static bool TestJsonVocabMissingFile() {
  try {
    JsonVocabTokenizer tok("models/tiny/nonexistent_vocab.json");
    MINI_LLAMA_ASSERT_FAIL("expected exception for missing vocab file");
  } catch (const std::runtime_error&) {
    // expected
  }
  return true;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------
static bool TestFactoryUsesJsonWhenExists() {
  auto tok = CreateTokenizer("models/tiny/vocab.json");
  MINI_LLAMA_ASSERT_TRUE(tok != nullptr);
  MINI_LLAMA_ASSERT_EQ(tok->vocab_size(), 128);
  return true;
}

static bool TestFactoryFallsBackToAscii() {
  auto tok = CreateTokenizer("");
  MINI_LLAMA_ASSERT_TRUE(tok != nullptr);
  MINI_LLAMA_ASSERT_EQ(tok->vocab_size(), 128);  // AsciiTokenizer
  return true;
}

static bool TestFactoryFallsBackWhenMissing() {
  auto tok = CreateTokenizer("models/tiny/does_not_exist.json");
  MINI_LLAMA_ASSERT_TRUE(tok != nullptr);
  MINI_LLAMA_ASSERT_EQ(tok->vocab_size(), 128);  // AsciiTokenizer fallback
  return true;
}

// ---------------------------------------------------------------------------
// BpeTokenizer
// ---------------------------------------------------------------------------
static bool TestBpeEncodeDecodeAndSpecialToken() {
  TempBpeFiles files = WriteTempBpeFiles();
  auto tok = CreateBpeTokenizer(files.vocab.string(), files.merges.string(),
                                files.special.string());
  MINI_LLAMA_ASSERT_TRUE(tok != nullptr);
  MINI_LLAMA_ASSERT_EQ(tok->bos_id(), 10);
  MINI_LLAMA_ASSERT_EQ(tok->eos_id(), 11);
  MINI_LLAMA_ASSERT_EQ(tok->unk_id(), 9);

  std::vector<int> hello = tok->Encode("hello");
  MINI_LLAMA_ASSERT_EQ(hello.size(), 1u);
  MINI_LLAMA_ASSERT_EQ(hello[0], 7);
  MINI_LLAMA_ASSERT_TRUE(tok->Decode(hello) == "hello");

  std::vector<int> special = tok->Encode("<|im_start|>");
  MINI_LLAMA_ASSERT_EQ(special.size(), 1u);
  MINI_LLAMA_ASSERT_EQ(special[0], 8);
  MINI_LLAMA_ASSERT_TRUE(tok->Decode(special) == "<|im_start|>");

  RemoveTempBpeFiles(files);
  return true;
}

// ---------------------------------------------------------------------------
// Auto-register
// ---------------------------------------------------------------------------
static struct TokenizerTestRegistrar {
  TokenizerTestRegistrar() {
    RegisterTest("ascii_encode_basic", TestAsciiEncodeBasic);
    RegisterTest("ascii_encode_empty", TestAsciiEncodeEmpty);
    RegisterTest("ascii_decode_token", TestAsciiDecodeToken);
    RegisterTest("ascii_decode_sequence", TestAsciiDecodeSequence);
    RegisterTest("ascii_vocab_size", TestAsciiVocabSize);
    RegisterTest("jsonvocab_loads", TestJsonVocabLoads);
    RegisterTest("jsonvocab_encode_basic", TestJsonVocabEncodeBasic);
    RegisterTest("jsonvocab_decode_token", TestJsonVocabDecodeToken);
    RegisterTest("jsonvocab_decode_sequence", TestJsonVocabDecodeSequence);
    RegisterTest("jsonvocab_matches_ascii", TestJsonVocabMatchesAscii);
    RegisterTest("jsonvocab_missing_file", TestJsonVocabMissingFile);
    RegisterTest("factory_uses_json_when_exists",
                 TestFactoryUsesJsonWhenExists);
    RegisterTest("factory_falls_back_to_ascii", TestFactoryFallsBackToAscii);
    RegisterTest("factory_falls_back_when_missing",
                 TestFactoryFallsBackWhenMissing);
    RegisterTest("bpe_encode_decode_and_special_token",
                 TestBpeEncodeDecodeAndSpecialToken);
  }
} tokenizer_test_registrar;
