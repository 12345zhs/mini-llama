#include "mini_llama/terminal.h"
#include "mini_llama/chat.h"
#include "mini_llama/context.h"
#include "mini_llama/model.h"
#include "mini_llama/tokenizer.h"
#include "mini_llama/gguf_loader.h"
#include "mini_llama/gguf_tokenizer.h"
#include "mini_llama/loader.h"
#include "mini_llama/prompt_builder.h"
#include "mini_llama/gguf.h"
#include "mini_llama/debug.h"
#include "mini_llama/backend.h"
#include "mini_llama/cuda_runtime.h"


#include <iostream>
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>
#include <string>

namespace mini_llama {

// 从命令行中解析整形
static bool ParseIntArg(const char* text, int& value) {
    char* end = nullptr;
    int64_t parsed = std::strtoll(text, &end, 10);   // end指向字符串里第一个不是数字的字符地址
    // 一个数字都没着找见
    if (end == text || *end != '\0') {
        return false;
    }
    if (parsed < 0 || parsed > 1000000) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}


// 解析backend的参数，转化为BackendConfig内部的变量
// --backend cpu | cuda
static bool ParseBackendArg(const char* text, BackendConfig& config) {
    BackendKind kind;
    if (!ParseBackendKind(text, kind)) {
        return false;
    }
    config.kind = kind;
    return true;
}


// 解析 --device 0
static bool ParseDeviceArg(const char* text, BackendConfig& config) {
    int device_id = 0;
    if (!ParseIntArg(text, device_id) || device_id < 0) {
        return false;
    }
    config.device_id = device_id;
    config.device_id_set = true;
    return true;
}

// 验证后端的config,是否有矛盾的参数
static bool ValidateBackendOrPrint(const BackendConfig& config) {
    try {
        ValidateBackend(config);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Backend setup failed: " << e.what() << std::endl;
        return false;
    }
}




// 加载的模型、tokenizer、chat
struct LoadedModel {
    MiniLlamaModel model;
    std::unique_ptr<ITokenizer> tokenizer;
    std::string chat_template;
};


// 判断gguf文件是否正确
static bool IsGgufFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        return false;
    }
    char magic[4];
    return f.read(magic, 4) && std::memcmp(magic, "GGUF", 4) == 0;
}


// 从文件夹中找.gguf文件
static std::string FindGgufInDirectory(const std::string& path) {
    std::vector<std::filesystem::path> candidates;
    std::error_code ec;

    // 扫描文件夹里的所有内容,并保存所有.gguf文件
    for (const auto& entry : std::filesystem::directory_iterator(path, ec)) {
        if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
            candidates.push_back(entry);
        }
    }
    if (candidates.empty()) {
        return "";
    }

    std::sort(candidates.begin(), candidates.end());   // 取字典序最小的

    return candidates.front().string();
}


// 根据文件路径，选择合适的分词器
static std::unique_ptr<ITokenizer> CreateTokenizerFromVocabHint(const std::string& vocab_path) {
    std::filesystem::path vocab(vocab_path);
    std::filesystem::path dir = vocab.parent_path();
    std::filesystem::path merges = dir / "merges.txt";
    std::filesystem::path special = dir / "special_tokens.json";

    if (std::filesystem::exists(merges)) {
        return CreateBpeTokenizer(vocab.string(), merges.string(), special.string());
    }
    return CreateTokenizer(vocab.string());
}

// 从配置文件中解析出来Tokenizer文件路径
static std::string ResolveManifestTokenizerPath(const std::string& config_path) {
    try {
        ModelManifest manifest = ParseManifest(config_path);
        if (manifest.tokenizer.type != "vocab_json" || manifest.tokenizer.path.empty()) {
            return "";
        }
        std::filesystem::path tokenizer_path(manifest.tokenizer.path);
        if (tokenizer_path.is_relative()) {
            // 是否为相对路径
            std::filesystem::path config_path(config_path);
            tokenizer_path = config_path.parent_path() / tokenizer_path;
        }
        return tokenizer_path.string();

    } catch (const std::exception& e) {
        return "";
    }
}


// 根据文件选择tokenizer
static LoadedModel LoadModelAndTokenizer(const std::string& path, const std::string& explicit_config_path = "", const std::string& explicit_tokenizer_path = "") {
    LoadedModel result;

    std::string model_path = path;
    std::string config_path;
    std::string tokenizer_path;

    if (std::filesystem::is_directory(path)) {
        std::filesystem::path bin_path = std::filesystem::path(path) / "model.bin";
        std::filesystem::path json_path = std::filesystem::path(path) / "model.json";
        std::string gguf_path = FindGgufInDirectory(path);

        // 二进制文件以及json文件不存在
        if (!std::filesystem::exists(bin_path) || !std::filesystem::exists(json_path) && !gguf_path.size()) {
            model_path = gguf_path;
        }
    }

    // 检查gguf是否可用
    bool is_gguf = IsGgufFFile(model_path);
    if (is_gguf) {
        result.model = LoadGgufModel(model_path);  // 加载权重
        if (!result.model.loaded) {
            return result;              // 加载模型失败直接返回           
        }

        // 加载分词器
        if (!explicit_tokenizer_path.empty()) {
            result.tokenizer = CreateTokenizerFromVocabHint(explicit_tokenizer_path);   // 根据文件选择ascii、json以及BPE
        } else {
            result.tokenizer = CreateGgufTokenizer(model_path);                         // 没传参数，就用gguf文件采用ggufTokenizer
        }

        // gguf里没有词表,就去当前的文件夹找
        if (!result.tokenizer) {
            std::filesystem::path gguf_dir = std::filesystem::path(model_path).parent_path();
            std::filesystem::path vocab_path = gguf_dir / "vocab.json";
            std::filesystem::path merges_path = gguf_dir / "merges.txt";
            std::filesystem::path special_path = gguf_dir / "special_tokens.json";

            if (std::filesystem::exists(vocab_path.string()) && std::filesystem::exists(merges_path.string()) && std::filesystem::exists(special_path.string())) {
                result.tokenizer = CreateBpeTokenizer(vocab_path.string(), merges_path.string(), special_path.string());
            }
        }

        // chat模板
        result.chat_template = LoadChatTemplateFromGguf(model_path);
    } else {
        // json+bin
        // 确认model.bin与model.json的路径
        if (std::filesystem::is_directory(path)) {
            // path 是文件夹
            model_path = path +  "/model.bin";
            config_path = path + "/model.json";
        } else {
            // path是文件
            model_path = path;
            if (!explicit_config_path.empty()) {
                config_path = explicit_config_path;
            } else {
                config_path = std::filesystem::path(path).parent_path().string() + "/model.json";
            }
        }
        // 加载权重
        result.model = LoadModel(config_path, model_path);
        if (!result.model.loaded) {
            return result;
        }

        // 加载分词器
        if (!explicit_tokenizer_path.empty()) {
            tokenizer_path = explicit_tokenizer_path;
        }
        // 第一次检查
        if (tokenizer_path.empty()) {
            std::filesystem::path dir = std::filesystem::is_directory(path) ? std::filesystem::path(path) : std::filesystem::path(path).parent_path();

            std::filesystem::path vocab_path = dir / "vocab.json";
            if (std::filesystem::exists(vocab_path)) {
                tokenizer_path = vocab_path.string();
            }
        }

        // 第二次检查,直接从配置文件中获取vocab
        if (tokenizer_path.empty()) {
            tokenizer_path = ResolveManifestTokenizerPath(config_path);
        }
        result.tokenizer = CreateTokenizer(tokenizer_path);

    }

    if (!result.tokenizer) {
        result.model.load_error = "Failed to load tokenizer";
        result.model.loaded = false;
        return result;
    }
         
    return result;
}


static void PrintRunUsage(const char* prog) {
  std::cout
      << "Usage: " << prog << " run <model-path|dir> [options]\n"
      << "Options:\n"
      << "  --temperature <T>  Sampling temperature (default: 0.0 = greedy)\n"
      << "  --top-k <k>        Top-k sampling (default: 0 = disabled)\n"
      << "  --seed <S>         Random seed (default: 0 = random)\n"
      << "  -n, --n-predict <n> Maximum response tokens per turn (default: "
         "64)\n"
      << "  --tokenizer <path> Path to vocab.json tokenizer file\n"
      << "  --backend cpu|cuda Execution backend (default: cpu; cuda requires "
         "-DMINI_LLAMA_CUDA=ON)\n"
      << "  --device <n>       CUDA device id for --backend cuda (default: 0)\n"
      << "  -h, --help         Show this help\n";
}


static int RunChat(int argc, char** argv) {

    if (argc >= 3 && (std::strcmp(argv[2], "-h") == 0 || std::strcmp(argv[2], "--help") == 0)) {
        PrintRunUsage(argv[0]);
        return 0;
    }

    if (argc < 3) {
        std::cerr << "Missing model directory.\n";
        PrintRunUsage(argv[0]);
        return 1;
    }

    std::string model_dir = argv[2];
    float temperature = 0.0f;
    int top_k = 0;
    unsigned int seed = 0;
    int max_response_tokens = 64;
    std::string tokenizer_path;



    LoadedModel  lm = LoadModelAndTokenizer(model_dir, "", tokenizer_path);
    if (!lm.model.loaded) {
        std::cerr << "Failed to load model: " << lm.model.load_error << "\n";
        return 1;
    }
    if (!lm.tokenizer) {
        std::cerr << "Failed to load tokenizer.\n";
        return 1;
    }
    if (lm.model.config.vocab_size < lm.tokenizer->vocab_size()) {
        std::cerr << "Model vocab_size must be at least " << lm.tokenizer->vocab_size() << " for the tokenizer.\n";
        return 1;
    }


    MiniLlamaModel& model = lm.model;
    Terminal term;
    ChatSession session;

    MiniLlamaContext ctx(&model);

    while (true) {
        term.PrintUserPrompt();
        std::string input = term.ReadLine();

        // 用户输入空或者输入已经到达了结尾
        if (input.empty() && std::cin.eof()) {
            break;
        }
        if (input == "/help") {
            term.PrintHelp();
            continue;
        }
        if (input == "/exit") {
            break;
        }

        if (input == "/clear") {
            session.Clear();                 // 清理本轮对话
            ctx = MiniLlamaContext(&model);  // 重新构建上下文
            if (lm.chat_template.empty()) {
                session.AddMessage("system", "You are a helpful assistant");
            }
            term.PrintMessage("Chat history cleared.\n");
            continue;
        } 
        if (input == "/stats") {
            term.PrintStats(session);
            continue;
        }
        if (input == "/params") {
            term.PrintParams(session.sampling_params);
        }
        if (!input.empty() && input[0] == '/') {
            term.PrintMessage("Unknown command: " + input + "\n");
            continue;
        }
        if (input.empty()) {
            continue;
        }
    }
}


// 解析gguf文件信息
static int RunInspectGguf(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " inspect-gguf <path>\n";        // 必须给出文件路径
        return 1;
    }
    GgufReader reader;
    if (!reader.Load(argv[2])) {
        std::cerr << "Failed to load GGUF: " << reader.load_error << "\n";
        return 1;
    }

    InspectGguf(reader);      // 打印解析信息
    return 0;
}


// benchmark的help指令
static void PrintBenchUsage(const char* prog) {
    std::cout
      << "Usage: " << prog << " bench <model-path|dir> [options]\n"
      << "Options:\n"
      << "  -p, --prompt <str>    Input prompt text (default: \"hello\")\n"
      << "  -n, --n-predict <n>   Number of tokens to generate (default: 64)\n"
      << "  --seed <S>            Random seed (default: 0 = random)\n"
      << "  --tokenizer <path>    Path to vocab.json tokenizer file\n"
      << "  --quant q8_0|q4_0     Quantize loaded Linear weights before "
         "benchmark\n"
      << "  --threads <n>         Number of threads for parallel ops (0 = "
         "auto)\n"
      << "  --backend cpu|cuda    Execution backend (default: cpu; cuda "
         "requires -DMINI_LLAMA_CUDA=ON)\n"
      << "  --device <n>          CUDA device id for --backend cuda (default: "
         "0)\n"
      << "  --verbose             Print debug dumps after each step\n"
      << "  -h, --help            Show this help\n"
      << "\n"
      << "CUDA benchmark metrics:\n"
      << "  uploaded weights: CUDA-resident Linear weights loaded before "
         "inference\n"
      << "  cuda Linear/activation/attention calls: GPU kernel coverage during "
         "forward\n"
      << "  cpu attention fallback calls: attention steps that returned to "
         "CPU\n"
      << "  host->device / device->host copies: runtime transfer count and "
         "bytes\n"
      << "  cuda fallback: unsupported or missing quantized Linear weights "
         "running on CPU\n";
}



static int RunBench(int argc, char** argv) {
    if (argc >=3 && (std::strcmp(argv[2], "-h") == 0 || std::strcmp(argv[2], "--help") == 0)) {
        PrintBenchUsage(argv[0]);
        return 0;
    }

    if (argc < 3) {
        std::cerr << "Missing model directory.\n";
        PrintBenchUsage(argv[0]);
        return 1;
    }

    // 接受指令
    // --prompt  --n-predict  --quant --verbose
    std::string model_dir = argv[2];   // 模型地址
    std::string prompt = "hello";
    int n_predict = 64;                   
    unsigned int seed = 0;
    std::string tokenizer_path;
    bool verbose = false;
    std::string quant_type;
    int n_threads = 0;


    // 加载模型并且设置分词器
    LoadedModel lm = LoadModelAndTokenizer(model_dir, "", tokenizer_path);
    // 校验正确性
    if (!lm.model.loaded) {
        std::cerr << "Failed to load model: " << lm.model.load_error << "\n";
        return 1;
    }
    if (!lm.tokenizer) {
        std::cerr << "Failed to load tokenizer.\n";
        return 1;
    }
    if (lm.model.config.vocab_size < lm.tokenizer->vocab_size()) {
        std::cerr << "Model vocab_size must be at least " << lm.tokenizer->vocab_size() << " for the tokenizer.\n";
        return 1;
    }


    MiniLlamaModel baseline_model = lm.model;
    MiniLlamaModel& model = lm.model;
    std::unique_ptr<ITokenizer>& tokenizer = lm.tokenizer;
    std::vector<int> tokens = tokenizer->Encode(prompt);

    if (tokens.size() > static_cast<size_t>(model.config.max_seq_len)) {
        std::cerr << "Prompt too long.\n";
        return 1;
    }
    if (tokens.size() + static_cast<size_t>(n_predict) > static_cast<size_t>(model.config.max_seq_len)) {
        n_predict = model.config.max_seq_len - tokens.size();
    }


    BenchmarkResult result = RunBenchmark(model, tokens, n_predict, seed, verbose);

    std::cout << "Results:\n";
    std::cout << "  prompt tokens:     " << result.n_prompt_tokens << "\n";
    std::cout << "  generated tokens:  " << result.n_generated_tokens << "\n";
    std::cout << "  Decode tokens:     " << result.n_decode_tokens << "\n";
    std::cout << "  prefill time:      " << std::fixed << std::setprecision(2) << result.prefill_ms << " ms\n";
    std::cout << "  Decode time:       " << std::fixed << std::setprecision(2) << result.decode_ms << " ms\n";
    std::cout << "  total time:        " << std::fixed << std::setprecision(2) << (result.prefill_ms + result.decode_ms) << " ms\n";
    std::cout << "  tokens/s (total):  " << std::fixed << std::setprecision(2) << result.tokens_per_sec() << "\n";
    std::cout << "  tokens/s (Decode): " << std::fixed << std::setprecision(2) << result.decode_tokens_per_sec() << "\n";


}

// 命令
static void PrintGlobalUsage(const char* prog) {
    std::cout << "Usage: " << prog << " <command> [options]\n"
              << "Commands:\n"
              << "  generate      One-shot text generation (default)\n"
              << "  run           Interactive chat mode\n"
              << "  inspect       Inspect model metadata\n"
              << "  inspect-gguf  Inspect GGUF file\n"
              << "  bench         Benchmark inference performance\n"
              << "  --cuda-info   Print CUDA device information for the selected "
                 "CUDA build\n"
              << "\n"
              << "Run \"" << prog << " generate --help\", \"" << prog
              << " run --help\", or \"" << prog
              << " bench --help\" for details.\n";
}


//  --cuda-info 命令
static int RunCudaInfo() {
    try {
        int count = CudaDeviceCount();
        std::cout << "CUDA devices: " << count << std::endl;
        for (int device_id = 0; device_id < count; ++device_id) {
            CudaDeviceInfo info = CudaGetDeviceInfo(device_id);
            std::cout << CudaFormatDeviceInfo(info) << std::endl;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "CUDA info failed: " << e.what() << "\n";
        return 1;
    }
}



int main(int argc, char** argv) {
    // 只有mini_llama一个参数，输出帮助文档
    if (argc < 2) {
        mini_llama::PrintGlobalUsage(argv[0]);
        return 1;
    }

    std::string command = argv[1];
    if (command == "generate") {

    } else if (command == "run") {

    } else if (command == "inspect") {
        
    } else if (command == "bench") {
        
    } else if (command == "inspect-gguf") {
        
    } else if (command == "--cuda-info") {
        
    } else if (command == "-h" || command == "--help") {
        
    } else if (!command.empty() && command[0] == '-') {
        
    } else  {
        std::cerr << "Unknown command: " << command << std::endl;
        mini_llama::PrintGlobalUsage(argv[0]);
        return 1;
    }
}
}