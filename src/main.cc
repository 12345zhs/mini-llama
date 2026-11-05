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
#include "mini_llama/batch.h"
#include "mini_llama/forward.h"
#include "mini_llama/request_context.h"
#include "mini_llama/thread_pool.h"


#include <iostream>
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <utility>
#include <filesystem>
#include <ctime>
#include <cstring>

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



// 解析单精度浮点数
static bool ParseFloatArg(const char* text, float& value) {
    char* end = nullptr;
    errno = 0;
    double parsed = std::strtod(text, &end);   // end指向字符串里第一个不是数字的字符
    if (end == text || *end != '\0') {
        return false;
    }

    if (errno == ERANGE || !std::isfinite(parsed) || parsed < 0.0 || parsed > 10000.0) {
        return false;
    }
    value = static_cast<float>(parsed);
    return true;
}



// 解析无符号整数
static bool ParseUintArg(const char* text, unsigned int& value) {
    if (text[0] == '-' || text[0] == '\0') {
        return false;
    }
    char* end = nullptr;
    errno = 0;
    uint64_t parsed = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') {
        return false;
    }
    if (errno == ERANGE || parsed > std::numeric_limits<unsigned int>::max()) {
        return false;
    }
    value = static_cast<unsigned int>(parsed);
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


// 输出后端信息
static void PrintBackendInfo(const BackendConfig& config) {
    std::cout << "backend: " << BackendKindName(config.kind) << std::endl;
    if (config.kind == BackendKind::kCuda) {
        std::cout << "cuda: " << CudaDeviceSummary(config.device_id) << std::endl;       
    } 
    std::cout << BackendExecutionNote(config) << std::endl;
}


// 输出多少MB
static std::string FormatMb(size_t bytes) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << (static_cast<double>(bytes) / (1024.0 * 1024.0)) << "MB";
    return out.str();
}


// 是否可以上传cuda权重
static bool PrepareCudaWeightsOrPrint(MiniLlamaModel& model, const BackendConfig& config) {
    if (config.kind != BackendKind::kCuda) {
        return true;
    }
    try {
        UploadModelWeightsToCuda(model, config.device_id);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "CUDA weight upload failed: " << e.what() << std::endl;
        return false;
    }
}



// 上传了多少权重，占了多少显存
static void PrintCudaWeightSummary(const MiniLlamaModel& model, const std::string& indent = "") {
    std::cout << indent << "uploaded weights: " << ModelCudaUploadedWeightCount(model) << std::endl;
    std::cout << indent <<  "gpu memory used: " << ModelCudaMemoryBytes(model) << " bytes (" << FormatMb(ModelCudaMemoryBytes(model)) << ")" << std::endl;
}


// 统计Linear的量化权重 
struct LinearWeightTypeCounts {
    size_t f32 = 0;
    size_t q8_0 = 0;
    size_t q4_0 = 0;
    size_t q4_1 = 0;

    size_t total() const {
        return f32 + q8_0 + q4_0 + q4_1;
    }
    size_t quantized() const {
        return q8_0 + q4_0 + q4_1;
    }
};


// LinearWeightTypeCounts的量化类型数量
static void CountLinearWeight(const QuantizedTensor& tensor, LinearWeightTypeCounts& counts) {
    switch (tensor.type) {
        case QuantType::kF32 : {
            counts.f32 += 1;
            return;
        }
        case QuantType::kQ80 : {
            counts.q8_0 += 1;
            return;
        }
        case QuantType::kQ40 : {
            counts.q4_0 += 1;
            return;
        }
        case QuantType::kQ41 : {
            counts.q4_1 += 1;
            return;
        }
    }
}


// 遍历linear里面的量化权重，每种类型有多少个
static LinearWeightTypeCounts ModelLinearWeightTypeCounts(const MiniLlamaModel& model) {
    LinearWeightTypeCounts counts;
    CountLinearWeight(model.lm_head, counts);
    for (const auto& lw : model.layers) {
        CountLinearWeight(lw.wq, counts);
        CountLinearWeight(lw.wk, counts);
        CountLinearWeight(lw.wv, counts);
        CountLinearWeight(lw.wo, counts);
        CountLinearWeight(lw.w_gate, counts);
        CountLinearWeight(lw.w_up, counts);
        CountLinearWeight(lw.w_down, counts);
    }
    return counts;
}


// 有多少个linear权重传给cuda，有多少个回退到cpu
static void PrintCudaExecutionSummary(const MiniLlamaModel& model, const std::string& indent = "") {
    PrintCudaWeightSummary(model, indent);
    LinearWeightTypeCounts counts = ModelLinearWeightTypeCounts(model);
    // 实际加载了多少量化权重
    size_t uploaded_f32 = ModelCudaUploadedF32WeightCount(model);
    size_t uploaded_q8_0 = ModelCudaUploadedQ80WeightCount(model);
    size_t uploaded_q4_0 = ModelCudaUploadedQ40WeightCount(model);
    size_t uploaded_q4_1 = ModelCudaUploadedQ41WeightCount(model);

    std::cout << indent << "cuda f32 Linear weights: " << uploaded_f32 << "/" << counts.total() << "\n";
    if (counts.q8_0 > 0) {
        std::cout << indent << "cuda q8_0 Linear weights: " << uploaded_q8_0 << "/" << counts.q8_0 << "\n";
    }
    if (counts.q4_0 > 0) {
        std::cout << indent << "cuda q4_0 Linear weights: " << uploaded_q4_0 << "/" << counts.q4_0 << "\n";
    }
    if (counts.q4_1 > 0) {
        std::cout << indent << "cuda q4_1 Linear weights: " << uploaded_q4_1 << "/" << counts.q4_1 << "\n";
    }

    size_t q8_0_fallback = counts.q8_0 >= uploaded_q8_0 ? counts.q8_0 - uploaded_q8_0 : 0;
    size_t q4_0_fallback = counts.q4_0 >= uploaded_q4_0 ? counts.q4_0 - uploaded_q4_0 : 0;
    size_t q4_1_fallback = counts.q4_1 >= uploaded_q4_1 ? counts.q4_1 - uploaded_q4_1 : 0;


    if (q8_0_fallback + q4_0_fallback + q4_1_fallback > 0) {
        std::cout << indent << "cuda fallback: unsupported or missing quantized Linear " "weights use CPU path" << " (Q8_0=" << q8_0_fallback << ", Q4_0=" << q4_0_fallback << ", Q4_1=" << q4_1_fallback << ")\n";
    }
}



// 打印cudaRuntime的情况
static void PrintCudaRuntimeSummary(const MiniLlamaModel& model, const std::string& indent = "") {
    std::cout << indent << "cuda Linear calls: " << ModelCudaLinearCalls(model) << "\n";
    std::cout << indent << "cuda activation calls: " << ModelCudaActivationCalls(model) << "\n";
    std::cout << indent << "cuda attention calls: " << ModelCudaAttentionCalls(model) << "\n";
    std::cout << indent << "cpu attention fallback calls: " << ModelCudaAttentionCpuFallbacks(model) << "\n";
    std::cout << indent << "cuda kv write bytes: " << ModelCudaKvCacheWriteBytes(model) << "\n";
    std::cout << indent << "cuda kv read bytes: " << ModelCudaKvCacheReadBytes(model) << "\n";
    std::cout << indent << "host->device copies: " << ModelCudaHostToDeviceCopies(model) << " (" << ModelCudaHostToDeviceBytes(model) << " bytes)\n";
    std::cout << indent << "device->host copies: " << ModelCudaDeviceToHostCopies(model) << " (" << ModelCudaDeviceToHostBytes(model) << " bytes)\n";
}




// 量化模型,把model中对应的权重量化为设定版本
static void ApplyQuantOverride(MiniLlamaModel& model, const std::string& quant_type) {
    if (quant_type.empty()) {
        return;
    }
    if (quant_type == "q8_0") {
        QuantizeModelToQ80(model);
        return;
    }
    if (quant_type == "q4_0") {
        QuantizeModelToQ40(model);
        return;
    }
    throw std::runtime_error("unsupported quant type: " + quant_type);
}



// 给定一个模型和一段 token，跑一次推理，返回最终输出的词表概率
static Tensor RunLogitsForTokens(const MiniLlamaModel& model, const std::vector<int>& tokens) {
    MiniLlamaContext ctx(&model);
    MiniBatch batch = MiniBatch::FromTokens(tokens, 0);
    return ForwardBatch(ctx, model, batch);
}


// 对比两个logits结果的差距
static std::pair<float, float> LogitsError(const Tensor& baseline, const Tensor& candidate) {
    if (baseline.shape != candidate.shape) {
        throw std::runtime_error("LogitsError: shape mismatch, baseline=" + baseline.ShapeToStringShort() + ", candidate=" + candidate.ShapeToStringShort());
    }   
    float max_err = 0.0f;
    float sum_err = 0.0f;
    for (size_t i = 0; i < baseline.size(); ++i) {
        float err = std::abs(baseline.data[i] - candidate.data[i]);
        max_err = std::max(err, max_err);
        sum_err += err;
    }
    return {max_err, sum_err / static_cast<size_t>(baseline.size())};
}



/*
-------------------------------------------------------------------------------------------------------------------------------------------
generate mode
-------------------------------------------------------------------------------------------------------------------------------------------
*/

// 打印generate mode的使用说明
static void PrintGenerateUsage(const char* prog) {
    std::cout
      << "Usage: " << prog << " generate [options]\n"
      << "Options:\n"
      << "  --model <path|dir>   Path to model weights binary or model "
         "directory (default: models/tiny/model.bin)\n"
      << "  --config <path>      Path to model config JSON (default: "
         "models/tiny/model.json)\n"
      << "  -p, --prompt <str>   Input prompt text (default: \"hello\")\n"
      << "  -n, --n-predict <n>  Number of tokens to generate (default: 16)\n"
      << "  --temperature <T>    Sampling temperature (default: 0.0 = greedy)\n"
      << "  --top-k <k>          Top-k sampling (default: 0 = disabled)\n"
      << "  --seed <S>           Random seed for reproducible sampling "
         "(default: 0 = random)\n"
      << "  --tokenizer <path>   Path to vocab.json tokenizer file\n"
      << "  --quant q8_0|q4_0    Quantize loaded Linear weights before "
         "generation\n"
      << "  --threads <n>        Number of threads for parallel ops (0 = "
         "auto)\n"
      << "  --backend cpu|cuda   Execution backend (default: cpu; cuda "
         "requires -DMINI_LLAMA_CUDA=ON)\n"
      << "  --device <n>         CUDA device id for --backend cuda (default: "
         "0)\n"
      << "  --dump-logits <dir>  Dump logits for each step to directory\n"
      << "  -h, --help           Show this help\n";
}



// 将tensor中的数据以二进制写入指定文件
static void DumpLogits(const Tensor& logits, const std::string path) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open logits dump file: " + path);
    }
    out.write(reinterpret_cast<const char*>(logits.data.data()), static_cast<std::streamsize>(logits.size() * sizeof(float)));
    if (!out.good()) {
        throw std::runtime_error("failed to write logits to: " + path);
    }
}




// 确保指定目录存在，验证是一个目录而非文件
static void EnsureDumpDirectory(const std::string& path) {
    std::error_code error;
    std::filesystem::create_directories(path, error);
    if (error) {
        throw std::runtime_error("failed to create logits dump directory: " + path + ": " + error.message());
    }
    if (!std::filesystem::is_directory(path)) {
        throw std::runtime_error("logits dump path is not a directory: " + path);
    }
}




// 将生成的 Token ID 列表以纯文本格式（空格分隔）写入文件
static void DumpGeneratedTokens(const std::vector<int>& generated, const std::string& path) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open generation token dump file: " + path);
    }

    for (size_t i = 0; i < generated.size(); ++i) {
        if (i > 0) {
            out << " ";
        }
        out << generated[i];
    }
    out << "\n";
    if (!out.good()) {
        throw std::runtime_error("failed to write generation tokens to: " + path);
    }
}



/*
-------------------------------------------------------------------------------------------------------------------------------------------
tokenizer path resolution
-------------------------------------------------------------------------------------------------------------------------------------------
*/

// 从 model.json 配置文件中读取分词器（Tokenizer）的路径，并将其转换为一个可以直接访问的绝对路径
static std::string ResolveManifestTokenizerPath(const std::string& config_path) {
    try {
        ModelManifest manifest = ParseManifest(config_path);
        if (manifest.tokenizer.type != "json_vocab" || manifest.tokenizer.path.empty()) {
            return "";
        }

        std::filesystem::path tokenizer_path(manifest.tokenizer.path);
        // 是否为相对地址
        if (tokenizer_path.is_relative()) {
            std::filesystem::path config_file(config_path);
            tokenizer_path = config_file.parent_path() / tokenizer_path;
        }
        return tokenizer_path.string();
    } catch (const std::exception&) {
        return "";
    }
}




/*
-------------------------------------------------------------------------------------------------------------------------------------------
Model loading helper: supports both JSON+BIN and GGUF
-------------------------------------------------------------------------------------------------------------------------------------------
*/

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
    bool is_gguf = IsGgufFile(model_path);
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


// 计算最长公共前缀
static size_t CommonPrefixLength(const std::vector<int>& a, const std::vector<int>& b) {
    size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) {
        ++i;
    }
    return i;
}



// 输出请求执行追踪日志
static void PrintRequestTrace(const RequestContext& request, std::ostream& out = std::cout) {
    for (const std::string& line : FormatRequestTraceEvents(request)) {
        out << line << "\n";
    }
    out << FormatRequestTraceSummary(request) << "\n";
}




// 运行generate命令，单次文本生成
static int RunGenerate(int argc, char** argv) {
    // 初始化所支持的命令行参数
    std::string model_path = "models/tiny/model.bin";
    std::string config_path = "models/tiny/model.json";
    bool config_path_set = false;
    std::string prompt = "hello";

    int n_predict = 16;
    float temperature = 0.0f;
    int top_k = 0;
    unsigned int seed = 0;

    std::string dump_logits_dir;
    std::string tokenizer_path;
    std::string quant_type;
    int n_threads = 0;
    BackendConfig backend_config;

    // 它从 argv[2] 开始遍历（因为 argv[0] 是程序名，argv[1] 是子命令 generate）
    for (int i = 2; i < argc; ++i) {
        
        // 指定模型权重文件（或目录）的路径
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            // 指定模型配置文件（model.json）的路径
            config_path = argv[++i];
            config_path_set = true;
        } else if ((std::strcmp(argv[i], "-p") == 0 || std::strcmp(argv[i], "--prompt") == 0) && i + 1 < argc) {
            // 指定输入给模型的提示词（Prompt）
            prompt = argv[++i];
        } else if ((std::strcmp(argv[i], "-n") == 0 || std::strcmp(argv[i], "--n-predict") == 0) && i + 1 < argc) {
            // 指定模型要生成的 Token 数量上限
            if (!ParseIntArg(argv[++i], n_predict)) {
                // 解析token数量
                std::cerr << "Invalid --n-predict value. Expected a non-negative integer.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            // 指定采样温度，控制生成文本的随机性
            if (!ParseFloatArg(argv[++i], temperature)) {
                // 解析temperature
                std::cerr << "Invalid --temperature value. Expected a non-negative float.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            // 指定 Top-K 采样的 K 值，限制候选词的数量
            if (!ParseIntArg(argv[++i], top_k)) {
                // 解析非负整数
                std::cerr << "Invalid --top-k value. Expected a non-negative integer.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            // 指定随机数种子，用于复现相同的生成结果
            if (!ParseUintArg(argv[++i], seed)) {
                std::cerr << "Invalid --seed value. Expected a non-negative integer.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--dump-logits") == 0 && i + 1 < argc) {
            // 将每一步的模型原始输出（Logits）以二进制形式保存到指定目录
            dump_logits_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
            // 显式指定分词器文件
            tokenizer_path = argv[++i];
        } else if (std::strcmp(argv[i], "--quant") == 0 && i + 1 < argc) {
            // 推理前对模型进行的量化类型
            quant_type = argv[++i];
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            // 指定计算时使用的 CPU 线程数
            if (!ParseIntArg(argv[++i], n_threads) || n_threads < 0) {
                std::cerr << "Invalid --threads value.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            // 指定执行后端，支持 cpu 或 cuda
            if (!ParseBackendArg(argv[++i], backend_config)) {
                std::cerr << "Invalid --backend value. Supported values: cpu, cuda.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            // 指定 CUDA 设备编号
            if (!ParseDeviceArg(argv[++i], backend_config)) {
                std::cerr << "Invalid --device value. Expected a non-negative integer.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            // 打印帮助文档并立即退出程序
            PrintGenerateUsage(argv[0]);
            return 0;
        } else {
            // 未知参数兜底
            std::cerr << "Unknown or incomplete argument: " << argv[i] << "\n";
            PrintGenerateUsage(argv[0]);
            return 1;
        }
    }

    // 量化类型检验
    if (!quant_type.empty() && quant_type != "q8_0" && quant_type != "q4_0") {
        std::cerr << "Invalid --quant value: " << quant_type << ". Supported values: q8_0, q4_0.\n";
        return 1;
    }

    // 后端校验
    if (!ValidateBackendOrPrint(backend_config)) {
        return 1;
    }

    std::cout << "mini-llama.cpp\n";
    std::cout << "==============\n\n";
    PrintBackendInfo(backend_config);


    // 日志导出目录准备
    if (!dump_logits_dir.empty()) {
        try {
            EnsureDumpDirectory(dump_logits_dir);
        } catch (const std::exception& e) {
            std::cerr << "Logits dump setup failed: " << e.what() << "\n";
        }
    }


    // 用户只指定了配置文件，没有指定模型权重文件
    if (config_path_set && model_path == "models/tiny/model.bin") {
        model_path = std::filesystem::path(config_path).parent_path().string() + "/model.bin";
    }


    RequestContext request = StartRequest("generate", BackendKindName(backend_config.kind), model_path);
    auto stage_start = RequestClock::now();
    LoadedModel lm = LoadModelAndTokenizer(model_path, config_path_set ? config_path : "", tokenizer_path); // 加载分词器
    request.model_load_ms = ElapsedMs(stage_start);      // 记录加载模型的时间
    request.RecordEvent("model_load", request.model_load_ms, 0, model_path);

    // 模型加载失败，并将当前请求的完整事件追踪打印到标准错误流
    if (!lm.model.loaded) {
        request.SetError("Failed to load tokenizer.");
        request.Finish();
        PrintRequestTrace(request, std::cerr);
        std::cerr << request.error << "\n";
        return 1;
    }

    // 分词器加载失败，记录错误、结束请求、打印追踪日志并退出
    if (!lm.tokenizer) {
        request.SetError("Failed to load tokenizer.");
        request.Finish();
        PrintRequestTrace(request, std::cerr);
        std::cerr << request.error << "\n";
        return 1;
    }


    // 词表一致性校验
    if (lm.model.config.vocab_size < lm.tokenizer->vocab_size()) {
        request.SetError("Model vocab_size must be at least " + std::to_string(lm.tokenizer->vocab_size()) + " for the tokenizer.");
        request.Finish();
        PrintRequestTrace(request, std::cerr);
        std::cerr << request.error << "\n";
        return 1;
    }

    MiniLlamaModel& model = lm.model;
    // 模型量化
    try {
        stage_start = RequestClock::now();
        ApplyQuantOverride(model, quant_type);
        request.RecordEvent("quantize", ElapsedMs(stage_start), 0, quant_type.empty() ? "model-naive" : quant_type);
    } catch (const std::exception& e) {
        request.SetError("quantization failed: " + std::string(e.what()));
        request.Finish();
        PrintRequestTrace(request, std::cerr);
        std::cerr << request.error << "\n";
        return 1;
    }

    // 是否可以加载cuda权重
    if (!PrepareCudaWeightsOrPrint(model, backend_config)) {
        request.SetError("CUDA weight preparation failed");
        request.Finish();
        PrintRequestTrace(request, std::cerr);
        return 1;
    }


    // 提示词编码
    std::unique_ptr<ITokenizer>& tokenizer = lm.tokenizer;
    stage_start = RequestClock::now();
    std::vector<int> tokens = tokenizer->Encode(prompt);

    request.tokenizer_ms = ElapsedMs(stage_start);    // 记录序列化时间
    request.prefill_tokens = static_cast<int>(tokens.size());
    request.RecordEvent("tokenize", request.tokenizer_ms, request.prompt_tokens, "prompt");
    std::cout << "prompt: " << prompt << "\n";
    std::cout << "tokens: [";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) {
            std::cout << " ";
        }
        std::cout << tokens[i];
    }
    std::cout << "]\n";

    // 线城池初始化+终端打印
    SetThreadCount(n_threads);
    std::cout << "sampling: temperature=" << temperature << ", top_k=" << top_k << ", seed=" << seed << "\n";
    std::cout << "quant: " << (quant_type.empty() ? "model-native" : quant_type) << "\n";
    std::cout << "threads: " << GetThreadCount() << "\n\n";
    if (backend_config.kind == BackendKind::kCuda) {
        PrintCudaExecutionSummary(model);
        std::cout << "\n";
    }


    // 上下文长度安全校验
    if (tokens.size() > static_cast<size_t>(model.config.max_seq_len)) {
        request.SetError("Prompt is too long for max_seq_len=" + std::to_string(model.config.max_seq_len) + ".");
        request.Finish();
        PrintRequestTrace(request, std::cerr);
        std::cerr << request.error << "\n";
        return 1;
    
    }
    if (tokens.size() + static_cast<size_t>(n_predict) > static_cast<size_t>(model.config.max_seq_len)) {
        request.SetError("Requested tokens exceed context window.")    ;
        request.Finish();
        PrintRequestTrace(request, std::cerr);
        std::cerr << request.error << "\n";
        return 1;    
    }


    // 初始化上下文以及采样器
    MiniLlamaContext ctx(&model);
    SamplingParams simpling_params;
    simpling_params.temperature = temperature;
    simpling_params.top_k = top_k;
    simpling_params.seed = seed;
    MiniSampler sampler(simpling_params);

    size_t prompt_len = tokens.size();    // 记录提示词长度
    int step = 0;

    try {
        Tensor logits;
        std::cout << "prefill...\n";
        stage_start = RequestClock::now();
        if (!dump_logits_dir.empty()) {
            // 逐词预填充
            for (size_t i = 0; i < prompt_len; ++i) {
                MiniBatch prefill_step = MiniBatch::FromTokens({tokens[i]}, static_cast<int>(i));
                logits = ForwardBatch(ctx, model, prefill_step);
                ++ctx.n_prefill_tokens;
                DumpLogits(logits, dump_logits_dir + "/logits_step" + std::to_string(step) + ".bin");
                ++step;
            } 
        } else {
            // 批次预处理
            MiniBatch prefill = MiniBatch::FromTokens(tokens, 0);
            logits = ForwardBatch(ctx, model, prefill);
            ctx.n_prefill_tokens += static_cast<int>(tokens.size());
        }
        request.prefill_ms = ElapsedMs(stage_start);
        request.prefill_tokens = static_cast<int>(prompt_len);
        request.RecordEvent("prefill", request.prefill_ms, request.prefill_tokens, dump_logits_dir.empty() ? "batch" : "stem_dump");

        if (n_predict > 0) {
        // decode
            stage_start = RequestClock::now();
            int next_token = sampler.Sample(logits, simpling_params);
            request.sample_ms += ElapsedMs(stage_start);
            tokens.push_back(next_token);

            std::cout << "decode loop...\n";
            for (int i = 1; i < n_predict; ++i) {
                MiniBatch decode_batch = MiniBatch::Single(tokens.back(), static_cast<size_t>(tokens.size() - 1));
                stage_start = RequestClock::now();
                logits = ForwardBatch(ctx, model, decode_batch);
                double decode_ms = ElapsedMs(stage_start);
                request.decode_ms += decode_ms;
                request.RecordEvent("decode", decode_ms, 1, "pos = " + std::to_string(tokens.size() - 1)); 

                ++ctx.n_decode_tokens;
                ++request.decode_tokens;

                if (!dump_logits_dir.empty()) {
                    DumpLogits(logits, dump_logits_dir + "/logits_step" + std::to_string(step) + ".bin");
                    ++step;
                }

                stage_start = RequestClock::now();
                next_token = sampler.Sample(logits, simpling_params);
                request.sample_ms += ElapsedMs(stage_start);
                tokens.push_back(next_token);
                if (next_token == tokenizer->eos_id()) {
                    break;
                }
            }
        } else {
            std::cout << "decode loop skipped.\n";
        }


    } catch (const std::exception& e) {
        request.SetError("Inference failed: " + std::string(e.what()));
        request.Finish();
        PrintRequestTrace(request, std::cerr);
        std::cerr << request.error << "\n";
        return 1;
    }

    
    // 导出生成的token
    if (!dump_logits_dir.empty()) {
        std::vector<int> generated(tokens.begin() + prompt_len, tokens.end());
        try {
            DumpGeneratedTokens(generated, dump_logits_dir + "/generation_tokens.txt");
        } catch (const std::exception& e) {
            request.SetError("Logits dump failed: " + std::string(e.what()));
            request.Finish();
            PrintRequestTrace(request, std::cerr);
            std::cerr << request.error << "\n";
            return 1;
        }
    }

    // 提取结果与结束请求
    std::vector<int> generated(tokens.begin() + prompt_len, tokens.end());
    request.generated_tokens = static_cast<int>(generated.size());
    request.RecordEvent("sample", request.sample_ms, request.generated_tokens, "generated_tokens");
    request.Finish();

    std::cout << "\ngenerated tokens: [";
    for (size_t i = 0; i < generated.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << generated[i];
    }
    std::cout << "]\n";

    std::string generated_text = tokenizer->Decode(generated);
    std::cout << "generated text: \"" << generated_text << "\"\n";
    if (backend_config.kind == BackendKind::kCuda) {
        std::cout << "\n";
        PrintCudaRuntimeSummary(model);
    }
    PrintRequestTrace(request);

    return 0;
}



/*
-------------------------------------------------------------------------------------------------------------------------------------------
Run mode (interactive chat)
-------------------------------------------------------------------------------------------------------------------------------------------
*/

// run模式的帮助文档
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


// 运行Run命令，循环生成
static int RunChat(int argc, char** argv) {

    // 帮助文档
    if (argc >= 3 && (std::strcmp(argv[2], "-h") == 0 || std::strcmp(argv[2], "--help") == 0)) {
        PrintRunUsage(argv[0]);
        return 0;
    }

    // 最少三个参数
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
    BackendConfig backend_config;

    // 从第三个参数开始
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            // 解析采样温度
            if (!ParseFloatArg(argv[++i], temperature)) {
                std::cerr << "Invalid --temperature value. Expected a non-negative float.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            // 解析 Top-K 采样值
            if (!ParseIntArg(argv[++i], top_k)) {
                std::cerr << "Invalid --top-k value. Expected a non-negative integer.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            // 解析随机种子
            if (!ParseUintArg(argv[++i], seed)) {
                std::cerr << "Invalid --seed value. Expected a non-negative integer.\n";
                return 1;
            }
        } else if ((std::strcmp(argv[i], "-n") == 0 || std::strcmp(argv[i], "--n-predict") == 0) && i + 1 < argc) {
            // 解析单轮最大回复 Token 数
            if (!ParseIntArg(argv[++i], max_response_tokens) || max_response_tokens <= 0) {
                std::cerr << "Invalid --n-predict value. Expected a positive integer.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
            // 记录用户显式指定的分词器路径
            tokenizer_path = argv[++i];
        } else if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            // 解析后端类型
            if (!ParseBackendArg(argv[++i], backend_config)) {
                std::cerr << "Invalid --backend value. Supported values: cpu, cuda.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            // 解析显卡编号
            if (!ParseDeviceArg(argv[++i], backend_config)) {
                std::cerr << "Invalid --device value. Expected a non-negative integer.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            // 如果是帮助标志则打印说明退出
            PrintRunUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << argv[i] << "\n";
            PrintRunUsage(argv[0]);
            return 1;
        }
    }


    if (!ValidateBackendOrPrint(backend_config)) {
        return 1;
    }

    // 加载分词器与模型
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
    if (!PrepareCudaWeightsOrPrint(model, backend_config)) {
        return 1;
    }
    std::unique_ptr<ITokenizer>& tokenizer = lm.tokenizer;


    // 标准提示词构建器
    PromptBuilder builder;
    if (!lm.chat_template.empty()) {
        builder.SetChatTemplate(lm.chat_template);
    }

    Terminal term;
    ChatSession session;
    session.sampling_params.temperature = temperature;
    session.sampling_params.top_k = top_k;
    session.sampling_params.seed = seed;

    // 如果模型没有自带对话模板，系统会手动为对话注入一条默认的系统提示词
    if (lm.chat_template.empty()) {
        session.AddMessage("system", "You are a helpful assistant.");
    }

    term.PrintMessage("mini-llama.cpp chat");
    term.PrintMessage("backend: " + BackendKindName(backend_config.kind));
    // 打印极其详细的显存与权重状态报告
    if (backend_config.kind == BackendKind::kCuda) {
        LinearWeightTypeCounts counts = ModelLinearWeightTypeCounts(model);
        size_t uploaded_f32 = ModelCudaUploadedF32WeightCount(model);
        size_t uploaded_q8_0 = ModelCudaUploadedQ80WeightCount(model);
        size_t uploaded_q4_0 = ModelCudaUploadedQ40WeightCount(model);
        size_t uploaded_q4_1 = ModelCudaUploadedQ41WeightCount(model);
        size_t q8_0_fallback = counts.q8_0 >= uploaded_q8_0 ? counts.q8_0 - uploaded_q8_0 : 0;
        size_t q4_0_fallback = counts.q4_0 >= uploaded_q4_0 ? counts.q4_0 - uploaded_q4_0 : 0;
        size_t q4_1_fallback = counts.q4_1 >= uploaded_q4_1 ? counts.q4_1 - uploaded_q4_1 : 0;
        term.PrintMessage("cuda: " + CudaDeviceSummary(backend_config.device_id));
        term.PrintMessage("uploaded weights: " + std::to_string(ModelCudaUploadedWeightCount(model)));
        term.PrintMessage("gpu memory used: " + std::to_string(ModelCudaMemoryBytes(model)) + " bytes (" + FormatMb(ModelCudaMemoryBytes(model)) + ")");
        term.PrintMessage("cuda f32 Linear weights: " + std::to_string(uploaded_f32) + "/" + std::to_string(counts.total()));
        if (counts.q8_0 > 0) {
            term.PrintMessage("cuda q8_0 Linear weights: " + std::to_string(uploaded_q8_0) + "/" + std::to_string(counts.q8_0));
        }
        if (counts.q4_0 > 0) {
            term.PrintMessage( "cuda q4_0 Linear weights: " + std::to_string(uploaded_q4_0) + "/" + std::to_string(counts.q4_0));
        }
        if (counts.q4_1 > 0) {
            term.PrintMessage("cuda q4_1 Linear weights: " + std::to_string(uploaded_q4_1) + "/" + std::to_string(counts.q4_1));
        }
        if (q8_0_fallback + q4_0_fallback + q4_1_fallback > 0) {
            term.PrintMessage(
                "cuda fallback: unsupported or missing quantized Linear weights use "
                "CPU path" +
                std::string(" (Q8_0=") + std::to_string(q8_0_fallback) +
                ", Q4_0=" + std::to_string(q4_0_fallback) +
                ", Q4_1=" + std::to_string(q4_1_fallback) + ")");
        }
    }

    term.PrintMessage(BackendExecutionNote(backend_config));
    term.PrintMessage("Type /help for commands, /exit to quit.\n");
    if (lm.chat_template.empty() && model.config.max_seq_len <= 256) {
        term.PrintMessage("Tiny teaching model: random weights, small context window, smoke-test ""output.");
        term.PrintMessage("Real chat demo: ./build/mini-llama run models/chat -n 8\n");
    }

    
    auto ctx = std::make_unique<MiniLlamaContext>(&model);

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
            ctx = std::make_unique<MiniLlamaContext>(&model);  // 重新构建上下文
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

        RequestContext request = StartRequest("run", BackendKindName(backend_config.kind), model_dir);
        // 准备完整的对话历史
        std::vector<ChatMessage> candidate_messages = session.messages; 
        candidate_messages.push_back({"user", input});
        

        // 把iuput构建为标准提示词
        auto stage_start = RequestClock::now();
        std::string prompt_text = builder.Build(candidate_messages);
        request.RecordEvent("prompt_build", ElapsedMs(stage_start), 0, "messages=" + std::to_string(candidate_messages.size()));


        // 序列化
        stage_start = RequestClock::now();
        std::vector<int> tokens = tokenizer->Encode(prompt_text);
        request.tokenizer_ms = ElapsedMs(stage_start);
        request.prompt_tokens = static_cast<int>(tokens.size());
        request.RecordEvent("tokenizer", request.tokenizer_ms, request.prompt_tokens, "prompt");

        // 上下文长度溢出校验
        if (tokens.size() >= static_cast<size_t>(model.config.max_seq_len)) {
            request.SetError("prompt uses " + std::to_string(tokens.size()) + " tokens, context window is " + std::to_string(model.config.max_seq_len) + ". Use /clear or a shorter prompt.");
            request.Finish();
            PrintRequestTrace(request);
            term.PrintMessage("Error: " + request.error + "\n");  
            continue;
        }

        // 计算剩余空间并限制上限
        int max_response = model.config.max_seq_len - static_cast<int>(tokens.size());
        if (max_response > max_response_tokens) {
            max_response = max_response_tokens;
        }


        // 构建采样器
        session.messages = candidate_messages;
        MiniSampler sampler(session.sampling_params);
        Tensor logits;

        auto start = std::chrono::steady_clock::now();

        // 计算最长公共前缀
        size_t cached_prefix_len = session.LongestCachePrefix(tokens);
        size_t context_prefix_len = CommonPrefixLength(ctx->token_history, tokens);
        size_t prefix_len = std::min(cached_prefix_len, context_prefix_len);

        if (prefix_len >= tokens.size()) {
            // 整个prompt都在kvcache里
            ctx = std::make_unique<MiniLlamaContext>(&model);
            prefix_len = 0;
        } else if (prefix_len == 0) {
            // 完全不匹配
            ctx = std::make_unique<MiniLlamaContext>(&model);
        } else if (prefix_len < ctx->token_history.size()) {
            ctx->token_history.resize(prefix_len);
            ctx->pos = static_cast<int>(prefix_len - 1);
        }

        // 把 Prompt 中前面匹配过的部分切掉，只保留需要新计算的部分，存入 new_prompt_tokens
        std::vector<int> new_prompt_tokens(tokens.begin() + static_cast<std::ptrdiff_t>(prefix_len), tokens.end());
        session.SetTokenHistory(tokens);
        

        // 增量填充
        {
            MiniBatch prefill = MiniBatch::FromTokens(new_prompt_tokens, static_cast<int>(prefix_len));
            stage_start = RequestClock::now();
            logits = ForwardBatch(*ctx, model, prefill);
            request.prefill_ms = ElapsedMs(stage_start);
            request.prefill_tokens = static_cast<size_t>(new_prompt_tokens.size());
            request.RecordEvent("prefill", request.prefill_ms, request.prefill_tokens, "radix_hit=" + std::to_string(cached_prefix_len) + ", prefix_reuse=" + std::to_string(prefix_len));
            ctx->n_prefill_tokens += static_cast<int>(new_prompt_tokens.size());
        }


        // 逐词生成与流式输出
        std::vector<int> generated_ids;
        std::string streamed_reply;
        int generated_count = 0;

        term.PrintAssistantPrefix();
        try {
            for (int i = 0; i < max_response; ++i) {
                // 采样下一个词
                stage_start = RequestClock::now();
                int next_token = sampler.Sample(logits, session.sampling_params);
                request.sample_ms += ElapsedMs(stage_start);
                tokens.push_back(next_token);
                session.AppendToken(next_token);
                ++generated_count;

                // 流式输出
                if (next_token != tokenizer->eos_id()) {
                    generated_ids.push_back(next_token);
                    std::string current_reply = tokenizer->Decode(generated_ids);
                    if (current_reply.size() > streamed_reply.size()) {
                        term.PrintTokenText(current_reply.substr(streamed_reply.size()));
                        term.Flush();
                        streamed_reply = current_reply;
                    }
                }


                // 前向传播下一个词的logits
                MiniBatch decode_batch = MiniBatch::Single(next_token, static_cast<int>(tokens.size() - 1));
                stage_start = RequestClock::now();
                logits = ForwardBatch(*ctx, model, decode_batch);
                double decode_ms = ElapsedMs(stage_start);
                request.decode_ms += decode_ms;
                request.RecordEvent("decode", decode_ms, 1, "pos= " + std::to_string(tokens.size() - 1));

                ++ctx->n_decode_tokens;
                ++request.decode_tokens;
                if (next_token == tokenizer->eos_id()) {
                    break;
                }
            }
        } catch (const std::exception& e) {
            term.NewLine();
            request.SetError("nference error: " + std::string(e.what()));
            request.Finish();
            PrintRequestTrace(request);
            term.PrintMessage(request.error + "\n");
            continue;
        }

        // 把这次生成的所有 Token（generated_ids）解码成完整的字符串
        // 因为在循环内部，每生成一个词就打印一次，可能会因为遇到 EOS 提前 break 退出循环，导致最后一个词还没来得及打印。这里对比一下完整字符串和已打印字符串的长度，如果有遗漏，立刻补打上去。
        std::string assistant_reply = tokenizer->Decode(generated_ids);
        if (assistant_reply.size() > streamed_reply.size()) {
            term.PrintTokenText(assistant_reply.substr(streamed_reply.size()));
        }
        term.NewLine();
        term.NewLine();

        auto end = std::chrono::steady_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

        session.AddMessage("assistant", assistant_reply);
        session.RecordTurn(static_cast<int>(new_prompt_tokens.size()), generated_count, elapsed_ms);
        request.generated_tokens = generated_count;
        request.RecordEvent("sample", request.sample_ms, request.generated_tokens, "generated_tokens");
        session.RecordPrefix(session.token_history);
        request.Finish();
        PrintRequestTrace(request);
    }

    term.PrintMessage("Goodbye.\n");
    return 0;
}




// 全局帮助菜单
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


// 查询打印显卡信息
// --cuda-info
static int RunCudaInfo() {
  try {
    int count = CudaDeviceCount();
    std::cout << "CUDA devices: " << count << "\n";
    for (int device_id = 0; device_id < count; ++device_id) {
      CudaDeviceInfo info = CudaGetDeviceInfo(device_id);
      double total_gb = static_cast<double>(info.total_memory_bytes) /
                        (1024.0 * 1024.0 * 1024.0);
      std::cout << "device " << info.id << ": " << info.name << "\n"
                << "  device name: " << info.name << "\n"
                << "  compute capability: " << info.compute_major << "."
                << info.compute_minor << "\n"
                << "  total memory: " << info.total_memory_bytes << " bytes ("
                << std::fixed << std::setprecision(2) << total_gb << " GB)\n"
                << "  driver version: " << info.driver_version << "\n"
                << "  runtime version: " << info.runtime_version << "\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "CUDA info failed: " << e.what() << "\n";
    return 1;
  }
}




/*
-------------------------------------------------------------------------------------------------------------------------------------------
Inspect mode
-------------------------------------------------------------------------------------------------------------------------------------------
*/

// 打印帮助信息
static void PrintInspectUsage(const char* prog) {
  std::cout << "Usage: " << prog << " inspect <model-path|dir> [options]\n"
            << "Options:\n"
            << "  --backend cpu|cuda   Execution backend (default: cpu; cuda "
               "requires -DMINI_LLAMA_CUDA=ON)\n"
            << "  --device <n>         CUDA device id for --backend cuda "
               "(default: 0)\n"
            << "  -h, --help           Show this help\n";
}



// 加载模型文件
static MiniLlamaModel LoadModelForInspect(const std::string& path) {
    std::string model_path = path;
    if (std::filesystem::is_directory(path)) {
        // 穿的是目录
        std::filesystem::path bin_path = std::filesystem::path(path) / "model.bin";
        std::filesystem::path json_path = std::filesystem::path(path) / "model.json";
        if (std::filesystem::exists(bin_path) && std::filesystem::exists(json_path)) {
            return LoadModel(json_path.string(), bin_path.string());
        }


        std::string gguf_path = FindGgufInDirectory(path);
        if (!gguf_path.empty()) {
            model_path = gguf_path;
        } else {
            return LoadModel(json_path.string(), bin_path.string());
        }
    }

    if (IsGgufFile(model_path)) {
        return LoadGgufModel(model_path);
    }

    std::filesystem::path model_file(model_path);
    std::string config_path = (model_file.parent_path() / "model.json").string();
    return LoadModel(config_path, model_path);
}


// 查看模型元数据
// --inspect
static int RunInspect(int argc, char** argv) {
    if (argc >= 3 && (std::strcmp(argv[2], "-h") == 0 || std::strcmp(argv[2], "--help") == 0)) {
        PrintInspectUsage(argv[0]);
        return 0;
    }

    // 必须大于三个参数
    if (argc < 3) {
        std::cerr << "Missing model path.\n";
        PrintInspectUsage(argv[0]);
        return 1;
    }

    std::string model_path = argv[2];
    BackendConfig backend_config;
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            // 解析后端
            if (!ParseBackendArg(argv[++i], backend_config)) {
                std::cerr << "Invalid --backend value. Supported values: cpu, cuda.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            // 解析后端编号
            if (!ParseDeviceArg(argv[++i], backend_config)) {
                std::cerr << "Invalid --device value. Expected a non-negative integer.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
                // 打印帮助文档
                PrintInspectUsage(argv[0]);
                return 0;
        } else {
            std::cerr << "Unknown argument: " << argv[i] << "\n";
            PrintInspectUsage(argv[0]);
            return 1;
        }
    }

    if (!ValidateBackendOrPrint(backend_config)) {
        return 1;
    }

    if (std::filesystem::is_directory(model_path)) {
        std::filesystem::path config_path = std::filesystem::path(model_path) / "model.json";
        if (std::filesystem::exists(config_path) && !mini_llama::InspectModel(config_path.string())) {
            return 1;
        }
    }

    MiniLlamaModel model = LoadModelForInspect(model_path);
    if (!model.loaded) {
        std::cerr << "Failed to load model: " << model.load_error << "\n";
        return 1;
    }


    std::cout << "backend: " << BackendKindName(backend_config.kind) << std::endl;
    if (backend_config.kind == BackendKind::kCuda) {
        std::cout << "cuda: " << CudaDeviceSummary(backend_config.device_id) << "\n";
        if (!PrepareCudaWeightsOrPrint(model, backend_config)) {
            return 1;
        }
        PrintCudaExecutionSummary(model);
    }
    return 0;
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




/*
-------------------------------------------------------------------------------------------------------------------------------------------
Bench mode
-------------------------------------------------------------------------------------------------------------------------------------------
*/

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




// 用于基准测试（Benchmark）的通用计时工具
static double BenchmarkRepeated (int warmup, int iterations, const std::function<void()>& fn) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        fn();
    }
    auto end = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return elapsed_ms / static_cast<double>(iterations);
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
    BackendConfig backend_config;


    // 解析参数
    for (int i = 3; i < argc; ++i) {
        if ((std::strcmp(argv[i], "-p") == 0 || std::strcmp(argv[i], "--prompt") == 0) && i + 1 < argc) {
            // 解析提示词
            prompt = argv[++i];
        } else if ((std::strcmp(argv[i], "-n") == 0 || std::strcmp(argv[i], "--n-predict") == 0) && i + 1 < argc) {
            // 解析每次最多生成的token数量
            if (!ParseIntArg(argv[++i], n_predict)) {
                std::cerr << "Invalid --n-predict value.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            // 解析随机数种子
            if (!ParseUintArg(argv[++i], seed)) {
                std::cerr << "Invalid --seed value.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
            // 解析分词器路径
            tokenizer_path = argv[++i];
        } else if (std::strcmp(argv[i], "--quant") == 0 && i + 1 < argc) {
            // 解析量化类型
            quant_type = argv[++i];
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            // 解析线程池线程数量
            if (!ParseIntArg(argv[++i], n_threads) || n_threads < 0) {
                std::cerr << "Invalid --threads value.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            // 解析后端
            if (!ParseBackendArg(argv[++i], backend_config)) {
                std::cerr << "Invalid --backend value. Supported values: cpu, cuda.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            // 解析设备标号
            if (!ParseDeviceArg(argv[++i], backend_config)) {
                std::cerr << "Invalid --device value. Expected a non-negative integer.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--verbose") == 0) {
            // 详细输出
            verbose = true;
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            // 打印帮助文档
            PrintBenchUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << argv[i] << "\n";
            PrintBenchUsage(argv[0]);
            return 1;
        }
    }


    // 量化类型合法性校验
    if (!quant_type.empty() && quant_type != "q8_0" && quant_type != "q4_0") {
        std::cerr << "Invalid --quant value: " << quant_type << ". Supported values: q8_0, q4_0.\n";
        return 1;
    }

    // 后端校验
    if (!ValidateBackendOrPrint(backend_config)) {
        return 1;
    }

    // 加载模型与分词器
    LoadedModel lm = LoadModelAndTokenizer(model_dir, "", tokenizer_path);
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

    try {
        ApplyQuantOverride(model, quant_type);
    } catch (const std::exception& e) {
        std::cerr << "Quantization failed: " << e.what() << "\n";
        return 1;
    }
    if (!PrepareCudaWeightsOrPrint(model, backend_config)) {
        return 1;
    }



    std::cout << "Benchmark: " << model_dir << "\n";
    std::cout << "  backend: " << BackendKindName(backend_config.kind) << "\n";
    if (backend_config.kind == BackendKind::kCuda) {
        std::cout << "  cuda: " << CudaDeviceSummary(backend_config.device_id) << "\n";
    }
    std::cout << "  " << BackendExecutionNote(backend_config) << "\n";
    std::cout << "  prompt: \"" << prompt << "\" (" << tokens.size() << " tokens)\n";
    SetThreadCount(n_threads);
    std::cout << "  n_predict: " << n_predict << "\n";
    std::cout << "  seed: " << seed << "\n";
    std::cout << "  quant: " << (quant_type.empty() ? "model-native" : quant_type) << "\n";
    std::cout << "  threads: " << GetThreadCount() << "\n";
    std::cout << "  verbose: " << (verbose ? "true" : "false") << "\n\n";
    if (backend_config.kind == BackendKind::kCuda) {
        PrintCudaExecutionSummary(model, "  ");
        std::cout << "\n";
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

    size_t actual_bytes = ModelWeightBytes(model);
    size_t f32_bytes = ModelWeightBytesF32(model);
    std::cout << "\n  weight memory:\n";
    std::cout << "    actual:    " << actual_bytes << " bytes (" << std::fixed
              << std::setprecision(2) << (actual_bytes / (1024.0 * 1024.0))
              << " MB)\n";
    std::cout << "    f32 equiv: " << f32_bytes << " bytes (" << std::fixed
              << std::setprecision(2) << (f32_bytes / (1024.0 * 1024.0))
              << " MB)\n";
    std::cout << "    savings:   " << std::fixed << std::setprecision(2)
              << (static_cast<double>(f32_bytes) / actual_bytes)
              << "x compression\n";
    if (backend_config.kind == BackendKind::kCuda) {
        std::cout << "\n  cuda runtime:\n";
        PrintCudaRuntimeSummary(model, "    ");
   }

    if (!quant_type.empty()) {
        try {
            Tensor baseline_logits = RunLogitsForTokens(baseline_model, tokens);
            Tensor quant_logits = RunLogitsForTokens(model, tokens);
            auto [max_err, mean_err] = LogitsError(baseline_logits, quant_logits);

            std::cout << "\n  logits error vs model-native:\n";
            std::cout << "    max:  " << std::scientific << std::setprecision(3) << max_err << "\n";
            std::cout << "    mean: " << std::scientific << std::setprecision(3) << mean_err << "\n";
        } catch (const std::exception& e) {
            std::cerr << "Failed to compute logits error: " << e.what() << "\n";
            return 1;
        }
    }

    return 0;
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
        return mini_llama::RunGenerate(argc, argv);
    } else if (command == "run") {
        return mini_llama::RunChat(argc, argv);
    } else if (command == "inspect") {
        return mini_llama::RunInspect(argc, argv);
    } else if (command == "bench") {
        return mini_llama::RunBench(argc, argv);
    } else if (command == "inspect-gguf") {
        return mini_llama::RunInspectGguf(argc, argv);
    } else if (command == "--cuda-info") {
        return mini_llama::RunCudaInfo();
    } else if (command == "-h" || command == "--help") {
        mini_llama::PrintGlobalUsage(argv[0]);
        return 0;
    } else if (!command.empty() && command[0] == '-') {
        // 没有子命令，只给了参数
        // 回退到genrate命令
        std::vector<char*> shifted_argv;
        shifted_argv.reserve(argc + 1);
        shifted_argv.push_back(argv[0]);
        shifted_argv.push_back(const_cast<char*>("generate"));
        for (int i = 1; i < argc; ++i) {
            shifted_argv.push_back(argv[i]);
        }
        return mini_llama::RunGenerate(argc + 1, shifted_argv.data());
    } else  {
        std::cerr << "Unknown command: " << command << std::endl;
        mini_llama::PrintGlobalUsage(argv[0]);
        return 1;
    }
}
