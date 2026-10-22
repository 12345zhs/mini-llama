#include "mini_llama/terminal.h"
#include "mini_llama/chat.h"
#include "mini_llama/context.h"
#include "mini_llama/model.h"
#include "mini_llama/tokenizer.h"
#include "mini_llama/gguf_loader.h"
#include "mini_llama/gguf_tokenizer.h"
#include "mini_llama/loader.h"

#include <iostream>
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <memory>

namespace mini_llama {

struct LoadedModel {
    MiniLlamaModel model;
    std::unique_ptr<ITokenizer> tokenizer;
    std::string chat_template;
};


// 判断gguf文件是否正确
static bool IsGgufFFile(const std::string& path) {
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
        if (entry.is_regular_file() && entry.path().extension() == ",gguf") {
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
        if (manifest.tokenizer.type != "vocab.json" || manifest.tokenizer.path.empty()) {
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
        result.chat_template = ;
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





static int RunChat(int argc, int** argv) {


    LoadedModel  lm = ;

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

}