#include "mini_llama/tokenizer.h"

#include <stdexcept>
#include <sstream>
#include <fstream>
#include <cctype>
#include <string>
#include <algorithm>
#include <memory>
#include <iostream>



namespace {
// 从json中解析出来一个字符值
static std::string ParseJsonString(const std::string& content, size_t pos) {
    while (pos < content.size() && content[pos] != '"') {
        ++pos;
    }
    if (pos >= content.size() || content[pos] != '"') {
        throw std::runtime_error("expected opening quote for string");
    }
    ++pos; // 跳过第一个起始"

    std::string result;
    // 逐字符读取，直到遇到闭合"
    while (pos < content.size() && content[pos] != '"') {
        // 遇到转义字符
        if (content[pos] == '\\' && pos + 1 < content.size()) {
            char next = content[pos + 1];
            if (next == 'n') {
                result += '\n';
            } else if (next == 't') {
                result += '\t';
            } else if (next == 'r') {
                result += '\r';
            } else if (next == '\\') {
                result += '\\';
            } else if (next == '"') {
                result += '"';
            } else if (next == 'u' && pos + 5 < content.size()) {
                // 处理unicode
                std::string hex = content.substr(pos + 2, 4);   // 获取\uXXXX的4位16进制数
                unsigned int codepoint = std::stoul(hex, nullptr, 16);     // 讲16进制数转化为无符号整数
                // 0~127,可以用AscII标识
                if(codepoint <= 0x7F) {
                    result += static_cast<char>(codepoint);
                } else {
                    result += '?';
                }
                pos += 6;
                continue;
            } else {
                result += next;
            }
            pos += 2;
        } else {
            result += content[pos];
            ++pos;
        }
    }

    if (pos >= content.size()) {
        throw std::runtime_error("unterminated string");
    }
    ++pos;   // 跳过闭合'"'
    return result;
}


// 从json中解析一个整数
static int ParseJosnInt(const std::string& content, size_t pos) {
    // 跳过非数字符号
    while (pos < content.size() && !std::isdigit(static_cast<unsigned char>(content[pos])) && content[pos] != '-') {
        ++pos;
    }

    int start = pos;
    if (content[pos] == '-') {
        ++pos;
    }
    while (pos < content.size() && std::isdigit(static_cast<unsigned char>(content[pos]))) {
        ++pos;
    }
    return std::stoi(content.substr(start, pos - start));
}


// 从json中解析一个布尔值
static bool ParseJsonBool(const std::string& content, size_t& pos) {
    if (content.substr(pos, 4) == "true") {
        pos += 4;
        return true;
    }
    if (content.substr(pos, 5) == "false") {
        pos += 5;
        return false;
    }
    throw std::runtime_error("expected true or false");
}

}

namespace mini_llama {

/*
[
  {"id": 0, "content": "<unk>", "special": true},
  {"id": 1, "content": "<bos>", "special": true},
  {"id": 3, "content": "hello", "special": false}
]
*/

// 构造函数，加载整个词表
JsonVocabTokenizer::JsonVocabTokenizer(const std::string& vocab_path) {
    // 打开文件
    std::ifstream f(vocab_path);
    if (!f.is_open()) {
        throw std::runtime_error("failed to open vocab: " + vocab_path);
    }

    // 读取文件内容到字符串流
    std::stringstream buffer;
    buffer << f.rdbuf();             // rdbuf是文件底层的读缓冲区，把文件内容送到字符串流中
    if (f.bad()) {
        throw std::runtime_error("failed to read vocab: " + vocab_path);
    }

    std::string content = buffer.str();            // 取出完成文本


    // 找到"["
    size_t pos = content.find('[');
    if (pos == std::string::npos) {
        throw std::runtime_error("vocab JSON must be an array");
    }
    ++pos;

    int max_id = -1;
    // 逐个寻找词条对象
    while (pos < content.size()) {
        while (pos < content.size() && content[pos] != '{') {
            if (content[pos] == ']') {
                break;
            }
            ++pos;
        }

        if (pos >= content.size() || content[pos] != '{') {
            break;
        }
        ++pos;

        // 遇到第一个词条对象，用两个标志位记录是否读到了必填字段
        VocabEntry entry;
        bool has_id = false;
        bool has_content = false;

        while (pos < content.size() && content[pos] != '}') {
            // 不处理空格和','
            while (pos < content.size() && (std::isspace(static_cast<unsigned char>(content[pos])) || content[pos] == ',')) {
                ++pos;
            }
            // 越界
            if (pos >= content.size() || content[pos] == '}') {
                break;
            }

            // 解析key,比如id/content/special
            std::string key = ParseJsonString(content, pos);

            // 跳过冒号
            while (pos < content.size() && content[pos] != ':') {
                ++pos;
            }
            if (pos < content.size()) {
                ++pos;
            }

            // 跳过空格
            while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) {
                ++pos;
            }

            if (key == "id") {
                entry.id = ParseJosnInt(content, pos);
                has_id = true;
            } else if (key == "content") {
                entry.content = ParseJsonString(content, pos);
                has_content = true;
            } else if (key == "special") {
                entry.special = ParseJsonBool(content, pos);
            } else {
                // 跳过未定义的值
                if (pos < content.size() && content[pos] == '"') {
                    ParseJsonString(content, pos);
                } else {
                    while (pos < content.size() && content[pos] != ',' && content[pos] != '}') {
                        ++pos;
                    }
                }
            }
        }
        
        // 当前{...}对象解析完毕
        if (pos < content.size() && content[pos] == '}') {
            ++pos;
        }

        if (!has_id || !has_content) {
            throw std::runtime_error("vocab entry missing id or content");
        }

        if (entry.id > max_id) {
            max_id = entry.id;
        }

        if (entry.id < 0) {
            throw std::runtime_error("vocab entry id must be positive");
        }

        if (entry.content == "<bos>") {
            bos_id_ = entry.id;
        } else if (entry.content == "<eos>") {
            eos_id_ = entry.id;
        } else if (entry.content == "<unk>") {
            unk_id_ = entry.id;
        }

        if (static_cast<size_t>(entry.id) >= id_to_entry_.size()) {
            id_to_entry_.reserve(entry.id + 1);
        }
        id_to_entry_[entry.id] = entry;
        content_to_id_.push_back({entry.content, entry.id});
    }

    vocab_size_ = max_id + 1;   // 因为用的是已经训练好的vocab,从0开始连续，所以vocab的大小为max_id + 1
    std::sort(content_to_id_.begin(), content_to_id_.end(), [](const auto& a, const auto& b) {
        return a.first.size() > b.first.size();
    });

}


std::vector<int> JsonVocabTokenizer::Encode(const std::string& text) const {
    std::vector<int> tokens;
    tokens.push_back(bos_id());
    
    size_t pos = 0;
    while (pos < text.size()) {
        bool is_match = false;
        for (const auto& pair : content_to_id_) {
            const std::string& content = pair.first;
            if (content.empty()) {
                continue;
            }

            if (text.compare(pos, content.size(), content) == 0) {
                tokens.push_back(pair.second);
                pos += content.size();
                is_match = true;
                break;
            }
        }

        // 没有匹配的字符串
        if (!is_match) {
            tokens.push_back(unk_id());
            ++pos;
        }
    }

    return tokens;
}

std::string JsonVocabTokenizer::DecodeToken(int token) const {
    if (token < 0 || static_cast<size_t>(token) >= id_to_entry_.size()) {
        return "<unk>";
    }
    return id_to_entry_[token].content;
}



std::string JsonVocabTokenizer::Decode(const std::vector<int>& tokens) const {
    std::string result;
    for (int token : tokens) {
        result += DecodeToken(token);
    }
    return result;
}


int JsonVocabTokenizer::vocab_size() const {
    return vocab_size_;
}
int JsonVocabTokenizer::bos_id() const {
    return bos_id_;
}
int JsonVocabTokenizer::eos_id() const {
    return eos_id_;
}
int JsonVocabTokenizer::unk_id() const {
    return unk_id_;
}


// 工厂模式，决定使用JsonVocabTokenizer/AsciiTokenizer
std::unique_ptr<ITokenizer> CreateToenizer(const std::string vocab_path) {
    if (!vocab_path.empty()) {
        std::ifstream f(vocab_path);
        if (f.good()) {
            try {
                return std::make_unique<JsonVocabTokenizer>(vocab_path);
            } catch (const std::exception& e) {
                std::cerr << "Warning: failed to load vocab: " << vocab_path 
                          << ", failing back to ASCII tokenizer: " << e.what() << std::endl; 
            }
        }
    }
    return std::make_unique<AsciiTokenizer>();
}

}