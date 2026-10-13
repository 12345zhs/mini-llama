#include"mini_llama/tokenizer.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cctype>
#include <cstring>
#include <limits>

namespace mini_llama {


// 把Unicode码点按UTF-8的规则编写成1~4字节的UTF-8
static std::string CodepointToUtf8(char32_t cp) {
    std::string result;
    if (cp <= 0x7F) {
        result += static_cast<char>(cp);
    } else if (cp <= 0x7FF) {
        result += static_cast<char>(0xC0 | (cp >> 6));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp <= 0xFFFF) {
        result += static_cast<char>(0xE0 | (cp >> 12));
        result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        result += static_cast<char>(0xF0 | (cp >> 18));
        result += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    }
  return result;
}
    

// 读出一个完整的UTF-8
static std::string NextUtf8Codepoint(const std::string& s, size_t& pos) {
    if (pos >= s.size()) {
        return "";
    }
    unsigned char first = s[pos];
    size_t len = 1;
    if ((first & 0x80) == 0) {
        len = 1;
    } else if ((first & 0xE0) == 0xC0) {
        len = 2;
    } else if ((first & 0xF0) == 0xE0) {
        len = 3;
    } else if ((first & 0xF8) == 0xF0) {
        len = 4;
    } else {
        // Invalid UTF-8, skip one byte
        ++pos;
        return "";
    }
    if (pos + len > s.size()) {
        ++pos;
        return "";
    }
    std::string result = s.substr(pos, len);
    pos += len;
    return result;
}


// GPT-2风格的BPE：byte->unicode映射表
static std::vector<std::string> BuildBytesToUnicode() {
    std::vector<std::string> map(256);         // 0~255,每个元素对应的UTF-8字节
    
    std::vector<int> bs;                       // 原样的字节,收集可打印，不会引起问题的字节值,没有控制字符、空格
    // 0x21~0x7E   
    for (int c = '!'; c <= '~'; ++c) {
        bs.push_back(c);
    }
    for (int c = 0xA1; c <= 0xAC; ++c) {
        bs.push_back(c);
    }
    for (int c = 0xAE; c <= 0xFF; ++c) {
        bs.push_back(c);
    }

    std::vector<int> cs = bs;    // 码点数组
    int n = 0;

    // 遍历256个字节，如果存在字节不在bs里（第一步没有放到bs中的，意味着不可打印/跳过的字节），要追加到bs中
    // 
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }

    for (size_t i = 0; i < bs.size(); ++i) {
        map[bs[i]] = CodepointToUtf8(static_cast<char32_t>(cs[i]));
    }
    return map;
} 


// 构建字节与Unicode之间的映射表
void BpeTokenizer::BuildByteMappings() {
    b2u_ = BuildBytesToUnicode();
    u2b_.clear();
    for (int b = 0; b < 256; ++b) {
        u2b_[b2u_[b]] = static_cast<uint8_t>(b);
    }
}


// 解析字符串
static bool ParseJsonString(const std::string& s, size_t& pos, std::string& out) {
    // 跳过空白
    while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) {
        ++pos;
    }
    if (pos >= s.size() || s[pos] != '"') {
        return false;
    }
    ++pos;  // 跳过引号

    out.clear();
    //逐字符读取
    while (pos < s.size() && s[pos] != '"') {
        if (s[pos] == '\\' && pos + 1 < s.size()) {
            char esc = s[pos + 1];
            switch(esc) {
                case '"' :
                    out += '"';
                    break;
                case '\\':
                    out += '\\';
                    break;
                case '/':
                    out += '/';
                    break;
                case 'b':
                    out += '\b';
                    break;
                case 'f':
                    out += '\f';
                    break;
                case 'n':
                    out += '\n';
                    break;
                case 'r':
                    out += '\r';
                    break;
                case 't':
                    out += '\t';
                    break;
                case 'u': {
                    if (pos + 5 < s.size()) {
                        std::string hex = s.substr(pos + 2, 4);
                        try {
                            char32_t cp = static_cast<char32_t>(std::stoul(hex, nullptr, 16));
                            out += CodepointToUtf8(cp);     // 之前ASCII只处理ASCII编码，非ASCII会用？，现在用UTF-8
                        } catch(...) {
                            out += '?';
                        }
                        pos += 4;
                    }
                    break;
                }
                default:
                    out += esc;
                    break;
            }
            pos += 2;
        } else {
            out += s[pos];
            ++pos;
        }
    }
    // 跳过'"'
    if (pos < s.size() && s[pos] == '"') {
        ++pos;
    }
    return true;
}


// 解析整形
static bool ParseJsonInt(const std::string& s, size_t& pos, int& out) {
    // 跳过空白
    while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) {
        ++pos;
    }
    if (pos >= s.size()) {
        return false;
    }

    size_t start = pos;
    if (s[pos] == '-') {
        ++pos;
    }
    while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) {
        ++pos;
    }
    if (pos == start) {
        return false;
    }

    try {
        out = std::stoi(s.substr(start, pos - start));
        return true;
    } catch(...) {
        return false;
    }
}


// 负责从三个文件中加载tokenizer的全部数据
// { "token": id, "token2": id2, ... }
bool BpeTokenizer::Load(const std::string& vocab_path, const std::string& merges_path, const std::string& special_path) {
    BuildByteMappings();    // 建立b2u_与u2b_的映射表

    // Load vocab.json
    {
        // 打开文件，并把文件内容读到content中
        std::ifstream f(vocab_path);
        if (!f.is_open()) {
            std::cerr << "Failed to open vocab: " << vocab_path << std::endl;
            return false;
        }
        std::stringstream buf;
        buf << f.rdbuf();
        std::string content = buf.str();


        // 跳过空白
        size_t pos = 0;
        while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) {
            pos++;
        }
        if (pos < content.size() && content[pos] == '{') {
            ++pos;
        }


        int max_val = 0;

        // 循环每个“键值对”
        while(pos < content.size()) {
            while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) {
                ++pos;
            }
            if (pos < content.size() && content[pos] == '}') {
                break;
            }

            std::string key;
            if (!ParseJsonString(content, pos, key)) {
                break;
            }

            // 跳过空白找冒号
            while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) {
                ++pos;
            }
            if (pos >= content.size() || content[pos] != ':') {
                break;
            }
            ++pos; //跳过冒号

            int val = 0;
            if (!ParseJsonInt(content, pos, val)) {
                break;
            }

            // 存进unordered_map中
            vocab_[key] = val;

            if (val > max_val) {
                max_val = val;
            }

            // 跳过空白和逗号
            while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) {
                ++pos;
            }
            if (pos < content.size() && content[pos] == ',') {
                ++pos;
            }
        }

        if (vocab_.empty()) {
            std::cerr << "Vocab is empty" << std::endl;
            return false;
        }

        id_to_token_.resize(max_val + 1);
        for (const auto& kv : vocab_) {
            id_to_token_[kv.second] = kv.first;
        }

        for (const auto& kv : vocab_) {
            if (kv.first.size() >= 4 && kv.first.front() == '<' && kv.first[1] == '|' && kv.first[kv.first.size() - 1] == '>' 
                && kv.first[kv.first.size() - 2] == '|') {
                    special_tokens_.push_back({kv.first, kv.second});
                }
        }

        std::sort(special_tokens_.begin(), special_tokens_.end(), [](const auto& a, const auto& b) {
            return a.first.size() > b.first.size();
        });
    }


    // Load merges.txt
    // "token1 token2"
    {
        std::ifstream f(merges_path);
        if (!f.is_open()) {
            std::cerr << "Failed to open merges: " << merges_path << std::endl;
            return false;
        }

        // 逐行读取
        std::string line;
        int rank = 0;
        while (std::getline(f, line)) {
            std::stringstream ss(line);
            std::string a, b;
            if (ss >> a >> b) {
                merge_ranks_[{a, b}] = rank++;
            }
        }
    }


    // load special_tokens.json
    {
        std::ifstream f(special_path);
        if (f.is_open()) {
            std::stringstream buf;
            buf << f.rdbuf();
            std::string content = buf.str();

            auto find_int = [&](const std::string& key) ->int {
                size_t p = content.find('"' + key + '"');
                if (p == std::string::npos) {
                    return -1;
                }

                p = content.find(':', p);    // 从紧接着的下标开始找
                if (p == std::string::npos) {
                    return -1;
                }

                size_t start = p + 1;
                // 跳过空白
                while (start < content.size() && std::isspace(static_cast<unsigned char>(content[start]))) {
                    start++;
                }
                size_t end = start;
                while (end < content.size() && (std::isdigit(static_cast<unsigned char>(content[end])) || content[end] == '-')) {
                    ++end;
                }
                if (start == end) {
                    return -1;
                }
                try {
                    return std::stoi(content.substr(start, end - start));
                } catch(...) {
                    return -1;
                }
            };

            bos_id_ = find_int("bos_id");
            eos_id_ = find_int("eos_id");
            unk_id_ = find_int("unk_id");
        }
    }



    std::cout << "BPE tokenizer loaded: vocab= " << vocab_.size() 
              << "merges = " << merge_ranks_.size() 
              << "bos = " << bos_id_
              << "eos = " << eos_id_
              << "unk = " << unk_id_ << std::endl;
    return true;
}



std::vector<int> BpeTokenizer::Encode(const std::string& text) const {
    if (text.empty()) {
        return {};
    }

    std::vector<int> result;
    int pos = 0;
    while (pos < text.size()) {
        bool is_match = false;

        // 优先处理special token
        for (const auto& st : special_tokens_) {
            const std::string& token_str = st.first;
            if (pos + token_str.size() <= text.size() && std::memcmp(text.data() + pos, token_str.data(), token_str.size()) == 0) {
                result.push_back(st.second);
                pos += token_str.size();
                is_match = true;
                break;
            }
        }

        if (is_match) {
            continue;
        }

        // 为了获得普通文本[pos, end]，进行第二次遍历
        size_t end = text.size();
        for (const auto& st : special_tokens_) {
            size_t p = text.find(st.first, pos);
            if (p != std::string::npos && p < end) {
                end = p;
            }
        }

        // 取出普通文本
        std::string segment = text.substr(pos, end - pos);
        pos = end;

        // 遍历segment的每一个字节，转化为unicode编码
        std::vector<std::string> word;
        for (unsigned char c : segment) {
            if (c >= b2u_.size()) {
                std::cerr << "ERROR: Byte " << static_cast<int>(c) << ">= b2u_ size" << b2u_.size() << std::endl;
                return {};
            }
            word.push_back(b2u_[c]);
        }


        // BPE合并
        while (word.size() > 1) {
            int best_rank = std::numeric_limits<int>::max();
            size_t best_idx = word.size();

            // 找出rank最小的那位
            for (size_t i = 0; i + 1 < word.size(); ++i) {
                auto it = merge_ranks_.find({word[i], word[i + 1]});
                if (it != merge_ranks_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_idx = i;
                }
            }

            // 没有任何对可以合并，直接跳出
            if (best_idx >= word.size()) {
                break;
            }

            // 将一对进行合并，并取出第二个
            word[best_idx] += word[best_idx + 1];
            word.erase(word.begin() + best_idx + 1);
        }


        // 映射到tokenID
        for (const auto& w : word) {
            auto it = vocab_.find(w);
            if (it != vocab_.end()) {
                result.push_back(it->second);
            } else {
                result.push_back(unk_id());
            }
        }
    }

    return result;
}


std::string BpeTokenizer::DecodeToken(int token) const {
    if (token >= 0 && token < static_cast<int>(id_to_token_.size())) {
        return id_to_token_[token];
    }
    return "";
}

std::string BpeTokenizer::Decode(const std::vector<int>& tokens) const {
    // 转化为unicode
    std::string text;
    for (int id : tokens) {
        text += DecodeToken(id);
    }

    // unicode-》原始字节
    std::string result;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t start = pos;
        std::string cp = NextUtf8Codepoint(text, pos);  // 每次取出来一个UTF-8的unicode
        if (cp.empty()) {
            if (start == pos) {
                ++pos;
            }
            continue;
        }

        // unicode-》原始字节
        auto it = u2b_.find(cp);
        if (it != u2b_.end()) {
            result += static_cast<char>(it->second);
        } else {
            result += cp;
        }
    }

    return result;
}



int BpeTokenizer::vocab_size() const {
    return static_cast<int>(id_to_token_.size());
}
int BpeTokenizer::bos_id() const {
    return bos_id_;
}
int BpeTokenizer::eos_id() const {
    return eos_id_;
}
int BpeTokenizer::unk_id() const {
    return unk_id_;
}




std::unique_ptr<ITokenizer> CreateBpeTokenizer(const std::string& vocab_path, const std::string& merges_path, const std::string& special_path) {
    auto tok = std::make_unique<BpeTokenizer>();
    if (tok->Load(vocab_path, merges_path, special_path)) {
        return tok;
    } 
    return nullptr;
}

}
