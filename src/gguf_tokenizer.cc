#include "mini_llama/gguf_tokenizer.h"
#include "mini_llama/gguf.h"

#include <cstring>
#include <algorithm>
#include <iostream>
#include <limits>

namespace mini_llama {

// 输入一个码点，输出1~4字节UTF-8
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


// byte->unicode的映射表
static std::vector<std::string> BuildBytesToUnicode() {
    std::vector<std::string> map(256);    // 结果数组
    std::vector<int> bs;                  // 保留原样字节

    // 下面给bs添加字节，但没有添加空格、控制符等
    for (int c = '!'; c <= '~'; c++) {
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

// 构建字节与unicode之间的双向映射
void GgufTokenizer::BuildByteMappings() {
    b2u_ = BuildBytesToUnicode();    // byte->unicode
    u2b_.clear();
    for (int b = 0; b < 256; ++b) {
        u2b_[b2u_[b]] = static_cast<uint8_t>(b);         // unicode->byte
    }
}



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


bool GgufTokenizer::Load(const std::string& gguf_path) {
    vocab_.clear();
    id_to_token_.clear();
    merge_ranks_.clear();
    special_tokens_.clear();

    bos_id_ = -1;
    eos_id_ = -1;
    unk_id_ = -1;

    BuildByteMappings();        // 建立unicode<->byte之间的双向映射

    // 读取gguf文件中的metadata的tokenizer.ggml.tokens
    std::vector<std::string> tokens;
    if (!GgufGetMetadataStringArray(gguf_path, "tokenizer.ggml.tokens", tokens)) {
        std::cerr << "GGUF missing tokenizer.ggml.tokens" << std::endl;
        return false;
    }
    // 按数组下标建立vocab和id_to_token
    id_to_token_.resize(tokens.size());
    for (size_t i  = 0; i < tokens.size(); ++i) {
        vocab_[tokens[i]] = static_cast<int>(i);
        id_to_token_[i] = tokens[i]; 
    }

    
    // 读gguf中有没有“tokenizer.ggml.token_type”
    std::vector<int32_t> token_types;
    bool has_token_types = GgufGetMetadataIntArray(gguf_path, "tokenizer.ggml.token_type", token_types);
    // 在gguf中，控制符的类型表示为3
    if (has_token_types && token_types.size() == tokens.size()) {
        for (size_t i = 0; i < tokens.size(); ++i) {
            if(token_types[i] == 3) {
                special_tokens_.push_back({tokens[i], static_cast<int>(i)});
            }
        }
    }

    // 控制符从长到短排序,控制符也要长词优先
    std::sort(special_tokens_.begin(), special_tokens_.end(), [](const auto& a, const auto& b) {
        return a.first.size() > b.first.size();
    });


    // 建立merge_ranks_
    // 读取有gguf的"tokenizer.ggml.merges"，里面全是字符串
    // "h e"    "l l"  "he llo"
    std::vector<std::string> merges;
    if (GgufGetMetadataStringArray(gguf_path, "tokenizer.ggml.merges", merges)) {
        for (int rank = 0; rank < static_cast<int>(merges.size()); ++rank) {
            const std::string& m = merges[rank];
            size_t pos = m.find(' ');
            if (pos == std::string::npos) {
                continue;
            }
            std::string left = m.substr(0, pos);
            std::string right = m.substr(pos + 1);
            merge_ranks_[{left, right}] = rank;
        }
    }


    // 查询bos、eos、unk
    int64_t bos = -1;
    int64_t eos = -1;
    int64_t unk = -1;
    if (GgufGetMetadataInt(gguf_path, "tokenizer.ggml.bos_token_id", bos)) {
        bos_id_ = static_cast<int>(bos);
    }
    if (GgufGetMetadataInt(gguf_path, "tokenizer.ggml.eos_token_id", eos)) {
        eos_id_ = static_cast<int>(eos);
    }
    if (GgufGetMetadataInt(gguf_path, "tokenizer.ggml.unknown_token_id", unk)) {
        unk_id_ = static_cast<int>(unk);
    }


    std::cout << "GGUF tokenizer loaded: vocab=" << vocab_.size()
            << " merges=" << merge_ranks_.size()
            << " special=" << special_tokens_.size() << " bos=" << bos_id_
            << " eos=" << eos_id_ << " unk=" << unk_id_ << std::endl;
    return true;
}


std::vector<int> GgufTokenizer::Encode(const std::string& text) const {
    if (text.empty()) {
        return {};
    }

    std::vector<int> result;
    size_t pos = 0;
     
    while (pos < text.size()) {
        // 先匹配控制符
        bool matched = false;
        for (const auto& st : special_tokens_) {
            const std::string token_str = st.first;
            if (pos + token_str.size() <= text.size() && std::memcmp(text.data() + pos, token_str.data(), token_str.size()) == 0) {
                result.push_back(st.second);
                pos += token_str.size();
                matched = true;
                break;
            }
        }

        if (matched) {
            continue;
        }

        // 当前位置不是特殊符，在剩下的文本里找最近的special_token位置
        size_t end = text.size();
        for (const auto& st : special_tokens_) {
            size_t p = text.find(st.first, pos);
            if (p != std::string::npos && p < end) {
                end = p;
            }
        }
        std::string segment = text.substr(pos, end - pos);  // 截取这段文本
        pos = end;


        std::vector<std::string> word;
        for (unsigned char c : segment) {
            if (c >= b2u_.size()) {
                std::cerr << "ERROR: byte " << static_cast<int>(c) << " >= b2u_ size " << b2u_.size() << "\n";
                return {};
            }
            word.push_back(b2u_[c]);
        }

        // 贪心合并
        while (word.size() > 1) {
            int best_rank = std::numeric_limits<int>::max();

            size_t best_idx = word.size();
            
            for (size_t i = 0; i + 1 < word.size(); ++i) {
                auto it = merge_ranks_.find({word[i], word[i + 1]});
                if (it != merge_ranks_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_idx = i;
                }
            }

            if (best_idx > word.size()) {
                break;
            }
            word[best_idx] += word[best_idx + 1];
            word.erase(word.begin() + static_cast<std::ptrdiff_t>(best_idx) + 1);
        }

        // 将合并的word映射到id
        for (const auto& w : word) {
            auto it = vocab_.find(w);
            if (it != vocab_.end()) {
                result.push_back(it->second);
            } else {
                result.push_back(unk_id_);
            }
        }
    }
    return result;
}



std::string GgufTokenizer::DecodeToken(int token) const {
    if (token >= 0 && token < static_cast<int>(id_to_token_.size())) {
        return id_to_token_[token];
    }
    return "";
}



std::string GgufTokenizer::Decode(const std::vector<int>& tokens) const {
    std::string text;
    for (int id : tokens) {
        if (id >= 0 && id <= static_cast<int>(id_to_token_.size())) {
            text += id_to_token_[id];
        }
    }


    // 把unicode还原
    std::string result;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t start = pos;
        std::string cp = NextUtf8Codepoint(text, pos);
        if (cp.empty()) {
            if (pos == start) {
                ++pos;
            }
            continue;
        } 
        auto it = u2b_.find(cp);
        if (it != u2b_.end()) {
            result += static_cast<char>(it->second);
        } else {
            result += cp;
        }
    }

    return result;
}


int GgufTokenizer::vocab_size() const {
    return id_to_token_.size();
}
int GgufTokenizer::bos_id() const {
    return bos_id_;
}
int GgufTokenizer::eos_id() const {
    return eos_id_;
}
int GgufTokenizer::unk_id() const {
    return unk_id_;
}


std::unique_ptr<ITokenizer> GreateGgufTokenizer(const std::string& gguf_path) {
    auto tok = std::make_unique<GgufTokenizer>();
    if (tok->Load(gguf_path)) {
        return tok;
    } 
    return nullptr;
}

}