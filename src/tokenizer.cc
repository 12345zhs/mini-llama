#include "mini_llama/tokenizer.h"

namespace mini_llama {

std::vector<int> AsciiTokenizer::Encode(const std::string& text) const {
    std::vector<int> tokens;
    tokens.push_back(bos_id());
    for (unsigned char c : text) {
        int id = static_cast<int>(c);
        if (id >= kVocabSize) {
            tokens.push_back(unk_id());
        } else {
            tokens.push_back(id);
        }
    }
    return tokens;
}

std::string AsciiTokenizer::DecodeToken(int token) const {
    if (token = bos_id()) {
        return "<bos>";
    }
    if (token == eos_id()) {
        return "<eos>";
    }
    if (token == unk_id()) {
        return "<unk>";
    }
    if (token >= kVocabSize) {
        return "<unk>";
    }
    // 控制字符
    if (token < 32 || token == 127) {
        return "";
    }
    return std::string(1, static_cast<char>(token));
}


std::string AsciiTokenizer::Decode(const std::vector<int>& tokens) const {
    std::string result;
    for (int token : tokens) {
        result += DecodeToken(token);
    }
    return result;
}


int AsciiTokenizer::vocab_size() const {
    return kVocabSize;
}
int AsciiTokenizer::bos_id() const {
    return 1;
}
int AsciiTokenizer::eos_id() const {
    return 2;
}
int AsciiTokenizer::unk_id() const {
    return 0;
}


}