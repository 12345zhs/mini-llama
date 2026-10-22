#pragma once

#include "mini_llama/tokenizer.h"



namespace mini_llama {

class GgufTokenizer : public ITokenizer {
public:
    GgufTokenizer() = default;
    ~GgufTokenizer() override = default;

    bool Load(const std::string& gguf_path);

    std::vector<int> Encode(const std::string& text) const override;
    std::string DecodeToken(int token) const override;
    std::string Decode(const std::vector<int>& tokens) const override;
    

    int vocab_size() const override;
    int bos_id() const override;
    int eos_id() const override;
    int unk_id() const override;

private:
    std::unordered_map<std::string, int> vocab_;
    std::vector<std::string> id_to_token_;

    std::map<std::pair<std::string, std::string>, int> merge_ranks_;

    std::vector<std::pair<std::string, int>> special_tokens_;


    std::vector<std::string> b2u_;
    std::unordered_map<std::string, uint8_t> u2b_;

    int bos_id_ = -1;
    int eos_id_ = -1;
    int unk_id_ = -1;

    void BuildByteMappings();
};

std::unique_ptr<ITokenizer> CreateGgufTokenizer(const std::string& gguf_path);

}