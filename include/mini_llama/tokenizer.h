#pragma once 

#include <vector>
#include <string>
#include <utility>
#include <unordered_map>
#include <memory>
#include <map>
#include <cstdint>
#include <memory>


namespace mini_llama {

class ITokenizer {
public:
    virtual ~ITokenizer() = default;

    virtual std::vector<int> Encode(const std::string& text) const = 0;
    virtual std::string DecodeToken(int token) const = 0;
    virtual std::string Decode(const std::vector<int>& tokens) const = 0;

    virtual int vocab_size() const = 0;
    virtual int bos_id() const = 0;
    virtual int eos_id() const = 0;
    virtual int unk_id() const = 0;
};


class AsciiTokenizer : public ITokenizer {
public:
    static constexpr int kVocabSize = 128;

    AsciiTokenizer() = default;
    std::vector<int> Encode(const std::string& text) const override;
    std::string DecodeToken(int token) const override;
    std::string Decode(const std::vector<int>& tokens) const override;

    int vocab_size() const override;
    int bos_id() const override;
    int eos_id() const override;
    int unk_id() const override;
};




class JsonVocabTokenizer : public ITokenizer {
public:
    explicit JsonVocabTokenizer(const std::string& vocab_path);

    std::vector<int> Encode(const std::string& text) const override;
    std::string DecodeToken(int token) const override;
    std::string Decode(const std::vector<int>& tokens) const override;

    int vocab_size() const override;
    int bos_id() const override;
    int eos_id() const override;
    int unk_id() const override;

private:
    struct VocabEntry{
        int id = 0;              // 词条的id
        std::string content;     // 词条的内容
        bool special = false;    // 是否为特殊词条
    };

    int vocab_size_ = 0;          // 词表的范围
    int bos_id_ = 1;
    int eos_id_ = 2;
    int unk_id_ = 0;

    
    std::vector<std::pair<std::string, int>> content_to_id_;   // 保存词条<->id的映射，编码使用
    std::vector<VocabEntry> id_to_entry_;     // 按id索引的数组，给解码使用
};



class BpeTokenizer : public ITokenizer {
public:
    BpeTokenizer() = default;
    ~BpeTokenizer() override = default;

    std::vector<int> Encode(const std::string& text) const override;
    std::string DecodeToken(int token) const override;
    std::string Decode(const std::vector<int>& tokens) const override;

    int vocab_size() const override;
    int bos_id() const override;
    int eos_id() const override;
    int unk_id() const override;

    // 依次建立字节映射，读取vocab.json的token与ID、读取merges.txt的合并规则， 并尝试从special.json中读取BOS/EOS/UNK的ID
    bool Load(const std::string& vocab_path, const std::string& merges_path, const std::string& special_path);

private:
    std::unordered_map<std::string, int> vocab_;                           // token字符串->id
    std::vector<std::string> id_to_token_;                                 // id->token字符串
    std::map<std::pair<std::string, std::string>, int> merge_ranks_;       // 相邻片段对->合并优先级

    int bos_id_ = -1;
    int eos_id_ = -1;
    int unk_id_ = -1;

    std::vector<std::pair<std::string, int>> special_tokens_;             // 特殊token及其id，长度从长到短

    std::vector<std::string> b2u_;                                        // 字节值->unicode
    std::unordered_map<std::string, uint8_t> u2b_;                        // unicode->字节值

    void BuildByteMappings();                                             // 建b2u_和u2b_
};

std::unique_ptr<ITokenizer> CreateTokenizer(const std::string vocab_path);

std::unique_ptr<ITokenizer> CreateBpeTokenizer(const std::string& vocab_path, const std::string& merges_path, const std::string& special_path);

}