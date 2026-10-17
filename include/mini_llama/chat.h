#pragma once

#include "mini_llama/radix_tree.h"
#include "mini_llama/sampler.h"

#include <string>
#include <vector>

namespace mini_llama {

// 具体的角色以及数据
struct ChatMessage {
    std::string role;      // "system", "user", "assistant"
    std::string content;   
};


// 会话管理
class ChatSession {
public:

    ChatSession() = default;

    std::vector<ChatMessage> messages;  // 完整对话历史
    std::vector<int> token_history;     // 已经计算过的token_id
    RadixTree prefix_cache;            // 记录已经计算过的prefill前缀
    SamplingParams sampling_params;     // 当前对话的采样参数

    // 统计
    int total_prompt_tokens = 0;
    int total_generated_tokens = 0;
    double total_time_ms = 0.0;
    
    
    // 清空
    void Clear();
    // 追加一条消息
    void AddMessage(const std::string& role, const std::string& content);
    // 回合统计，用于/status显式
    void RecordTurn(int prompt_tokens, int generated_tokens, double time_ms);
    // 替换token_history为指定token，每轮把prompt+回复的完整token存进去，作为下一次的算前缀基础
    void SetTokenHistory(const std::vector<int>& tokens);
    // 追加一个生成的token
    void AppendToken(int token);
    // 把之前算的token存到RadixTree中
    void RecordPrefix(const std::vector<int>& tokens);
    // 给定token序列，查询再RadixTree中能匹配多长的前缀
    size_t LongestCachePrefix(const std::vector<int>& tokens) const;

private:

};

}