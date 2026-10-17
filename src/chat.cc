#include "mini_llama/chat.h"

namespace mini_llama {
    
// 清空
void ChatSession::Clear() {
    messages.clear();
    token_history.clear();
    prefix_cache.Clear();

    total_prompt_tokens = 0;
    total_generated_tokens = 0;
    total_time_ms = 0;
}


// 追加一条消息
void ChatSession::AddMessage(const std::string& role, const std::string& content) {
    messages.push_back({role, content});
}


// 回合统计，用于/status显式
void ChatSession::RecordTurn(int prompt_tokens, int generated_tokens, double time_ms) {
    total_prompt_tokens += prompt_tokens;
    total_generated_tokens += generated_tokens;
    total_time_ms += time_ms;
}


// 替换token_history为指定token，每轮把prompt+回复的完整token存进去，作为下一次的算前缀基础
void ChatSession::SetTokenHistory(const std::vector<int>& tokens) {
    token_history = tokens;
}


// 追加一个生成的token
void ChatSession::AppendToken(int token) {
    token_history.push_back(token);
}

// 把之前算的token存到RadixTree中
void ChatSession::RecordPrefix(const std::vector<int>& tokens) {
    prefix_cache.Insert(tokens);
}


// 给定token序列，查询再RadixTree中能匹配多长的前缀
size_t ChatSession::LongestCachePrefix(const std::vector<int>& tokens) const {
    return prefix_cache.LongestPrefix(tokens);
}


}