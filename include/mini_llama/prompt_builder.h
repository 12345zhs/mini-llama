#pragma once

#include "mini_llama/chat.h"

#include <vector>
#include <memory>
#include <string>

namespace mini_llama {


// 将chat中的messages拍平,组装为模型训练设定的格式
class PromptBuilder {
public:
    PromptBuilder() = default;

    // 设置模板字符串
    void SetChatTemplate(const std::string& template_str);

    // 把messages拼成prompt字符串
    std::string Build(const std::vector<ChatMessage>& message) const;


private:
    std::string chat_template_;  // 模板字符串

    // 朴素拼接
    std::string BuildPlain(const std::vector<ChatMessage>& messages) const;

    // Qwen2专用拼接
    std::string BuildQwen2(const std::vector<ChatMessage>& messages) const;
};


// 从gguf文件里读取chat template字符串
std::string LoadChatTemplateFromGguf(const std::string& gguf_path);

}