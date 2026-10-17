#pragma once 


#include "mini_llama/chat.h"
#include "mini_llama/model.h"
#include "mini_llama/sampler.h"

#include <string>

namespace mini_llama {

class Terminal {

public:
    Terminal() = default;

    // 打印用户输入的提示词
    void PrintUserPrompt() const;

    // 从std::cin中读一行输入
    std::string ReadLine() const;

    // 打印助手回复前缀
    void PrintAssistantPrefix() const;

    // 打印已解码的token片段
    void PrintTokenText(const std::string& text) const;

    // 刷新stdout缓冲区
    void Flush() const;

    // 打印一个换行
    void NewLine() const;


    // 打印/help文本
    void PrintHelp() const;

    // 打印会话统计
    void PrintStats(const ChatSession& session) const;

    // 打印当前采样参数
    void PrintParams(const SamplingParams& params) const;

    // 打印一条通用消息
    void PrintMessage(const std::string& msg) const;

private:
};

}