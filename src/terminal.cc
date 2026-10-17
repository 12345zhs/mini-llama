#include "mini_llama/terminal.h"

#include <iostream>
#include <iomanip>

namespace mini_llama {

// 打印用户输入的提示词
void Terminal::PrintUserPrompt() const {
    std::cout << "mini_llama > ";
    std::cout.flush();               // 把stdout缓冲区里还没有输出的内容立刻送到终端
}

// 从std::cin中读一行输入
std::string Terminal::ReadLine() const {
    std::string line;
    if (std::getline(std::cin, line)) {
        return line;
    }
    return "";
}

// 打印助手回复前缀
void Terminal::PrintAssistantPrefix() const {
    std::cout << "assistant >";
    std::cout.flush();
}

// 打印已解码的token片段
void Terminal::PrintTokenText(const std::string& text) const {
    std::cout << text;
}

// 刷新stdout缓冲区
void Terminal::Flush() const {
    std::cout.flush();
}

// 打印一个换行
void Terminal::NewLine() const {
    std::cout << "\n";
}


// 打印/help文本
void Terminal::PrintHelp() const {
    std::cout << "Commands:\n"
              << " /help   show this help message\n"
              << " /clear  clear chat history and context\n"
              << " /stats  show session statistics\n"
              << " /params show current sampling parameters\n"
              << " /exit   exit chat\n";
}

// 打印会话统计
void Terminal::PrintStats(const ChatSession& session) const {
    std::cout << "Session stats:\n"
              << "  message:             " << session.messages.size() << "\n"
              << "  context tokens       " << session.token_history.size() << "\n"
              << "  total prompt tokens: " << session.total_prompt_tokens << "\n"
              << "  total generated:     " << session.total_generated_tokens << "\n"
              << "  total time:          " << std::fixed << std::setprecision(2) << session.total_time_ms << " ms\n"
              << "  tokens/s:            ";
    if (session.total_time_ms > 0.0) {
        std::cout << std::fixed << std::setprecision(2) << (session.total_generated_tokens * 1000.0 / session.total_time_ms);
    } else {
        std::cout << "n/a";
    }

    std::cout << std::defaultfloat << "\n";
}

// 打印当前采样参数
void Terminal::PrintParams(const SamplingParams& params) const {
    std::cout << std::defaultfloat << "Sampling params:\n"
              << "  temperature: " << params.temperature << "\n"
              << "  top_k:       " << params.top_k << "\n"
              << "  seed:        " << params.seed << "\n";
}

// 打印一条通用消息
void Terminal::PrintMessage(const std::string& msg) const {
    std::cout << msg << "\n";
}

}