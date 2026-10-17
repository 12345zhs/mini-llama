#include "mini_llama/terminal.h"
#include "mini_llama/chat.h"
#include "mini_llama/context.h"
#include "mini_llama/model.h"
#include "mini_llama/tokenizer.h"

#include <iostream>

namespace mini_llama {

struct LoadedModel {
    MiniLlamaModel model;
    std::unique_ptr<ITokenizer> tokenizer;
    std::string chat_template;
};

static int RunChat(int argc, int** argv) {


    LoadedModel lm = ;

    MiniLlamaModel& model = lm.model;
    Terminal term;
    ChatSession session;

    MiniLlamaContext ctx(&model);

    while (true) {
        term.PrintUserPrompt();
        std::string input = term.ReadLine();

        // 用户输入空或者输入已经到达了结尾
        if (input.empty() && std::cin.eof()) {
            break;
        }
        if (input == "/help") {
            term.PrintHelp();
            continue;
        }
        if (input == "/exit") {
            break;
        }

        if (input == "/clear") {
            session.Clear();                 // 清理本轮对话
            ctx = MiniLlamaContext(&model);  // 重新构建上下文
            if (lm.chat_template.empty()) {
                session.AddMessage("system", "You are a helpful assistant");
            }
            term.PrintMessage("Chat history cleared.\n");
            continue;
        } 
        if (input == "/stats") {
            term.PrintStats(session);
            continue;
        }
        if (input == "/params") {
            term.PrintParams(session.sampling_params);
        }
        if (!input.empty() && input[0] == '/') {
            term.PrintMessage("Unknown command: " + input + "\n");
            continue;
        }
        if (input.empty()) {
            continue;
        }
    }
}

}