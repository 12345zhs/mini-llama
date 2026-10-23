#include "mini_llama/prompt_builder.h"
#include "mini_llama/gguf.h"

#include <stack>
#include <cstring>


namespace mini_llama {

struct EvalContext {
    const std::vector<ChatMessage>* messages = nullptr;      // 全部消息
    const ChatMessage* current_message = nullptr;            // 当前循环的消息
    size_t message_index = 0;                                // 当前消息的下标
    bool add_generation_prompt = false;                      // 是否添加generation开关
};


// 一条指令
struct Instr {
    // 决定执行时怎么处理，对应Jinja2的六种结构
    enum Type {
        kText,                   // 纯文本
        kOutput,                 // 输出
        kIf,                     // if
        kEndIf,                  // endif
        kFor,                    // for
        kEndFor                  // end for
    } type;
    std::string text;             // 文本内容
    size_t jump = 0;              // 跳转目标
};


// 循环时的运行状态
struct ForFrame {
    size_t pc = 0;                // for指令的pc(类似程序计数器，判断是不是同一个for指令)
    size_t index = 0;             // 当前迭代到第几个元素
};


// 去掉两边的空白位置
static std::string Trim(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) {
        a++;
    } 
    size_t b = 0;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) {
        --b;
    }
    return s.substr(a, b - a);
}

// Jinja2的语法，里面可能会有'-'（吃掉空白）
static std::string StripJinjaTriMakers(const std::string& raw) {
    std::string value = Trim(raw);

    // 删去第一个-
    if (!value.empty() && value.front() == '-') {
        value = Trim(value.substr(1));
    }

    //删去最后一个'-'
    if (!value.empty() && value.back() == '-') {
        value.pop_back();
        value = Trim(value);          // 可能还有空格
    }

    return value;
}


// 把表达式转化为实际字符串
static std::string EvalExpr(const std::string& expr_raw, const EvalContext& ctx) {
    std::string expr = Trim(expr_raw);
    if (expr.empty()) {
        return "";
    }

    // 各段求值结果
    std::vector<std::string> parts;
    size_t pos = 0;
    while (pos < expr.size()) {

        // 跳过空格
        while (pos < expr.size() && std::isspace(static_cast<unsigned char>(expr[pos]))) {
            ++pos;
        }
        if (pos >= expr.size()) {
            break;
        }

        // 遇到'
        if (expr[pos] == '\'') {
            ++pos;         // 跳过'

            std::string lit;

            // 逐字符拷贝
            while (pos < expr.size() && expr[pos] != '\'') {
                lit += expr[pos];
                ++pos;
            }

            if (pos < expr.size() && expr[pos] =='\'') {
                ++pos;
            }
            parts.push_back(lit);
        } else {
            // 读一个token
            size_t start = pos;
            while (pos < expr.size() && expr[pos] != '+' && expr[pos] != '\'') {
                if (expr[pos] == '[') {
                    // 遇到 [ ，一直吃到 ]，整个 [...] 算 token 一部分
                    ++pos;
                    while (pos < expr.size() && expr[pos] != ']') {
                        ++pos;
                    }
                    if (pos < expr.size()) {
                        ++pos;
                    }
                } else if (expr[pos] == '.') {
                    // 遇到 . ，吃到下一个分隔符，整个 .xxx 算 token 一部分
                    ++pos;
                    while (pos < expr.size() && !std::isspace(static_cast<unsigned char>(expr[pos])) && expr[pos] != '+' && expr[pos] != '\'' && expr[pos] != '[' && expr[pos] != '.') {
                        ++pos;
                    }
                } else {
                    ++pos;
                }
            }
            std::string token = Trim(expr.substr(start, pos - start));
            if (!token.empty()) {
                // Evaluate variable/token
                if (token == "add_generation_prompt") {
                    parts.push_back(ctx.add_generation_prompt ? "true" : "false");
                } else if (token == "message['role']" || token == "message[\"role\"]") {
                    parts.push_back(ctx.current_message ? ctx.current_message->role : "");
                } else if (token == "message['content']" || token == "message[\"content\"]") {
                    parts.push_back(ctx.current_message ? ctx.current_message->content : "");
                } else if (token.find("messages[") == 0) {
                    // messages[0]['role'] or similar
                    size_t close = token.find(']');
                    if (close != std::string::npos && close + 1 < token.size() &&
                        token[close + 1] == '[') {
                        std::string idx_str = Trim(token.substr(9, close - 9));
                        std::string key;
                        size_t key_start = close + 2;
                        size_t key_end = token.find(']', key_start);
                        if (key_end != std::string::npos) {
                            key = token.substr(key_start, key_end - key_start);
                            // Remove quotes
                            if ((key.front() == '\'' && key.back() == '\'') || (key.front() == '"' && key.back() == '"')) {
                                key = key.substr(1, key.size() - 2);
                            }
                        }
                        try {
                            int idx = std::stoi(idx_str);
                            if (ctx.messages && idx >= 0 && idx < static_cast<int>(ctx.messages->size())) {
                                const ChatMessage& msg = (*ctx.messages)[idx];
                                if (key == "role") {
                                    parts.push_back(msg.role);
                                } else if (key == "content") {
                                    parts.push_back(msg.content);
                                } else {
                                    parts.push_back("");
                                }
                            } else {
                                parts.push_back("");
                            }
                        } catch (...) {
                            parts.push_back("");
                        }
                    } else {
                        parts.push_back("");
                    }
                } else if (token == "loop.first") {
                    parts.push_back(ctx.message_index == 0 ? "true" : "false");
                } else {
                    parts.push_back("");
                }
            }
        }

        // Skip + operator
        while (pos < expr.size() && std::isspace(static_cast<unsigned char>(expr[pos]))) {
            ++pos;
        }
        if (pos < expr.size() && expr[pos] == '+') {
            ++pos;
        }
    }

    // Concatenate all parts
    std::string result;
    for (const auto& p : parts) {
        result += p;
    }
    return result;
}



static bool EvalCondition(const std::string& cond_raw, const EvalContext& ctx) {
    std::string cond = Trim(cond_raw);
    if (cond.empty()) {
        return false;
    }

    // Handle 'and'
    size_t and_pos = cond.find(" and ");
    while (and_pos != std::string::npos) {
        std::string left = Trim(cond.substr(0, and_pos));
        std::string right = Trim(cond.substr(and_pos + 5));
        return EvalCondition(left, ctx) && EvalCondition(right, ctx);
    }

    // Handle 'or'
    size_t or_pos = cond.find(" or ");
    while (or_pos != std::string::npos) {
        std::string left = Trim(cond.substr(0, or_pos));
        std::string right = Trim(cond.substr(or_pos + 4));
        return EvalCondition(left, ctx) || EvalCondition(right, ctx);
    }

    // Handle == and !=
    size_t neq_pos = cond.find(" != ");
    if (neq_pos != std::string::npos) {
        std::string left = EvalExpr(cond.substr(0, neq_pos), ctx);
        std::string right = EvalExpr(cond.substr(neq_pos + 4), ctx);
        return left != right;
    }
    size_t eq_pos = cond.find(" == ");
    if (eq_pos != std::string::npos) {
        std::string left = EvalExpr(cond.substr(0, eq_pos), ctx);
        std::string right = EvalExpr(cond.substr(eq_pos + 4), ctx);
        return left == right;
    }

    // Single variable
    std::string val = EvalExpr(cond, ctx);
    if (val == "true" || val == "True") {
        return true;
    }
    if (val == "false" || val == "False") {
        return false;
    }
    return !val.empty();
}

// 编译Jinja2的模板，把模板字符切成指令，并回填好所有跳转地址
static std::vector<Instr> CompileTemplate(const std::string& tmpl) {
    std::vector<Instr> instrs;           // 输出指令序列
    std::stack<size_t> if_stack;         // if栈
    std::stack<size_t> for_stack;        // for栈

    size_t pos = 0;

    while (pos < tmpl.size()) {
        if (pos + 1 < tmpl.size() && tmpl[pos] == '{' && tmpl[pos + 1] == '%') {
            pos += 2;   // 跳过"{%"

            // 找到闭合的"%}",没有就直接结束
            size_t end = tmpl.find("%}", pos);
            if (end == std::string::npos) {
                break;
            }

            // 取出{% ... %}之间的内容
            std::string stmt = StripJinjaTriMakers(tmpl.substr(pos, end - pos));
            pos = end + 2;                // 跳转到%}之后


            // 以for开头
            if (stmt.rfind("for ", 0) == 0) {
                instrs.push_back({Instr::kFor, stmt});  // 生成kfor指令
                for_stack.push(instrs.size() - 1);      // 当前for指令在Instr的下标
            } else if (stmt == "endfor") {
                // 没有配对的指令
                if (for_stack.empty()) {
                    continue;
                }
                size_t for_pc = for_stack.top();                   // 对应for的Instr下标
                for_stack.pop();
                instrs.push_back({Instr::kEndFor, "", for_pc});   
                instrs[for_pc].jump = instrs.size();               // for循环执行完后的跳转下标
            } else if (stmt.rfind("if ", 0) == 0) {
                instrs.push_back({Instr::kIf, stmt.substr(3)});
                if_stack.push(instrs.size() - 1);
            } else if (stmt == "endif") {
                if (if_stack.empty()) {
                    continue;
                }

                size_t if_pc = if_stack.top();
                if_stack.pop();

                instrs.push_back({Instr::kEndIf});
                instrs[if_pc].jump = instrs.size();
            }
        } else if (pos + 1 < tmpl.size() && tmpl[pos] == '{' && tmpl[pos + 1] == '{') {
            // 表达式
            pos += 2;
            size_t end = tmpl.find("}}", pos);
            if (end == std::string::npos) {
                break;
            }
            std::string expr = StripJinjaTriMakers(tmpl.substr(pos, end - pos));
            pos = end + 2;

            instrs.push_back({Instr::kOutput, expr});
        } else {
            // 纯文本
            size_t end = pos;
            while (end < tmpl.size() && !(tmpl[end] == '{' && end + 1 < tmpl.size() && (tmpl[end + 1] == '%' || tmpl[end + 1] == '{'))) {
                ++end;
            }
            std::string text = tmpl.substr(pos, end - pos);
            pos = end;

            if (!text.empty()) {
                instrs.push_back({Instr::kText, text});
            }
        }
    }

    // 如果if/for都没有配对，保证跳转地址指向指令末尾
    while (!if_stack.empty()) {
        instrs[if_stack.top()].jump = instrs.size();
        if_stack.pop();
    }
    while (!for_stack.empty()) {
        instrs[for_stack.top()].jump = instrs.size();
        for_stack.pop();
    }

    return instrs;
}


static std::string ExecuteTemplate(const std::vector<Instr>& instrs, const std::vector<ChatMessage>& message, bool add_gen) {
    std::string output;
    EvalContext ctx;            
    ctx.messages = &message;                    // 指向全部消息
    ctx.add_generation_prompt = add_gen;

    std::stack<ForFrame> for_stack;
    size_t pc = 0;

    while (pc < instrs.size()) {
        const Instr& instr = instrs[pc];
        switch (instr.type) {
            case Instr::kText :
            // 纯文本直接拼
                output += instr.text;
                ++pc;
                break;
            case Instr::kOutput :
            // 表达式，求值再拼
                output += EvalExpr(instr.text, ctx);
                ++pc;
                break;
            case Instr::kIf :
                // 条件假，跳到endif后
                if (!EvalCondition(instr.text, ctx)) {
                    pc = instr.jump;
                } else {
                    // 条件真，顺序执行指令
                    ++pc;
                }
                break;
            case Instr::kEndIf :
                // 什么都不做
                ++pc;
                break;
            case Instr::kFor :
                if (for_stack.empty() || for_stack.top().pc != pc) {
                    // 首次进入循环
                    ForFrame frame;
                    frame.pc = pc;
                    frame.index = 0;
                    for_stack.push(frame);
                    if (message.empty()) {
                        // message为空，循环体一次都不执行，弹栈
                        for_stack.pop();
                        pc = instr.jump;
                    } else {
                        ctx.current_message = &message[0];
                        ctx.message_index = 0;
                        ++pc;
                    }
                } else {
                    // 循环体执行完，要开始下一轮循环
                    ForFrame& frame = for_stack.top();
                    ++frame.index;
                    if (frame.index > message.size()) {
                        // 迭代完了，弹栈
                        for_stack.pop();
                        pc = instr.jump;
                    } else {
                        ctx.current_message = &message[frame.index];
                        ctx.message_index = frame.index;
                        ++pc;
                    }
                }
                break;
            case Instr::kEndFor :
                pc = instr.jump;   //跳回for指令
                break;
        }
    }   

    return output;
}

// 设置模板字符串
void PromptBuilder::SetChatTemplate(const std::string& template_str) {
    chat_template_ = template_str;
}

// 把messages拼成prompt字符串
std::string PromptBuilder::Build(const std::vector<ChatMessage>& message) const {
    // 如果没有templat,就plain拼接
    if (chat_template_.empty()) {
        return BuildPlain(message);
    }


    // 如果以{%或{{开头，就是用Jinja2模板
    if (chat_template_.size() >= 2 && chat_template_[0] == '{' && (chat_template_[1] == '%' || chat_template_[1] == '{')) {
        auto instrs = CompileTemplate(chat_template_);
        return ExecuteTemplate(instrs, message, true);
    }

    // Qwen2模板
    if (chat_template_ == "qwen2") {
        return BuildQwen2(message);
    }

    //其他 
    return BuildPlain(message);
}


// 朴素拼接
std::string PromptBuilder::BuildPlain(const std::vector<ChatMessage>& messages) const {
    std::string prompt;
    for (const auto& msg : messages) {
        if (msg.role == "system") {
            prompt += "System: " + msg.content + "\n";
        } else if (msg.role == "user") {
            prompt += "User: " + msg.content + "\n";
        } else if (msg.role == "assistant") {
            prompt += "Assistant: " + msg.content + "\n";
        }
    }
    prompt += "Assistant:";      // 提示模型该说话了
    return prompt; 
}

// Qwen2专用拼接
/*
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！<|im_end|>
<|im_start|>assistant
*/
std::string PromptBuilder::BuildQwen2(const std::vector<ChatMessage>& messages) const {
    bool has_system= false;
    std::string prompt;
    // Qwen2必须要有ssytem信息，没有就要自己补
    for (const auto& msg : messages) {
        if (msg.role == "system") {
            has_system = true;
            break;
        }
    }
    if (!has_system) {
        prompt += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
    }
    for (const auto& msg : messages) {
        if (msg.role != "system" && msg.role != "user" && msg.role != "assistant") {
            continue;
        }
        prompt += "<|im_start|>" + msg.role + "\n" + msg.content + "<|im_end|>\n";
    }

    prompt += "<|im_start|>assistant\n";
    return prompt;
}


// 从gguf文件里读取chat template字符串
std::string LoadChatTemplateFromGguf(const std::string& gguf_path) {
    std::string tmpl;
    if (!GgufGetMetadataString(gguf_path, "tokenizer.chat_template", tmpl)) {
        // gguf里没有模板，看架构名字
        std::string architecture;
        if (!GgufGetMetadataString(gguf_path, "general.architecture", architecture) && std::strcmp(architecture.c_str(), "qwen2") == 0) {
            return "qwen2";
        }
        return "";
    }

    // gguf里有模板，看看是不是qwen2的特征
    if (tmpl.find("{% for message in messages %}") != std::string::npos && tmpl.find("<|im_start|>") != std::string::npos && tmpl.find("<|im_end|>") != std::string::npos) {
        return tmpl;
    }

    return "";

}

}