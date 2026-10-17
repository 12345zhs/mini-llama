// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <string>
#include <vector>

#include "mini_llama/chat.h"
#include "mini_llama/prompt_builder.h"
#include "mini_llama/sampler.h"
#include "mini_llama/terminal.h"
#include "tests/test_main.h"
#include "tests/test_names.h"

// ---------------------------------------------------------------------------
// ChatMessage
// ---------------------------------------------------------------------------
static bool TestChatMessageStoresRoleAndContent() {
  ChatMessage msg{"user", "hello"};
  MINI_LLAMA_ASSERT_TRUE(msg.role == "user");
  MINI_LLAMA_ASSERT_TRUE(msg.content == "hello");
  return true;
}

// ---------------------------------------------------------------------------
// ChatSession
// ---------------------------------------------------------------------------
static bool TestChatSessionStartsEmpty() {
  ChatSession session;
  MINI_LLAMA_ASSERT_TRUE(session.messages.empty());
  MINI_LLAMA_ASSERT_TRUE(session.token_history.empty());
  MINI_LLAMA_ASSERT_EQ(session.total_prompt_tokens, 0);
  MINI_LLAMA_ASSERT_EQ(session.total_generated_tokens, 0);
  MINI_LLAMA_ASSERT_NEAR(session.total_time_ms, 0.0, 1e-6);
  return true;
}

static bool TestChatSessionAddMessageAppends() {
  ChatSession session;
  session.AddMessage("system", "You are helpful.");
  session.AddMessage("user", "hi");
  MINI_LLAMA_ASSERT_EQ(session.messages.size(), 2);
  MINI_LLAMA_ASSERT_TRUE(session.messages[0].role == "system");
  MINI_LLAMA_ASSERT_TRUE(session.messages[0].content == "You are helpful.");
  MINI_LLAMA_ASSERT_TRUE(session.messages[1].role == "user");
  MINI_LLAMA_ASSERT_TRUE(session.messages[1].content == "hi");
  return true;
}

static bool TestChatSessionClearResets() {
  ChatSession session;
  session.AddMessage("user", "test");
  session.SetTokenHistory({1, 2, 3});
  session.RecordPrefix({1, 2, 3});
  session.RecordTurn(10, 5, 100.0);
  session.Clear();
  MINI_LLAMA_ASSERT_TRUE(session.messages.empty());
  MINI_LLAMA_ASSERT_TRUE(session.token_history.empty());
  MINI_LLAMA_ASSERT_EQ(session.LongestCachedPrefix({1, 2, 3}), 0);
  MINI_LLAMA_ASSERT_EQ(session.total_prompt_tokens, 0);
  MINI_LLAMA_ASSERT_EQ(session.total_generated_tokens, 0);
  MINI_LLAMA_ASSERT_NEAR(session.total_time_ms, 0.0, 1e-6);
  return true;
}

static bool TestChatSessionTracksTokenHistory() {
  ChatSession session;
  session.SetTokenHistory({1, 10, 11});
  session.AppendToken(12);
  MINI_LLAMA_ASSERT_EQ(session.token_history.size(), 4);
  MINI_LLAMA_ASSERT_EQ(session.token_history[0], 1);
  MINI_LLAMA_ASSERT_EQ(session.token_history[3], 12);
  return true;
}

static bool TestChatSessionRecordsPrefixCache() {
  ChatSession session;
  session.RecordPrefix({1, 2, 3, 4});
  session.RecordPrefix({1, 2, 9});
  MINI_LLAMA_ASSERT_EQ(session.LongestCachedPrefix({1, 2, 3}), 3);
  MINI_LLAMA_ASSERT_EQ(session.LongestCachedPrefix({1, 2, 9, 8}), 3);
  MINI_LLAMA_ASSERT_EQ(session.LongestCachedPrefix({1, 8}), 1);
  return true;
}

static bool TestChatSessionRecordTurnAccumulates() {
  ChatSession session;
  session.RecordTurn(10, 5, 100.0);
  session.RecordTurn(20, 8, 200.0);
  MINI_LLAMA_ASSERT_EQ(session.total_prompt_tokens, 30);
  MINI_LLAMA_ASSERT_EQ(session.total_generated_tokens, 13);
  MINI_LLAMA_ASSERT_NEAR(session.total_time_ms, 300.0, 1e-6);
  return true;
}

// ---------------------------------------------------------------------------
// PromptBuilder
// ---------------------------------------------------------------------------
static bool TestPromptBuilderEmptyHistory() {
  PromptBuilder builder;
  std::vector<ChatMessage> messages;
  std::string prompt = builder.Build(messages);
  MINI_LLAMA_ASSERT_TRUE(prompt == "Assistant:");
  return true;
}

static bool TestPromptBuilderMultiTurn() {
  PromptBuilder builder;
  std::vector<ChatMessage> messages = {{"system", "You are helpful."},
                                       {"user", "hi"},
                                       {"assistant", "Hello!"},
                                       {"user", "bye"}};
  std::string prompt = builder.Build(messages);
  MINI_LLAMA_ASSERT_TRUE(prompt ==
                         "System: You are helpful.\n"
                         "User: hi\n"
                         "Assistant: Hello!\n"
                         "User: bye\n"
                         "Assistant:");
  return true;
}

static bool TestPromptBuilderIgnoresUnknownRoles() {
  PromptBuilder builder;
  std::vector<ChatMessage> messages = {{"unknown", "ignored"},
                                       {"user", "hello"}};
  std::string prompt = builder.Build(messages);
  MINI_LLAMA_ASSERT_TRUE(prompt == "User: hello\nAssistant:");
  return true;
}

static bool TestPromptBuilderQwen2TemplateInjectsSystemAndPrefix() {
  PromptBuilder builder;
  builder.SetChatTemplate("qwen2");
  std::vector<ChatMessage> messages = {{"user", "你好"}};
  std::string prompt = builder.Build(messages);
  MINI_LLAMA_ASSERT_TRUE(prompt ==
                         "<|im_start|>system\n"
                         "You are a helpful assistant.<|im_end|>\n"
                         "<|im_start|>user\n"
                         "你好<|im_end|>\n"
                         "<|im_start|>assistant\n");
  return true;
}

static bool TestPromptBuilderQwen2TemplateKeepsExistingSystem() {
  PromptBuilder builder;
  builder.SetChatTemplate("qwen2");
  std::vector<ChatMessage> messages = {{"system", "Be concise."},
                                       {"user", "hi"}};
  std::string prompt = builder.Build(messages);
  MINI_LLAMA_ASSERT_TRUE(prompt ==
                         "<|im_start|>system\n"
                         "Be concise.<|im_end|>\n"
                         "<|im_start|>user\n"
                         "hi<|im_end|>\n"
                         "<|im_start|>assistant\n");
  return true;
}

// ---------------------------------------------------------------------------
// Terminal (smoke tests — just verify no crash)
// ---------------------------------------------------------------------------
static bool TestTerminalPrintUserPrompt() {
  Terminal term;
  term.PrintUserPrompt();
  return true;
}

static bool TestTerminalPrintAssistantPrefix() {
  Terminal term;
  term.PrintAssistantPrefix();
  return true;
}

static bool TestTerminalPrintHelp() {
  Terminal term;
  term.PrintHelp();
  return true;
}

static bool TestTerminalPrintStats() {
  Terminal term;
  ChatSession session;
  session.RecordTurn(10, 5, 250.0);
  term.PrintStats(session);
  return true;
}

static bool TestTerminalPrintParams() {
  Terminal term;
  SamplingParams params;
  params.temperature = 0.7f;
  params.top_k = 40;
  params.seed = 42;
  term.PrintParams(params);
  return true;
}

static bool TestTerminalPrintMessage() {
  Terminal term;
  term.PrintMessage("test message");
  return true;
}

static bool TestTerminalNewLine() {
  Terminal term;
  term.NewLine();
  return true;
}

// ---------------------------------------------------------------------------
// Auto-register
// ---------------------------------------------------------------------------
static struct ChatTestRegistrar {
  ChatTestRegistrar() {
    RegisterTest("chat_message_stores_role_and_content",
                 TestChatMessageStoresRoleAndContent);
    RegisterTest("chat_session_starts_empty", TestChatSessionStartsEmpty);
    RegisterTest("chat_session_add_message_appends",
                 TestChatSessionAddMessageAppends);
    RegisterTest("chat_session_clear_resets", TestChatSessionClearResets);
    RegisterTest("chat_session_record_turn_accumulates",
                 TestChatSessionRecordTurnAccumulates);
    RegisterTest("chat_session_tracks_token_history",
                 TestChatSessionTracksTokenHistory);
    RegisterTest("chat_session_records_prefix_cache",
                 TestChatSessionRecordsPrefixCache);
    RegisterTest("prompt_builder_empty_history", TestPromptBuilderEmptyHistory);
    RegisterTest("prompt_builder_multi_turn", TestPromptBuilderMultiTurn);
    RegisterTest("prompt_builder_ignores_unknown_roles",
                 TestPromptBuilderIgnoresUnknownRoles);
    RegisterTest("prompt_builder_qwen2_template_injects_system_and_prefix",
                 TestPromptBuilderQwen2TemplateInjectsSystemAndPrefix);
    RegisterTest("prompt_builder_qwen2_template_keeps_existing_system",
                 TestPromptBuilderQwen2TemplateKeepsExistingSystem);
    RegisterTest("terminal_print_user_prompt", TestTerminalPrintUserPrompt);
    RegisterTest("terminal_print_assistant_prefix",
                 TestTerminalPrintAssistantPrefix);
    RegisterTest("terminal_print_help", TestTerminalPrintHelp);
    RegisterTest("terminal_print_stats", TestTerminalPrintStats);
    RegisterTest("terminal_print_params", TestTerminalPrintParams);
    RegisterTest("terminal_print_message", TestTerminalPrintMessage);
    RegisterTest("terminal_new_line", TestTerminalNewLine);
  }
} chat_test_registrar;
