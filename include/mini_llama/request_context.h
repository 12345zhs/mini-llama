#pragma once

#include <chrono>
#include <vector>
#include <string>

namespace mini_llama {

using RequestClock = std::chrono::steady_clock;

struct TraceEvent {
    std::string stage;
    double elapsed_ms = 0.0;
    int tokens = 0;
    std::string detail;
};

struct RequestContext {
    // 请求的身份
    std::string trace_id;
    std::string mode;
    std::string backend;
    std::string model_path;

    // 汇总指标
    RequestClock::time_point start_time;
    double total_ms = 0.0;

    int prompt_tokens = 0;
    int prefill_tokens = 0;
    int decode_tokens = 0;
    int generated_tokens = 0;

    double model_load_ms = 0.0;
    double tokenizer_ms = 0.0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double sample_ms = 0.0;

    std::string error;
    std::vector<TraceEvent> events;

    void Start();
    void Finish();
    void RecordEvent(const std::string& stage, double elapsed_ms, int tokens = 0, const std::string& detail = "");
    void SetError(const std::string& message);
    bool ok() const;

};


std::string NewTraceId();
double ElapsedMs(RequestClock::time_point start);
RequestContext StartRequest(const std::string& mode, const std::string& backend, const std::string& mode_path);
std::string FormatRequestTraceSummary(const RequestContext& request);
std::vector<std::string> FormatRequestTraceEvents(const RequestContext& request);



}