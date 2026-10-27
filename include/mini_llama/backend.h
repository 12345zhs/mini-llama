#pragma once 

#include <string>

namespace mini_llama {


// 采用哪种后端
enum class BackendKind{
    kCpu,
    kCuda,
};


// 后端配置
struct BackendConfig {
    BackendKind kind = BackendKind::kCpu;
    int device_id = 0;                  // 支持多卡环境，通过device_id指定要使用的GPU
    int device_id_set = false;
};

bool ParseBackendKind(const std::string& text, BackendKind& kind);
std::string BackendKindName(BackendKind kind);

bool CudaBackendBuild();
bool CudaBackendAvailable(int device_id = 0);
std::string CudaDeviceSummary(int device_id = 0);

void ValidateBackend(const BackendConfig& config);


std::string BackendExecutionNote(const BackendConfig& config);


}