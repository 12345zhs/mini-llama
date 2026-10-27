#include "mini_llama/backend.h"
#include "mini_llama/cuda_runtime.h"

#include <stdexcept>

namespace mini_llama {


// 解析后端的模式
bool ParseBackendKind(const std::string& text, BackendKind& kind) {
    if (text == "cpu") {
        kind = BackendKind::kCpu;
        return true;
    }
    if (text == "cuda") {
        kind = BackendKind::kCuda;
        return true;
    }
    return false;
}

// 打印后端模式
std::string BackendKindName(BackendKind kind) {
    switch (kind) {
        case BackendKind::kCpu : {
            return "cpu";
        }
        case BackendKind::kCuda : {
            return "cuda";
        }
    }
    return "unknow";
}


// cuda后端是否建立成功
bool CudaBackendBuild() {
    return CudaRuntimeBuild();
}


bool CudaBackendAvailable(int device_id) {
    try {
        int count = CudaDeviceCount();
        return device_id >= 0 && device_id < count;
    } catch (std::exception&){
        return false;
    }

}


std::string CudaDeviceSummary(int device_id = 0) {
    try {
        return CudaFormatDeviceInfo(CudaGetDeviceInfo(device_id));
    } catch (const std::exception& e) {
        return e.what();
    }
}


// 检查环境参数，是否有不该输入的参数
void ValidateBackend(const BackendConfig& config) {
    // cpu模式没有GPU编号
    if (config.kind == BackendKind::kCpu) {
        if (config.device_id_set) {
            throw std::runtime_error("--device can only be used with --backend cuda.");
        }
        return;
    }

    if (config.kind == BackendKind::kCuda) {
        if (!CudaBackendBuild()) {
            throw std::runtime_error("CUDA backend was not built. Reconfigure with -DMINI_LLAMA_CUDA=ON " "on a NVIDIA CUDA machine.");
        }
        if (!CudaBackendAvailable(config.device_id)) {
            throw std::runtime_error(CudaDeviceSummary(config.device_id));
        }
        return;
    }

    throw std::runtime_error("unknown backend");
}


// gpu负责的部分
std::string BackendExecutionNote(const BackendConfig& config) {
    if (config.kind == BackendKind::kCpu) {
        return "compute: cpu";
    }
    return "compute: cuda Linear (F32/Q8_0/Q4_0/Q4_1 where uploaded) + CUDA " "attention over GPU KV cache + device-resident Decode path; CPU " "sampler";
}

}