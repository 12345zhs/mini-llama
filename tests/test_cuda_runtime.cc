#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mini_llama/cuda_runtime.h"
#include "tests/test_names.h"
#include "tests/test_main.h"

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool MessageContains(const std::exception& e, const std::string& text) {
  return std::string(e.what()).find(text) != std::string::npos;
}

void TestCpuBuildReportsMissingCuda() {
  Require(!CudaRuntimeBuilt(), "CudaRuntimeBuilt should be false in CPU build");

  bool threw = false;
  try {
    (void)CudaDeviceCount();
  } catch (const std::exception& e) {
    threw = true;
    Require(MessageContains(e, "CUDA backend was not built"),
            "CPU build error message should name CUDA build flag");
  }
  Require(threw, "CudaDeviceCount should throw in CPU build");
}

void TestCudaBuildDeviceInfoAndBuffer() {
  Require(CudaRuntimeBuilt(), "CudaRuntimeBuilt should be true in CUDA build");

  int count = CudaDeviceCount();
  Require(count >= 1, "CUDA build should see at least one device");

  CudaDeviceInfo info = CudaGetDeviceInfo(0);
  Require(info.id == 0, "device id should round trip");
  Require(!info.name.empty(), "device name should be non-empty");
  Require(info.compute_major > 0, "compute major should be positive");
  Require(info.total_memory_bytes > 0, "total memory should be positive");
  Require(info.driver_version > 0, "driver version should be positive");
  Require(info.runtime_version > 0, "runtime version should be positive");

  std::string formatted = CudaFormatDeviceInfo(info);
  Require(formatted.find("compute capability") != std::string::npos,
          "formatted info should include compute capability");
  Require(formatted.find("total memory") != std::string::npos,
          "formatted info should include total memory");

  std::vector<float> host = {1.0f, 2.0f, 3.5f, -4.0f};
  std::vector<float> out(host.size(), 0.0f);
  CudaDeviceBuffer buffer(host.size() * sizeof(float), 0);
  Require(!buffer.empty(), "device buffer should own memory");
  Require(buffer.bytes() == host.size() * sizeof(float),
          "device buffer should store byte size");
  Require(buffer.device_id() == 0, "device buffer should store device id");
  buffer.Upload(host.data(), host.size() * sizeof(float));
  buffer.Download(out.data(), out.size() * sizeof(float));
  Require(out == host, "device buffer upload/download should round trip");

  CudaDeviceBuffer moved = std::move(buffer);
  Require(buffer.empty(), "moved-from buffer should be empty");
  Require(!moved.empty(), "moved-to buffer should own memory");
  moved.Reset();
  Require(moved.empty(), "Reset should release memory");
}

}  // namespace


static struct CudaRuntimeRegistrar {
  CudaRuntimeRegistrar() {
    // 把原来的测试逻辑包装成一个 lambda 或静态函数注册进去
    RegisterTest("cuda_runtime_test", []() {
      try {
#ifdef MINI_LLAMA_USE_CUDA
        TestCudaBuildDeviceInfoAndBuffer();
#else
        TestCpuBuildReportsMissingCuda();
#endif
        return true;
      } catch (const std::exception& e) {
        MINI_LLAMA_ASSERT_FAIL(std::string("cuda_runtime: ") + e.what());
        return false;
      }
    });
  }
} cuda_runtime_registrar;