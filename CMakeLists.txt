cmake_minimum_required(VERSION 3.17)
project(mini_llama CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# cuda选项与检测
option(MINI_LLAMA_CUDA "Build NVIDIA CUDA backend" OFF)

if(MINI_LLAMA_CUDA)
    enable_language(CUDA)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 89) # 默认针对 40系显卡(8.9)，可按需修改
    endif()
    find_package(CUDAToolkit REQUIRED)
endif()

# 头文件搜索路径
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR})


# 核心源代码
set(MINI_LLAMA_CORE_SOURCES 
    src/tensor.cc
    src/ops.cc
    src/matmul_optimized.cc
    src/tokenizer.cc
    src/json_vocab_tokenizer.cc
    src/bpe_tokenizer.cc
    src/radix_tree.cc
    src/request_context.cc
    src/sampler.cc
    src/chat.cc
    src/prompt_builder.cc
    src/terminal.cc
    src/gguf.cc
    src/gguf_tokenizer.cc
    src/gguf_loader.cc
    src/quant.cc
    src/quantized_tensor.cc
    src/thread_pool.cc
    

    # gpu的运行文件
    src/cuda_runtime.cc
    src/backend.cc
)


if(MINI_LLAMA_CUDA)
    list(APPEND MINI_LLAMA_CORE_SOURCES 
        
    )
endif()


# 测试所用到的源代码
set(TEST_SOURCES
    ${MINI_LLAMA_CORE_SOURCES}
    tests/test_main.cc
    tests/test_tensor.cc
    tests/test_ops.cc
    tests/test_tokenizer.cc
    tests/test_radix_tree.cc
    tests/test_request_context.cc
    tests/test_sampler.cc
    tests/test_chat.cc
    tests/test_gguf.cc
    tests/test_gguf_tokenizer.cc
    tests/test_forward.cc
    tests/test_kv_cache.cc
    tests/test_loader.cc
    tests/test_batch.cc
    tests/test_debug.cc
    tests/test_gguf_loader.cc
    tests/test_quant.cc
    tests/test_threaded_matmul.cc

    tests/test_cuda_runtime.cc
)

# 建立可执行文件
add_executable(mini-llama-tests ${TEST_SOURCES})


# 设置目标属性
set_target_properties (mini-llama-tests PROPERTIES
    OUTPUT_NAME mini-llama-tests
)


# 开启 CUDA 时，给测试目标打上宏并链接动态库
if(MINI_LLAMA_CUDA)
    target_compile_definitions(mini-llama-tests PRIVATE MINI_LLAMA_USE_CUDA=1)
    target_link_libraries(mini-llama-tests PRIVATE CUDA::cudart CUDA::cublas)
endif()

#启动CTest测试
enable_testing()


# 注册一个CTest测试
add_test (
    NAME mini-llama-tests
    COMMAND mini-llama-tests
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
