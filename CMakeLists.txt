cmake_minimum_required(VERSION 3.17)
project(mini_llama CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
)


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
)

# 建立可执行文件
add_executable(mini-llama-tests ${TEST_SOURCES})


# 设置目标属性
set_target_properties (mini-llama-tests PROPERTIES
    OUTPUT_NAME mini-llama-tests
)

#启动CTest测试
enable_testing()


# 注册一个CTest测试
add_test (
    NAME mini-llama-tests
    COMMAND mini-llama-tests
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
