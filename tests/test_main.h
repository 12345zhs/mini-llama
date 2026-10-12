#pragma once

#include "tests/test_names.h"

#include <string>
#include <functional>
#include <iostream>
#include <cmath>
#include <vector>


// 测试记录
struct TestCase {
  std::string name;           // 测试名称
  std::function<bool()> fn;   // 要执行的函数
};


// 生成注册表,保存所有的测试
std::vector<TestCase>& TestRegistry();

// 注册接口，用于添加测试
void RegisterTest(const std::string& name, std::function<bool()> fn);



// 通用条件检查函数
//                   条件计算后的结果       条件文字          文件          行
inline bool AssertTrue(bool cond, const char* expr, const char* file, int line) {
  if (!cond) {
    std::cerr << " ASSERTION FAILED: " << expr << " at " << file << " : " << line << std::endl;
    return false;
  }
  return true;
}



// 浮点数误差检查函数
inline bool AssertEqFloat(float a, float b, float tol, const char* expr_a, const char* expr_b, const char* file, int line) {
  if (std::fabs(a - b) > tol) {
    std::cerr << " ASSERTION FAILED: |" << expr_a << " - " << expr_b << "| = |" 
              << a << " - " << b << "| = " << std::fabs(a - b) << " > " << tol 
              << " at " << file << " : " << line << std::endl;
    return false;
  }
  return true;
}

#define MINI_LLAMA_ASSERT_TRUE(cond)                         \
  do {                                                       \
    if (!AssertTrue((cond), #cond, __FILE__, __LINE__)) {    \
      return false;                                          \
    }                                                        \
  } while (0)                                                




#define MINI_LLAMA_ASSERT_EQ(a, b)                                  \
  do {                                                              \
    if (!AssertTrue((a) == (b), #a "==" #b, __FILE__, __LINE__)) {  \
      return false;                                                 \
    }                                                               \
  } while (0)


#define MINI_LLAMA_ASSERT_NEAR(a, b, tol)                                  \
  do {                                                                     \
    if (!AssertEqFloat((a), (b), (tol), #a, #b, __FILE__, __LINE__)) {     \
      return false;                                                        \
    }                                                                      \
  } while (0)

#define MINI_LLAMA_ASSERT_FAIL(msg)                                                                   \
  do {                                                                                                \
    std::cerr << " ASSERTION FAILED: " << msg << " at " << __FILE__ << " : " << __LINE__ << std::endl;\
    return false;                                                                                     \
  } while (0)
  