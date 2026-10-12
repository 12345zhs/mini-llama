#include <tests/test_main.h>

// 申请注册表
std::vector<TestCase>& TestRegistry() {
  static std::vector<TestCase> registry;
  return registry;
}


// 添加测试用例
void RegisterTest(const std::string& name, std::function<bool()> fn) {
  TestRegistry().push_back({name, fn});
}

int main(int argc, char** argv) {
  const auto& tests = TestRegistry();
  int passed = 0;
  int failed = 0;

  std::cout << "Running" << tests.size() << " test..." << std::endl;
  std::cout << "======================================" << std::endl;

  for (const auto& t : tests) {
    std::cout << "[RUN]" << t.name << std::endl;
    bool ok = t.fn();
    if (ok) {
      std::cout << "[PASS]" << t.name << std::endl;
      ++passed;
    }  else {
      std::cout << "[FAIL]" << t.name << std::endl;
      ++failed;
    }
  }

  std::cout << "=======================================" <<std::endl;
  std::cout << "Results: " << passed << " passed, " << failed << " failed" << std::endl;
  return failed > 0 ? 1 : 0; 
}
