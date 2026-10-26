#include "mini_llama/thread_pool.h"

#include <vector>
#include <mutex>


namespace mini_llama {

static int g_thread_count = 0;

int GetThreadCount() {
    if (g_thread_count > 0) {
        return g_thread_count;
    }

    unsigned int hw = std::thread::hardware_concurrency();
    return hw > 0? static_cast<int>(hw) : 0;
}

void SetThreadCount(int n) {
    g_thread_count = n > 0 ? n  : 0;
}

void ParallelFor(int n, std::function<void(int begin, int end)>& fn) {
    if (n < 0) {
        return;
    }

    // 获取线程数量
    int n_threads = GetThreadCount();

    constexpr int kMinChunk = 16;     // 每个线程最少处理的任务数
    // 任务总数小于n_threads * kMinChunk，不至于用线程池
    if (n < n_threads * kMinChunk) {
        fn(0, n);     // 之间在当前线程调用fn 
        return;
    }

    // 线程数不能大于任务数
    if (n_threads > n) {
        n_threads = n;
    }

    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    std::exception_ptr first_exception = nullptr;    // 某个线程出现异常，不能闪退，记录下来
    std::mutex exception_mutex;                      // 防止多个线程同时抛异常，写冲突

    int chunk = n / n_threads;           // 每个线程的任务书
    int remainder = n % n_threads;       // 任务的零头

    int start = 0;
    for (int t = 0; t < n_threads; ++t) {
        int count = chunk + (t < remainder ? 1 : 0);
        int end = start + count;

        if (count > 0) {
            threads.emplace_back(
                [start, end, &fn, &first_exception, &exception_mutex]() {
                    try {
                        fn(start, end);
                    } catch (...) {
                        std::lock_guard<std::mutex> lock(exception_mutex);
                        if (!first_exception) {
                            first_exception = std::current_exception();     // 捕获异常
                        }
                    }
                }
            );
        }
        start = end;
    }

    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    if (first_exception) {
        std::rethrow_exception(first_exception);     // 将异常抛出
    }
}

}