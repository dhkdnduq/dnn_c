#include "pch.h"
/*
perf_timer<std::chrono::milliseconds>::duration_p([&](){batch_size = dl_trt->predict_category_classification(rst_list); });
*/

template <typename Time = std::chrono::milliseconds,
          typename Clock = std::chrono::high_resolution_clock>
struct perf_timer {
  template <typename F, typename... Args>
  static Time duration(F&& f, Args... args) {
    auto start = Clock::now();

    std::invoke(std::forward<F>(f), std::forward<Args>(args)...);

    auto end = Clock::now();

    return std::chrono::duration_cast<Time>(end - start);
  };
  template <typename F, typename... Args>
  static void duration_p(F&& f, Args... args) {
    auto start = Clock::now();

    std::invoke(std::forward<F>(f), std::forward<Args>(args)...);

    auto end = Clock::now();

    cout << std::chrono::duration_cast<Time>(end - start).count() << endl;
  };
};
