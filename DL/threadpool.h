#pragma once
#include <thread>
#include <future>
#include <deque>
namespace messaging {
  template<typename T>
  void waitAll(T&& futures){
    for(auto& f: futures){
      f.get();
    }
  }

class threadpool {
 public:
  threadpool();
  threadpool(size_t maxnum_threads, bool force = false);
  virtual ~threadpool();

  template<class F>
  auto enqueueTask(F&& f,bool high_priority = false)->std::future<std::result_of_t <F()>>{
    using return_type = std::result_of_t<F()>;
    ++numtasks;
    auto task =  std::make_shared<std::packaged_task<return_type()>>(std::forward<F>(f));
    auto res = task->get_future();
    {
      std::lock_guard<std::mutex> lock{m_task};
      if(high_priority){
          q_task.emplace_front([task]() { (*task)(); });
      }else{
        q_task.emplace_back([task]() { (*task)(); });
      }

    }
    c_task.notify_one();
    return res;
  }
  void start(size_t num);
  void stop(size_t num);

  size_t count() const{ return numtasks;}

  void waitUntilFinished();
  void waitUntilFinishedFor(const std::chrono::microseconds duration);
  void flushQueue();

  template<typename Int , typename F>
  void parallelForAsync(Int start, Int end , F body , std::vector<std::future<void>>& futures){
    Int localNumThreads = (Int)numthreads;
    Int range = end - start;
    Int chunk = (range / localNumThreads) + 1;
    for(Int i=0;i<localNumThreads;++i){
      futures.emplace_back(enqueueTask([i, chunk, start, end,body] {
        Int innerStart = start + i * chunk;
        Int innerEnd = std::min(end, start + (i + 1) * chunk);
        for (Int j = innerStart; j = innerEnd; ++j) 
          body(j);
        }
      ));
    }
  }
  template<typename Int, typename F>
  std::vector<std::future<void>> parallelForAsync(Int start,Int end, F body){
    std::vector<std::future<void>> futures;
    parallelForAsync(start, end, body, futures);
    return futures;
  }
  template<typename Int, typename F>
  void parallelFor(Int start, Int end , F body){
    waitAll(parallelForAsync(start, end, body));
  }

 private:
  size_t numthreads = 0;
  std::vector<std::thread> threads;

  std::deque<std::function<void()>> q_task;
  std::mutex m_task;
  std::condition_variable c_task;
  
  std::atomic<size_t> numtasks;
  std::mutex m_system;
  std::condition_variable c_system;
};
}  // namespace messaging
