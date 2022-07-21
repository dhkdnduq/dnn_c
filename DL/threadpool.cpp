#include "pch.h"
#include "threadpool.h"

using namespace std;
namespace messaging
{
  threadpool::threadpool() : threadpool{thread::hardware_concurrency()} {}
  threadpool::threadpool(size_t maxnum_threads,bool force){
    if(!force){
      maxnum_threads =
          std::min((size_t)thread::hardware_concurrency(), maxnum_threads);
    }
    start(maxnum_threads);
    numtasks.store(0);
  }
  threadpool::~threadpool(){
    stop(threads.size());
  }
  void threadpool::start(size_t num){ 
    numthreads += num;
    for(size_t i = threads.size(); i< numthreads;++i){
      threads.emplace_back([ this, i ]
      {
        while (true) {
          unique_lock<mutex> lock({m_task});

          while (i < numthreads && q_task.empty()) {
            c_task.wait(lock);
          }
          if (i >= numthreads) break;

          function<void()> task(move(q_task.front()));
          q_task.pop_front();

          lock.unlock();
          task();
          numtasks--;
          { 
            unique_lock<mutex> localLock{m_system}; 
            if (numtasks == 0) c_system.notify_all();
          }

        }
      }
      );
    }
  }

  void threadpool::stop(size_t num){
    auto numToClose = std::min(num, numthreads);

    { 
      lock_guard<mutex> lock{m_task};
      numthreads -= numToClose;
    }

    c_task.notify_all();
    for(auto i = 0u; i< numToClose;++i){
      threads.back().join();
      threads.pop_back();
    }
  }
  void threadpool::waitUntilFinished(){
    unique_lock<mutex> lock{m_system};
    if (numtasks == 0) return;
    c_system.wait(lock);
  }
  void threadpool::waitUntilFinishedFor(const std::chrono::microseconds duration){
    unique_lock<mutex> lock{m_system};
    if (numtasks == 0) return;
    c_system.wait_for(lock, duration);
  }
  void threadpool::flushQueue(){
    lock_guard<mutex> lock{m_task};
    numtasks -= q_task.size();
    q_task.clear();
  }
}
