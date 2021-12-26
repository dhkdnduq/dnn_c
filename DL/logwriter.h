#pragma once
#include <future>
#include "tempdispatcher.hpp"
#include "sender.hpp"
#include "dispatcher.hpp"
#include "receiver.hpp"
#include "msgstructure.h"
#include "tsqueue.hpp"
using namespace std;
using namespace messaging;

class logwriter {
  class worker {
   private:
    worker(worker const&);
    worker& operator=(worker const&);

   public:
    std::future<void> t;
    receiver incoming;

    wstring filepath;
    std::ofstream of;
    messaging::tsqueue dataq;
    bool isopen() {
      if (of.is_open()) return true;

      of = std::ofstream(filepath, std::ios::out | ios::app);

      if (of.fail()) return false;

      return true;
    }

   public:
    void run() {
      for (;;) {
        auto data = dataq.wait_and_pop();
        if (!isopen()) continue;

        if (wrapped_message<write_s_text>* wrapper =
                dynamic_cast<wrapped_message<write_s_text>*>(data.get())) {
          of << wrapper->contents.text;
        } else if (wrapped_message<close_queue>* wrapper =
                       dynamic_cast<wrapped_message<close_queue>*>(
                           data.get())) {
          break;
        }
      }
    }
    void stackdata(write_s_text const& msg) { dataq.push(msg); }
    void stop() {
      dataq.push(close_queue());
      t.wait();
      of.close();
    }

   public:
    worker(wstring filepath_) {
      filepath = filepath_;
      of = std::ofstream(filepath, std::ios::out | std::ios::app);
      t = std::async(&logwriter::worker::run, this);
    }
    ~worker() {}
    sender get_sender() { return incoming; }
  };

 private:
  logwriter(logwriter const&);
  logwriter& operator=(logwriter const&);
  receiver incoming;
  sender logdispatcher;
  lookup_table<wstring, worker*> bucketlist;
  void (logwriter::*state)();

 protected:
  std::future<void> waitwork;
  sender get_sender();
  void done();
  void waiting_for_data();
  void distribute_data(write_ws_text const& msg);
  void distribute_data(write_s_text const& msg);

  void add_bucket(wstring wspath, worker* sener);
  void notify_bucket(worker* sender, write_s_text const& msg);
  bool is_exist_bucket(wstring wspath);
  worker* make_bucket(wstring wspath);
  worker* build_bucket(wstring wspath, worker* work_);
  worker* find_bucket(wstring wspath);
  string wstos(wstring const& ws);

 public:
  logwriter(sender logdispatcher_) : logdispatcher(logdispatcher_){};
  ~logwriter();
  void run();
  void stop();
  friend class logdispatcher;
};


