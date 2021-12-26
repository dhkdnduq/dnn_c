#pragma once
#include <future>
using namespace messaging;

  class logsender{
  private:
  logsender(logsender const&);
  logsender& operator=(logsender const&);
  void (logsender::*state)();
  
  std::future<void> waitwork;
  receiver incoming;
  sender logdispatch;

  protected:
  void run();
  void waiting_for_text();
  void done_processing();

 public:
  sender get_sender();
  logsender(sender logdispatch_) : logdispatch(logdispatch_){};
 ~logsender();
  void done();
  void write(wstring const& path , wstring const& msg);
  void write(wstring const& path, string const& msg);
  void stop();
  void test();

  };
