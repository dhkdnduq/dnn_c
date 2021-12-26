#pragma once
#include "receiver.hpp"
#include "sender.hpp"
#include <future>
class logwriter;
using namespace messaging;

  class logdispatcher {
 public:
   logdispatcher();
   ~logdispatcher();
   messaging::sender get_sender() { return incoming; }
  void done();
  void run();
  void stop();

 private:
  logwriter* lwriter;
  messaging::receiver incoming;
  messaging::sender fwriter;
  std::future<void> waitwork;

  };

