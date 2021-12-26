#pragma once
#include "tsqueue.hpp"
#include "sender.hpp"
#include "dispatcher.hpp"

namespace messaging {
class receiver {
  tsqueue q;
 public:
  operator sender() { 
    return sender(&q);
  }
  dispatcher wait() {
    return dispatcher(&q);
  }
};
}  // namespace messaging