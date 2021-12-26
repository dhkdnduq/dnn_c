#pragma once

namespace messaging {

class sender {
  tsqueue* q;

 public:
  sender() : q(nullptr) {}
  explicit sender(tsqueue* q_) : q(q_) {}
  template <typename Message>
  void send(Message const& msg) {
    if (q) {
      q->push(msg);
    }
  }
};
}  // namespace messaging
