#pragma once

namespace messaging {
  class close_queue {};
  class dispatcher {
   private:
    tsqueue* q;
    bool chained;
    dispatcher(dispatcher const&);
    dispatcher& operator=(dispatcher const&);

    template<typename Dispatcher , typename Msg, typename Func>
    friend class temp_dispatcher;

    public:
      dispatcher(dispatcher&& other) : q(other.q), chained(other.chained) {
       other.chained = true;
      }
      explicit dispatcher(tsqueue* q_) : q(q_), chained(false) {}
      template<typename Message , typename Func>
      temp_dispatcher<dispatcher, Message, Func>handle(Func&& f) {
        return temp_dispatcher<dispatcher, Message, Func>(
            q, this, std::forward<Func>(f));
      }
      __declspec(nothrow) ~dispatcher() { 
        if (!chained) {
          wait_and_dispatch();
        }
      }

      void wait_and_dispatch() {
        for (;;) {
          auto msg = q->wait_and_pop();
          dispatch(msg);
        }
      }
      bool dispatch(std::shared_ptr<message_base> const& msg) {
        if (dynamic_cast<wrapped_message<close_queue>*>(msg.get())) {
          throw close_queue();
        }
        return false;
      }
  };
}