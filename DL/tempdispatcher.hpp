#pragma once
class tsqueue;
class message_base;
namespace messaging 
{
template <typename prev_dispatcher, typename Msg, typename Func>
class temp_dispatcher {
 public:
  temp_dispatcher(temp_dispatcher&& other)
      : q(other.q),
        prev(other.prev),
        f(std::move(other.f)),
        chained(other.chained) {
    other.chained = true;
  }
  temp_dispatcher(tsqueue* q_, prev_dispatcher* prev_, Func&& f_)
    : q(q_), prev(prev_), f(std::forward<Func>(f_)),chained(false) {
    prev_->chained = true;
  }
  template<typename OtherMsg , typename OtherFunc> 
  temp_dispatcher<temp_dispatcher , OtherMsg , OtherFunc> 
    handle(OtherFunc&& of) {
    return temp_dispatcher<temp_dispatcher, OtherMsg, OtherFunc>(
        q, this, std::forward<OtherFunc>(of));
  }
  bool chained;
  __declspec(nothrow) ~temp_dispatcher() {
    if(!chained) {
      wait_and_dispatch();
    }
  }

  private:
   tsqueue* q;
   prev_dispatcher* prev;
   Func f;
  

   temp_dispatcher(temp_dispatcher const&);
   temp_dispatcher& operator=(temp_dispatcher const&);

   template<typename Dispatcher,typename OtherMsg , typename OtherFunc>
   friend class temp_dispatcher;

   void wait_and_dispatch() { 
     for (;;) {
       auto msg = q->wait_and_pop();
       if (dispatch(msg)) 
         break;
     }
   }

   bool dispatch(std::shared_ptr<message_base> const& msg) { 
     if (wrapped_message<Msg>* wrapper =  dynamic_cast<wrapped_message<Msg>* > (msg.get())) {
       f(wrapper->contents);
       return true;
       
     }else {
       return prev->dispatch(msg);
     }
   }
};
} // namespace messaging