#include "pch.h"

logdispatcher::logdispatcher()
{
  lwriter = new logwriter(get_sender());
  fwriter = lwriter->get_sender();
}

logdispatcher::~logdispatcher()
{
  delete lwriter;
}
void logdispatcher::done()
{
  waitwork = std::async(&logdispatcher::run , this);
  lwriter->done();
}
void logdispatcher::run()
{
  try
  {
    while (true) {
      incoming.wait()
          .handle<write_ws_text>([&](write_ws_text const& msg) {
            fwriter.send(msg);
          })
          .handle<write_s_text>(
              [&](write_s_text const& msg) { 
            fwriter.send(msg); 
          });
    };
  }
  catch (close_queue const&)
  {
  	
  }
}

void logdispatcher::stop()
{
  get_sender().send(close_queue());
  waitwork.wait();
  lwriter->stop();
}