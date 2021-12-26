#include "pch.h"
#include "logsender.h"

logsender::~logsender()
{
}

void logsender::done() { waitwork = std::async(&logsender::run, this); }

void logsender::run() { 
  state = &logsender::waiting_for_text;
  try
  {
    for(;;)
    {
      (this->*state)();
    }
  }
  catch (close_queue const&)
  {
  }
}
sender logsender::get_sender()
{ return incoming; }

void logsender::waiting_for_text()
{

  incoming.wait().handle<write_ws_text>(
    [&](write_ws_text const& msg)
    {
        logdispatch.send(write_ws_text(msg));
      
    }).handle<write_s_text>(
    [&](write_s_text const& msg)
    {
      logdispatch.send(write_s_text(msg));

    });
}

void logsender::done_processing()
{ state = &logsender::waiting_for_text; }

void logsender::write(wstring const& path, wstring const& msg) {
  get_sender().send(write_ws_text(path, msg));
}
void logsender::write(wstring const& path, string const& msg) {
  get_sender().send(write_s_text(path, msg));
}

void logsender::stop()
{
  get_sender().send(close_queue());
  waitwork.wait();
}
void logsender::test()
{
  get_sender().send(close_queue()); 
}