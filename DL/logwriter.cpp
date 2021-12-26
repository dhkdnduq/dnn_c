#include "pch.h"
#include "logwriter.h"
#include <sstream>
#include <ctime>
logwriter::~logwriter() { 
  waitwork.wait();
}

sender logwriter::get_sender() {
  return incoming;
}
void logwriter::done() {
  waitwork = std::async(&logwriter::run , this);
}

void logwriter::run() { 
  state = &logwriter::waiting_for_data; 
  try
  {
    for (;;) (this->*state)();
  }
  catch (close_queue const&)
  {
  	
  }
}
void logwriter::stop() { 
  get_sender().send(messaging::close_queue());
  waitwork.wait();
  
  for (auto& thwork : bucketlist.get_bucket()) {
    auto& bucket_list = thwork.get()->get_bucket_list();
    for (auto& bucket : bucket_list) {
      worker* _work = dynamic_cast<worker*>(bucket.second);
      _work->stop();
      delete _work;
    }
  }
  
}

void logwriter::waiting_for_data() {
 
  incoming.wait().handle<write_ws_text>([&](write_ws_text const& msg)
  {
    distribute_data(msg);
  })
  .handle<write_s_text>([&](write_s_text const& msg)
  {
    distribute_data(msg);
  }
  ); 
}


string logwriter::wstos(wstring const& ws) {
  string s;
  s.assign(ws.begin(),ws.end());
  return s;
}
void logwriter::distribute_data(write_s_text const& msg) {
  wstring wspath = msg.path;
  worker* newbucket = make_bucket(wspath);
  notify_bucket(newbucket, write_s_text(msg.path, msg.text));
}


void logwriter::distribute_data(write_ws_text const& msg) {
  wstring wspath = msg.path;
  worker * newbucket = make_bucket(wspath);
  notify_bucket(newbucket,write_s_text(msg.path,wstos(msg.text)));
}

logwriter::worker* logwriter::make_bucket(wstring wspath) {
  worker * newbucket = nullptr;
  if (!is_exist_bucket(wspath)) {
    newbucket = build_bucket(wspath,newbucket);
    add_bucket(wspath,newbucket);
  }
  else {
    newbucket = find_bucket(wspath);
  }
  return newbucket;

 }

bool logwriter::is_exist_bucket(wstring wspath) {
  return bucketlist.find_mapping(wspath);
}
logwriter::worker* logwriter::find_bucket(wstring wspath) {
  return bucketlist.value_for(wspath);
}

logwriter::worker* logwriter::build_bucket(wstring wspath, worker* work_) {
  worker* newworker = new worker(wspath);
  return newworker;
}

void logwriter::add_bucket(wstring wspath, worker* sender) {
  bucketlist.add_or_update_mapping(wspath,sender);
}
void logwriter::notify_bucket(worker* sender, write_s_text const& msg) {
  sender->stackdata(msg);
}
