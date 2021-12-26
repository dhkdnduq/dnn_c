#include "pch.h"
#include "logmanager.h"
#include "logdispatcher.h"
#include "logsender.h" 
#include <random>
#include <functional>
#include <ctime>
#include <iomanip>
#include <sstream>

logmanager::logmanager(void)
{
  m_isrun = true;
}

logmanager::~logmanager(void)
{ delete ldispatcher;
  delete lsender;
}

void logmanager::start()
{
  ldispatcher = new logdispatcher();
  ldispatcher->done();
  lsender = new logsender(ldispatcher->get_sender());
  lsender->done();
  m_isrun = true;
}

void logmanager::stop()
{
  ldispatcher->stop();
  lsender->stop();
  m_isrun = false;
}
logmanager& logmanager::getinst()
{
  static logmanager inst;
  return inst;
}
void logmanager::writelog(std::wstring const& path, std::string const& msg) {
  lsender->write(path, msg);
}

void logmanager::writelog(std::wstring const& path, std::wstring const& msg) {
  lsender->write(path, msg);
}

wstring logmanager::date_format(wstring const& ws) {
  wstringstream ss;
  std::time_t rawtime;
  std::tm* timeinfo;
  char buffer[255];

  std::time(&rawtime);
  timeinfo = std::localtime(&rawtime);

  std::strftime(buffer, 255, "%Y-%m-%d-%H-%M-%S", timeinfo);
  std::puts(buffer);

  ss << "[ " << buffer << " ]>>" << ws << "\r\n";
  return ss.str();
}

void logmanager::testbegin()
{ 
  int threadcount = 10;

  bool bRun = true;
  vector<std::thread> vtest;

  auto run = [=](int thereadnum) {
    const wchar_t charset[] =
        L"0123456789"
        L"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        L"abcdefghijklmnopqrstuvwxyz";

    int ntestseconds = 20;

    wstring filepath = L"D:\\logtest\\";
    bool bMultiPathMode = false;
    if (bMultiPathMode) {
      filepath.append(std::to_wstring(thereadnum));
    }
    filepath.append(L"_logtest.txt");
    std::chrono::system_clock::time_point _begin =
        std::chrono::system_clock::now();
    for (;;) {
      chrono::seconds _checkTime = chrono::duration_cast<chrono::seconds>(
          std::chrono::system_clock::now() - _begin);

      if (_checkTime.count() > ntestseconds) break;

      auto seed = std::chrono::system_clock::now().time_since_epoch().count();
      std::mt19937_64 ranNum(seed);

      // sleep time
      std::uniform_int_distribution<int> sleepfor(1, 10);
      auto sleeptime = std::bind(sleepfor, ranNum);
      std::this_thread::sleep_for(std::chrono::milliseconds(0));

      // make random str
      
      wstring str;
      for (int i = 0; i < sleeptime(); i++) {
        str.append(charset);
      }
      str.append(L"\n");
      date_format(str);
      // TRACE ( L"thread id : %d \n",thereadnum);
      writelog(filepath, str);
    }
  };
  for (int i = 0; i < threadcount; i++) {
    vtest.emplace_back(std::thread(run, std::ref(i)));
  }

  for (int i = 0; i < threadcount; i++) {
    vtest[i].join();
  }

}