#pragma once

class logsender;
class logdispatcher;
/*
* log_instance.start();
  log_instance.testbegin();
  log_instance.stop();
*/
  class logmanager {
  public:
    logmanager(void);
   ~logmanager(void);
    void start();
    void stop();
    bool isrun() { return m_isrun; }
    static logmanager& getinst();
    wstring date_format(wstring const& ws);
    void writelog(std::wstring const& path, std::wstring const& msg);
    void writelog(std::wstring const& path, std::string const& msg);

    void testbegin();
  private:
    logdispatcher* ldispatcher;
    logsender* lsender;
    bool m_isrun;
  };




