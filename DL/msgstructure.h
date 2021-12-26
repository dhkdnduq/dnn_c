#pragma once
#include <string>
struct write_ws_text {
  std::wstring path;
  std::wstring text;
  write_ws_text(std::wstring const& path_, std::wstring const& text_)
      : path(path_), text(text_) {}
};

struct write_s_text {
  std::wstring path;
  std::string text;
  write_s_text(std::wstring const& path_, std::string const& text_)
      : path(path_), text(text_) {}
};