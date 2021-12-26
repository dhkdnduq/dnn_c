#pragma once
#include "dl_base.h"

namespace DL {
class  Onnx : public dl_base {
 private:
  dl_base* dl_;

 public:
  Onnx();
  ~Onnx();

  bool load_model(string configpath = "dnn_setting.json");
  
  void detectYolact(unsigned char* buf, int buf_w, int buf_h, int buf_channel );

};
}  // namespace DL

