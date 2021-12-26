#pragma once
#include "dl_base.h"

namespace DL {
class TensorRT : public dl_base {
 private:
  dl_base* dl_;

 public:
  TensorRT();
  ~TensorRT();

  bool load_model(string configpath = "dnn_setting.json");
  int predict_category_classification(unsigned char* buf, int buf_w, int buf_h,
                                      int buf_channel);
  bool predict_binary_classification(unsigned char* buf, int buf_w, int buf_h,
                                     int buf_channel);
  void detectYolact(unsigned char* buf, int buf_w, int buf_h, int buf_channel);
};
}  // namespace DL
