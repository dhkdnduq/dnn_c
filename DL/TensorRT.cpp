#include "pch.h"
#include "TensorRT.h"
#include "dnn_module_tensorrt.h"

using namespace DL;

TensorRT::TensorRT() { dl_ = new dnn_module_tensorrt(); }
TensorRT::~TensorRT() {
  if (dl_) delete dl_;
}

bool TensorRT::load_model(string configpath) {
  return dl_->load_model(configpath);
}

int TensorRT::predict_category_classification(unsigned char* buf, int buf_w,
                                             int buf_h, int buf_channel) {
  auto trt = dynamic_cast<dnn_module_tensorrt*>(dl_);
  return trt->predict_category_classification(buf, buf_w, buf_h, buf_channel);
}


void TensorRT::detectYolact(unsigned char* buf, int buf_w, int buf_h, int buf_channel) {

  auto torch = dynamic_cast<dnn_module_tensorrt*>(dl_);
  return torch->detectYolact(buf, buf_w, buf_h, buf_channel);

}
