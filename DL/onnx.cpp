#include "pch.h"
#include "Onnx.h"
#include "dnn_module_onnx.h"
using namespace DL;

Onnx::Onnx() { dl_ = new dnn_module_onnx(); }
Onnx::~Onnx() {
  if (dl_) delete dl_;
}

bool Onnx::load_model(string configpath) {
  return dl_->load_model(configpath);
}



void Onnx::detectYolact(unsigned char* buf, int buf_w, int buf_h, int buf_channel ) {
  auto onnx = dynamic_cast<dnn_module_onnx*>(dl_);
  return onnx->yolacttest();
}

