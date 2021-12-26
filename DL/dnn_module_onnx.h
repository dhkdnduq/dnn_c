#pragma once

#include <memory>
#include <string>
#include "dnn_impl.h"
#include "model_config.h"
#include "opencv2/opencv.hpp"
#include <onnxruntime_cxx_api.h>



/************************************************************************/
/* TRT 사용함에 따라 사용 안함                                          */
/************************************************************************/

class dnn_module_onnx : public dnn_impl {

 private:

  Ort::Session session_{nullptr};
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "Default"};


  cv::Mat get_inputs();
  void createSession(string model_path);
 public:
  dnn_module_onnx();
  ~dnn_module_onnx();
  virtual bool load_model(string configpath);

  void yolacttest();
};
