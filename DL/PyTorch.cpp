#include "pch.h"
#include "PyTorch.h"
#include "dnn_module_torch.h"

using namespace DL;

PyTorch::PyTorch() { 
 dl_ = new dnn_module_torch();
}
PyTorch::~PyTorch() {
  if (dl_) delete dl_;
}


bool PyTorch::load_model(string configpath) {
  return dl_->load_model(configpath);
}

int PyTorch::predict_category_classification(category_rst_list& rst_container) {
  auto torch = dynamic_cast<dnn_module_torch*>(dl_);
  return torch->predict_category_classification(rst_container);
}
int PyTorch::predict_binary_classification(binary_rst_list& rst_container) {
  auto torch = dynamic_cast<dnn_module_torch*>(dl_);
  return torch->predict_binary_classification(rst_container);
}


int PyTorch::predict_object_detection_efficientdet(bbox_t_container_rst_list& rst_container) {
  auto torch  = dynamic_cast<dnn_module_torch*>(dl_);
  return torch->predict_object_detection_efficientdet(rst_container);
}

int PyTorch::predict_object_detection_yolact(bbox_t_container_rst_list& rst_container) {
  auto torch  = dynamic_cast<dnn_module_torch*>(dl_);
  return torch->predict_object_detection_yolact(rst_container);
}



  int PyTorch::predict_anomaly_detection(binary_rst_list& rst_container)
  {
    auto torch = dynamic_cast<dnn_module_torch*>(dl_);
    return torch->predict_anomaly_detection(rst_container);
  }

  void PyTorch::detectYolact(float* loc_name, float* conf_name,
                             float* mask_name, float* priors_name,
                             float* proto_name, int view_size, int proto_size) {

     cv::Mat mat = dnn_module_torch::detectYolact( loc_name, conf_name, mask_name, priors_name,
                                   proto_name,view_size,proto_size);
 }

  void PyTorch::beginPeriod() { dl_->start_timer(); }
void PyTorch::endPeriod() { dl_->end_timer(); }

