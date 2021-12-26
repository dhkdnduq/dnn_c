#pragma once
#include "dl_base.h"

namespace DL {
class DLDLL PyTorch : public dl_base {
 private:
  dl_base* dl_;

 public:
  PyTorch();
  ~PyTorch();

  bool load_model(string configpath = "dnn_setting.json");
  int predict_category_classification(category_rst_list& rst_container);
  int predict_binary_classification(binary_rst_list& rst_container);
  int predict_object_detection_efficientdet(bbox_t_container_rst_list& rst_container);
  int predict_object_detection_yolact(bbox_t_container_rst_list& rst_container);
  int predict_anomaly_detection(binary_rst_list& rst_container);


  static void detectYolact(float* loc_name, float* conf_name, float* mask_name, float* priors_name,float* proto_name ,int view_size , int proto_size);

  void beginPeriod();
  void endPeriod();
};

}  // namespace DL

