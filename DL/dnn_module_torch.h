#pragma once
#include <string>
#include <memory>
#include "model_config.h"
#include "dnn_impl.h"
#include "structure.h"
#include <torch/script.h>
#include "opencv2/opencv.hpp"
#include <torchvision/csrc/models/resnet.h>

using namespace torch::indexing;
using namespace std;

class  dnn_module_torch : public dnn_impl
{
 private:
 
  void loadlibrary();
  void load_anomaly_detection_padim();
  void load_anomaly_detection_patchcore();
  torch::Tensor embedding_concat(torch::Tensor x, torch::Tensor y);
  static cv::Mat tensor1dToMat(torch::Tensor t);
  static cv::Mat tensor2dToMat(torch::Tensor t);
  vector<cv::Mat> tensor3dToMat(torch::Tensor t);
  static void tensor2dToImageInfo(torch::Tensor t , image_info& imageinfo);
  torch::Tensor mahalanobis(torch::Tensor u, torch::Tensor v,torch::Tensor cov_inv);
 
 protected:
  std::vector<torch::jit::IValue> get_inputs();
 
 public:
  dnn_module_torch();
  ~dnn_module_torch();
  virtual bool load_model(string configpath);
  static void gen_colors();
  int predict_category_classification(category_rst_list& rst_container);
  int predict_binary_classification(binary_rst_list& rst_container);
  int predict_object_detection_efficientdet(bbox_t_container_rst_list& rst_container);
  int predict_anomaly_detection_padim(segm_t_container_rst_list& rst_container , int category);
  int predict_anomaly_detection_patchcore(segm_t_container_rst_list& rst_container , int category);

  static int detectYolact(model_config& cfg, float* loc_name, float* conf_name,
                           float* mask_name, float* priors_name, float* proto_name,int class_num, int view_size, int proto_size,
                          vector<cv::Mat>& origin_image, segm_t_container_rst_list& rst_containert);
  static int detectYolov5(model_config& cfg, float* prediction_ptr, 
                          int output_dim1_size,
                          int output_dim2_size,vector<cv::Mat>& origin_image,
                          bbox_t_container_rst_list& rst_containert);
  static void nhwc_blob_from_images(vector<cv::Mat> buffers, float* hostDataBuffer);

  private:
  torch::jit::script::Module module_;
  torch::jit::script::Module module_backbone;
  static c10::DeviceType default_dev;
  // for anomaly
  vision::models::WideResNet50_2 module_wideresnet_50_;

  vector<torch::Tensor> anomaly_feature_patchcore;
  torch::Tensor anomaly_conv_inv;
  torch::Tensor anomaly_mean_inv;
  torch::Tensor anomaly_rand_index;

  cv::cuda::GpuMat anomaly_mean_mat;
  vector<cv::cuda::GpuMat> anomaly_conv_inv_mat;
  static bool isinit_;
  static vector<torch::Tensor> mask_colors;
};
