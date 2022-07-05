#pragma once


#include "structure.h"
#include "model_config.h"
#include "opencv2/opencv.hpp"

class  dnn_impl {
 
 protected:
  void gen_dl_image_gpu(cv::Mat frame);
  void gen_dl_image_cpu(cv::Mat frame);
  vector<cv::Mat>& get_origin_image_buffers() { return ori_image_buffer; }
  vector<cv::Mat>& get_preprocess_image_buffers() { return preprocess_image_buffer; }

 public:
	dnn_impl();



  static void matToImageinfo(cv::Mat src, image_info& imInfo);
  static cv::Mat imageinfoToMat(image_info& imInfo );


  void clear_buffer() {
    ori_image_buffer.clear();
    preprocess_image_buffer.clear();
  }
	virtual bool load_model(string configpath = "dnn_setting.json") = 0;
  int get_batch_size() { return cfg_.batchSize; };
  bool file_exists(const std::string fileName, bool verbose);
  bool add_image(const char* filepath);
  bool add_image(const cv::Mat data);
  bool add_image(unsigned char* buf, const size_t data_length);
  void fill_dummy_image_if_not_enough();
  //multi thread 사용시 lock 필요 
  static int predict_yolact(model_config& cfg, float* loc_name,
                              float* conf_name, float* mask_name,
                              float* priors_name, float* proto_name,int class_num,
                            int view_size, int proto_size,vector<cv::Mat>& origin_image,segm_t_container_rst_list& rst_container);
  static int predict_yolov5(model_config& cfg, float* prediction,int output_dim1_size,int output_dim2_size,
                            vector<cv::Mat>& origin_image,
                            bbox_t_container_rst_list& rst_container);

  static void nhwc_blob_from_images(vector<cv::Mat> buffers,float* hostDataBuffer);

  protected:
  vector<cv::Mat> ori_image_buffer;
  vector<cv::Mat> preprocess_image_buffer;

  model_config cfg_;
};

