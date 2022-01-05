#include "pch.h"
#include "DL.h"
#include "opencv2/opencv.hpp"
#include "dnn_base.hpp"
#include "logmanager.h"
#include "perf_timer.hpp"

using namespace cv;
static dnn_base<dnn_module_torch> dl_torch1;
static dnn_base<dnn_module_tensorrt> dl_trt1;
static dnn_base<dnn_module_torch> dl_torch2;
static dnn_base<dnn_module_tensorrt> dl_trt2;
static logmanager log_instance;


void release_image_info(image_info& info) {
  info.clear();
}
void release_segm_container(segm_t_container_rst_list & rst_list) {
  rst_list.clear();
}
bool torch_init(const char* configurationFilename, int gpu ) {
  if (gpu == 0)
    dl_torch1.init();
  else
    dl_torch2.init();
  auto dl_torch = gpu == 0 ? dl_torch1.func() : dl_torch2.func();
 
  return  dl_torch->load_model(configurationFilename);
 }
int torch_effdet(bbox_t_container_rst_list& rst_list , int gpu, bool is_clear_buffer) {
   rst_list.clear();
  auto dl_torch = gpu == 0 ? dl_torch1.func() : dl_torch2.func();
  
  int batch_size =  dl_torch->predict_object_detection_efficientdet(rst_list);
  if (is_clear_buffer)  dl_torch->clear_buffer();
  rst_list.cnt = batch_size;
  return batch_size;
 }
void torch_clear_buffer(int gpu) {
   auto dl_torch = gpu == 0 ? dl_torch1.func() : dl_torch2.func();
  dl_torch->clear_buffer();
}
 
int torch_binary_classification(binary_rst_list& rst_list, int gpu, bool is_clear_buffer) {
  rst_list.clear();
  auto dl_torch = gpu == 0 ? dl_torch1.func() : dl_torch2.func();
 
  int batch_size =  dl_torch->predict_binary_classification(rst_list);
  if (is_clear_buffer) dl_torch->clear_buffer();
  rst_list.cnt = batch_size;
  return batch_size;
}
int torch_category_classification(category_rst_list& rst_list,
                                  bool is_clear_buffer, int gpu) {
  rst_list.clear();
  auto dl_torch = gpu == 0 ? dl_torch1.func() : dl_torch2.func();
 
  int batch_size = dl_torch->predict_category_classification(rst_list);
  if (is_clear_buffer) dl_torch->clear_buffer();
  rst_list.cnt = batch_size;
  return batch_size;
}
int torch_anomaly_detection(segm_t_container_rst_list& rst_list, int category ,int gpu, bool is_clear_buffer) {
  rst_list.clear();
  
  auto dl_torch = gpu == 0 ? dl_torch1.func() : dl_torch2.func();
  //#endif
  int batch_size = dl_torch->predict_anomaly_detection_patchcore(rst_list,category);
  if (is_clear_buffer) dl_torch->clear_buffer();
  rst_list.cnt = batch_size;
  //#endif
  return batch_size;
}
bool torch_add_image_file(const char* filename, int gpu) {
  auto dl_torch = gpu == 0 ? dl_torch1.func() : dl_torch2.func();
  return dl_torch->add_image(filename);
  ;
}
bool torch_add_encoded_image(unsigned char* buf, const size_t data_length, int gpu) {
  auto dl_torch = gpu == 0 ? dl_torch1.func() : dl_torch2.func();
  return dl_torch->add_image(buf, data_length);
}
bool torch_add_buffer(unsigned char* buf, int buf_w, int buf_h,
                          int buf_channel, int gpu) {
  auto dl_torch = gpu == 0 ? dl_torch1.func() : dl_torch2.func();
 
  return dl_torch->add_image(
      cv::Mat(buf_h, buf_w, buf_channel == 3 ? CV_8UC3 : CV_8UC1, buf));
}

int torch_dispose(int gpu) {
 
  gpu == 0 ? dl_torch1.reset():dl_torch2.reset();
  return 1;
}

bool trt_init(const char* configurationFilename, int gpu) {
  if (gpu == 0)
    dl_trt1.init();
  else
    dl_trt2.init();
  auto dl_trt = gpu == 0 ? dl_trt1.func() : dl_trt2.func();
  return dl_trt->load_model(configurationFilename);
}
int trt_category_classification(category_rst_list& rst_list, int gpu ,bool is_clear_buffer) {
  rst_list.clear();
  auto dl_trt = gpu == 0 ? dl_trt1.func() : dl_trt2.func();
  int batch_size;
  perf_timer<std::chrono::milliseconds>::duration_p([&](){ batch_size = dl_trt->predict_category_classification(rst_list); });
  if (is_clear_buffer) dl_trt->clear_buffer();
  rst_list.cnt = batch_size;
  return batch_size;
}
int trt_yolov5(bbox_t_container_rst_list & rst_list, int gpu ,bool is_clear_buffer) {
  auto dl_trt = gpu == 0 ? dl_trt1.func() : dl_trt2.func();
  rst_list.clear();
  int batch_size = dl_trt->predict_yolov5(rst_list);
  if (is_clear_buffer) dl_trt->clear_buffer();
  rst_list.cnt = batch_size;
  return batch_size;
}
int trt_yolact(segm_t_container_rst_list& rst_list, int gpu, bool is_clear_buffer) {
  auto dl_trt = gpu == 0 ? dl_trt1.func() : dl_trt2.func();
  rst_list.clear();
  int batch_size = dl_trt->predict_yolact(rst_list);
  if (is_clear_buffer) dl_trt->clear_buffer();
  rst_list.cnt = batch_size;
  return batch_size;
}

int trt_dispose(int gpu) {
  gpu == 0 ? dl_trt1.reset() : dl_trt2.reset();
  return 1;
}


void trt_clear_buffer(int gpu) { 
   auto dl_trt = gpu == 0 ? dl_trt1.func() : dl_trt2.func();
  dl_trt->clear_buffer();
}

bool trt_add_image_file(const char* filename, int gpu) {
  auto dl_trt = gpu == 0 ? dl_trt1.func() : dl_trt2.func();
   return dl_trt->add_image(filename);
 }
bool trt_add_encoded_image(unsigned char* buf, const size_t data_length, int gpu) {
   auto dl_trt = gpu == 0 ? dl_trt1.func() : dl_trt2.func();
   return dl_trt->add_image(buf, data_length);
 }
bool trt_add_buffer(unsigned char* buf, int buf_w, int buf_h, int buf_channel, int gpu) {
   auto dl_trt = gpu == 0 ? dl_trt1.func() : dl_trt2.func();
   return dl_trt->add_image(cv::Mat(buf_h, buf_w, buf_channel == 3 ? CV_8UC3 : CV_8UC1, buf));
 }
