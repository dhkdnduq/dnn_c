#include "pch.h"
#include "dnn_impl.h"
#include <opencv2/cudaarithm.hpp>
#include "opencv2/cudawarping.hpp"
#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "dnn_module_torch.h"


namespace fs = std::filesystem;
using namespace cv;


dnn_impl::dnn_impl()
{

}

bool dnn_impl::file_exists(const std::string fileName, bool verbose) {
  if (!fs::exists( fs::path(fileName))) {
    if (verbose) std::cout << "File does not exist : " << fileName << std::endl;
    return false;
  }
  return true;
}

void dnn_impl::gen_dl_image(cv::Mat frame) {
   ori_image_buffer.push_back(frame);

   try {
     cv::cuda::GpuMat gpu_frame;
     gpu_frame.upload(frame);
     auto input_width = cfg_.dnn_width;
     auto input_height = cfg_.dnn_height;
     auto input_size = cv::Size(input_width, input_height);

     if (cfg_.dnn_chnnel == 3) {
       cv::cuda::GpuMat resized;
       cv::cuda::resize(gpu_frame, resized, input_size, 0, 0);
       cv::cuda::GpuMat flt_image;

       if (frame.channels() == 1)
         cv::cuda::cvtColor(resized, resized, cv::COLOR_GRAY2BGR);
       //if (cfg_.dnn_scale_div != 1)
       resized.convertTo(flt_image, CV_32FC3, 1.f / cfg_.dnn_scale_div);
       int channel = flt_image.channels();
       // subtract mean
       vector<cv::cuda::GpuMat> img_channels;
       if (cfg_.is_use_mean_sub) {
         cv::cuda::GpuMat subtract, divide;
         cv::cuda::subtract(
             flt_image, cv::Scalar(cfg_.mean[2], cfg_.mean[1], cfg_.mean[0]),
             subtract);
         cv::cuda::split(subtract, img_channels);
         cv::cuda::divide(img_channels[0], cfg_.std[2], img_channels[0]);
         cv::cuda::divide(img_channels[1], cfg_.std[1], img_channels[1]);
         cv::cuda::divide(img_channels[2], cfg_.std[0], img_channels[2]);

         cv::cuda::merge(img_channels, flt_image);
       }

       //must be converted to mat, because of gpumat stride too long
     
       preprocess_image_buffer.push_back(cv::Mat(flt_image));
    

     }
   } catch (...) {
     ::cout << "failed preprocessImage\n";
   }
}

bool dnn_impl::add_image(const char* filepath)
{
  if (file_exists(filepath,true)) {
    Mat mat = imread(filepath, ImreadModes::IMREAD_COLOR);
    if (mat.empty()) return false;
    gen_dl_image(mat);
  } else {
    return false;
  }
  return true;
}
bool dnn_impl::add_image(const cv::Mat mat)
{
  if (mat.empty()) {
    cout << "buffer is empty";
    return false;
  }
  gen_dl_image(mat);
  return true;
}
bool dnn_impl::add_image(unsigned char* data, const size_t data_length)
{
  std::vector<char> vdata(data, data + data_length);
  cv::Mat mat = imdecode(cv::Mat(vdata), ImreadModes::IMREAD_COLOR);
  if (mat.empty()) {
    cout << "buffer is empty";
    return false;
  }
  gen_dl_image(mat);
  return true;
}


cv::Mat dnn_impl::imageinfoToMat(image_info& imInfo) { 
  std::vector<char> data(imInfo.data, imInfo.data + imInfo.size);
  return imdecode(data, cv::IMREAD_COLOR);
 }
void dnn_impl::matToImageinfo(cv::Mat src, image_info& imInfo) {
   if (src.empty()) return;
 

  vector<unsigned char> buf;
  imencode(".bmp", src, buf);
  imInfo.gen(buf);
 
 }

int  dnn_impl::predict_yolact(model_config& cfg, float* loc_name,
  float* conf_name, float* mask_name,
  float* priors_name, float* proto_name,int class_num,
  int view_size, int proto_size,vector<cv::Mat>& origin_image,segm_t_container_rst_list& rst_container) {
  return dnn_module_torch::detectYolact(cfg, loc_name, conf_name, mask_name,
                                        priors_name, proto_name,class_num, view_size,
                                        proto_size,origin_image,rst_container);
 }


int dnn_impl::predict_yolov5(model_config& cfg, float* prediction, int output_dim1_size,int output_dim2_size,vector<cv::Mat>& origin_image, bbox_t_container_rst_list& rst_container) {
   return dnn_module_torch::detectYolov5(
      cfg, prediction, output_dim1_size ,output_dim2_size,origin_image, rst_container);

 }


void dnn_impl::nhwc_blob_from_images(vector<cv::Mat> buffers,float* hostDataBuffer) {
   dnn_module_torch::nhwc_blob_from_images(buffers,hostDataBuffer);
 }
