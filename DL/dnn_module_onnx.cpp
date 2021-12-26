#include "pch.h"
#include "dnn_module_onnx.h"
#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "dnn_module_torch.h"
cv::Mat dnn_module_onnx::get_inputs() {

  cv::Mat blob;
  auto buffers = get_preprocess_image_buffers();
  if (buffers.size() == 0) return cv::Mat();
  blob = cv::dnn::blobFromImages(buffers);

  return blob;
 }

dnn_module_onnx::dnn_module_onnx() {

}
dnn_module_onnx::~dnn_module_onnx() {

}

void dnn_module_onnx::createSession(string model_path) {
  ;
  wstring onnx_path;
  onnx_path.assign(cfg_.modelFileName.begin(), cfg_.modelFileName.end());
  session_ = Ort::Session(env_, onnx_path.c_str(), Ort::SessionOptions{nullptr});
}


bool dnn_module_onnx::load_model(string configpath) {

  if (!cfg_.load_config(configpath)) {
    return false;
  }

  cfg_.modelFileName =   "D:\\deeplearning\\yolact-onnx\\yolact-onnx\\yolact.onnx";
  //cfg_.inputTensorNames = "input.1";
  cfg_.dnn_width = 550;
  cfg_.dnn_height = 550;
  cfg_.dnn_chnnel = 3;
  cfg_.is_nchw = true;

  createSession(cfg_.modelFileName);
  return true;
}



void dnn_module_onnx::yolacttest() {

  try {
    // pred_onx = sess.run([loc_name, conf_name, mask_name, priors_name,
    //([1, 19248, 4]) ,([1, 19248, 81]), ([1, 19248, 32]), ([19248, 4]) ,([1, 138, 138, 32])
     //([1, 19248, 4]) ,([1, 19248, 81]), ([1, 19248, 32]), ([19248, 4]) ,([1, 138, 138, 32])
    const char* input_names[] = {"input.1"};
    const char* output_names[] = {"792", "801", "794", "992", "596"};
    int output_name_n = 5;
    int cnt = 0;

    cv::Mat input_image_ = get_inputs();
    
    std::array<int64_t, 4> input_shape_{1, input_image_.channels(), input_image_.rows, input_image_.cols};
    
    auto memory_info =  Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    size_t bytes = input_image_.total() * input_image_.elemSize();
    
    auto input_tensor_ = Ort::Value::CreateTensor<float>(
        memory_info, reinterpret_cast<float*>(input_image_.data), bytes,
        input_shape_.data(),
        input_shape_.size());
    auto rst = session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_, 1, output_names, output_name_n);
  
    float* loc_name = rst[0].GetTensorMutableData<float>();
    float* conf_name = rst[1].GetTensorMutableData<float>();
    float* mask_name = rst[2].GetTensorMutableData<float>();
    float* priors_name = rst[3].GetTensorMutableData<float>();
    float* proto_name = rst[4].GetTensorMutableData<float>();

  
    //dnn_module_torch::detectYolact(this->cfg_,loc_name, conf_name, mask_name, priors_name,  proto_name, 19248, 138);
  
  } catch (const Ort::Exception& exception) {
    cout << exception.what() << endl;
  }
 }
