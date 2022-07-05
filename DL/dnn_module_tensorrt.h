#pragma once
#include "model_config.h"
#include "dnn_impl.h"
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <random>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include "opencv2/opencv.hpp"
#include <filesystem>
using namespace sample;
class dnn_module_tensorrt : public dnn_impl
{
private:
  template <typename T>
  using TRTUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

 public:
  dnn_module_tensorrt():mEngine(nullptr) {
 
  }
  ~dnn_module_tensorrt();

 protected:
  void nchw_from_images(float* dst);
  void nhwc_from_images(float* dst);
  void blob_from_images(float* dst);
  bool build();

 private:
  bool constructNetwork(TRTUniquePtr<nvinfer1::IBuilder>& builder,
                        TRTUniquePtr<nvinfer1::INetworkDefinition>& network,
                        TRTUniquePtr<nvinfer1::IBuilderConfig>& config,
                        TRTUniquePtr<nvonnxparser::IParser>& parser);


  int verifyCategoryClassification(const samplesCommon::BufferManager& buffers, category_rst_list& rst_container);
  int verifyBinaryClassification(const samplesCommon::BufferManager& buffers,   binary_rst_list& rst_container);
  
  int verifyYolact(const samplesCommon::BufferManager& buffers, segm_t_container_rst_list& rst_container);
  int verifyYolov5(const samplesCommon::BufferManager& buffers, bbox_t_container_rst_list& rst_container);
  bool saveEngine(const std::string& fileName);

  ICudaEngine* loadEngine(const std::string& fileName, int DLACore);
	 
public:
	virtual bool load_model(string configpath);
	int predict_category_classification(category_rst_list& rst_list);
  int predict_binary_classification(binary_rst_list& rst_list);

  int predict_yolact(segm_t_container_rst_list& rst_container);
  int predict_yolov5(bbox_t_container_rst_list& rst_container);

 private:
  nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
  vector<nvinfer1::Dims>  mOutputDims;  //!< The dimensions of the output to the network.
  shared_ptr<nvinfer1::ICudaEngine>  mEngine;  //!< The TensorRT engine used to run the network
  shared_ptr<samplesCommon::BufferManager> buffers;
  TRTUniquePtr<nvinfer1::IExecutionContext> context;
  string inputName;
  string mEngineName;
  vector<string> outputNames;
};
