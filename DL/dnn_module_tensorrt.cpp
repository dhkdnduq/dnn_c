#include "pch.h"
#include "dnn_module_tensorrt.h"
#include <opencv2/cudaarithm.hpp>
#include "opencv2/cudawarping.hpp"
#include "calibrator.hpp"
#include "common.h"
#include "logger.h"

const std::string gLoggerName = "TensorRT.DNN";
auto log_ = sample::gLogger.defineTest(gLoggerName, 0, nullptr);
  
dnn_module_tensorrt::~dnn_module_tensorrt() {}


bool dnn_module_tensorrt::load_model(string configpath)
{
	if (!cfg_.load_config(configpath))
	{
		return false;
	}

	sample::gLogger.reportTestStart(log_);

	if (!build())
	{
		return gLogger.reportFail(log_);
	}

	return true;
}

int dnn_module_tensorrt::predict_category_classification(category_rst_list& rst_list )
{
  float* hostDataBuffer =static_cast<float*>(buffers->getHostBuffer(inputName));
  blob_from_images(hostDataBuffer);
  // Memcpy from device output buffers to host output buffers
  buffers->copyInputToDevice();
  bool status = context->executeV2(buffers->getDeviceBindings().data());
  if (!status) {
    sample::gLogger.reportFail(log_);
    return false;
  }
  buffers->copyOutputToHost();
  return verifyClassification(*buffers, rst_list);
}
int dnn_module_tensorrt::predict_yolov5(bbox_t_container_rst_list& rst_container)
{
  float* hostDataBuffer = static_cast<float*>(buffers->getHostBuffer(inputName));
  blob_from_images(hostDataBuffer);
  buffers->copyInputToDevice();
  bool status = context->executeV2(buffers->getDeviceBindings().data());
  if (!status) {
    sample::gLogger.reportFail(log_);
    return false;
  }

  // Memcpy from device output buffers to host output buffers
  buffers->copyOutputToHost();

  return verifyYolov5(*buffers, rst_container);
}
int dnn_module_tensorrt::predict_yolact(segm_t_container_rst_list& rst_container ) {

  float* hostDataBuffer =   static_cast<float*>(buffers->getHostBuffer(inputName));
  blob_from_images(hostDataBuffer);
  buffers->copyInputToDevice();
  bool status = context->executeV2(buffers->getDeviceBindings().data());
  if (!status) {
    sample::gLogger.reportFail(log_);
    return false;
  }

  // Memcpy from device output buffers to host output buffers
  buffers->copyOutputToHost();
 
 return verifyYolact(*buffers,rst_container);

}
void dnn_module_tensorrt::nhwc_from_images(float* dst) {
  auto buffers = get_preprocess_image_buffers();
  nhwc_blob_from_images(buffers, dst);  
}

void dnn_module_tensorrt::nchw_from_images(float* dst) {
  cv::Mat blob;
  auto buffers = get_preprocess_image_buffers();
  if (buffers.size() == 0) return;
  blob = cv::dnn::blobFromImages(buffers);
  size_t bytes = 0;
  bytes = blob.total() * blob.elemSize();
  memcpy(dst, blob.data, bytes);
}

void dnn_module_tensorrt::blob_from_images(float* dst) {
  if (!cfg_.is_nchw) {
    nhwc_from_images(dst);
  } else {
    nchw_from_images(dst);
  }
}

bool dnn_module_tensorrt::build() {
  auto builder = TRTUniquePtr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
  if (!builder) {
    return false;
  }
  namespace fs = std::filesystem;

  if(!fs::exists(cfg_.modelFileName)) {
    return false;
  }

  mEngineName = fs::path(cfg_.modelFileName).replace_extension(".trt").string();

  const auto explicitBatch =
      1U << static_cast<uint32_t>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

  auto network = TRTUniquePtr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2(explicitBatch));
  if (!network) {
    return false;
  }

  auto config =
      TRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
   if (!config) {
    return false;
  }

  auto parser = TRTUniquePtr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
  if (!parser) {
    return false;
  }

  auto constructed = constructNetwork(builder, network, config, parser);
  if (!constructed) {
    return false;
  }
  inputName = network->getInput(0)->getName();
  nvinfer1::Dims input_dims =  network->getInput(0)->getDimensions();
  IOptimizationProfile* profile = builder->createOptimizationProfile();
  if (input_dims.d[1] <= 4 ){  //1 or 3 
    cfg_.is_nchw = true;
    cfg_.dnn_width = input_dims.d[3];
    cfg_.dnn_height = input_dims.d[2];
    cfg_.dnn_chnnel = input_dims.d[1];
  }
  else  {
    cfg_.is_nchw = false;
    cfg_.dnn_width = input_dims.d[2];
    cfg_.dnn_height = input_dims.d[1];
    cfg_.dnn_chnnel = input_dims.d[3];
  }

  
  if (cfg_.is_nchw == false) {
    profile->setDimensions(inputName.c_str(), OptProfileSelector::kMIN,
                           Dims4(cfg_.batchSize, cfg_.dnn_width,
                                 cfg_.dnn_height, cfg_.dnn_chnnel));
    profile->setDimensions(inputName.c_str(), OptProfileSelector::kOPT,
                           Dims4(cfg_.batchSize, cfg_.dnn_width,
                                 cfg_.dnn_height, cfg_.dnn_chnnel));
    profile->setDimensions(inputName.c_str(), OptProfileSelector::kMAX,
                           Dims4(cfg_.batchSize, cfg_.dnn_width,
                                 cfg_.dnn_height, cfg_.dnn_chnnel));

  } else {
    profile->setDimensions(inputName.c_str(), OptProfileSelector::kMIN,
                           Dims4(cfg_.batchSize, cfg_.dnn_chnnel,
                                 cfg_.dnn_width, cfg_.dnn_height));
    profile->setDimensions(inputName.c_str(), OptProfileSelector::kOPT,
                           Dims4(cfg_.batchSize, cfg_.dnn_chnnel,
                                 cfg_.dnn_width, cfg_.dnn_height));
    profile->setDimensions(inputName.c_str(), OptProfileSelector::kMAX,
                           Dims4(cfg_.batchSize, cfg_.dnn_chnnel,
                                 cfg_.dnn_width, cfg_.dnn_height));
  }

  config->addOptimizationProfile(profile);
  
  ifstream engineFile(mEngineName, ios::binary);
  if (engineFile.is_open()) {
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        loadEngine(mEngineName, -1), samplesCommon::InferDeleter());
  }
  else 
  {
    // calibration
    if (cfg_.fp16) {
      config->setFlag(BuilderFlag::kFP16);
    }
    std::unique_ptr<Int8EntropyCalibrator> calibrator;
    if (cfg_.int8) {
      config->setFlag(BuilderFlag::kINT8);
      config->setCalibrationProfile(profile);
      string imgpath = "test";
      auto calibration_images =  Int8EntropyCalibrator::getCalibrationFiles(imgpath);
      ImageStream stream(cfg_, input_dims, calibration_images);
      calibrator = std::unique_ptr<Int8EntropyCalibrator>( new Int8EntropyCalibrator(stream, cfg_.modelFileName,cfg_.modelFileName+"_clb"));
      config->setInt8Calibrator(calibrator.get());
    }


    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config),
        samplesCommon::InferDeleter());
   
    if (!mEngine) {
      return false;
    }
    if (!saveEngine(mEngineName)) {
      return false;
    }
  }

  assert(network->getNbInputs() == 1);
  mInputDims = network->getInput(0)->getDimensions();
  assert(mInputDims.nbDims == 4);

  buffers = std::shared_ptr<samplesCommon::BufferManager>(
      new samplesCommon::BufferManager(mEngine/*, cfg_.batchSize*/));

  context = TRTUniquePtr<nvinfer1::IExecutionContext>(
      mEngine->createExecutionContext());
  if (!context) {
    return false;
  }

  mInputDims.d[0] = cfg_.batchSize;
  context->setBindingDimensions(0, mInputDims);
  // We can only run inference once all dynamic input shapes have been
  // specified.
  if (!context->allInputDimensionsSpecified()) {
    return false;
  }

  mOutputDims.clear();
  outputNames.clear();
  for (int i = 0; i < network->getNbOutputs(); i++) {
    outputNames.push_back(network->getOutput(i)->getName());
    int output_index = mEngine->getBindingIndex(outputNames[i].c_str());
    mOutputDims.push_back(mEngine->getBindingDimensions(output_index));
  }
 
  return true;
}

//!
//! \brief Uses a ONNX parser to create the Onnx MNIST Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx
//! MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool dnn_module_tensorrt::constructNetwork(
    TRTUniquePtr<nvinfer1::IBuilder>& builder,
    TRTUniquePtr<nvinfer1::INetworkDefinition>& network,
    TRTUniquePtr<nvinfer1::IBuilderConfig>& config,
    TRTUniquePtr<nvonnxparser::IParser>& parser) {
  auto parsed =  parser->parseFromFile(cfg_.modelFileName.c_str(), static_cast<int>(gLogger.getReportableSeverity()));
  if (!parsed) {
    return false;
  }

  // const char * test = network->getOutput(0)->getName();

  builder->setMaxBatchSize(cfg_.batchSize);

//#ifdef INCREASE_STACK_SIZE
//  config->setMaxWorkspaceSize(6_GiB);
  //#else
  config->setMaxWorkspaceSize(6_GiB);
  //#endif
  samplesCommon::enableDLA(builder.get(), config.get(), cfg_.dlaCore);

  return true;
}

bool dnn_module_tensorrt::saveEngine(const std::string& fileName) {
  std::ofstream engineFile(fileName, std::ios::binary);
  if (!engineFile) {
    cout << "Cannot open engine file:" << fileName << std::endl;
    return false;
  }
  TRTUniquePtr<IHostMemory> serializedEngine(mEngine->serialize());
  if (serializedEngine == nullptr) {
    cout << "Engine serialization failed" << fileName << std::endl;
  }

  engineFile.write(static_cast<char*>(serializedEngine->data()),
                   serializedEngine->size());
  return !engineFile.fail();
}

ICudaEngine* dnn_module_tensorrt::loadEngine(const std::string& fileName, int DLACore) {
  std::ifstream engineFile(fileName, std::ios::binary);
  if (!engineFile) {
    cout << "Error opening engine file: " << fileName << std::endl;
    return nullptr;
  }

  engineFile.seekg(0, engineFile.end);
  long int fsize = engineFile.tellg();
  engineFile.seekg(0, engineFile.beg);

  std::vector<char> engineData(fsize);
  engineFile.read(engineData.data(), fsize);
  if (!engineFile) {
    cout << "Error loading engine file: " << fileName << endl;
    return nullptr;
  }

  TRTUniquePtr<IRuntime> runtime(createInferRuntime(gLogger.getTRTLogger()));
  if (DLACore != -1) {
    runtime->setDLACore(DLACore);
  }

  return runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr);
}
int dnn_module_tensorrt::verifyYolov5(const samplesCommon::BufferManager& buffers, bbox_t_container_rst_list& rst_container) {

 /*85 is 80classes + 4bbox +   1score.
 25200 is the combined layer of preliminary output layers that are 20x20, 40x40,
 80x80(with the default 3 scales P3, P4, P5 model) and 3 anchors
 (20 ^ 2 + 40 ^ 2 + 80 ^ 2)* 3 = 25200 You can see those 3 layers in netron.  */
  float* prediction = static_cast<float*>( buffers.getHostBuffer(outputNames[0]));  //([1, 25200, 4])

  return dnn_impl::predict_yolov5(this->cfg_, prediction,mOutputDims[0].d[1],mOutputDims[0].d[2],ori_image_buffer,rst_container);
 }
int dnn_module_tensorrt::verifyYolact( const samplesCommon::BufferManager& buffers, segm_t_container_rst_list& rst_container) {


  float* loc_name = static_cast<float*>(
      buffers.getHostBuffer(outputNames[0]));  //([1, 19248, 4])
  float* conf_name = static_cast<float*>(
      buffers.getHostBuffer(outputNames[1]));  //([1, 19248, 81])
  float* mask_name = static_cast<float*>(
      buffers.getHostBuffer(outputNames[2]));  //([1, 19248, 32])
  float* priors_name = static_cast<float*>(
      buffers.getHostBuffer(outputNames[3]));  //([19248, 4])
  float* proto_name = static_cast<float*>(
      buffers.getHostBuffer(outputNames[4]));  //([1, 138, 138, 32])

  int view_size = mOutputDims[0].d[1];   // 19248
  int proto_size = mOutputDims[4].d[1];  // 138
  int class_num = mOutputDims[1].d[2]; 
  return dnn_impl::predict_yolact(this->cfg_, loc_name, conf_name, mask_name, priors_name, proto_name, class_num,view_size, proto_size,
                          ori_image_buffer, rst_container);

}

//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
int dnn_module_tensorrt::verifyClassification(const samplesCommon::BufferManager& buffers, category_rst_list& rst_container) {
 
  int batchSize = cfg_.batchSize;
  int classNum = mOutputDims[0].d[1];
  float* outputs = static_cast<float*>(buffers.getHostBuffer(outputNames[0]));
  
  for (int batch_index = 0; batch_index < cfg_.batchSize; batch_index++) {
      float* output = &(outputs[batch_index * classNum]);
      
      float val{0.0f};
      int idx{0};
      /*
      float sum{0.0f};
      for (int i = 0; i < outputSize; i++) {
        output[i] = exp(output[i]);
        sum += output[i];
      }
      
      for (int i = 0; i < outputSize; i++) {
        output[i] /= sum;
        val = std::max(val, output[i]);
        if (val == output[i]) {
          idx = i;
        }
        cout << " Prob " << i << "  " << std::fixed << std::setw(5)
                         << std::setprecision(4) << output[i] << " "
                         << "Class " << i << ": "
                         << std::string(int(std::floor(output[i] * 10 + 0.5f)),
                                        '*')
                         << std::endl;
      }
      */
      
      for (int category = 0; category < classNum; category++) {
        float prob = output[category];
        if (val < prob) {
          idx = category;
          val = prob;
        }
      }
      rst_container[batch_index] = idx;
  }
  rst_container.cnt = batchSize;
  return batchSize;
}
