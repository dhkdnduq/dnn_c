#include "pch.h"
#include "trtonnx.h"
#include "dl_base.h"
#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"

bool TRTOnnx::build()
{
    auto builder = TRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatch =
        1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
   

    auto network = TRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = TRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser = TRTUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }

    IOptimizationProfile* profile = builder->createOptimizationProfile();
    
    if (cfg_.is_nhcw == 0 ) {
      profile->setDimensions(
          cfg_.inputTensorNames.c_str(), OptProfileSelector::kMIN,
          Dims4(1, cfg_.dnn_width, cfg_.dnn_height, cfg_.dnn_chnnel));
      profile->setDimensions(
          cfg_.inputTensorNames.c_str(), OptProfileSelector::kOPT,
          Dims4(1, cfg_.dnn_width, cfg_.dnn_height, cfg_.dnn_chnnel));
      profile->setDimensions(
          cfg_.inputTensorNames.c_str(), OptProfileSelector::kMAX,
          Dims4(1, cfg_.dnn_width, cfg_.dnn_height, cfg_.dnn_chnnel));
    
    }
    else
    {
      profile->setDimensions(
          cfg_.inputTensorNames.c_str(), OptProfileSelector::kMIN,
          Dims4(1, cfg_.dnn_chnnel, cfg_.dnn_width, cfg_.dnn_height));
      profile->setDimensions(
          cfg_.inputTensorNames.c_str(), OptProfileSelector::kOPT,
          Dims4(1, cfg_.dnn_chnnel, cfg_.dnn_width, cfg_.dnn_height));
      profile->setDimensions(
          cfg_.inputTensorNames.c_str(), OptProfileSelector::kMAX,
          Dims4(1, cfg_.dnn_chnnel, cfg_.dnn_width, cfg_.dnn_height));
    }
   
    
    config->addOptimizationProfile(profile);

    ifstream engineFile (mEngineName,ios::binary);
	  if (engineFile)
	  {
          mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(loadEngine(mEngineName, -1), samplesCommon::InferDeleter());
	  }
    else
    {
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

    
	// Create RAII buffer manager object
    auto bex = mEngine->hasImplicitBatchDimension();

    buffers = std::shared_ptr<samplesCommon::BufferManager>(new samplesCommon::BufferManager(mEngine, 0));

	  context = TRTUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
	  if (!context)
	  {
		  return false;
	  }

	  // fix batchsize  1 
	  mInputDims.d[0] = 1;
	  context->setBindingDimensions(0, mInputDims);
	  // We can only run inference once all dynamic input shapes have been specified.
	  if (!context->allInputDimensionsSpecified())
	  {
		  return false;
	  }

    mOutputDims.clear();

    for (int i = 0; i < network->getNbOutputs(); i++) {
      mOutputDims.push_back(network->getOutput(i)->getDimensions());
    }
   

    /*
    for (int i = 0; i < network->getNbOutputs(); i++)
    {
        cout << "Name:" << network->getOutput(i)->getName() << " : " << network->getOutput(i)->getDimensions() << endl;

    }
    */
    return true;
}

//!
//! \brief Uses a ONNX parser to create the Onnx MNIST Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool TRTOnnx::constructNetwork(TRTUniquePtr<nvinfer1::IBuilder>& builder,
    TRTUniquePtr<nvinfer1::INetworkDefinition>& network, TRTUniquePtr<nvinfer1::IBuilderConfig>& config,
    TRTUniquePtr<nvonnxparser::IParser>& parser)
{
    auto parsed = parser->parseFromFile(cfg_.modelFileName.c_str(), static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    // const char * test = network->getOutput(0)->getName();

     // Attach a softmax layer to the end of the network.
     //auto softmax = network->addSoftMax(*network->getOutput(0));

    builder->setMaxBatchSize(cfg_.batchSize);
    config->setMaxWorkspaceSize(2_GiB);
   
    if (cfg_.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (cfg_.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), cfg_.dlaCore);

    return true;
}
bool TRTOnnx::inferClassification(cv::Mat& frame, int& result) {
  // while (true)
  {
    // preprocessYolactImage(frame);
    preprocessImage(frame);
    /*
    if (!processInput(buffers))
    {
        return false;
    }
    */
    // Memcpy from host input buffers to device input buffers
    buffers->copyInputToDevice();
    bool status = context->executeV2(buffers->getDeviceBindings().data());
    if (!status) {
      return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers->copyOutputToHost();
  }

  // Verify results
  result = verifyClassification(*buffers);
  // result = verifyClassification(*buffers);
  return true;

}
//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool TRTOnnx::inferYolact(cv::Mat& frame, int& result)
{
    //while (true)
    {
      // preprocessYolactImage(frame);
      preprocessImage(frame);
        /*
        if (!processInput(buffers))
        {
            return false;
        }
        */
        // Memcpy from host input buffers to device input buffers
        buffers->copyInputToDevice();
        bool status = context->executeV2(buffers->getDeviceBindings().data());
        if (!status)
        {
            return false;
        }

        // Memcpy from device output buffers to host output buffers
        buffers->copyOutputToHost();
    }

    // Verify results
    result = verifyYolact(frame,*buffers);
    //result = verifyClassification(*buffers);
    return true;
}

void TRTOnnx::preprocessImage(cv::Mat frame) {
  // vector<double> mean{103.94, 116.78, 123.68};
  // vector<double> std{57.38, 57.12, 58.4};
  vector<double> mean = cfg_.mean;
  vector<double> std = cfg_.std;
  cv::cuda::GpuMat gpu_frame;
  gpu_frame.upload(frame);

  int input_width, input_height;
  input_width = input_height = 0;

  if (cfg_.is_nhcw == 0) {
    input_width = mInputDims.d[2];
    input_height = mInputDims.d[1];
  }
  // nchw
  else if (cfg_.is_nhcw == 1) {
    input_width = mInputDims.d[3];
    input_height = mInputDims.d[2];
  }

  auto input_size = cv::Size(input_width, input_height);

  cv::cuda::GpuMat resized;
  cv::cuda::resize(gpu_frame, resized, input_size, 0, 0, cv::INTER_LINEAR);
  cv::cuda::GpuMat flt_image;

  if (frame.channels() == 1)
    cv::cuda::cvtColor(resized, resized, cv::COLOR_GRAY2BGR);

  resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
  int channel = flt_image.channels();
  // subtract mean
  vector<cv::cuda::GpuMat> img_channels;

  if (cfg_.mean_sub_enable) {
    // bgr
    cv::cuda::GpuMat subtract, divide;
    // bgr
    cv::cuda::subtract(flt_image, cv::Scalar(mean[0], mean[1], mean[2]),
                       subtract);
    cv::cuda::split(subtract, img_channels);
    cv::cuda::divide(img_channels[0], std[0], img_channels[0]);
    cv::cuda::divide(img_channels[1], std[1], img_channels[1]);
    cv::cuda::divide(img_channels[2], std[2], img_channels[2]);

    cv::cuda::merge(img_channels, flt_image);
  }
  cv::Mat cpumat(flt_image);

   // HWC to NCHW
  if (cfg_.is_nhcw)
     cv::dnn::blobFromImage(cpumat, cpumat);

  size_t bytes = cpumat.total() * cpumat.elemSize();
  // cv::Mat output(input_size,CV_32FC3,cpumat.data);

  float* hostDataBuffer = static_cast<float*>(buffers->getHostBuffer(cfg_.inputTensorNames));

  bytes = cpumat.total() * cpumat.elemSize();
  memcpy(hostDataBuffer, cpumat.data, bytes);

}


void TRTOnnx::preprocessYolactImage(cv::Mat frame) {

  //vector<double> mean{103.94, 116.78, 123.68};
  //vector<double> std{57.38, 57.12, 58.4};
  vector<double> mean = cfg_.mean;
  vector<double> std = cfg_.std;
   cv::cuda::GpuMat gpu_frame;
  gpu_frame.upload(frame);

  int input_width, input_height;
  input_width = input_height = 0;

  if (cfg_.is_nhcw == false) {
    input_width = mInputDims.d[2];
    input_height = mInputDims.d[1];
  }
  // nchw
  else if (cfg_.is_nhcw == true) {
    input_width = mInputDims.d[3];
    input_height = mInputDims.d[2];
  }

   auto input_size = cv::Size(input_width, input_height);

  cv::cuda::GpuMat resized;
  cv::cuda::resize(gpu_frame, resized, input_size, 0, 0, cv::INTER_LINEAR);
  cv::cuda::GpuMat flt_image;

  if (frame.channels() == 1)
    cv::cuda::cvtColor(resized, resized, cv::COLOR_GRAY2BGR);

  resized.convertTo(flt_image, CV_32FC3);
  int channel = flt_image.channels();
  // subtract mean
  vector<cv::cuda::GpuMat> img_channels;

  // bgr
  cv::cuda::GpuMat subtract, divide;
  // bgr
  cv::cuda::subtract(flt_image, cv::Scalar(mean[0], mean[1], mean[2]),
                     subtract);
  cv::cuda::split(subtract, img_channels);
  cv::cuda::divide(img_channels[0], std[0], img_channels[0]);
  cv::cuda::divide(img_channels[1], std[1], img_channels[1]);
  cv::cuda::divide(img_channels[2], std[2], img_channels[2]);
  cv::cuda::merge(img_channels, flt_image);
  std::vector<cv::cuda::GpuMat> chw;
  int channels = frame.channels();
  for (size_t i = 0; i < channels; ++i) {
    chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1));
  }

  cv::cuda::split(flt_image, chw);
  cv::Mat cpumat;
  cv::cuda::merge(chw, cpumat);
  cv::cvtColor(cpumat, cpumat, cv::COLOR_BGR2RGB);
  // HWC to CHW
  cv::dnn::blobFromImage(cpumat, cpumat);

  size_t bytes = cpumat.total() * cpumat.elemSize();
  // cv::Mat output(input_size,CV_32FC3,cpumat.data);

  float* hostDataBuffer =
      static_cast<float*>(buffers->getHostBuffer(cfg_.inputTensorNames));

  bytes = cpumat.total() * cpumat.elemSize();
  memcpy(hostDataBuffer, cpumat.data, bytes);
  
}

bool TRTOnnx::saveEngine(const std::string& fileName)
{
    std::ofstream engineFile(fileName, std::ios::binary);
    if (!engineFile)
    {
        cout << "Cannot open engine file:" << fileName << std::endl;
        return false;
    }
    TRTUniquePtr<IHostMemory> serializedEngine(mEngine->serialize());
    if (serializedEngine == nullptr)
    {
        cout << "Engine serialization failed" << fileName << std::endl;

    }

    engineFile.write(static_cast<char*>(serializedEngine->data()), serializedEngine->size());
    return !engineFile.fail();
}

ICudaEngine* TRTOnnx::loadEngine(const std::string& fileName, int DLACore)
{
    std::ifstream engineFile(fileName, std::ios::binary);
    if (!engineFile)
    {
        cout << "Error opening engine file: " << fileName << std::endl;
        return nullptr;
    }

    engineFile.seekg(0, engineFile.end);
    long int fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);
    if (!engineFile)
    {
        cout << "Error loading engine file: " << fileName << endl;
        return nullptr;
    }

    TRTUniquePtr<IRuntime> runtime(createInferRuntime(gLogger.getTRTLogger()));
    if (DLACore != -1)
    {
        runtime->setDLACore(DLACore);
    }

    return runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr);
}

int TRTOnnx::verifyYolact(cv::Mat& frame,const samplesCommon::BufferManager& buffers)
{
    float* loc_name = static_cast< float*>(buffers.getHostBuffer(cfg_.loc_name)); //([1, 19248, 4])
    float* conf_name = static_cast< float*>(buffers.getHostBuffer(cfg_.conf_name));//([1, 19248, 81])
    float* mask_name = static_cast< float*>(buffers.getHostBuffer(cfg_.mask_name));//([1, 19248, 32])
    float* priors_name = static_cast< float*>(buffers.getHostBuffer(cfg_.priors_name));//([19248, 4])
    float* proto_name = static_cast< float*>(buffers.getHostBuffer(cfg_.proto_name));//([1, 138, 138, 32])

    int view_size = mOutputDims[3].d[0];//19248
    int proto_size = mOutputDims[4].d[1];//138
    
    dl_base::predict_yolact(this->cfg_,loc_name, conf_name, mask_name,
                              priors_name, proto_name, view_size,proto_size);


  return 0;
}

    //!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
int TRTOnnx::verifyClassification(const samplesCommon::BufferManager& buffers)
{
  int outputSize = mOutputDims[0].d[1];
  //-1로 나옴.3으로 교환. class 개수만큼
  outputSize = cfg_.classNum;

  float* output =  static_cast<float*>(buffers.getHostBuffer(cfg_.outputTensorNames));

  float val{0.0f};
  int idx{0};

  // Calculate Softmax
  float sum{0.0f};
  for (int i = 0; i < outputSize; i++) {
    float score = output[i];
    output[i] = exp(output[i]);
    sum += output[i];
  }
  #ifdef DEBUG
  gLogInfo << "Output:" << "\n";
  #endif
  for (int i = 0; i < outputSize; i++) {
    output[i] /= sum;
    val = std::max(val, output[i]);
    if (val == output[i]) {
      idx = i;
    }
#ifdef DEBUG
    gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5)
             << std::setprecision(4) << output[i] << " "
             << "Class " << i << ": "
             << std::string(int(std::floor(output[i] * 10 + 0.5f)), '*')
             << "\n";
#endif
  }
#ifdef DEBUG
  gLogInfo <<"\n";
#endif
  return idx;
}