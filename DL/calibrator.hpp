#pragma once
#include <assert.h>
#include <algorithm>
#include <iterator>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <vector>
#include "NvInfer.h"
using namespace std;
using namespace cv;
using namespace nvinfer1;
class ImageStream{
 public:
  ImageStream(model_config& _cfg, Dims _inputDims,
              const vector<string> _calibrationImages)
      : batchSize(_cfg.batchSize),
        calibrationImages(_calibrationImages),
        currentBatch(0),
        maxBatches(_calibrationImages.size() / _cfg.batchSize) {
    if (_inputDims.d[1] <= 4) {  // 1 or 3
      inputDims = {3, _inputDims.d[1], _inputDims.d[2], _inputDims.d[3]};
    } else {
      inputDims = {3, _inputDims.d[3], _inputDims.d[1], _inputDims.d[2]};
    }

    batch.resize(batchSize * inputDims.d[0] * inputDims.d[1] * inputDims.d[2]);
    is_use_mean_sub = _cfg.is_use_mean_sub;
    if (is_use_mean_sub)
    {
      mean = {(float)_cfg.mean[0], (float)_cfg.mean[1], (float)_cfg.mean[2]};
      std = {(float)_cfg.std[0], (float)_cfg.std[1], (float)_cfg.std[2]};
    }else
    {
      mean = {0.f, 0.f, 0.f};
      std = {1.f, 1.f, 1.f};
    }
  }

  int getBatchSize() const noexcept { return batchSize; }
  int getMaxBatches() const { return maxBatches; }
  float* getBatch() noexcept { return &batch[0]; }
  Dims getInputDims() { return inputDims; }
  bool next(){

    if (currentBatch == maxBatches) return false;

    for(int i=0;i<batchSize;i++)
    {
      auto image = imread(calibrationImages[batchSize * currentBatch + i].c_str(), IMREAD_COLOR);
      resize(image, image, Size(inputDims.d[2], inputDims.d[1]));
      cv::Mat pixels;
      image.convertTo(pixels, CV_32FC3, 1.0 / 255.0);
      
      vector<float> img;
      if (pixels.isContinuous())
        img.assign((float*)pixels.datastart, (float*)pixels.dataend);
      else
        return false;

      auto hw = inputDims.d[1] * inputDims.d[2];
      auto channels = inputDims.d[0];
      auto vol = channels * hw;

      for(int c = 0;c<channels;c++){
        for(int j = 0; j < hw; j++){
          batch[i * vol + c * hw + j] = (img[channels * j + 2 - c ] - mean[c]) / std[c];
        }
      }

    }
    currentBatch++;
    return true;
  }
  void reset() { 
    currentBatch = 0;
  }
  
private:
    int batchSize;
    vector<string> calibrationImages;
    int currentBatch;
    int maxBatches;
    Dims inputDims;
    bool is_use_mean_sub;
    vector<float> mean{0.485, 0.456, 0.406};
    vector<float> std{0.229, 0.224, 0.225};
    vector<float> batch;
};

class Int8EntropyCalibrator : public IInt8EntropyCalibrator2{
 public:
  Int8EntropyCalibrator(ImageStream& _stream, const string _networkName,
                        const string _calibrationCacheName,
                        bool _readCache = true)
      : stream(_stream),
        networkName(_networkName),
        calibrationCacheName(_calibrationCacheName),
        readCache(_readCache) {
    Dims d = stream.getInputDims();
    inputCount = stream.getBatchSize() * d.d[0] * d.d[1] * d.d[2];
    cudaMalloc(&deviceInput, inputCount * sizeof(float));
  }
  int getBatchSize() const noexcept override { return stream.getBatchSize(); }
  virtual ~Int8EntropyCalibrator() { cudaFree(deviceInput); }
  bool getBatch(void* bindings[] , const char * name[] , int nBindings) noexcept override{
    if (!stream.next()) return false;

    cudaMemcpy(deviceInput, stream.getBatch(), inputCount * sizeof(float) , cudaMemcpyHostToDevice);
    bindings[0] = deviceInput;
    return true;
  }
  static vector<String> getCalibrationFiles(string dir_path) {
    vector<String> files;
    if(dir_path.back()!='/') dir_path+="/";
    glob((dir_path + "*").c_str(), files, false);
    vector<string> calibration_files;
    for(int i=0;i<files.size();i++)
    {
      calibration_files.push_back(files[i]);
    }
    return calibration_files;
  }
  const void * readCalibrationCache(size_t & length) noexcept{
    calibrationCache.clear();
    std::ifstream input(calibrationTableName(), ios::binary);
    input >> noskipws;
    if(readCache && input.good())
      copy(istream_iterator<char>(input),istream_iterator<char>(),back_inserter(calibrationCache));

    length = calibrationCache.size();
    return length ? &calibrationCache[0] : nullptr;
  }

  void writeCalibrationCache(const void * cache , size_t length ) noexcept{
    std::ofstream output(calibrationTableName(), std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
  }

 private:
  std::string calibrationTableName() {
    // Use calibration cache if provided
    if (calibrationCacheName.length() > 0) return calibrationCacheName;

    assert(networkName.length() > 0);
    Dims d = stream.getInputDims();
    return std::string("Int8CalibrationTable_") + networkName +
           to_string(d.d[2]) + "x" + to_string(d.d[3]) + "_" +
           to_string(stream.getMaxBatches());
  }

  ImageStream stream;
  const string networkName;
  const string calibrationCacheName;
  bool readCache{true};
  size_t inputCount;
  void* deviceInput{nullptr};
  vector<char> calibrationCache;
  };

