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
  ImageStream(int batchSize, Dims inputDims,
              const vector<string> calibrationImages)
      : _batchSize(batchSize),
        _calibrationImages(calibrationImages),
        _currentBatch(0),
        _maxBatches(_calibrationImages.size() / _batchSize){
    if (inputDims.d[1] <= 4) {  // 1 or 3
      _inputDims = { 3,inputDims.d[1], inputDims.d[2], inputDims.d[3]};
    } else {
      _inputDims = { 3,inputDims.d[3], inputDims.d[1], inputDims.d[2]};
    }

    _batch.resize(batchSize * _inputDims.d[0] * _inputDims.d[1] *  _inputDims.d[2]);
    
  }

  int getBatchSize() const noexcept { return _batchSize; }
  int getMaxBatches() const { return _maxBatches; }
  float* getBatch() noexcept { return &_batch[0]; }
  Dims getInputDims() { return _inputDims; }
  bool next(){

    if (_currentBatch == _maxBatches) return false;

    for(int i=0;i<_batchSize;i++)
    {
      auto image = imread(_calibrationImages[_batchSize * _currentBatch + i].c_str(), IMREAD_COLOR);
      resize(image, image, Size(_inputDims.d[2], _inputDims.d[1]));
      cv::Mat pixels;
      image.convertTo(pixels, CV_32FC3, 1.0 / 255.0);
      
      vector<float> img;
      if (pixels.isContinuous())
        img.assign((float*)pixels.datastart, (float*)pixels.dataend);
      else
        return false;

      auto hw = _inputDims.d[1] * _inputDims.d[2];
      auto channels = _inputDims.d[0];
      auto vol = channels * hw;

      for(int c = 0;c<channels;c++){
        for(int j = 0; j < hw; j++){
          _batch[i * vol + c * hw + j] = (img[channels * j + 2 - c ] - _mean[c]) / _std[c];
        }
      }

    }
    _currentBatch++;
    return true;
  }
  void reset() { 
    _currentBatch = 0;
  }
  
private:
    int _batchSize;
    vector<string> _calibrationImages;
    int _currentBatch;
    int _maxBatches;
    Dims _inputDims;

    vector<float> _mean{0.485, 0.456, 0.406};
    vector<float> _std{0.229, 0.224, 0.225};
    vector<float> _batch;
};

class Int8EntropyCalibrator : public IInt8EntropyCalibrator2{
 public:
  Int8EntropyCalibrator(ImageStream& stream, const string networkName,
                        const string calibrationCacheName,
                        bool readCache = true)
      : _stream(stream),
        _networkName(networkName),
        _calibrationCacheName(calibrationCacheName),
        _readCache(readCache){
    Dims d = _stream.getInputDims();
    _inputCount = _stream.getBatchSize() * d.d[0] * d.d[1] * d.d[2];
    cudaMalloc(&_deviceInput, _inputCount * sizeof(float));
  }
  int getBatchSize() const noexcept override { return _stream.getBatchSize(); }
  virtual ~Int8EntropyCalibrator() { cudaFree(_deviceInput); }
  bool getBatch(void* bindings[] , const char * name[] , int nBindings) noexcept override{
    if (!_stream.next()) return false;

    cudaMemcpy(_deviceInput, _stream.getBatch(), _inputCount * sizeof(float) , cudaMemcpyHostToDevice);
    bindings[0] = _deviceInput;
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
    _calibrationCache.clear();
    std::ifstream input(calibrationTableName(), ios::binary);
    input >> noskipws;
    if(_readCache && input.good())
      copy(istream_iterator<char>(input),istream_iterator<char>(),back_inserter(_calibrationCache));

    length = _calibrationCache.size();
    return length ? &_calibrationCache[0] : nullptr;
  }

  void writeCalibrationCache(const void * cache , size_t length ) noexcept{
    std::ofstream output(calibrationTableName(), std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
  }

 private:
  std::string calibrationTableName() {
    // Use calibration cache if provided
    if (_calibrationCacheName.length() > 0) return _calibrationCacheName;

    assert(_networkName.length() > 0);
    Dims d = _stream.getInputDims();
    return std::string("Int8CalibrationTable_") + _networkName +
           to_string(d.d[2]) + "x" + to_string(d.d[3]) + "_" +
           to_string(_stream.getMaxBatches());
  }

  ImageStream _stream;
  const string _networkName;
  const string _calibrationCacheName;
  bool _readCache{true};
  size_t _inputCount;
  void* _deviceInput{nullptr};
  vector<char> _calibrationCache;
  };

