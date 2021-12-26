/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//!
//! sampleOnnxMNIST.cpp
//! This file contains the implementation of the ONNX MNIST sample. It creates the network using
//! the MNIST onnx model.
//! It can be run with the following command line:
//! Command: ./sample_onnx_mnist [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//! [--useDLACore=<int>]
//!

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <chrono>
#include <filesystem>
#include "model_config.h"
using namespace sample;

 
//!
//! \details It creates the network using an ONNX model
//!
//! 
//! 

class TRTOnnx
{
    template <typename T>
    using TRTUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    TRTOnnx(const model_config& cfg)
        : cfg_(cfg)
        , mEngine(nullptr)
    {
   	namespace fs = std::filesystem;
        mEngineName = fs::path(cfg.modelFileName).replace_extension(".trt").string();
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool inferYolact(cv::Mat& frame,int& result);
    bool inferClassification(cv::Mat& frame, int& result);


private:
  

    model_config cfg_;

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    vector<nvinfer1::Dims> mOutputDims; //!< The dimensions of the output to the network.
    string mEngineName;
    shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
    shared_ptr<samplesCommon::BufferManager> buffers;
    TRTUniquePtr<nvinfer1::IExecutionContext> context;
    //TRTUniquePtr<nvinfer1::IExecutionContext> context{ nullptr };
    //!
    //! \brief Parses an ONNX model for CNN and creates a TensorRT network
    //!
    bool constructNetwork(TRTUniquePtr<nvinfer1::IBuilder>& builder,
        TRTUniquePtr<nvinfer1::INetworkDefinition>& network, TRTUniquePtr<nvinfer1::IBuilderConfig>& config,
        TRTUniquePtr<nvonnxparser::IParser>& parser);
 
    void preprocessImage(cv::Mat frame);
    void preprocessYolactImage(cv::Mat frame);
 
    //!
    //! \brief Classifies digits and verify result
    //!
   
    int verifyClassification(const samplesCommon::BufferManager& buffers);
    int verifyYolact(cv::Mat& frame,const samplesCommon::BufferManager& buffers);
   
    bool saveEngine(const std::string& fileName);

    ICudaEngine* loadEngine(const std::string& fileName,int DLACore);

};
