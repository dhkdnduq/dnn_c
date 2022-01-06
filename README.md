# DeepLearning Edge for Windows
Don't use Python for performance.

## Build
**System Packages**
* [TensorRT](https://github.com/NVIDIA/TensorRT) >= v8.2 (for yolov5)
* [Opencv(cuda,dnn,..)](https://github.com/opencv/opencv) >= v4.5
* c++ >= v17
* [libtorch](https://pytorch.org/) >= v1.7.1
* [torchvision](https://github.com/pytorch/vision/releases) 
* [cuda](https://developer.nvidia.com/cuda-toolkit) >= v11.2 

## Distribution
* dl.h 
* structure.h
* *.dll

## Setting
* include (dl.h  , structure.h)
* link *.dll 
* If you need a large stack vs project->linker->system->stack reserve size: >=10485760  & structure.h change  MAX_BATCH_SIZE 10-> x 


## Usage 
* main_yolov5.cpp (https://github.com/ultralytics/yolov5)
* main_effdet.cpp(https://github.com/rwightman/efficientdet-pytorch)
* main_trt_classification.cpp(https://github.com/NVIDIA/TensorRT)
* main_yolact.cpp(https://github.com/dbolya/yolact)
* main_anomaly.cpp(https://github.com/dhkdnduq/PatchCore_anomaly_detection) 


