
//#include "TensorRT.h"
//#include "Onnx.h"
#include <chrono>
#include <experimental/filesystem>
#include <iostream>
#include <opencv2/core.hpp>

#include "dl.h"
#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;
namespace fs = std::experimental::filesystem;



int main__yolact() {
  fs::path cwd = fs::current_path();
  cout << "current dir : " << cwd.string() << endl;
  bool bload = trt_init("dnn_setting_yolact.json");
  int nbatchsize = 1;
  string dir_path = "D:\\deeplearning\\image-to-coco-json-converter-master\\dataset\\val";
  for (auto& p : fs::directory_iterator(dir_path)) {
    fs::path path = p;
    string filepath = path.u8string();
      
    for (int i = 0; i < nbatchsize; i++) {
      trt_add_image_file(filepath.c_str());
    }
    if (bload) {
      segm_t_container_rst_list bbox_container;
      int batch_size = trt_yolact(bbox_container);
      // continue;
      for (int i = 0; i < batch_size; i++) {
        segm_t_container& segm = bbox_container[i];
        if (segm.cnt == 0) continue;

        //show result
        /*
        std::vector<unsigned char> dispdata(segm.display_image.data,segm.display_image.data + segm.display_image.size); 
        cv::Mat dispmat = imdecode(dispdata,-1); 
        cv::imshow("detection", dispmat); 
        cv::waitKey(-1);

        std::vector<unsigned char> maskdata(segm.mask_image.data, segm.mask_image.data + segm.mask_image.size); 
        cv::Mat maskmat = imdecode(maskdata,-1); 
        cv::imshow("mask", maskmat);
        cv::waitKey(-1);

        */
      }
     
    }
  }
  trt_dispose();
  getchar();
  return 0;
}