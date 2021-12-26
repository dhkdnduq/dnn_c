
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

int main() {
  fs::path cwd = fs::current_path();
  cout << "current dir : " << cwd.string() << endl;
  bool bload = trt_init("dnn_setting_yolov5.json");
  int img_index = 0; 
  int batch_size = 1;
  vector<string> vpath;
  string dir_path = "D:\\deeplearning\\yolov5-master\\yolov5-master\\data\\images";
  for (auto& p : fs::directory_iterator(dir_path)) {
    fs::path path = p;
    string filepath = path.u8string();
    bbox_t_container_rst_list bbox_container;
    trt_add_image_file(path.string().c_str());
    trt_yolov5(bbox_container, 0);
    auto mat = cv::imread(path.string().c_str(), IMREAD_UNCHANGED);
    
    
    bbox_t_container bbox = bbox_container[0];
    std::map<int, cv::Scalar> colormap;
    RNG rng(12345);
    for (int i = 0; i < bbox.cnt; i++) {
      bbox_t t = bbox.candidates[i];
      cv::Rect rt;
      rt.x = t.x;
      rt.y = t.y;
      rt.width = t.w;
      rt.height = t.h;
      if (colormap.find(t.obj_id) == colormap.end()) {
        Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
                              rng.uniform(0, 255));
        colormap.insert(make_pair(t.obj_id, color));
      };

      auto color = colormap[t.obj_id];
      cv::rectangle(mat, rt, color, 5);
    }
    continue;
    if (bbox.cnt > 0) {
      cv::resize(mat, mat, {446, 446});
      cv::imshow("temp", mat);
      cv::waitKey(0);
      cv::destroyAllWindows();
    }
    
  }
  
   
 
  trt_dispose();
  getchar();
  return 0;
}