

#include <chrono>
#include <experimental/filesystem>
#include <iostream>
#include "dl.h"
#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;
namespace fs = std::experimental::filesystem;

int main_edet() {
  fs::path cwd = fs::current_path();
  cout << "current dir : " << cwd.string() << endl;
  bool bload = torch_init("dnn_setting_effdet.json");
 
  torch_add_image_file("eff1.jpg");
  torch_add_image_file("eff2.jpg");
  torch_add_image_file("eff1.jpg");
  torch_add_image_file("eff2.jpg");

  if (bload) {
    bbox_t_container_rst_list bbox_container;
    //while (true) {
      int batch_size = torch_effdet(bbox_container);

      for (int batch_index=0;batch_index<batch_size;batch_index++) {
        auto mat = cv::imread("eff1.jpg",IMREAD_UNCHANGED);

        bbox_t_container bbox = bbox_container[batch_index];
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

        if (bbox.cnt > 0) {
          cv::resize(mat, mat, {446, 446});
          cv::imshow("temp", mat);
          cv::waitKey(0);
          cv::destroyAllWindows();
        }

      }
    //}
  } else {
    ;
  }
  getchar();
 
  return 0;
}