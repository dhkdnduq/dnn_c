
//#include "TensorRT.h"
//#include "Onnx.h"
#include <chrono>
#include <experimental/filesystem>
#include <iostream>

#include "dl.h"
#include "opencv2/highgui.hpp"
using namespace std;
namespace fs = std::experimental::filesystem;

int main_ano()
{
  fs::path cwd = fs::current_path();
  cout << "current dir : " << cwd.string() << endl;
  bool bload = torch_init("dnn_setting_anomaly_patchcore_bmw.json");
  string dir_path = "D:\\deeplearning\\data\\mvtec_anomaly_detection\\bmw\\test\\ng\\";
  string save_dir_path = "bmw\\";

  for (auto& p : fs::directory_iterator(dir_path)) {
    
      fs::path path = p;
      string filename = path.filename().string();
      string filepath = path.u8string();
      segm_t_container_rst_list binary_container;
      torch_add_image_file(filepath.c_str());
      torch_anomaly_detection(binary_container,0);
      int batch_index = 0;
      cout << binary_container[batch_index].candidates[0].prob << ","<< (float)binary_container[batch_index].candidates[0].obj_id<<endl;
      //segm_t_container& segm = binary_container[batch_index];
      //std::vector<unsigned char> dispdata(binary_container[batch_index].display_image.data, segm.display_image.data + segm.display_image.size);
      //cv::Mat dispmat = cv::imdecode(dispdata, -1); 
      //cv::imwrite(save_dir_path + filename, dispmat);

  }
  torch_dispose();
  getchar();
	return 0;

}