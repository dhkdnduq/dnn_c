#include <chrono>
#include <experimental/filesystem>
#include <iostream>

#include "dl.h"
#include "opencv2/highgui.hpp"
using namespace std;
namespace fs = std::experimental::filesystem;

int main()
{
  fs::path cwd = fs::current_path();
  cout << "current dir : " << cwd.string() << endl;
  bool bload = torch_init("dnn_setting_anomaly_patchcore_bmw.json");
  string dir_path = "anomaly";
  string save_dir_path = "bmw\\";
  int batch_size = 3;
 
  for(int i=0;i<3;i++)
  {
    int idx = 0;
    for (auto& p : fs::directory_iterator(dir_path)) {
      fs::path path = p;
      string filename = path.filename().string();
      string filepath = path.u8string();
      torch_add_image_file(filepath.c_str());
      if (++idx >= 3) break;
    }
    segm_t_container_rst_list segm_container;
    torch_anomaly_detection(segm_container);
    
    int batch_index = 0;
    for (int batch_index = 0; batch_index < batch_size; batch_index++) {
      // show result
      segm_t_container& segm = segm_container[batch_index];
      std::vector<unsigned char> dispdata(
          segm.display_image.data,
          segm.display_image.data + segm.display_image.size);
      cv::Mat dispmat = cv::imdecode(dispdata, -1);
      cv::imshow("detection", dispmat);
      cv::waitKey(-1);
    }
    // cout << binary_container[batch_index].candidates[0].prob << ","<<
    // (float)binary_container[batch_index].candidates[0].obj_id<<endl;
    // cv::imdecode(dispdata, -1); cv::imwrite(save_dir_path + filename,
    // dispmat);

  }
  


  torch_dispose();
  getchar();
	return 0;

}
