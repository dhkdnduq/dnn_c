
#include <experimental/filesystem>
#include <iostream>
#include "dl.h"
using namespace std;
namespace fs = std::experimental::filesystem;

int main(){
  fs::path cwd = fs::current_path();
  cout << "current dir : " << cwd.string() << endl;
  bool bload = trt_init("dnn_setting_trt_classification_1.json",0);
  int batch_size = 3;
  string dir_path =   "test\\";
  vector<string> vpath;
  int img_index = 0;
  for (auto& p : fs::directory_iterator(dir_path)) {
    if (img_index >= batch_size) break;
    img_index++;
    fs::path path = p;
    string filepath = path.u8string();
    vpath.push_back(filepath);
    trt_add_image_file(filepath.c_str(),0);
  }
  if (bload) {
    category_rst_list rst_list;
    int batch_size = trt_category_classification(rst_list);
    for (int i = 0; i < img_index; i++) {
      int rst = rst_list[i];
      cout << vpath[i] << ", class :" << (rst == 1 ? "ng" : "ok") << endl;
    }
      
  } 
  trt_dispose();
  return 0;
}
