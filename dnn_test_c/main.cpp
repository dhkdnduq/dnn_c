

#include <chrono>
#include <experimental/filesystem>
#include <iostream>
#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include "dl.h"
using namespace cv;
using namespace cv::cuda;
using namespace std;
namespace fs = std::experimental::filesystem;

int main___()
{
	try
	{
    /*
		cout << getBuildInformation() << endl;;
		printShortCudaDeviceInfo(getDevice());
  
	  int cuda_devices_number = getCudaEnabledDeviceCount();
	  cout << "CUDA Device(s) Number: " << cuda_devices_number << endl;
	  DeviceInfo _deviceInfo;
	  bool _isd_evice_compatible = _deviceInfo.isCompatible();
	  cout << "CUDA Device(s) Compatible: " << _isd_evice_compatible << endl;
	
	  fs::path cwd = fs::current_path();
	  cout << "current dir_: " << cwd.string() << endl;;
	  */
	}
	catch (...)
	{

	}
  
  fs::path cwd = fs::current_path();
  cout << "current dir_: " << cwd.string() << endl;;
  string js = "dnn_setting.json";


  //  DL::Onnx trdl;
//  DL::TensorRT dl_;
  /*
  dl_.load_model(js);
  Mat mat2 = imread("1.jpg", IMREAD_COLOR);
 
   while (true) {
     dl_.begin_period();
     dl_.predict_category_classification(mat2.data, mat2.cols, mat2.rows,
                                       mat2.channels());
     dl_.end_period();
  }
  
  */
  /*
  DL::PyTorch dl_;
  Mat mat2 = imread("1.png", IMREAD_COLOR);
  while (true) {
    if (dl_.load_model(js)) {
      dl_.beginPeriod();
      cout << dl_.predict_anomaly_detection(mat2.data, mat2.cols, mat2.rows,
                                            mat2.channels())
           << "              ";
    }
    dl_.endPeriod();
  }
 */

  /*
  //  DL::Onnx trdl;
   DL::TensorRT trdl;
   trdl.load_model(js);
   Mat mat2 = imread("D:\\deeplearning\\yolact\\weights\\dog.jpg",IMREAD_COLOR);
   while (true) {
    trdl.begin_period();
    trdl.detectYolact(mat2.data, mat2.cols, mat2.rows, mat2.channels());
    trdl.end_period();
  }
  */
  //return 0;
  
  //init_torch(js.c_str());
  //while (true) {
  /*
  if (dl_.load_model(js))
	{
			cout << "load model" << endl;
     //string dir_path = "D:\\deeplearning\\data\\pim\\aug_image\\"; 
     //string dir_path ="D:\\deeplearning\\data\\mvtec_anomaly_detection\\bottle\\test\\good\\"; 
     //string save_dir_path = "D:\\deeplearning\\data\\pim\\rst\\";
     //string dir_path =  "D:\\deeplearning\\data\\magnet\\crop_ng";
			
      string dir_path =  "test\\";
      
      std::map<int,cv::Scalar> colormap;
      RNG rng(12345);
      int i = 0;           
       //while (true) 
       {
   
      for (auto& p : fs::directory_iterator(dir_path))
			{
             i++;
              fs::path path = p;
              string filepath = path.u8string();
              cout << filepath << ",";
              Mat mat2 = imread(filepath, IMREAD_COLOR);
              if (mat2.empty())
								continue;;
              dl_.start_timer();
              int predict  =dl_.predict_category_classification(mat2.data, mat2.cols,
                                                  mat2.rows, mat2.channels());
             
              dl_.end_timer();
             /*
              #pragma region det
              
              dl_.beginPeriod();
              bbox_t_container bbox;
             // cout << path.filename() << ",";
              int cnt = dl_.predict_object_detection_efficientdet(
                  mat2.data, mat2.cols, mat2.rows, mat2.channels(), bbox);
              cout << cnt<<" , ";
              dl_.endPeriod();
              
              //continue;
              for (int i = 0; i < cnt; i++) {
                bbox_t t = bbox.candidates[i];
                cv::Rect rt;
                rt.x = t.x;
                rt.y = t.y;
                rt.width = t.w;
                rt.height = t.h;
                if (colormap.find(t.obj_id) == colormap.end()) {
                  Scalar color =  Scalar(rng.uniform(0, 255), rng.uniform(0, 255),rng.uniform(0, 255));
                  colormap.insert(make_pair(t.obj_id, color));
                };
                //cout << rt.x << "," << rt.y << "," << rt.width<<","<<rt.height<<","<<t.obj_id<<endl;
                
                auto color = colormap[t.obj_id];
                cv::rectangle(mat2, rt,color,5);
              }
              if (cnt > 0) 
                cv::imwrite(save_dir_path + path.filename().u8string(), mat2); 
              continue;
              if (cnt > 0) {
                cv::resize(mat2, mat2, {446, 446});
                cv::imshow("temp", mat2);
                cv::waitKey(0);
                cv::destroyAllWindows();
                                                        
							}
              #pragma endregion 
              

              #pragma region net
              
              
              dl_.beginPeriod();
              cout<<dl_.predict_category_classification( mat2.data, mat2.cols, mat2.rows, mat2.channels())<<"              ";
              dl_.endPeriod();
              
              #pragma endregion

              #pragma region anomaly
      
              dl_.beginPeriod();
              cout << dl_.predict_anomaly_detection(mat2.data, mat2.cols, mat2.rows, mat2.channels())<< "              ";
              dl_.endPeriod();
              
              #pragma endregion
      }
    
	   }
		//dl_.predict_category_classification();
		
	}
	else
	{
		cout << "model load failed, current dir : "<<cwd.string() << endl;;
	}
  //}
        
	getchar();
  */
	return 0;
  
}


