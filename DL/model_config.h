#pragma once
#include "structure.h"

class  model_config {
public:
	model_config();
	bool load_config(string configpath);
public:

	int dnn_width;
	int dnn_height;
	int dnn_chnnel;
	float dnn_scale_div;
	float threshold = 0.5f;
  float iou_threshold = 0.5;
  int batchSize{ 1 };                  //!< Number of inputs in a batch
	int dlaCore{ -1 };                   //!< Specify the DLA core to run network on.
	bool int8{ false };                  //!< Allow runnning the network in Int8 mode.
	bool fp16{ false };                  //!< Allow running the network in FP16 mode.
  bool mean_sub_enable {true};
  bool is_nchw;//(tensorflow model == false) (pytorch model == true) 
	std::vector<std::string> dataDirs; //!< Directory paths where sample data files are stored
	string modelFileName;
	bool anomalyEnable = false;;
  vector<anomaly_var> vanomaly;
  std::vector<double> mean = {0.5, 0.5, 0.5};
  std::vector<double> std = {0.5, 0.5 ,0.5};

	vector<double> tokenize_d(const string& data, const char delimiter);
  vector<string> tokenize_s(const string& data, const char delimiter);
	//for yolact

  int yolact_max_size;
  int yolact_min_size ;

};

