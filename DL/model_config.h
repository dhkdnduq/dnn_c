#pragma once
#include "structure.h"

class  model_config {
public:
	model_config();
	bool load_config(string configpath);
	vector<double> tokenize_d(const string& data, const char delimiter);
  vector<string> tokenize_s(const string& data, const char delimiter);

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
  bool is_use_mean_sub {true};
  bool is_nchw;
	string modelFileName;
  bool anomalyEnable = false;;
  vector<anomaly_var> vanomaly;

  vector<double> mean{0.485, 0.456, 0.406};
  vector<double> std{0.229, 0.224, 0.225};
	//for yolact
  int yolact_max_size;
  int yolact_min_size ;

};

