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
        int batchSize{1};
        int dlaCore{-1};
        bool int8{false};
        bool fp16{false};                
  bool is_use_mean_sub {true};
  bool is_nchw;
  bool letterboxing{false};
  bool use_gpu_image_process{true};
	string modelFileName;
  bool anomalyEnable = false;;
  vector<int> order_of_feature_index_to_batch;
  vector<anomaly_var> anomaly_feature;

  vector<double> mean{0.485, 0.456, 0.406};
  vector<double> std{0.229, 0.224, 0.225};
	//for yolact
  int yolact_max_size;
  int yolact_min_size ;
  unsigned long long G_ib;


};

