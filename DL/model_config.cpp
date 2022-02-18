#include "pch.h"
#include "model_config.h"
#include "json/json.h"


template <typename T>
vector<T> tokenize(const string& data, const char delimiter) {
  vector<T> result;
  std::string token;
  std::stringstream ss(data);
  while (getline(ss, token, delimiter)) {
    if (std::is_same_v<T, float>)
      result.push_back(std::stof(token));
    else if (std::is_same_v<T, double>)
      result.push_back(std::stod(token));
    else if (std::is_same_v<T, int>)
      result.push_back(std::stoi(token));
  }
  return result;
}

 model_config::model_config()
{
	dnn_width = 224;
  dnn_height = 224;
	dnn_chnnel = 3;
	dnn_scale_div = 255.;
	threshold = 0.5f;
}

bool model_config::load_config(string configpath)
{
	std::ifstream ifs;
	ifs.open(configpath, ios_base::in);
	if (!ifs.is_open())
		return false;

	stringstream ss;
	string errors;
	ss << ifs.rdbuf();
  
	
  Json::CharReaderBuilder builder;
	std::unique_ptr<Json::CharReader> const reader(builder.newCharReader());

	Json::Value root;
  if (reader->parse(ss.str().c_str(),
                    ss.str().c_str() + ss.str().length(), &root,
                    &errors)) {

		Json::Value& jcommon = root["common"];
    if (jcommon) {
      this->dnn_width = std::stoi(jcommon.get("width", "100").asString().c_str());
      this->dnn_height =std::stoi(jcommon.get("height", "100").asString().c_str());
      this->dnn_scale_div =std::stof(jcommon.get("scale_div", "1").asString().c_str());
      this->dnn_chnnel = std::stoi(jcommon.get("chn", "3").asString().c_str());
      this->modelFileName = jcommon["model_path"].asString();
      this->threshold =std::stof(jcommon.get("threshold", "0.5").asString().c_str());
      this->iou_threshold =std::stof(jcommon.get("iou_threshold", "0.5").asString().c_str());
      this->batchSize = std::stoi(jcommon.get("batchsize", "1").asString().c_str());
      this->is_use_mean_sub = jcommon.get("mean_sub_enable", "false").asString() == "true" ?  true : false;
      string s_mean = jcommon.get("mean", "0.5,0.5,0.5").asString();
      this->mean = tokenize<double>(s_mean,',');
      string s_std = jcommon.get("std", "0.5,0.5,0.5").asString();
      this->std = tokenize<double>(s_std, ',');
    }

    Json::Value& jyolact = root["yolact"];
    if (jyolact) {
      this->fp16 =   jyolact.get("fp16", "false").asString() == "true" ? true : false;
      this->yolact_max_size = std::stoi(jyolact.get("max_size", "550").asString());
      this->yolact_min_size = std::stoi(jyolact.get("min_size", "200").asString());
     
    }
   
    Json::Value& janomaly = root["anomaly"];
    anomaly_feature.clear();
    order_of_feature_index_to_batch.clear();
    if (janomaly) {
      this->anomalyEnable = janomaly.get("enable", "false").asString() == "true" ?  true : false;
      string s_order_index = janomaly.get("order_of_feature_index_to_batch", "0").asString();
      this->order_of_feature_index_to_batch = tokenize<int>(s_order_index, ',');
      for(auto featit :janomaly["features"]) {
      
        anomaly_var anomaly_temp;
        anomaly_temp.anomalyFeatureFileName =
            featit.get("feature", "").asString();
        anomaly_temp.anomalyMaxScore =
            std::stof(featit.get("max_score", "32").asString().c_str());
        anomaly_temp.anomalyMinScore =
            std::stof(featit.get("min_score", "7").asString().c_str());
        anomaly_temp.anomalyThreshold =
            std::stof(featit.get("threshold", "20").asString().c_str());
        anomaly_temp.batch_idx =
            std::stoi(featit.get("index", "0").asString().c_str());
        auto jdef_threshold = featit["defect_extraction"];
        if (jdef_threshold) {
          anomaly_temp.defect_extraction_enable =
              jdef_threshold.get("enable", "false").asString() == "true"    ? true   : false;
          anomaly_temp.defect_extraction_threshold = std::stof(
              jdef_threshold.get("threshold", "0.8").asString().c_str());
          anomaly_temp.defect_extraction_jud_area_ratio = std::stof(
              jdef_threshold.get("judge_area_ratio", "8").asString().c_str());
        }
        anomaly_feature.push_back(anomaly_temp);

        
        
      }
    }
   
    ifs.close();
  }

	else
	{
		return false;
	}
	return true;
}
