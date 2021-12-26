#include "pch.h"
#include "model_config.h"
#include "json/json.h"

 model_config::model_config()
{
	dnn_width = 224;
  dnn_height = 224;
	dnn_chnnel = 3;
	dnn_scale_div = 255.;
	threshold = 0.5f;
}

 vector<string> model_config::tokenize_s(const string& data,  const char delimiter) {
  vector<string> result;
  string token;
  stringstream ss(data);
  while (getline(ss, token, delimiter)) {
    result.push_back(token);
  }
  return result;
 }
vector<double> model_config::tokenize_d(const string& data, const char delimiter) {
  vector<double> result;
  string token;
  stringstream ss(data);
  while (getline(ss, token, delimiter))
  {
    result.push_back(std::stof(token));
  }
  return result;
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
      this->mean_sub_enable = jcommon.get("mean_sub_enable", "false").asString() == "true" ?  true : false;
      string s_mean = jcommon.get("mean", "0.5,0.5,0.5").asString();
      this->mean = tokenize_d(s_mean,',');
      string s_std = jcommon.get("std", "0.5,0.5,0.5").asString();
      this->std = tokenize_d(s_std, ',');
    }

    Json::Value& jyolact = root["yolact"];
    if (jyolact) {
      this->fp16 =   jyolact.get("fp16", "false").asString() == "true" ? true : false;
      this->yolact_max_size = std::stoi(jyolact.get("max_size", "550").asString());
      this->yolact_min_size = std::stoi(jyolact.get("min_size", "200").asString());
     
    }
   
    Json::Value& janomaly = root["anomaly"];
    vanomaly.clear();
    if (janomaly) {
      this->anomalyEnable = janomaly.get("enable", "false").asString() == "true" ?  true : false;
      for(auto featit :janomaly["features"]) {
        anomaly_var anomaly_temp;
        anomaly_temp.anomalyFeatureFileName =  featit.get("feature", "").asString();
        anomaly_temp.anomalyMaxScore = std::stof(featit.get("max_score", "32").asString().c_str());
        anomaly_temp.anomalyMinScore = std::stof(featit.get("min_score", "7").asString().c_str());
        anomaly_temp.anomalyThreshold =  std::stof(featit.get("threshold", "20").asString().c_str());
        auto jdef_threshold = featit["defect_extraction"];
        if (jdef_threshold) {
          anomaly_temp.defect_extraction_enable =  jdef_threshold.get("enable", "false").asString()  == "true" ? true : false;
          anomaly_temp.defect_extraction_threshold = std::stof(jdef_threshold.get("threshold", "0.8").asString().c_str()); 
          anomaly_temp.defect_extraction_jud_area_ratio = std::stof(jdef_threshold.get("judge_area_ratio", "8").asString().c_str());
        }
        vanomaly.push_back(anomaly_temp);
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
