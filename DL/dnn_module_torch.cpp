#include "pch.h"
#include "dnn_module_torch.h"
#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"
#include <cuda.h>
#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <random>
#include <algorithm>
#include <nms.h>
#include <torchvision/csrc/cuda/vision_cuda.h>

using Eigen::MatrixXd;
bool dnn_module_torch::isinit_ = false;
namespace F = torch::nn::functional;
using namespace torch::nn;
using namespace std;
c10::DeviceType dnn_module_torch::default_dev = at::kCUDA;
vector<torch::Tensor> dnn_module_torch::mask_colors;
dnn_module_torch::dnn_module_torch() {  }

dnn_module_torch::~dnn_module_torch() 
{
 
}
void dnn_module_torch::load_anomaly_detection_patchcore() {
  torch::load(module_wideresnet_50_, cfg_.modelFileName);
  module_wideresnet_50_->eval();
  module_wideresnet_50_->to(at::kCUDA);

  for (int i = 0; i < cfg_.anomaly_feature.size(); i++) {
    auto anomaly_features = torch::jit::load(cfg_.anomaly_feature[i].anomalyFeatureFileName);
    anomaly_feature_patchcore.push_back(anomaly_features.attr("feature").toTensor().to(at::kCUDA));
    cfg_.anomaly_feature[i].batch_idx = i;
  }
}

 void dnn_module_torch::loadlibrary() {
  if (!isinit_) {

    HANDLE h = LoadLibrary(L"torchvision.dll");
    if (h == NULL) {
      cout << "failed\n";
      return;
    }
    auto& ops = torch::jit::getAllOperators();
    std::cout << "torch jit operators\n";
    for (auto& op : ops) {
      auto& name = op->schema().name();
      if (name.find("torchvision") != std::string::npos)
        std::cout << "op : " << op->schema().name() << "\n";
    }
    std::cout << "\n";
   
    isinit_ = true;
  }
}


cv::Mat dnn_module_torch::tensor1dToMat(torch::Tensor t) {
  int H = t.size(0);
  at::ScalarType type = t.scalar_type();
  cv::Mat mat;
  if (type == at::ScalarType::Float) {
    mat = cv::Mat(H, 1, CV_32F);
    std::memcpy(mat.data, t.data_ptr(), sizeof(float) * t.numel());
  }

  else if (type == at::ScalarType::Char) {
    mat = cv::Mat(H, 1, CV_8UC1);
    std::memcpy(mat.data, t.data_ptr(), sizeof(char) * t.numel());
  }
  return mat;
}

cv::Mat dnn_module_torch::tensor2dToMat(torch::Tensor t) {
  int H = t.size(0);
  int W = t.size(1);
  int C = 1;
  
  if (t.sizes().size() > 2)
   C = t.size(2);
  at::ScalarType type = t.scalar_type();
  cv::Mat mat;
  if (type == at::ScalarType::Float) {
    mat = cv::Mat(H, W, C == 1 ?  CV_32FC1 : CV_32FC3);
    std::memcpy(mat.data, t.data_ptr(), sizeof(float) * t.numel());
  } 
 
  else if (type == at::ScalarType::Char) {
    mat = cv::Mat(H, W, C == 1 ? CV_8UC1 : CV_8UC3);
    std::memcpy(mat.data, t.data_ptr(), sizeof(char) * t.numel());
  } else if (type == at::ScalarType::Bool) {
    mat = cv::Mat(H, W, C == 1 ? CV_8UC1 : CV_8UC3);
    std::memcpy(mat.data, t.data_ptr(), sizeof(char) * t.numel());
  } 

  return mat;
}


vector<cv::Mat> dnn_module_torch::tensor3dToMat(torch::Tensor t) {
  
  vector<cv::Mat> temp; 
  int W = t.size(0);
  int H = t.size(1);
  int C = t.size(2);
  
  for (int i = 0; i < C; i++) {
    auto tensor2d =  t.index({Slice(None, None), Slice(None, None), i});
    temp.push_back(tensor2dToMat(tensor2d));
  }
  return temp;
}

void dnn_module_torch::tensor2dToImageInfo(torch::Tensor t , image_info& imageinfo) {
 
  int H = t.size(0);
  int W = t.size(1);
  int C = 1;
  if (t.sizes().size() > 2) C = t.size(2);
  at::ScalarType type = t.scalar_type();
  cv::Mat mat;
  if (type == at::ScalarType::Float) {
    mat = cv::Mat(H, W, C == 1 ? CV_32FC1 : CV_32FC3);
    std::memcpy(mat.data, t.data_ptr(), sizeof(float) * t.numel());
  }
  else if (type == at::ScalarType::Char) {
    mat = cv::Mat(H, W, C == 1 ? CV_8UC1 : CV_8UC3);
    std::memcpy(mat.data, t.data_ptr(), sizeof(char) * t.numel());
  } else if (type == at::ScalarType::Bool) {
    mat = cv::Mat(H, W, C == 1 ? CV_8UC1 : CV_8UC3);
    std::memcpy(mat.data, t.data_ptr(), sizeof(char) * t.numel());
  }
  matToImageinfo(mat, imageinfo);

}
  at::Tensor dnn_module_torch::embedding_concat(torch::Tensor x, torch::Tensor y) {

  int64 B1 = x.size(0), C1 = x.size(1), H1 = x.size(2), W1 = x.size(3);
  int64 B2 = y.size(0), C2 = y.size(1), H2 = y.size(2), W2 = y.size(3);
  int64 s = H1 / H2;

  x = F::unfold(x, F::UnfoldFuncOptions(s).dilation(1).stride(s));
  x = x.view({B1, C1, -1, H2, W2});
  auto z = torch::zeros({B1, C1 + C2, x.size(2), H2, W2});
  for (int i = 0; i < x.size(2); i++) {
   // z.index({Slice(None, None, i,None,None)}) = torch::cat(x.index({Slice(None, None, i),y}), 1);
    auto temp =  x.index({Slice(None, None), Slice(None, None), i, Slice(None, None),Slice(None, None)});
    z.index({Slice(None, None), Slice(None, None), i, Slice(None, None),Slice(None, None)}) = torch::cat( {temp,y},1);
  }

  z = z.view({B1, -1, H2 * W2});
  z = F::fold(z, F::FoldFuncOptions({H1, W1}, {s, s}).stride(s)).to(at::kCUDA);

  //cout << z.sizes() << endl;
  
  return z;
 }

 void dnn_module_torch::gen_colors() {

   if (mask_colors.size() == 0) {
     for (int i = 0; i < MAX_OBJECTS; i++) {
       auto rand_color =
           torch::randint(0, 255, 3).toType(c10::ScalarType::Long).to(at::kCPU);
       mask_colors.push_back(
           (rand_color.to(at::ScalarType::Float) / 255.).view({1, 1, 1, 3}));
     }
   }
 }

 bool dnn_module_torch::load_model(string configpath) 
{ 
  if (!cfg_.load_config(configpath)) {
    return false;
  }

  loadlibrary();
  cfg_.is_nchw = true;
  try
  {
    if (cfg_.anomalyEnable) {
      load_anomaly_detection_patchcore();
  
    } else {
      if (!file_exists(cfg_.modelFileName, true))
        return false;
       module_ = torch::jit::load(cfg_.modelFileName);
       module_.to(at::kCUDA);
     
    }
  }
  catch (const c10::Error& e) {
    std::cout << e.msg() << endl;
    return false;
  }
  return true;

}

int dnn_module_torch::predict_category_classification(category_rst_list& rst_container) {
  int batchsize = 0;
  auto __softmax = [](vector<float> unnorm_probs) -> vector<float> {
    int n_classes = unnorm_probs.size();

    // 1. Partition function
    float log_sum_of_exp_probs = 0;
    for (auto& n : unnorm_probs) {
      log_sum_of_exp_probs += std::exp(n);
    }
    log_sum_of_exp_probs = std::log(log_sum_of_exp_probs);

    // 2. normalize
    std::vector<float> probs(n_classes);
    for (int class_idx = 0; class_idx != n_classes; class_idx++) {
      probs[class_idx] =
          std::exp(unnorm_probs[class_idx] - log_sum_of_exp_probs);
    }
    return probs;
  };

    const int batchSize = cfg_.batchSize;
    auto inputs = get_inputs();
    auto outputs = module_.forward(inputs).toTensor();
  
  for (int batch_index = 0; batch_index < batchsize; batch_index++) {
      try {
      auto output = outputs[batch_index];
      int ndim = output.ndimension();
      assert(ndim == 2);

      torch::ArrayRef<int64_t> sizes = output.sizes();
      int n_samples = sizes[0];
      int n_classes = sizes[1];

      auto cpu_output = output.to(at::kCPU);

      std::vector<float> unnorm_probs(  cpu_output.data_ptr<float>(), cpu_output.data_ptr<float>() + (n_samples * n_classes));

      vector<float> probs;
      probs = __softmax(unnorm_probs);
      auto prob = std::max_element(probs.begin(), probs.end());
      auto label_idx = std::distance(probs.begin(), prob);
      float prob_float = *prob;
      rst_container[batch_index] = label_idx;
    } catch (const c10::Error& e) {
      std::cout << e.msg() << endl;
    } catch (...) {
      ::cout << "failed predict_object_detection_efficientdet\n";
    }
    
  } 
  return batchSize;
}

int dnn_module_torch::predict_binary_classification(binary_rst_list& rst_container) {

  const int batchSize = cfg_.batchSize;
  
  auto inputs = get_inputs();
  auto outputs = module_.forward(inputs).toTensor();
  
  for (int batch_index = 0; batch_index < batchSize; batch_index++) {
    try {
      auto output = torch::sigmoid(outputs[batch_index]);
      output = torch::sigmoid(output);
      torch::Tensor classification = output.argmax(1);
      int32_t classificationWinner = classification.item().toInt();
      rst_container[batch_index] = classificationWinner == 1 ? true : false;
    } catch (const c10::Error& e) {
      std::cout << e.msg() << endl;
    } catch (...) {
      ::cout << "failed predict_object_detection_efficientdet\n";
    }
  }
  return batchSize;
 
}
void dnn_module_torch::nhwc_blob_from_images(vector<cv::Mat> buffers, float* hostDataBuffer) {
  cv::Mat image0 = buffers[0];
  int rows_ = image0.rows;
  int cols_ = image0.cols;
  int channel_ = image0.channels();
  int batch_ = buffers.size();
  int nimages_ = buffers.size();
  int ddepth = CV_32F;
  cv::Mat blob_ = cv::dnn::blobFromImages(buffers);
  
  auto tensor = torch::from_blob(blob_.data, {batch_, channel_, rows_, cols_}).to(at::kCPU);
  tensor = tensor.permute({0,2,3,1}).contiguous();
  std::memcpy(hostDataBuffer, tensor.data_ptr(), sizeof(float) * tensor.numel());
 
}
std::vector<torch::jit::IValue> dnn_module_torch::get_inputs() {
  auto buffers = get_preprocess_image_buffers();
  std::vector<at::Tensor> inputs_vec;
  for (int i = 0; i < buffers.size();i++) 
  {
    try {
        cv::cuda::GpuMat gpu_frame;
        auto frame = buffers[i];
        auto size = frame.size();
        auto nChannels = frame.channels();
        auto tensor = torch::from_blob(frame.data,  {1, size.height, size.width,nChannels });
        inputs_vec.emplace_back(tensor.permute({0, 3, 1, 2}).contiguous().to(at::kCUDA));
       
    } catch (const c10::Error& e) {
      std::cout << e.msg() << endl;
    } catch (...) {
      std::cout << "failed preprocessImage\n";
      
    }
  }
  at::Tensor input_ = torch::cat(inputs_vec);
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(input_);
  return inputs;
}

int dnn_module_torch::predict_object_detection_efficientdet(bbox_t_container_rst_list& rst_container) 
{
  
  const int batchSize = cfg_.batchSize;
  
  auto inputs = get_inputs();
  auto outputs = module_.forward({inputs}).toTensor();
  
  float dnn_width = cfg_.dnn_width;
  float dnn_height = cfg_.dnn_height;
  auto origin_image = get_origin_image_buffers();

  for (int batch_index = 0; batch_index < batchSize; batch_index++) {
    try{
      cv::Mat& src = origin_image[batch_index];
      auto output = outputs[batch_index];  // batch
      auto boxes = output.index({Slice(None, None), Slice(None, 4)});
      auto scores = output.index({Slice(None, None), 4});
      auto labels = output.index({Slice(None, None), 5});
      auto indexs = (scores > cfg_.threshold).toType(c10::ScalarType::Bool);
      //cout << boxes.sizes() << "," << scores.sizes() << endl;
      boxes = boxes.index(indexs);
      scores = scores.index(indexs);
      labels = labels.index(indexs);
     
      for (int i = 0; i < boxes.size(0) && i < MAX_OBJECTS; i++) {
        bbox_t box;
        float lx = boxes[i][0].item<float>() / dnn_width * src.cols;
        float ly = boxes[i][1].item<float>() / dnn_height * src.rows;
        float rx = boxes[i][2].item<float>() / dnn_width * src.cols;
        float ry = boxes[i][3].item<float>() / dnn_height * src.rows;

        box.x = lx;
        box.y = ly;
        box.w = (rx - lx);
        box.h = (ry - ly);
        box.obj_id = labels[i].item<float>();
        box.prob = scores[i].item<float>();
        rst_container[batch_index].candidates[i] = box;
    
        // cout << lx << " , " << ly << " , " << rx << " , " << ry << " ,
        // "<<box.prob<<" , "<<box.obj_id<<"\n ";
      }
      rst_container[batch_index].cnt = std::min(MAX_OBJECTS,(int)boxes.size(0));
    }
    catch (const c10::Error& e) {
      std::cout << e.msg() << endl;
    }
    catch (...) {
      ::cout << "failed predict_object_detection_efficientdet\n";
    }
  }
  rst_container.cnt = batchSize;
  return batchSize;
}


torch::Tensor dnn_module_torch::mahalanobis(torch::Tensor u, torch::Tensor v,
                                            torch::Tensor cov_inv) {
  auto delta = u  - v;
  auto mul = torch::matmul(delta, cov_inv);
  auto m = torch::dot(delta, mul);
  return torch::sqrt(m);
 

}
int dnn_module_torch::predict_anomaly_detection_patchcore(segm_t_container_rst_list& rst_container) 
{
  const int batchSize = cfg_.batchSize;
     try {
       auto distance_matix = [](torch::Tensor x, torch::Tensor features,
                                int p = 2) -> torch::Tensor {
         int n = x.size(0);
         int m = features.size(0);
         int d = x.size(1);
         auto x_ = x.unsqueeze(1).expand({n, m, d});
         auto features_ = features.unsqueeze(0).expand({n, m, d});
         return torch::pow(x_ - features_, p).sum(2);
       };

       auto inputs = get_inputs();
       auto x = module_wideresnet_50_->conv1->forward(inputs[0].toTensor());
       x = module_wideresnet_50_->bn1->forward(x).relu_();
       x = torch::max_pool2d(x, 3, 2, 1);

       auto outputs1 = module_wideresnet_50_->layer1->forward(x);
       auto outputs2 = module_wideresnet_50_->layer2->forward(outputs1);
       auto outputs3 = module_wideresnet_50_->layer3->forward(outputs2);

       auto m = AvgPool2d(AvgPool2dOptions(3).stride(1).padding(1));
       auto embed1 = m(outputs2);
       auto embed2 = m(outputs3);

       

       auto embedding_vectors = embedding_concat(embed1, embed2);
       //reshape_embedding
      
       for (int batch_idx = 0; batch_idx < batchSize; batch_idx++) 
       {
         int feature_idx = cfg_.order_of_feature_index_to_batch[batch_idx];; 
         auto embedding_vector_batch = embedding_vectors[batch_idx];

         embedding_vector_batch = embedding_vector_batch.reshape(
             {embedding_vector_batch.size(0),
              embedding_vector_batch.size(1) * embedding_vectors.size(2)});
         embedding_vector_batch = embedding_vector_batch.permute({1, 0});
         int p = 2;
         int k = 9;
        
         auto dist = torch::cdist(embedding_vector_batch,
                                  anomaly_feature_patchcore[feature_idx], p);
         auto knn = std::get<0>(dist.topk(k, -1, false));
         int block_size = static_cast<int>(std::sqrt(knn.size(0)));
         auto anomaly_map = knn.index({Slice(None, None), 0})
                                .reshape({block_size, block_size});
         double max_score = cfg_.anomaly_feature[feature_idx].anomalyMaxScore;
         double min_score = cfg_.anomaly_feature[feature_idx].anomalyMinScore;
         auto scores = (anomaly_map - min_score) / (max_score - min_score);

         auto scores_resized =
             F::interpolate(scores.unsqueeze(0).unsqueeze(0),
                            F::InterpolateFuncOptions()
                                .size(std::vector<int64_t>{cfg_.dnn_height,
                                                           cfg_.dnn_width})
                                .align_corners(false)
                                .mode(torch::kBilinear)).squeeze().squeeze();

         if (cfg_.anomaly_feature[feature_idx].defect_extraction_enable) {
           auto anomal_index =
               scores_resized <
               cfg_.anomaly_feature[feature_idx].defect_extraction_threshold;
           scores_resized.index_put_(anomal_index,
                                     torch::zeros(1).to(at::kCUDA));
           cfg_.anomaly_feature[feature_idx].defect_extraction_jud_area_ratio;
         }

         auto anomaly_mat = tensor2dToMat(scores_resized.to(at::kCPU));

         cv::Mat anomaly_colormap, anomaly_mat_scaled;
         anomaly_mat.at<float>(0, 0) = 1;
         anomaly_mat.convertTo(anomaly_mat_scaled, CV_8UC3, 255.f);

         applyColorMap(anomaly_mat_scaled, anomaly_colormap, cv::COLORMAP_JET);
         cv::Mat anomaly_mat_origin_size;
         cv::resize(anomaly_colormap, anomaly_mat_origin_size,
                    {cfg_.dnn_height, cfg_.dnn_width});
         auto origin_mat = get_origin_image_buffers()[batch_idx];

         cv::resize(origin_mat, anomaly_mat_origin_size,
                    {cfg_.dnn_height, cfg_.dnn_width});
         cv::Mat dst;
         cv::addWeighted(anomaly_mat_origin_size, 0.5, anomaly_colormap,
                         1 - 0.5, 0, dst);

         segm_t_container& segm = rst_container[batch_idx];
         segm.cnt = 1;
         segm.candidates[0].prob = anomaly_map.mean().item<float>();
         segm.candidates[0].obj_id =
             anomaly_map.mean().item<float>() >   cfg_.anomaly_feature[feature_idx].anomalyThreshold ? false  : true;

         if (cfg_.anomaly_feature[feature_idx].defect_extraction_enable) {
           auto defect_area =  scores_resized >     cfg_.anomaly_feature[feature_idx].defect_extraction_threshold;
           int defect_area_count = defect_area.count_nonzero().item<int>();
           float defect_area_ratio =   defect_area_count / (float)(cfg_.dnn_height * cfg_.dnn_width);
           segm.candidates[0].prob = defect_area_ratio;
           segm.candidates[0].obj_id =     defect_area_ratio > cfg_.anomaly_feature[feature_idx].defect_extraction_jud_area_ratio   ? false : true;
         }

         matToImageinfo(anomaly_colormap, segm.display_image);

       }
     } catch (const c10::Error& e) {
       std::cout << e.msg() << endl;
       return false;
     }
  
  rst_container.cnt = batchSize;

 
  return 0;
}
   
int dnn_module_torch::detectYolov5(model_config& cfg, float* prediction_ptr,
                   int output_dim1_size, int output_dim2_size,
                   vector<cv::Mat>& origin_image,
                   bbox_t_container_rst_list& rst_container) {
  const int batchSize = cfg.batchSize;
  auto prediction =  torch::from_blob(prediction_ptr,
                       {batchSize, output_dim1_size, output_dim2_size})
          .to(default_dev);
  auto xywh2xyxy = [](torch::Tensor x) -> torch::Tensor {
    auto y = x.clone();
    y.index({Slice(None, None), 0}) =   x.index({Slice(None, None), 0}) -  x.index({Slice(None, None), 2}) / 2.;  // top left x
    y.index({Slice(None, None), 1}) = x.index({Slice(None, None), 1}) -   x.index({Slice(None, None), 3}) / 2.;  // top left y
    y.index({Slice(None, None), 2}) =   x.index({Slice(None, None), 0}) + x.index({Slice(None, None), 2}) / 2.;  // bottom right x
    y.index({Slice(None, None), 3}) = x.index({Slice(None, None), 1}) +  x.index({Slice(None, None), 3}) / 2.;  // bottom right y
    return y;
  };
  auto xyxy2xywh = [](torch::Tensor x) -> torch::Tensor {
    auto y = x.clone();
    y.index({Slice(None, None), 0}) =
        x.index({Slice(None, None), 0}) +
        x.index({Slice(None, None), 2}) / 2.;  // x c
    y.index({Slice(None, None), 1}) =
        x.index({Slice(None, None), 1}) +
        x.index({Slice(None, None), 3}) / 2.;  // y c
    y.index({Slice(None, None), 2}) =
        x.index({Slice(None, None), 2}) - x.index({Slice(None, None), 0});  // w
    y.index({Slice(None, None), 3}) =
        x.index({Slice(None, None), 3}) - x.index({Slice(None, None), 1});  // h
    return y;
  };

  auto box_area = [](torch::Tensor box) -> torch::Tensor {
    return (box.index({2}) - box.index({0})) *
           (box.index({3}) - box.index({1}));
  };
  auto box_iou = [&](torch::Tensor box1, torch::Tensor box2) {
    auto area1 = box_area(box1.t());
    auto area2 = box_area(box2.t());

    auto inter =
        (torch::min(box1.index({Slice(None, None), None, Slice(2, None)}),
                    box2.index({Slice(None, None), Slice(2, None)})) -
         torch::max(box1.index({Slice(None, None), None, Slice(None, 2)}),
                    box2.index({Slice(None, None), Slice(None, 2)})))
            .clamp(0)
            .prod(2);
    return inter / (area1.index({Slice(None, None), None}) + area2 - inter);
  };
  auto non_max_suppression =
      [&](float conf_thres = 0.25, float iou_thres = 0.45,
          bool agnostic = false, bool multi_label = false,
          int max_det = MAX_OBJECTS) -> vector<torch::Tensor> {
    auto nc = prediction.size(2) - 5;
    auto xc = prediction.index({"...", 4}) > conf_thres;
    // Settings
    int min_wh = 2,  max_wh = 4096;    //(pixels)minimum and maximum box width and height
    int max_nms = 30000;  // maximum number of boxes into torchvision.ops.nms()
    float time_limit = 10.0;  // seconds to quit after

    bool redundant = true;    // require redundant detections
    multi_label &= nc > 1;    // multiple labels per box(adds 0.5ms / img)
    bool merge = false;       // use merge - NMS
    vector<torch::Tensor> output;
    int n_batch_size = prediction.size(0);
    for (int xi = 0; xi < n_batch_size; xi++) {
      auto x = prediction[xi];
      x = x.index({xc.index({xi})});
      /*
      Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)
      */
     if (x.size(0) == 0) {
        output.push_back(x);
        continue;
      }

      x.index({Slice(None, None), Slice(5, None)}) *=
          x.index({Slice(None, None), Slice(4, 5)});
      auto box = xywh2xyxy(x.index({Slice(None, None), Slice(None, 4)}));
      bool multi_label = false;
      if (multi_label) {
        auto rst = x.index({Slice(None, None), Slice(5, None)}) > conf_thres;
        auto non_zero = rst.nonzero().t();
        auto i = non_zero.index({0});
        auto j = non_zero.index({1});
        x = torch::cat(
            {box[i], x.index({i, j + 5, None}),
             x.index({Slice(None, None), None}).to(c10::ScalarType::Float)},
            1);
      } else {  // best class only
        auto tup = x.index({Slice(None, None), Slice(5, None)}).max(1, true);
        auto conf = std::get<0>(tup);
        auto j = std::get<1>(tup).to(c10::ScalarType::Float);
        x = torch::cat({box, conf, j}, 1).index({conf.view(-1) > conf_thres});
      }
      
      // Filter by class;
      bool classes = false;
      if (classes) {
      }

      auto n = x.size(0);
      if (n == 0) {
        output.push_back(x);
        continue;
      }
      else if (n > max_nms)
        x = x.index({x.index({Slice(None, None), 4})
                         .argsort(true)
                         .index({Slice(None, max_nms)})});
      bool agnostic = false;
      int n_agnostic = agnostic ? 0 : max_wh;
      auto c = x.index({Slice(None, None), Slice(5, 6)}) * n_agnostic;
      auto boxes = x.index({Slice(None, None), Slice(None, 4)}) + c;
      auto scores = x.index({Slice(None, None), 4});

      auto i = nms(boxes, scores, (double)iou_thres);
      
      bool merge = false;
      if (i.size(0) > max_det) i = i.index({Slice(None, max_det)});
      if (merge && 1 < n < 3E3) {
        auto iou = box_iou(boxes.index({i}), boxes) > iou_thres;
        auto weights = iou * scores.index({None});
        x.index({i, Slice(None, 4)}) =
            torch::mm(weights, x.index({Slice(None, None), Slice(None, 4)}))
                .to(c10::ScalarType::Float) /
            weights.sum(1, true);
        bool redundant = true;
        if (redundant) {
          i = i.index({iou.sum(1) > 1});
        }
      }

      output.push_back(x.index({i}));
    };
    return output;
  };
  auto clip_coords = [](torch::Tensor boxes, c10::IntArrayRef img_shape) {
    auto tensorIsNan = at::isnan(boxes).any().item<bool>();
    if (!tensorIsNan) {
      boxes.index({Slice(None, None), 0}).clamp_(0, img_shape[1]);  // x1
      boxes.index({Slice(None, None), 1}).clamp_(0, img_shape[0]);  // y1
      boxes.index({Slice(None, None), 2}).clamp_(0, img_shape[1]);  // x2
      boxes.index({Slice(None, None), 3}).clamp_(0, img_shape[0]);  // y2

    } else {
      boxes.index({Slice(None, None), Slice(0, 2)}) =
          boxes.index({Slice(None, None), Slice(0, 2)}).clip(0, img_shape[1]);
      boxes.index({Slice(None, None), Slice(1, 3)}) =
          boxes.index({Slice(None, None), Slice(1, 3)}).clip(0, img_shape[0]);
    }
  };
  auto scale_coords = [&](c10::IntArrayRef img1_shape, torch::Tensor coords,
                          c10::IntArrayRef img0_shape) {
    float gain = std::min(img1_shape[0] / (float)img0_shape[0],
                          img1_shape[1] / (float)img0_shape[1]);
    float pad_w = (img1_shape[1] - img0_shape[1] * gain) / 2;
    float pad_h = (img1_shape[0] - img0_shape[0] * gain) / 2.;

    coords.index_put_(
        {Slice(None, None), torch::tensor({0, 2})},
        coords.index({Slice(None, None), torch::tensor({0, 2})})) -= pad_w;
    coords.index_put_(
        {Slice(None, None), torch::tensor({1, 3})},
        coords.index({Slice(None, None), torch::tensor({1, 3})})) -= pad_h;
    // coords[:, [0, 2]] -= pad[0]  # x padding
    // coords[:, [1, 3]] -= pad[1]  # y padding
    coords.index({Slice(None, None), Slice(None, 4)}) /= gain;
    clip_coords(coords, img0_shape);

    return coords;
  };
  auto apply_classifier = [&](vector<torch::Tensor> x) {
    // not complete
    for (int i = 0; i < x.size(); i++) {
      cv::Mat& src = origin_image[i];

      auto d = x[i].clone();
      auto b = xyxy2xywh(d.index({Slice(None, None), Slice(None, 4)}));
      b.index({Slice(None, None), Slice(2, None)}) =
          std::get<0>(b.index({Slice(None, None), Slice(2, None)}).max(1))
              .unsqueeze(1);
      b.index({Slice(None, None), Slice(2, None)}) =
          b.index({Slice(None, None), Slice(2, None)}) * 1.3 + 30;  // pad
      b.index({Slice(None, None), Slice(None, 4)}) =
          xywh2xyxy(b).to(c10::ScalarType::Long);
      scale_coords({cfg.dnn_height, cfg.dnn_width},
                   d.index({Slice(None, None), Slice(None, 4)}),
                   {cfg.dnn_height, cfg.dnn_width, cfg.dnn_chnnel});

      // classes
      auto pred_cls1 =
          d.index({Slice(None, None), 5}).to(c10::ScalarType::Long);
    }
  };

  auto plot_one_box = [](torch::Tensor x) -> std::pair<cv::Point, cv::Point> {
    cv::Point pt1, pt2;
    pt1 = {int(x.index({0}).item<float>()), int(x.index({1}).item<float>())};
    pt2 = {int(x.index({2}).item<float>()), int(x.index({3}).item<float>())};
    return make_pair(pt1, pt2);  // lt,rb
  };

  try {
    auto pred = non_max_suppression(cfg.threshold, cfg.iou_threshold);
    /*
    bool classify = false;
    if (classify) {
      apply_classifier(pred);
    }
    */
    for (int batch_index = 0; batch_index < pred.size(); batch_index++) {
      cv::Mat& src = origin_image[batch_index];
      auto gn = torch::tensor({src.rows, src.cols, cfg.dnn_chnnel})
                    .index({torch::tensor({1, 0, 1, 0})});
      auto det = pred[batch_index];
      int n_det = det.size(0);
      if (n_det > 0) {
        det.index({Slice(None, None), Slice(None, 4)}) =
            scale_coords({cfg.dnn_height, cfg.dnn_width},
                         det.index({Slice(None, None), Slice(None, 4)}),
                         {cfg.dnn_height, cfg.dnn_width, cfg.dnn_chnnel})
                .round();
        auto uniq = torch::_unique(det.index({Slice(None, None), -1}));
        auto classes = std::get<0>(uniq);
        set<int> vnumber;
        for (int c = 0; c < classes.size(0); c++) {
          int n = (det.index({Slice(None, None), -1}) == c).sum().item<int>();
          vnumber.insert(n);
        }
        
        for (int det_i = 0; det_i < n_det && det_i < MAX_OBJECTS; det_i++) {
          auto xyxy = det.index({det_i, Slice(0, 4)});
          auto conf = det.index({det_i, Slice(4, 5)}).item<float>();
          auto cls = det.index({det_i, Slice(5, 6)}).item<float>();
          auto ltrb = plot_one_box(xyxy);
          bbox_t box;
          float lx = ltrb.first.x /  (float) cfg.dnn_width  * src.cols ;
          float ly = ltrb.first.y /  (float) cfg.dnn_height * src.rows ;
          float rx = ltrb.second.x / (float) cfg.dnn_width * src.cols;;
          float ry = ltrb.second.y / (float) cfg.dnn_height * src.rows;

          box.x = lx;
          box.y = ly;
          box.w = (rx - lx);
          box.h = (ry - ly);
          box.obj_id = cls;
          box.prob = conf;
          rst_container[batch_index].candidates[det_i] = box;
        }
        rst_container[batch_index].cnt = std::min(MAX_OBJECTS, n_det);
      }
    }

    rst_container.cnt = batchSize;
  } catch (const c10::Error& e) {
    std::cout << e.msg() << endl;
    return false;
  } catch (...) {
    std::cout << "failed predict_anomaly_detection\n";
    return false;
  };

  return batchSize;
}



int dnn_module_torch::detectYolact(model_config& cfg, float* loc_name,
                                    float* conf_name, float* mask_name,
                                       float* priors_name, float* proto_name,int class_num,
                                       int view_size, int proto_size,vector<cv::Mat>& origin_image,segm_t_container_rst_list& rst_container) {
  
  gen_colors();
  const int batchSize = cfg.batchSize;
  int num_classes = class_num;
  const int background_label = 0;
  const float conf_thresh = 0.05;
  const float nms_thresh = 0.5;
  const float score_threshold = cfg.threshold;
  const int max_num_detections = 200;
  bool preserve_aspect_ratio = false;
  int top_k = 200;
  bool cropmask = true;

  auto decode = [](torch::Tensor loc , torch::Tensor priors) -> torch::Tensor {
      float variances[2] = {0.1, 0.2};
      auto temp1 = priors.index({Slice(None, None), Slice(None, 2)}) +
                     loc.index({Slice(None, None), Slice(None, 2)}) *
                         variances[0] *
                         priors.index({Slice(None, None), Slice(2, None)});
      auto temp2 = priors.index({Slice(None, None), Slice(2, None)}) *
                   torch::exp(loc.index({Slice(None, None), Slice(2, None)}) *
                              variances[1]);
      auto boxes = torch::cat({temp1, temp2}, 1);
      boxes.index({Slice(None, None), Slice(None, 2)})-= boxes.index({Slice(None, None), Slice(2, None)}) / 2;
      boxes.index({Slice(None, None), Slice(2,None)}) += boxes.index({Slice(None, None), Slice(None,2)});
      return boxes;

  };
  auto intersect = [](torch::Tensor box_a, torch::Tensor box_b)  -> torch::Tensor{
    auto n = box_a.size(0);
    auto A = box_a.size(1);
    auto B = box_b.size(1);

    auto max_xy = torch::min(box_a.index({Slice(None, None),Slice(None, None),Slice(2, None)}).unsqueeze(2).expand({n, A, B, 2}),
                       box_b.index({Slice(None, None),Slice(None, None),Slice(2, None)}).unsqueeze(1).expand({n, A, B, 2}));
    auto min_xy = torch::max(box_a.index({Slice(None, None),Slice(None, None),Slice(None, 2)}).unsqueeze(2).expand({n, A, B, 2}),
                       box_b.index({Slice(None, None),Slice(None, None),Slice(None, 2)}).unsqueeze(1).expand({n, A, B, 2}));
    auto inter = torch::clamp((max_xy - min_xy), 0);
    return inter.index({Slice(None, None),Slice(None, None),Slice(None,None),0})
      * inter.index({Slice(None, None),Slice(None, None),Slice(None,None),1});
  };
   
  auto jaccard = [&](torch::Tensor box_a, torch::Tensor box_b, bool iscrowd = false) -> torch::Tensor{
    bool use_batch = true;
    if (box_a.dim() == 2) {
      use_batch = false;
      box_a = box_a.index({None, "..."});
      box_b = box_b.index({None, "..."});
    }
      auto inter = intersect(box_a, box_b);

      auto area_a = ((box_a.index({Slice(None, None), Slice(None, None), 2}) -
                      box_a.index({Slice(None, None), Slice(None, None), 0})) *
                     (box_a.index({Slice(None, None), Slice(None, None), 3}) -
                      box_a.index({Slice(None, None), Slice(None, None), 1})))
                        .unsqueeze(2)
                        .expand_as(inter);

      auto area_b = ((box_b.index({Slice(None, None), Slice(None, None), 2}) -
                      box_b.index({Slice(None, None), Slice(None, None), 0})) *
                     (box_b.index({Slice(None, None), Slice(None, None), 3}) -
                      box_b.index({Slice(None, None), Slice(None, None), 1})))
                        .unsqueeze(1)
                        .expand_as(inter);
      auto union_ab = area_a + area_b - inter;
      torch::Tensor out;
      if (iscrowd) {
        out = inter / area_a;
      } else {
        out = inter / union_ab;
      }
      if (use_batch)
        return out;
      else
        return out.squeeze(0);
    
  };
     
  auto fast_nms = [&](torch::Tensor boxes, torch::Tensor masks, torch::Tensor scores, float iou_threshold = 0.5, int top_k = 200, bool second_threshold = false)
      ->vector<torch::Tensor> { 

    try
    {
      auto score_tup = scores.sort(1, true);
      auto idx = std::get<1>(score_tup);
      auto scores_ = std::get<0>(score_tup);
      idx = idx.index({Slice(None, None), Slice(None, top_k)}).contiguous();
      scores_ = scores_.index({Slice(None, None), Slice(None, top_k)});
      auto num_classes = idx.sizes().at(0);
      auto num_dets = idx.sizes().at(1);
      boxes = boxes.index({idx.view(-1), Slice(None, None)})
                  .view({num_classes, num_dets, 4});
      masks = masks.index({idx.view(-1), Slice(None, None)})
                  .view({num_classes, num_dets, -1});

      
      auto dev = boxes.device();
      auto iou = jaccard(boxes, boxes);
    
      iou.triu_(1);
      auto iou_max = std::get<0>(iou.max(1));
      auto keep = (iou_max <= iou_threshold);
   
      if (second_threshold) {
        keep *= (scores_ > conf_thresh);
      }
     
      auto classes = torch::arange(num_classes)
                         .to(dev)
                         .index({Slice(None, None), None})
                         .expand_as(keep);
      classes = classes.index({keep});
      boxes = boxes.index({keep});
      masks = masks.index({keep});
      scores_ = scores_.index({keep});
      score_tup = scores_.sort(0, true);
      scores_ = std::get<0>(score_tup);
      idx = std::get<1>(score_tup);
      idx = idx.index({Slice(None, max_num_detections)});
      scores_ = scores_.index({Slice(None, max_num_detections)});
      classes = classes.index({idx});
      boxes = boxes.index({idx});
      masks = masks.index({idx});
      
      
      return std::vector<torch::Tensor>({boxes, masks, classes, scores_});
    }
     catch (const c10::Error& e) {
      std::cout << e.msg() << endl;
      return std::vector<torch::Tensor>();
    }
    
  };
  auto detect = [&](int batch_idx, torch::Tensor conf_preds, torch::Tensor decoded_boxes, torch::Tensor mask_data)
    -> vector<torch::Tensor> {
    try{

     auto cur_scores = conf_preds.index({batch_idx , Slice(1, None) ,Slice(None, None)});
     auto conf_scores = std::get<0>(torch::max(cur_scores, 0));
    
     auto keep = (conf_scores > conf_thresh);
     auto scores = cur_scores.index({Slice(None, None), keep});
     auto boxes = decoded_boxes.index({keep, Slice(None, None)}); 
     auto masks = mask_data.index({batch_idx, keep, Slice(None, None)});
     
     if (scores.size(1) == 0)
       return vector<torch::Tensor>();
     return fast_nms(boxes, masks, scores, nms_thresh, top_k);
  } catch (const c10::Error& e) {
    std::cout << e.msg() << endl;
    return std::vector<torch::Tensor>();
  }
     
  };
  auto sanitize_coordinates = [&](torch::Tensor _x1, torch::Tensor _x2, int img_size, int padding = 0, bool cast = true)->std::tuple<torch::Tensor,torch::Tensor> 
  {
    _x1 *= img_size;
    _x2 *= img_size;

    if (cast) {
      _x1 = _x1.to(at::ScalarType::Long);
      _x2 = _x2.to(at::ScalarType::Long);
    }

    auto x1 = torch::min(_x1, _x2);
    auto x2 = torch::max(_x1, _x2);
    x1 = torch::clamp(x1 - padding, 0);
    x2 = torch::clamp(x2 + padding, c10::nullopt, img_size);

    
    //cout << x1.sizes() << "," << x2.sizes() << "," << _x1.sizes() << "," << _x2.sizes()<< endl;

    return make_tuple(x1, x2);
  };
    
  auto crop = [&](torch::Tensor masks, torch::Tensor boxes, int padding = 1) ->torch::Tensor{
    try
    {
      torch::NoGradGuard no_grad;
      int h, w, n;
      h = masks.size(0);
      w = masks.size(1);
      n = masks.size(2);
      boxes = boxes.clone();
      auto x_ = sanitize_coordinates(boxes.index({Slice(None, None), 0}),
                                     boxes.index({Slice(None, None), 2}), w,
                                     padding, true);
      auto y_ = sanitize_coordinates(boxes.index({Slice(None, None), 1}),
                                     boxes.index({Slice(None, None), 3}), h,
                                     padding, true);
      auto x1 = std::get<0>(x_);
      auto x2 = std::get<1>(x_);
      auto y1 = std::get<0>(y_);
      auto y2 = std::get<1>(y_);

      auto rows = torch::arange(w)
                      .index({None, Slice(None, None), None})
                      .expand({h, w, n}).to(default_dev);
      auto cols = torch::arange(h)
                      .index({Slice(None, None), None, None})
                      .expand({h, w, n}).to(default_dev);

      auto masks_left = rows >= x1.index({None, None, Slice(None, None)});
      auto masks_right = rows < x2.index({None, None, Slice(None, None)});
      auto masks_up = cols >= y1.index({None, None, Slice(None, None)});
      auto masks_down = cols < y2.index({None, None, Slice(None, None)});
      auto crop_mask = masks_left * masks_right * masks_up * masks_down;

      return masks * crop_mask.to(at::ScalarType::Float);

     } catch (const c10::Error& e) {
    std::cout << e.msg() << endl;
  }
     return torch::Tensor();

  };

  
  auto center_size=[](torch::Tensor boxes)->torch::Tensor {
    return torch::cat({
         (boxes.index({Slice(None, None) ,Slice(2, None)}) + boxes.index({Slice(None, None) ,Slice(None, 2)})) / 2,
          boxes.index({Slice(None, None) ,Slice(2, None)}) - boxes.index({Slice(None, None) ,Slice(None, 2)})},1);
  };
   
   

  auto faster_rcnn_scale = [&](int width, int height, int min_size, int max_size) -> tuple<int, int> {
    auto min_scale = min_size / (float)std::min(width, height);
    width *= min_scale;
    height *= min_scale;

   auto max_scale = max_size / (float)std::max(width, height);
       if (max_scale < 1)
       {
         width *= max_scale;
         height *= max_scale;

       }
       
       return make_tuple(width, height);
  };
     
  auto postprocess = [&](vector<torch::Tensor> dets, int w,int h ,bool crop_masks=true,float threshold = 0)->vector<torch::Tensor>
  {
    try {
    bool visualize_lincomb = false;
    //bool preserve_aspect_ratio = false;
    //boxes, masks, class,scores,proto
   
    auto keep = dets[3] > threshold;
   //skip proto
    for (int i=0;i<4;i++) 
    {
      dets[i] = dets[i].index({keep});
    }
    float min_size = cfg.yolact_min_size;
    float max_size = cfg.yolact_max_size;
    int b_w = w, b_h = h;
    int r_w = 0, r_h = 0;
    /*
    if (preserve_aspect_ratio) {
      auto r_ = faster_rcnn_scale(w, h, min_size, max_size);
      r_w = std::get<0>(r_);
      r_h = std::get<1>(r_);
      auto boxes = dets[0];
      boxes = center_size(boxes);
      auto s_w = r_w / (float)max_size;
      auto s_h = r_h / (float)max_size;
      auto not_outside = ((boxes.index({Slice(None, None), 0}) > s_w) +
                          (boxes.index({Slice(None, None), 1}) > s_h)) < 1;

      for (int i = 0; i < 4; i++) {
        dets[i] = dets[i].index({not_outside});
      }

       b_w = max_size / (float)(r_w * w);
       b_h = max_size / (float)(r_h * h);
    };
     */

    auto classes = dets[2];
    auto boxes = dets[0];
    auto scores = dets[3];
    auto masks = dets[1];
    auto proto_data = dets[4];

    masks = torch::matmul(proto_data, masks.t());
    masks = torch::sigmoid(masks);
    if (crop_masks)
            masks = crop(masks, boxes);
    masks = masks.permute({2, 0, 1}).contiguous();
 
    /*
    if (preserve_aspect_ratio)
            masks = masks.index({Slice(None, None), Slice(None, int(r_h/max_size*proto_data.size(1))), Slice(None, int(r_w/max_size*proto_data.size(2)))});
    */
    
    masks = F::interpolate(masks.unsqueeze(0),F::InterpolateFuncOptions().size(std::vector<int64_t>{h,w}).align_corners(false).mode(torch::kBilinear)).squeeze(0);
    masks = masks.gt(0.5).to(at::ScalarType::Float);
    auto x_ = sanitize_coordinates(boxes.index({Slice(None, None), 0}),
                                   boxes.index({Slice(None, None), 2}), b_w, 0, false);
    auto y_ = sanitize_coordinates(boxes.index({Slice(None, None), 1}),
                                   boxes.index({Slice(None, None), 3}), b_h, 0, false);

    boxes.index({Slice(None, None), 0}) = std::get<0>(x_);
    boxes.index({Slice(None, None), 2}) = std::get<1>(x_);
    boxes.index({Slice(None, None), 1}) = std::get<0>(y_);
    boxes.index({Slice(None, None), 3}) = std::get<1>(y_);

    boxes = boxes.to(at::ScalarType::Long);
   
    return vector<torch::Tensor>({classes, scores, boxes, masks});
    } catch (const c10::Error& e) {
      std::cout << e.msg() << endl;
    }
    return vector<torch::Tensor>();
    
  };

  
  auto prep_display = [&](vector<torch::Tensor> dets, cv::Mat frame,
                          segm_t_container& segm, bool class_color = false,
                          float mask_alpha = 0.45) {
    try
    {
      
      cv::cuda::GpuMat gpu_frame, flt_image;
      gpu_frame.upload(frame);
  
    if (frame.channels() == 1)
      cv::cuda::cvtColor(gpu_frame, gpu_frame, cv::COLOR_GRAY2BGR);

    gpu_frame.convertTo(flt_image, CV_32FC3, 1.f / 255.f);

    int channel = flt_image.channels();
    
    cv::Mat cpumat(flt_image);

    auto size = cpumat.size();
    auto nChannels = cpumat.channels();
    auto tensor = torch::from_blob(cpumat.data,
                                    {size.height, size.width, nChannels});
    auto img_gpu = tensor.contiguous().to(default_dev);   

    auto h = gpu_frame.rows;
    auto w = gpu_frame.cols;
    // classes, scores, boxes, masks
    auto t = postprocess(dets, w, h, cropmask, score_threshold);
    if (t.size() == 0) 
      return ;
    auto masks = t[3].index({Slice(None, top_k)});
    auto classes = t[0].index({Slice(None, top_k)});
    auto scores = t[1].index({Slice(None, top_k)});
    auto boxes = t[2].index({Slice(None, top_k)});
    int classes_shape = classes.size(0);
    auto num_dets_to_consider = std::min(top_k, classes_shape);
    for (int j = 0; j < num_dets_to_consider;j++) {
      if (scores[j].item<float>() < score_threshold) {
        num_dets_to_consider = j;
        break;
      }
    }
    if (num_dets_to_consider == 0) 
      return ;
    
    masks = masks.index({Slice(None, num_dets_to_consider),  Slice(None, None), Slice(None, None), None});
    vector<torch::Tensor> vdetect;
    for (int i = 0; i < num_dets_to_consider; i++) {
      vdetect.push_back(mask_colors[i]);
    }

    auto colors = torch::cat(vdetect, 0).to(default_dev);
 
    auto masks_color = masks.repeat({1, 1, 1, 3}) * colors * mask_alpha;
    auto inv_alph_masks = masks * (-mask_alpha) + 1;
    auto masks_color_summand = masks_color[0];
      
    if(num_dets_to_consider > 1) {
      auto inv_alph_cumul = inv_alph_masks.index({Slice(None,(num_dets_to_consider - 1))}).cumprod(0);
      auto masks_color_cumul = masks_color.index({Slice(1, None)}) * inv_alph_cumul;
      masks_color_summand += masks_color_cumul.sum(0);
    }
   
    auto rst = (img_gpu * inv_alph_masks.prod(0) + masks_color_summand) * 255.;
    rst=  rst.toType(c10::ScalarType::Char);
    masks_color_summand = masks_color_summand.permute({2, 0, 1});
    masks_color_summand = masks_color_summand.index({0});
    masks_color_summand = (masks_color_summand* 255.).toType(c10::ScalarType::Char);
   
    
    for (int i = 0;i<num_dets_to_consider;i++) {
      auto pos = boxes.index({i, Slice(None, None)});
      int x1 = pos[0].item<int>();
      int y1 = pos[1].item<int>();
      int x2 = pos[2].item<int>();
      int y2 = pos[3].item<int>();
      auto score = scores[i].item<float>();
      int class_id = classes[i].item<int>();
      double r = (double)(vdetect[i][0][0][0][0].item<float>());
      double g = (double)(vdetect[i][0][0][0][1].item<float>());
      double b = (double)(vdetect[i][0][0][0][2].item<float>());
      /*
      cv::rectangle(rst_mat, {x1, y1}, {x2, y2},
                    cv::Scalar{r,g,b},
                    1);*/
      string score_text = std::to_string(class_id) +":"+  std::to_string(score);
      int baseline = 0;
      auto text_size = cv::getTextSize(score_text, cv::FONT_HERSHEY_DUPLEX,0.6,1,&baseline);
      auto text_pt = cv::Size(x1, y1 - 3);
      //cv::putText(rst_mat, score_text.c_str(), text_pt, cv::FONT_HERSHEY_DUPLEX, 0.6,{255,255,255} , 1, cv::LINE_AA);
      bbox_t& box_t =segm.candidates[i];
      box_t.obj_id = class_id;
      box_t.prob = score;
      box_t.x = x1;
      box_t.y = y1;
      box_t.w = (x2 - x1);
      box_t.h = (y2 - y1); 
      box_t.x_3d = (torch::tensor(r * mask_alpha * 255.).toType(c10::ScalarType::Char).item<float>());
      box_t.y_3d = (torch::tensor(g * mask_alpha * 255.).toType(c10::ScalarType::Char).item<float>());
      box_t.z_3d = (torch::tensor(b * mask_alpha * 255.).toType(c10::ScalarType::Char).item<float>());
   
    }

    //rst_mat.convertTo(rst_mat, CV_8UC3, 255);
    //matToImageinfo(mask_b, segm.mask_image);
   // matToImageinfo(rst_mat, segm.display_image);
  
    tensor2dToImageInfo(rst.to(at::kCPU), segm.display_image);
    tensor2dToImageInfo(masks_color_summand.to(at::kCPU), segm.mask_image);
    

    segm.cnt = num_dets_to_consider;
 
    } catch (const c10::Error& e) {
      std::cout << e.msg() << endl;
    }
  };
  
  auto loc_data = torch::from_blob(loc_name, {batchSize, view_size, 4}).to(default_dev);
  auto conf_data = torch::from_blob(conf_name, {batchSize, view_size, num_classes}).to(default_dev);
  auto mask_data = torch::from_blob(mask_name, {batchSize, view_size, 32}).to(default_dev);
  auto priors_data = torch::from_blob(priors_name, {view_size, 4}).to(default_dev);
  auto proto_data = torch::from_blob(proto_name, {batchSize, proto_size, proto_size, 32}).to(default_dev);
 
  int num_priors = priors_data.size(0);
  auto conf_preds = conf_data.view({batchSize, num_priors, num_classes}).transpose(2, 1).contiguous();
 // vector<vector<torch::Tensor>> preds;
  for (int batch_idx = 0; batch_idx < batchSize; batch_idx++) {
    try
    {
      auto decoded_boxes = decode(loc_data[batch_idx], priors_data);
      auto pred = detect(batch_idx, conf_preds, decoded_boxes, mask_data);

      if (pred.size() == 0 || proto_data.size(0)-1 < batch_idx) continue;

      pred.push_back(proto_data[batch_idx]);
      segm_t_container& segm = rst_container[batch_idx];
     
      prep_display(pred, origin_image[batch_idx], segm);
      
    } catch (const c10::Error& e) {
      std::cout << e.msg() << endl;
    }
  
  // preds.push_back(pred);
  }
  rst_container.cnt = batchSize;
  return batchSize;
 }
