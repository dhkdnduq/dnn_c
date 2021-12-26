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


bool dnn_module_torch::isinit_ = false;

namespace F = torch::nn::functional;
using namespace torch::nn;
using namespace cv::cuda;
dnn_module_torch::dnn_module_torch() {

}

dnn_module_torch::~dnn_module_torch() 
{
 
}
void dnn_module_torch::load_anomaly_detection() {

  torch::load(module_wideresnet_50_, cfg_.modelFileName);
  module_wideresnet_50_->eval();
  module_wideresnet_50_->to(at::kCUDA);
  auto anomaly_features = torch::jit::load(cfg_.anomalyFeatureFileName);
  anomaly_mean = anomaly_features.attr("mean").toTensor().to(at::kCPU);
  anomaly_conv_inv = anomaly_features.attr("conv_inv").toTensor().to(at::kCPU);
  
  
  anomaly_mean_mat =tensor2dToMat(anomaly_mean);
  anomaly_conv_inv_mat = tensor3dToMat(anomaly_conv_inv);



  int range = 1752;
  int size = 550;

  // anomaly_rand_index = torch::randint(0, range,
  // size).toType(c10::ScalarType::Long).to(at::kCPU);

  anomaly_rand_index = torch::tensor(
      {1632, 39,   990,  1740, 797,  661,  1698, 1623, 1782, 1064, 205,  911,
       1050, 749,  1654, 1478, 1652, 1466, 755,  795,  1626, 196,  1457, 1525,
       287,  1633, 1787, 209,  838,  317,  893,  970,  913,  1504, 1075, 1726,
       710,  1262, 1244, 180,  72,   171,  1303, 1576, 168,  1764, 741,  1411,
       684,  6,    759,  616,  373,  276,  326,  796,  585,  329,  186,  469,
       1680, 1193, 247,  941,  1396, 1150, 48,   1428, 840,  1650, 44,   1694,
       43,   505,  1540, 1696, 1762, 683,  704,  944,  1250, 356,  1125, 1416,
       1634, 1707, 872,  124,  137,  1685, 785,  201,  1488, 1221, 1736, 383,
       438,  1445, 70,   1716, 1265, 1639, 1346, 676,  1788, 1723, 951,  1056,
       1747, 1273, 742,  194,  525,  78,   1629, 142,  1570, 1627, 1222, 1739,
       257,  1084, 994,  1641, 663,  868,  1712, 724,  159,  1189, 1008, 1594,
       734,  1103, 350,  1731, 502,  1042, 1132, 1578, 1328, 860,  1689, 1045,
       1009, 1341, 277,  17,   453,  633,  765,  1550, 305,  1460, 1147, 419,
       220,  696,  524,  581,  978,  53,   1471, 689,  57,   1124, 784,  290,
       1655, 927,  1529, 439,  198,  294,  1505, 323,  128,  1111, 1185, 1334,
       11,   1211, 1295, 1714, 743,  999,  1283, 236,  426,  809,  926,  139,
       1321, 447,  1292, 997,  1546, 515,  1377, 688,  961,  1591, 546,  1458,
       1249, 884,  1338, 405,  1535, 1790, 832,  887,  565,  244,  643,  702,
       846,  719,  1596, 1082, 1233, 29,   1036, 264,  571,  1240, 81,   746,
       269,  512,  321,  260,  343,  573,  424,  184,  699,  870,  601,  457,
       1513, 1744, 158,  230,  1069, 513,  1097, 632,  134,  30,   932,  826,
       1554, 229,  278,  727,  1606, 804,  1486, 1391, 15,   853,  1357, 1063,
       1209, 1080, 1539, 1769, 1566, 1526, 938,  1597, 1491, 851,  800,  77,
       552,  772,  798,  358,  1434, 758,  852,  404,  473,  1568, 1200, 1085,
       1153, 1530, 1307, 928,  506,  918,  397,  18,   297,  624,  620,  1263,
       1686, 491,  478,  1624, 420,  352,  537,  686,  391,  202,  788,  1277,
       285,  738,  902,  187,  674,  723,  415,  1039, 1608, 679,  318,  470,
       1146, 712,  576,  237,  769,  602,  210,  1010, 963,  1410, 1016, 75,
       618,  924,  1326, 1166, 598,  1422, 231,  189,  1248, 71,   1038, 1371,
       1612, 1593, 441,  1757, 80,   708,  395,  874,  917,  412,  611,  1679,
       1439, 522,  340,  543,  328,  262,  1783, 233,  195,  50,   623,  362,
       1522, 622,  1614, 830,  1666, 726,  203,  677,  1386, 409,  1631, 678,
       915,  1374, 22,   138,  135,  652,  1062, 1502, 1444, 1279, 1534, 1395,
       774,  1141, 25,   488,  639,  371,  238,  1021, 507,  614,  1638, 1288,
       775,  781,  754,  1474, 1180, 1380, 149,  812,  1758, 1743, 1177, 84,
       1588, 199,  922,  1759, 1436, 834,  1480, 1628, 1237, 1446, 820,  109,
       204,  1538, 850,  564,  1363, 122,  863,  1771, 845,  1201, 1620, 251,
       605,  1745, 494,  19,   408,  815,  106,  980,  1314, 792,  273,  1332,
       402,  208,  1733, 808,  151,  435,  271,  898,  716,  1289, 1581, 575,
       1565, 780,  1692, 1369, 1385, 1342, 971,  331,  957,  909,  1706, 1372,
       1186, 1389, 1168, 480,  206,  1364, 1642, 1636, 308,  770,  617,  418,
       544,  720,  490,  1784, 1,    866,  1572, 510,  1414, 1653, 92,   163,
       1267, 1044, 1464, 973,  875,  1646, 1005, 1034, 590,  99,   1242, 956,
       1705, 154,  1390, 862,  1055, 756,  1427, 931,  709,  1644, 1256, 1126,
       539,  1670, 527,  958,  13,   1145, 166,  1604, 722,  1518, 1378, 514,
       744,  211,  1475, 1665, 1290, 879,  1217, 1319, 153,  1284});
       
  
  
  
  int max_count = omp_get_max_threads();
  omp_set_num_threads(max_count);

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


cv::cuda::GpuMat dnn_module_torch::tensor1dToMat(torch::Tensor t) {
  int H = t.size(0);
  at::ScalarType type = t.scalar_type();
  cv::Mat mat;
  GpuMat g;
  if (type == at::ScalarType::Float) {
    mat = cv::Mat(H, 1, CV_32F);
    std::memcpy(mat.data, t.data_ptr(), sizeof(float) * t.numel());
  }

  else if (type == at::ScalarType::Int) {
    mat = cv::Mat(H, 1, CV_32SC1);
    std::memcpy(t.data_ptr(), mat.data, sizeof(int) * t.numel());
  }
  g.upload(mat);
  return g;
}

cv::cuda::GpuMat dnn_module_torch::tensor2dToMat(torch::Tensor t) {
  int H = t.size(0);
  int W = t.size(1);
  at::ScalarType type = t.scalar_type();
  cv::Mat mat;
  GpuMat g;
  if (type == at::ScalarType::Float) {
    mat = cv::Mat(H, W, CV_32F);
    std::memcpy(mat.data, t.data_ptr(), sizeof(float) * t.numel());
  } 
 
  else if (type == at::ScalarType::Int) {
    mat = cv::Mat(H, W, CV_32SC1);
    std::memcpy(t.data_ptr(), mat.data, sizeof(int) * t.numel());
  }
  g.upload(mat);
  return g;
}


vector<cv::cuda::GpuMat> dnn_module_torch::tensor3dToMat(torch::Tensor t) {
  
  vector<cv::cuda::GpuMat> temp; 
  int W = t.size(0);
  int H = t.size(1);
  int C = t.size(2);
  
  for (int i = 0; i < C; i++) {
    auto tensor2d =  t.index({Slice(None, None), Slice(None, None), i});
    temp.push_back(tensor2dToMat(tensor2d));
  }
  return temp;
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

bool dnn_module_torch::load_model(string configpath) 
{ 
  if (!cfg_.load_config(configpath)) {
    return false;
  }
  loadlibrary();
  try
  {
    if (!cfg_.anomalyFeatureFileName.empty()) {
      load_anomaly_detection();
  
    } else {
      //module_ = torch::jit::load("D:\\deeplearning\\data\\wheat\\train\\traced_effdet_model.pt");
      //module_ = torch::jit::load("D:\\deeplearning\\data\\magnet\\logs\\trace\\traced-forward.pth");
    
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

int dnn_module_torch::predict_category_classification(unsigned char* buf, int buf_w, int buf_h, int buf_channel) {
  
 

  cv::Mat frame = cv::Mat(buf_h, buf_w, buf_channel == 3 ? CV_8UC3 : CV_8UC1, buf);
  torch::Tensor input;
  preprocessImage(frame, input);
 
  try
  {
    auto output = module_.forward({input}).toTensor();
    int ndim = output.ndimension();
    assert(ndim == 2);

    torch::ArrayRef<int64_t> sizes = output.sizes();
    int n_samples = sizes[0];
    int n_classes = sizes[1];

    auto cpu_output = output.to(at::kCPU);

    std::vector<float> unnorm_probs(
        cpu_output.data_ptr<float>(),
        cpu_output.data_ptr<float>() + (n_samples * n_classes));

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

    vector<float> probs;
    probs = __softmax(unnorm_probs);
    auto prob = std::max_element(probs.begin(), probs.end());
    auto label_idx = std::distance(probs.begin(), prob);
    float prob_float = *prob;
    return label_idx;
  } catch (const c10::Error& e) {
    std::cout << e.msg() << endl;
    return -1;
  }
  return 0;
}

bool dnn_module_torch::predict_binary_classification(unsigned char* buf, int buf_w, int buf_h, int buf_channel) {

  cv::Mat frame = cv::Mat(buf_h, buf_w, buf_channel == 3 ? CV_8UC3 : CV_8UC1, buf);
  torch::Tensor input;
  preprocessImage(frame, input);
  
  auto output = module_.forward({input}).toTensor();
  //cout << output << endl;


  output = torch::sigmoid(output);
  torch::Tensor classification = output.argmax(1);
  int32_t classificationWinner = classification.item().toInt();
  return classificationWinner == 1 ? true : false;
}


bool dnn_module_torch::preprocessImage(cv::Mat frame, torch::Tensor& out) {

  try {
    cv::cuda::GpuMat gpu_frame;
    gpu_frame.upload(frame);
    auto input_width = cfg_.dnn_width;
    auto input_height = cfg_.dnn_height;
    auto input_size = cv::Size(input_width, input_height);

    if (cfg_.dnn_chnnel == 3) {
      cv::cuda::GpuMat resized;
      cv::cuda::resize(gpu_frame, resized, input_size, 0, 0, cv::INTER_NEAREST);
      cv::cuda::GpuMat flt_image;

      if (frame.channels() == 1)
        cv::cuda::cvtColor(resized, resized, cv::COLOR_GRAY2BGR);
      
      resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
     
      int channel = flt_image.channels();
      // subtract mean
      vector<cv::cuda::GpuMat> img_channels;
      if (cfg_.mean_sub_enable) {
        // bgr
          cv::cuda::GpuMat subtract, divide;
        // bgr
        cv::cuda::subtract(flt_image,cv::Scalar(cfg_.mean[2], cfg_.mean[1], cfg_.mean[0]), subtract);
        cv::cuda::split(subtract, img_channels);
        cv::cuda::divide(img_channels[0], cfg_.std[2], img_channels[0]);
        cv::cuda::divide(img_channels[1], cfg_.std[1], img_channels[1]);
        cv::cuda::divide(img_channels[2], cfg_.std[0], img_channels[2]);
       
        cv::cuda::merge(img_channels, flt_image);
      }  

      //gpumat stride 때문에 from_blob에서 문제생김 ->mat변환
      cv::Mat cpumat(flt_image);
    
      auto size = cpumat.size();
      auto nChannels = cpumat.channels();
      auto tensor = torch::from_blob(cpumat.data,
                                     {1, size.height, size.width, nChannels});
      out = tensor.permute({0, 3, 1, 2}).contiguous().to(at::kCUDA);//bgr 2 rgb
    }
  }
  catch (const c10::Error& e) {
    std::cout << e.msg() << endl;
    return false;
  }
  catch (...) {
    ::cout << "failed preprocessImage\n";
    return false;
  }
  return true;
}

int dnn_module_torch::predict_object_detection_efficientdet(unsigned char* buf, int buf_w, int buf_h, int buf_channel,bbox_t_container& rst_container) 
{
  int cnt = 0;
  try {
    cv::Mat frame = cv::Mat(buf_h, buf_w, buf_channel == 3 ? CV_8UC3 : CV_8UC1, buf);
    torch::Tensor input;
    preprocessImage(frame, input);
   
    auto output = module_.forward({input}).toTensor();
    int batchsize = 1;
    output = output[batchsize - 1];  // batch
    /*
       scores = det[i].detach().cpu().numpy()[:,4]    #모든 행의 4번째
       indexes = np.where(scores > score_thres)[0] 
       boxes =
       boxes[indexes] boxes[:, 2] = boxes[:, 2] + boxes[:, 0] boxes[:, 3] =
       
    */
    auto boxes = output.index({Slice(None, None), Slice(None, 4)});
    auto scores = output.index({Slice(None, None), 4});
    auto labels = output.index({Slice(None, None), 5});
    auto indexs = (scores > cfg_.threshold).toType(c10::ScalarType::Bool);
    boxes = boxes.index(indexs);
    scores = scores.index(indexs);
    labels = labels.index(indexs);
    float dnn_width = cfg_.dnn_width;
    float dnn_height = cfg_.dnn_height;
    for (int i = 0; i < boxes.size(0) && i < C_SHARP_MAX_OBJECTS; i++) {
      bbox_t box;
      
      float lx = boxes[i][0].item<float>() / dnn_width * buf_w;
      float ly = boxes[i][1].item<float>() / dnn_width * buf_h;
      float rx = boxes[i][2].item<float>() / dnn_width * buf_w;
      float ry = boxes[i][3].item<float>() / dnn_width * buf_h;
     
      box.x = lx;
      box.y = ly;
      box.w = (rx - lx);
      box.h = (ry - ly);
      box.obj_id = labels[i].item<float>();
      box.prob = scores[i].item<float>();
      rst_container.candidates[i] = box;
     
      //cout << lx << " , " << ly << " , " << rx << " , " << ry << " , "<<box.prob<<" , "<<box.obj_id<<"\n ";
      
    }
    cnt = boxes.size(0);
  }
  catch (const c10::Error& e) {
    std::cout << e.msg() << endl;
  }
  catch (...) {
  ::cout << "failed predict_object_detection_efficientdet\n";
  }
  return cnt;
  
}



int dnn_module_torch::predict_object_detection_yolact(
    unsigned char* buf, int buf_w, int buf_h, int buf_channel,
    bbox_t_container& rst_container) {
  int cnt = 0;
  try {
    cv::Mat frame =
        cv::Mat(buf_h, buf_w, buf_channel == 3 ? CV_8UC3 : CV_8UC1, buf);
    torch::Tensor input;
    preprocessImage(frame, input);

    auto output = module_.forward({input}).toTensor();
    int batchsize = 1;
    output = output[batchsize - 1];  // batch
    /*
       scores = det[i].detach().cpu().numpy()[:,4]    #모든 행의 4번째
       indexes = np.where(scores > score_thres)[0]
       boxes =
       boxes[indexes] boxes[:, 2] = boxes[:, 2] + boxes[:, 0] boxes[:, 3] =

    */
    auto boxes = output.index({Slice(None, None), Slice(None, 4)});
    auto scores = output.index({Slice(None, None), 4});
    
    auto labels = output.index({Slice(None, None), 5});
    auto indexs = (scores > cfg_.threshold).toType(c10::ScalarType::Bool);
    

    boxes = boxes.index(indexs);
    scores = scores.index(indexs);
    labels = labels.index(indexs);
    float dnn_width = cfg_.dnn_width;
    float dnn_height = cfg_.dnn_height;
    for (int i = 0; i < boxes.size(0) && i < C_SHARP_MAX_OBJECTS; i++) {
      bbox_t box;

      float lx = boxes[i][0].item<float>() / dnn_width * buf_w;
      float ly = boxes[i][1].item<float>() / dnn_width * buf_h;
      float rx = boxes[i][2].item<float>() / dnn_width * buf_w;
      float ry = boxes[i][3].item<float>() / dnn_width * buf_h;

      box.x = lx;
      box.y = ly;
      box.w = (rx - lx);
      box.h = (ry - ly);
      box.obj_id = labels[i].item<float>();
      box.prob = scores[i].item<float>();
      rst_container.candidates[i] = box;

      // cout << lx << " , " << ly << " , " << rx << " , " << ry << " ,
      // "<<box.prob<<" , "<<box.obj_id<<"\n ";
    }
    cnt = boxes.size(0);
  } catch (const c10::Error& e) {
    std::cout << e.msg() << endl;
  } catch (...) {
    std::cout << "failed predict_object_detection_efficientdet\n";
  }
  return cnt;
}

torch::Tensor dnn_module_torch::mahalanobis(torch::Tensor u, torch::Tensor v,
                                            torch::Tensor cov_inv) {
  auto delta = u  - v;
  auto mul = torch::matmul(cov_inv, delta);
  auto m = torch::dot(delta, mul);
  /*
  cout << mul.mean().item<float>() << "," << u.mean().item<float>() << ","
       << cov_inv.mean().item<float>() << "," << v.mean().item<float>() << ","
       << m.mean().item<float>() << ","
       << delta.mean().item<float>() << endl;
       */
  return torch::sqrt(m);
 

}
bool dnn_module_torch::predict_anomaly_detection(unsigned char* buf, int buf_w, int buf_h, int buf_channel) 
{
  int cnt = 0;
  try {
    cv::Mat frame =
        cv::Mat(buf_h, buf_w, buf_channel == 3 ? CV_8UC3 : CV_8UC1, buf);
    torch::Tensor input;
    preprocessImage(frame, input);

    auto x = module_wideresnet_50_->conv1->forward(input);
    x = module_wideresnet_50_->bn1->forward(x).relu_();
    x = torch::max_pool2d(x, 3, 2, 1);

    auto output1 = module_wideresnet_50_->layer1->forward(x);
    auto output2 = module_wideresnet_50_->layer2->forward(output1);
    auto output3 = module_wideresnet_50_->layer3->forward(output2);

    output1 = torch::cat(output1, 0);
    output2 = torch::cat(output2, 0);
    output3 = torch::cat(output3, 0);

    auto embedding_vectors = output1;
    embedding_vectors = embedding_concat(embedding_vectors, output2);
    embedding_vectors =
        embedding_concat(embedding_vectors, output3).to(at::kCPU);
    embedding_vectors =
        torch::index_select(embedding_vectors, 1, anomaly_rand_index);

 
    int B = embedding_vectors.size(0);
    int C = embedding_vectors.size(1);
    int H = embedding_vectors.size(2);
    int W = embedding_vectors.size(3);

    embedding_vectors = embedding_vectors.view({B, C, H * W}).to(at::kCPU);
    ;  //[1,550,3136]
     auto embedding_vectors_mat = tensor2dToMat(embedding_vectors[0]);
  
    //begin_period();
    vector<double> dist_list(H * W);

    
#pragma omp parallel for
    for (int i = 0; i < H * W; i++) {
      auto sample =
          embedding_vectors[0].index({Slice(None, None), i}).to(at::kCUDA);
      auto mean = anomaly_mean.index({Slice(None, None), i});
      auto conv_inv =
          anomaly_conv_inv.index({Slice(None, None), Slice(None, None), i});
      auto dist = mahalanobis(sample, mean, conv_inv);
      dist_list[i] = dist.item<double>();
    }

    
    // float average = accumulate(dist_list.begin(), dist_list.end(), 0.0
    // )/dist_list.size(); cout << "The average is " << average << endl;
    
    // cout <<dist.mean() << endl;
    //end_period();
    /*
 #pragma omp parallel for
    for (int i = 0; i < H * W; i++) {
      auto sample = embedding_vectors_mat.col(i);
      auto mean = anomaly_mean_mat.col(i);;
      auto conv_inv = anomaly_conv_inv_mat[i];
      cv::cuda::GpuMat delta,multiply,m;
      cv::cuda::subtract(sample, mean, delta);
      cv::cuda::multiply(delta, conv_inv,multiply);
      cv::cuda::multiply(multiply,delta)

      double dist = cv::Mahalanobis(sample, mean, conv_inv);
      dist_list[i] = dist;
    }
    */
   
    auto dist = torch::tensor(dist_list);
   
    /* 별 차이없음
    dist = dist.reshape({H, W}).unsqueeze(0).unsqueeze(0); //
    [56,56]->[1,1,56,56] auto score_map =
    F::interpolate(dist,F::InterpolateFuncOptions().size(std::vector<int64_t>{cfg_.dnn_width,cfg_.dnn_height}).align_corners(false).mode(torch::kBilinear)).squeeze();
    cout <<score_map.mean().item<float>()<< endl;
    */

  } catch (const c10::Error& e) {
    std::cout << e.msg() << endl;
    return false;
  } catch (...) {
    std::cout << "failed predict_anomaly_detection\n";
    return false;
  }
  return true;
}