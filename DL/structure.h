#pragma once

#define DLLEXPORT
#ifdef DLLEXPORT
#define DLDLL __declspec(dllexport)
#else
#define DLDLL __declspec(dllimport)
#endif

#define MAX_OBJECTS 10
#define INCREASE_STACK_SIZE
#ifdef INCREASE_STACK_SIZE
#define MAX_BATCH_SIZE 300 
#else
#define MAX_BATCH_SIZE 10 
#endif
#define MAX_CLASS 100

using namespace std;
struct image_info {
  unsigned char* data  = nullptr;
  int size =0;
  void clear () {
    if(data != nullptr) {
      delete data;
      data = nullptr;
    }
    size = 0;
  }
  void gen(vector<unsigned char>& buf) {
    clear();
    size = (int)buf.size();
    data = (unsigned char*)calloc(size, sizeof(unsigned char));
    std::copy(buf.begin(), buf.end(), data);
  }
};


struct bbox_t {
 public:
  unsigned int x, y, w, h;
  float prob;
  unsigned int obj_id;
  float x_3d, y_3d, z_3d;
};

struct bbox_t_container {
  int cnt = 0;
  bbox_t candidates[MAX_OBJECTS];
  bbox_t& operator[](int index) {
    if (index >= MAX_OBJECTS) {
      cout << "Array index out of bound, exiting";
      exit(0);
    }
    return candidates[index];
  }
  void clear() {
    cnt = 0;
  }
};

struct bbox_t_container_rst_list {
  int cnt = 0;
  bbox_t_container container_list[MAX_BATCH_SIZE];
  bbox_t_container& operator[](int index) {
    if (index >= MAX_BATCH_SIZE) {
      cout << "Array index out of bound, exiting";
      exit(0);
    }
    return container_list[index];
  }
  void clear() {
    for (int i = 0; i < cnt; i++) {
      container_list[i].clear();
    }
  }
};

struct binary_rst_list {
  int cnt = 0;
  bool rst[MAX_BATCH_SIZE];
  bool& operator[](int index) {
    if (index >= MAX_BATCH_SIZE) {
      cout << "Array index out of bound, exiting";
      exit(0);
    }
    return rst[index];
  }
  void clear() { cnt = 0; }
};

struct category_rst_list {
  int cnt = 0;
  int rst[MAX_BATCH_SIZE];
  int& operator[](int index) {
    if (index >= MAX_BATCH_SIZE) {
      std::cout << "Array index out of bound, exiting";
      exit(0);
    }
    return rst[index];
  }
  void clear() {
    cnt = 0;
  }
};



struct segm_t_container {
  int cnt = 0;
  image_info mask_image;
  image_info display_image;
  bbox_t candidates[MAX_OBJECTS];
  bbox_t& operator[](int index) {
    if (index >= MAX_OBJECTS) {
      cout << "Array index out of bound, exiting";
      exit(0);
    }
    return candidates[index];
  }
  void clear() {
    cnt = 0;
    mask_image.clear();
    display_image.clear();
  }

};

struct segm_t_container_rst_list {
  int cnt = 0;
  segm_t_container container_list[MAX_BATCH_SIZE];
  segm_t_container& operator[](int index) {
    if (index >= MAX_BATCH_SIZE) {
      cout << "Array index out of bound, exiting";
      exit(0);
    }
    return container_list[index];
  }
  void clear() {
    for (int i = 0; i < cnt; i++) {
      container_list[i].clear();
    }
    cnt = 0;
  }
  ~segm_t_container_rst_list() {
    clear();
  }
};


struct anomaly_var {

  bool anomalyEnable = false;
  string anomalyFeatureFileName;
  float anomalyMaxScore = 0.f;
  float anomalyMinScore = 0.f;
  float anomalyThreshold = 0.f;
  bool  defect_extraction_enable = false;
  float defect_extraction_threshold = 0.8f;  //(0~1)
  float defect_extraction_jud_area_ratio = 0.1f;

};