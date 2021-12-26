#pragma once
#include "structure.h"

/************************************************************************/
/* Common Module                                                      */
/************************************************************************/
extern "C" DLDLL void release_image_info(image_info &info);
extern "C" DLDLL void release_segm_container(segm_t_container_rst_list& rst_list);


/************************************************************************/
/* PyTorch Module                                                       */
/************************************************************************/
extern "C" DLDLL bool torch_init(const char *configurationFilename,int gpu = 0);
extern "C" DLDLL bool torch_add_image_file(const char *filename, int gpu = 0);
extern "C" DLDLL bool torch_add_encoded_image(unsigned char *buf, const size_t data_length,int gpu = 0);
extern "C" DLDLL bool torch_add_buffer(unsigned char *buf, int buf_w, int buf_h,int buf_channel,int gpu = 0);
extern "C" DLDLL void torch_clear_buffer(int gpus = 0);
extern "C" DLDLL int  torch_effdet(bbox_t_container_rst_list& rst_list,int gpu = 0 ,bool is_clear_buffer = true);
extern "C" DLDLL int  torch_binary_classification(binary_rst_list& rst_list,int gpu = 0, bool is_clear_buffer = true);
extern "C" DLDLL int  torch_anomaly_detection(segm_t_container_rst_list &rst_list,int category ,int gpu = 0, bool is_clear_buffer = true);
extern "C" DLDLL int  torch_dispose(int gpu = 0);


/************************************************************************/
/* TensorRT Module                                                      */
/************************************************************************/

extern "C" DLDLL bool trt_init(const char *configurationFilename, int gpu = 0);
extern "C" DLDLL bool trt_add_image_file(const char *filename, int gpu = 0);
extern "C" DLDLL bool trt_add_encoded_image(unsigned char *buf, const size_t data_length, int gpu = 0);
extern "C" DLDLL bool trt_add_buffer(unsigned char *buf, int buf_w, int buf_h,int buf_channel, int gpu = 0);
extern "C" DLDLL void trt_clear_buffer(int gpu = 0);
extern "C" DLDLL int trt_category_classification(category_rst_list& rst_list, int gpu = 0,bool is_clear_buffer = true);
extern "C" DLDLL int trt_yolact(segm_t_container_rst_list& rst_list, int gpu = 0 , bool is_clear_buffer = true);
extern "C" DLDLL int trt_yolov5(bbox_t_container_rst_list& rst_list, int gpu = 0 , bool is_clear_buffer = true);
extern "C" DLDLL int trt_dispose(int gpu = 0);


