#pragma once
#include <driver_types.h>
namespace dnn {
namespace cuda {

int iou(const void *const *inputs, void *const *outputs, int num_boxes,
        int num_anchors, cudaStream_t stream);
} 
} 
