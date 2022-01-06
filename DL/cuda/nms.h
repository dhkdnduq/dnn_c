#pragma once
#include <driver_types.h>
namespace dnn {
namespace cuda {

int nms(int batchSize, const void *const *inputs, void *const *outputs,
        size_t count, int detections_per_im, float nms_thresh, void *workspace,
        size_t workspace_size, cudaStream_t stream);

}
}  // namespace dnn
