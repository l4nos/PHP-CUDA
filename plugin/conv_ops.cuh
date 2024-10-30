#ifndef CONV_OPS_CUH
#define CONV_OPS_CUH

#include <cuda_runtime.h>
#include <cudnn.h>
#include "cuda_utils.cuh"

extern "C" {
    cudaError_t cuda_batch_convolution_kernel(
        cudnnHandle_t handle,
        const float* const input_arrays[],
        const float* const filter_arrays[],
        float* const output_arrays[],
        int batch_count,
        int batch_size,
        int in_channels,
        int height,
        int width,
        int filter_count,
        int filter_height,
        int filter_width,
        cudaStream_t stream
    );
}

#endif // CONV_OPS_CUH
