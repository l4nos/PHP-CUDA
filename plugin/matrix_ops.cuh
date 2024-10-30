#ifndef MATRIX_OPS_CUH
#define MATRIX_OPS_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "cuda_utils.cuh"

extern "C" {
    cudaError_t cuda_batch_matrix_multiply_kernel(
        cublasHandle_t handle,
        const float* const array_a[],
        const float* const array_b[],
        float* const array_c[],
        int batch_size,
        int m, int n, int k,
        cudaStream_t stream
    );
}

#endif // MATRIX_OPS_CUH
