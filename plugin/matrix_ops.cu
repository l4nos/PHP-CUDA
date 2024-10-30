#include "matrix_ops.cuh"

extern "C" cudaError_t cuda_batch_matrix_multiply_kernel(
    cublasHandle_t handle,
    const float* const array_a[],
    const float* const array_b[],
    float* const array_c[],
    int batch_size,
    int m, int n, int k,
    cudaStream_t stream
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cublasStatus_t status = cublasSgemmBatched(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        k, m, n,
        &alpha,
        array_b, k,
        array_a, n,
        &beta,
        array_c, k,
        batch_size
    );
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        return cudaErrorUnknown;
    }
    
    if (stream == 0) {
        return cudaDeviceSynchronize();
    }
    
    return cudaSuccess;
}
