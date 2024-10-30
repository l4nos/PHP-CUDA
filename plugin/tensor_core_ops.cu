#include "tensor_core_ops.cuh"
#include <cuda_fp16.h>

// Helper function to check Tensor Core support
bool check_tensor_core_support(cudaDeviceProp& prop) {
    return prop.major >= 7;  // Volta or newer
}

extern "C" cudaError_t cuda_tensorcore_matmul(
    cublasHandle_t handle,
    const void* A,
    const void* B,
    void* C,
    int m, int n, int k,
    TensorCoreConfig* config
) {
    if (!config->enabled) {
        return cudaErrorNotSupported;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m, n, k,
        &alpha,
        A, config->input_type, m,
        B, config->input_type, k,
        &beta,
        C, config->output_type, m,
        config->compute_type,
        config->algo
    );

    return (status == CUBLAS_STATUS_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
}

extern "C" cudaError_t cuda_mixed_precision_matmul(
    cublasHandle_t handle,
    const half* A,
    const half* B,
    float* C,
    int m, int n, int k
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m, n, k,
        &alpha,
        A, CUDA_R_16F, m,
        B, CUDA_R_16F, k,
        &beta,
        C, CUDA_R_32F, m,
        CUBLAS_COMPUTE_32F_FAST_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    return (status == CUBLAS_STATUS_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
}

extern "C" cudaError_t cuda_tensorcore_autotune(
    int m, int n, int k,
    TensorCoreConfig* config
) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    if (!check_tensor_core_support(prop)) {
        config->enabled = false;
        return cudaSuccess;
    }

    // Auto-select precision based on size
    if (m * n * k > 1024 * 1024) {  // Large matrix
        config->input_type = CUDA_R_16F;
        config->output_type = CUDA_R_32F;
        config->compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
    } else {  // Small matrix
        config->input_type = CUDA_R_32F;
        config->output_type = CUDA_R_32F;
        config->compute_type = CUBLAS_COMPUTE_32F;
    }

    // Select algorithm based on matrix size
    if (m % 8 == 0 && n % 8 == 0 && k % 8 == 0) {
        config->algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    } else {
        config->algo = CUBLAS_GEMM_DEFAULT;
    }

    config->enabled = true;
    return cudaSuccess;
}
