#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

// Include order matters - system headers first
#ifdef _WIN32
#include <windows.h>
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <string.h>
#include <stdio.h>

// Version compatibility check
#if CUDART_VERSION < 8000
#error "CUDA 8.0 or higher is required"
#endif

#if CUDNN_MAJOR < 7
#error "cuDNN 7.0 or higher is required"
#endif

// Error checking macros with more detailed information
#define CUDA_CHECK_ERROR(err) \
    do { \
        cudaError_t error = (err); \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s (%d): %s\n", \
                    __FILE__, __LINE__, cudaGetErrorName(error), error, \
                    cudaGetErrorString(error)); \
            return error; \
        } \
    } while(0)

#define CUBLAS_CHECK_ERROR(err) \
    do { \
        cublasStatus_t error = (err); \
        if (error != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error in %s:%d: %d\n", \
                    __FILE__, __LINE__, error); \
            return cudaErrorUnknown; \
        } \
    } while(0)

#define CUDNN_CHECK_ERROR(err) \
    do { \
        cudnnStatus_t error = (err); \
        if (error != CUDNN_STATUS_SUCCESS) { \
            fprintf(stderr, "cuDNN error in %s:%d: %s\n", \
                    __FILE__, __LINE__, cudnnGetErrorString(error)); \
            return cudaErrorUnknown; \
        } \
    } while(0)

// Thread-safe error buffer
static __thread char cuda_last_error_msg[1024];

// Utility functions
inline const char* get_last_cuda_error_msg() {
    return cuda_last_error_msg;
}

inline void set_last_cuda_error_msg(const char* msg) {
    strncpy(cuda_last_error_msg, msg, sizeof(cuda_last_error_msg) - 1);
    cuda_last_error_msg[sizeof(cuda_last_error_msg) - 1] = '\0';
}

// RAII-style initializer for CUDA API
class CudaInitializer {
public:
    CudaInitializer() {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to initialize CUDA: %s\n", 
                    cudaGetErrorString(err));
            throw err;
        }
    }
    ~CudaInitializer() {
        cudaDeviceReset();
    }
};

#endif // CUDA_UTILS_CUH
