#ifndef LOGGER_CUH
#define LOGGER_CUH

#include <cuda_runtime.h>
#include "cuda_utils.cuh"

// Log levels
enum LogLevel {
    LOG_DEBUG,
    LOG_INFO,
    LOG_WARNING,
    LOG_ERROR
};

// Log configuration
struct LogConfig {
    LogLevel level;
    bool enable_api_logging;
    bool enable_memory_logging;
    bool enable_kernel_logging;
    const char* log_file;
};

extern "C" {
    // Logger management
    cudaError_t cuda_logger_init(const LogConfig* config);
    cudaError_t cuda_logger_shutdown();
    
    // Logging functions
    void cuda_log_message(LogLevel level, const char* message);
    void cuda_log_api_call(const char* api_name, cudaError_t result);
    void cuda_log_memory_operation(const char* op_type, size_t size, cudaError_t result);
    void cuda_log_kernel_launch(const char* kernel_name, dim3 grid, dim3 block);
    
    // Debug utilities
    void cuda_debug_memory_info();
    void cuda_debug_device_info();
    void cuda_debug_kernel_info(const char* kernel_name);
}

// Macro for automatic API logging
#define CUDA_LOG_API(api_call) \
    do { \
        cudaError_t result = api_call; \
        cuda_log_api_call(#api_call, result); \
        if (result != cudaSuccess) return result; \
    } while(0)

#endif // LOGGER_CUH
