#ifndef MEMORY_UTILS_CUH
#define MEMORY_UTILS_CUH

#include <cuda_runtime.h>
#include "cuda_utils.cuh"

// Memory types
enum MemoryType {
    MEMORY_PAGEABLE,
    MEMORY_PINNED,
    MEMORY_UNIFIED
};

struct MemoryConfig {
    MemoryType type;
    cudaMemoryAdvise advice;
    int preferred_location;  // device ID
    bool read_mostly;
};

extern "C" {
    // Pinned memory operations
    cudaError_t cuda_pinned_malloc(void** ptr, size_t size);
    cudaError_t cuda_pinned_free(void* ptr);
    
    // Unified memory operations
    cudaError_t cuda_unified_malloc(void** ptr, size_t size, MemoryConfig* config);
    cudaError_t cuda_unified_free(void* ptr);
    cudaError_t cuda_unified_prefetch(void* ptr, size_t size, int device);
    
    // Memory hints and optimization
    cudaError_t cuda_set_memory_hint(void* ptr, size_t size, cudaMemoryAdvise advice);
    cudaError_t cuda_optimize_memory_access(void* ptr, size_t size, int device);
    
    // Memory bandwidth test
    cudaError_t cuda_measure_memory_bandwidth(size_t size, float* bandwidth);
}

#endif // MEMORY_UTILS_CUH
