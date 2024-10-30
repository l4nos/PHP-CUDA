#ifndef PROFILER_CUH
#define PROFILER_CUH

#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include "cuda_utils.cuh"

// Performance profiling and monitoring
struct ProfilerEvent {
    const char* name;
    cudaEvent_t start;
    cudaEvent_t stop;
    float duration;
};

extern "C" {
    // Profiler management
    cudaError_t cuda_profiler_start();
    cudaError_t cuda_profiler_stop();
    
    // Event management
    cudaError_t cuda_event_create(ProfilerEvent** event, const char* name);
    cudaError_t cuda_event_destroy(ProfilerEvent* event);
    cudaError_t cuda_event_record_start(ProfilerEvent* event);
    cudaError_t cuda_event_record_stop(ProfilerEvent* event);
    float cuda_event_elapsed_time(ProfilerEvent* event);
    
    // Memory tracking
    cudaError_t cuda_memory_get_info(size_t* free, size_t* total);
    cudaError_t cuda_memory_get_peak_usage();
    
    // Performance metrics
    cudaError_t cuda_get_device_utilization();
    cudaError_t cuda_get_memory_utilization();
    cudaError_t cuda_get_kernel_metrics(const char* kernel_name);
}

// RAII-style profiler marker
class ProfilerMarker {
public:
    ProfilerMarker(const char* name) {
        nvtxRangePushA(name);
    }
    ~ProfilerMarker() {
        nvtxRangePop();
    }
};

#define PROFILE_SCOPE(name) ProfilerMarker __profiler_marker__(name)
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)

#endif // PROFILER_CUH
