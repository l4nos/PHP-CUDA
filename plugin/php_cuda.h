#ifndef PHP_CUDA_H
#define PHP_CUDA_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

extern zend_module_entry cuda_module_entry;
#define phpext_cuda_ptr &cuda_module_entry

#define PHP_CUDA_VERSION "1.0.0"
#define PHP_CUDA_EXTNAME "cuda"

/* Memory pool configuration */
#define CUDA_MEMORY_POOL_INITIAL_SIZE (1024 * 1024 * 64)  // 64MB
#define CUDA_MEMORY_POOL_GROWTH_FACTOR 2
#define CUDA_MAX_MEMORY_POOLS 8

/* Multi-GPU configuration */
#define CUDA_MAX_DEVICES 8
#define CUDA_LOAD_BALANCE_THRESHOLD 0.8

/* Enhanced error handling macros with detailed messages */
#define CUDA_CHECK_ERROR(err) do { \
    if (err != cudaSuccess) { \
        char error_msg[256]; \
        snprintf(error_msg, sizeof(error_msg), "[%s:%d] CUDA error: %s", \
                 __FILE__, __LINE__, cudaGetErrorString(err)); \
        php_error_docref(NULL, E_WARNING, "%s", error_msg); \
        return; \
    } \
} while(0)

#define CUDA_CHECK_ERROR_RET(err) do { \
    if (err != cudaSuccess) { \
        char error_msg[256]; \
        snprintf(error_msg, sizeof(error_msg), "[%s:%d] CUDA error: %s", \
                 __FILE__, __LINE__, cudaGetErrorString(err)); \
        php_error_docref(NULL, E_WARNING, "%s", error_msg); \
        RETURN_FALSE; \
    } \
} while(0)

/* Resource management structures */
typedef struct _cuda_memory_block {
    void* ptr;
    size_t size;
    int device_id;
    bool in_use;
    struct _cuda_memory_block* next;
} cuda_memory_block;

typedef struct _cuda_memory_pool {
    cuda_memory_block* blocks;
    size_t total_size;
    size_t used_size;
    pthread_mutex_t mutex;
} cuda_memory_pool;

typedef struct _cuda_device_info {
    int device_id;
    size_t total_memory;
    size_t free_memory;
    float utilization;
    cudaStream_t compute_stream;
    cudaStream_t transfer_stream;
    cublasHandle_t cublas_handle;
    cudnnHandle_t cudnn_handle;
    cuda_memory_pool* memory_pool;
    pthread_mutex_t mutex;
} cuda_device_info;

typedef struct _cuda_context {
    cuda_device_info devices[CUDA_MAX_DEVICES];
    int num_devices;
    int current_device;
    bool gpu_available;
    pthread_mutex_t global_mutex;
} cuda_context;

/* Global module variables */
ZEND_BEGIN_MODULE_GLOBALS(cuda)
    cuda_context* ctx;
    zend_bool enable_cpu_fallback;
    zend_bool enable_error_checking;
    zend_bool enable_memory_pool;
    zend_long batch_size;
    HashTable* active_streams;
    HashTable* allocated_memory;
ZEND_END_MODULE_GLOBALS(cuda)

ZEND_EXTERN_MODULE_GLOBALS(cuda)

#define CUDA_G(v) ZEND_MODULE_GLOBALS_ACCESSOR(cuda, v)

/* CPU fallback functions */
void cpu_matrix_multiply(const float* a, const float* b, float* c, int m, int n, int k);
void cpu_convolution(const float* input, const float* filter, float* output,
                    int batch_size, int channels, int height, int width,
                    int filter_size, int stride, int padding);

/* Memory pool functions */
PHP_FUNCTION(cuda_memory_pool_init);
PHP_FUNCTION(cuda_memory_pool_destroy);
PHP_FUNCTION(cuda_memory_pool_allocate);
PHP_FUNCTION(cuda_memory_pool_free);
PHP_FUNCTION(cuda_memory_pool_stats);

/* Multi-GPU management */
PHP_FUNCTION(cuda_get_optimal_device);
PHP_FUNCTION(cuda_set_device_affinity);
PHP_FUNCTION(cuda_get_device_stats);
PHP_FUNCTION(cuda_sync_devices);

/* Batch processing */
PHP_FUNCTION(cuda_batch_matrix_multiply);
PHP_FUNCTION(cuda_batch_convolution_forward);
PHP_FUNCTION(cuda_batch_gemm);

/* Asynchronous operations */
PHP_FUNCTION(cuda_async_memcpy);
PHP_FUNCTION(cuda_async_matrix_multiply);
PHP_FUNCTION(cuda_async_convolution);
PHP_FUNCTION(cuda_stream_wait_event);
PHP_FUNCTION(cuda_stream_query);

/* Thread safety */
PHP_FUNCTION(cuda_lock_device);
PHP_FUNCTION(cuda_unlock_device);
PHP_FUNCTION(cuda_is_device_locked);

/* Configuration */
PHP_FUNCTION(cuda_set_cpu_fallback);
PHP_FUNCTION(cuda_get_cpu_fallback);
PHP_FUNCTION(cuda_set_memory_pool);
PHP_FUNCTION(cuda_get_memory_pool);

/* Existing function declarations... */
PHP_MINIT_FUNCTION(cuda);
PHP_MSHUTDOWN_FUNCTION(cuda);
PHP_RINIT_FUNCTION(cuda);
PHP_RSHUTDOWN_FUNCTION(cuda);
PHP_MINFO_FUNCTION(cuda);

/* Device Management */
PHP_FUNCTION(cuda_device_count);
PHP_FUNCTION(cuda_device_properties);
PHP_FUNCTION(cuda_set_device);
PHP_FUNCTION(cuda_get_device);
PHP_FUNCTION(cuda_device_reset);
PHP_FUNCTION(cuda_device_synchronize);

/* Memory Management */
PHP_FUNCTION(cuda_malloc);
PHP_FUNCTION(cuda_free);
PHP_FUNCTION(cuda_memcpy_host_to_device);
PHP_FUNCTION(cuda_memcpy_device_to_host);
PHP_FUNCTION(cuda_memcpy_device_to_device);
PHP_FUNCTION(cuda_memset);

/* Stream Management */
PHP_FUNCTION(cuda_stream_create);
PHP_FUNCTION(cuda_stream_destroy);
PHP_FUNCTION(cuda_stream_synchronize);

/* cuBLAS Operations */
PHP_FUNCTION(cuda_cublas_create);
PHP_FUNCTION(cuda_cublas_destroy);
PHP_FUNCTION(cuda_cublas_matrix_multiply);
PHP_FUNCTION(cuda_cublas_matrix_multiply_ex);
PHP_FUNCTION(cuda_cublas_gemm);

/* cuDNN Operations */
PHP_FUNCTION(cuda_cudnn_create);
PHP_FUNCTION(cuda_cudnn_destroy);
PHP_FUNCTION(cuda_cudnn_convolution_forward);
PHP_FUNCTION(cuda_cudnn_convolution_backward_data);
PHP_FUNCTION(cuda_cudnn_convolution_backward_filter);
PHP_FUNCTION(cuda_cudnn_pooling_forward);
PHP_FUNCTION(cuda_cudnn_pooling_backward);
PHP_FUNCTION(cuda_cudnn_activation_forward);
PHP_FUNCTION(cuda_cudnn_activation_backward);

/* Basic CUDA Operations */
PHP_FUNCTION(cuda_matrix_multiply);
PHP_FUNCTION(cuda_vector_add);

/* Error Handling */
PHP_FUNCTION(cuda_get_last_error);
PHP_FUNCTION(cuda_get_error_string);
PHP_FUNCTION(cuda_get_error_name);

#if defined(ZTS) && defined(COMPILE_DL_CUDA)
ZEND_TSRMLS_CACHE_EXTERN()
#endif

#endif /* PHP_CUDA_H */
