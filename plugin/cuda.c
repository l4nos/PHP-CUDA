#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "php.h"
#include "ext/standard/info.h"
#include "php_cuda.h"
#include <cuda_runtime.h>

ZEND_DECLARE_MODULE_GLOBALS(cuda)

/* Resource destructors */
static void cuda_stream_dtor(zend_resource *rsrc) {
    cuda_stream_resource *stream_res = (cuda_stream_resource*)rsrc->ptr;
    if (stream_res) {
        cudaStreamDestroy(stream_res->stream);
        efree(stream_res);
    }
}

static void cuda_memory_dtor(zend_resource *rsrc) {
    cuda_memory_resource *mem_res = (cuda_memory_resource*)rsrc->ptr;
    if (mem_res) {
        cudaSetDevice(mem_res->device_id);
        cudaFree(mem_res->ptr);
        efree(mem_res);
    }
}

/* Argument information */
ZEND_BEGIN_ARG_INFO_EX(arginfo_void, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_device, 0, 0, 1)
    ZEND_ARG_INFO(0, device_id)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_malloc, 0, 0, 1)
    ZEND_ARG_INFO(0, size)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_memcpy, 0, 0, 3)
    ZEND_ARG_INFO(0, dst)
    ZEND_ARG_INFO(0, src)
    ZEND_ARG_INFO(0, size)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_matrix_multiply, 0, 0, 3)
    ZEND_ARG_ARRAY_INFO(0, matrix_a, 0)
    ZEND_ARG_ARRAY_INFO(0, matrix_b, 0)
    ZEND_ARG_INFO(1, result)
ZEND_END_ARG_INFO()

/* Module entry */
static const zend_function_entry cuda_functions[] = {
    PHP_FE(cuda_device_count, arginfo_void)
    PHP_FE(cuda_device_properties, arginfo_device)
    PHP_FE(cuda_set_device, arginfo_device)
    PHP_FE(cuda_get_device, arginfo_void)
    PHP_FE(cuda_device_reset, arginfo_void)
    PHP_FE(cuda_device_synchronize, arginfo_void)
    PHP_FE(cuda_malloc, arginfo_malloc)
    PHP_FE(cuda_free, arginfo_device)
    PHP_FE(cuda_memcpy_host_to_device, arginfo_memcpy)
    PHP_FE(cuda_memcpy_device_to_host, arginfo_memcpy)
    PHP_FE(cuda_memcpy_device_to_device, arginfo_memcpy)
    PHP_FE(cuda_matrix_multiply, arginfo_matrix_multiply)
    PHP_FE(cuda_get_last_error, arginfo_void)
    PHP_FE(cuda_get_error_string, arginfo_void)
    PHP_FE_END
};

zend_module_entry cuda_module_entry = {
    STANDARD_MODULE_HEADER,
    PHP_CUDA_EXTNAME,
    cuda_functions,
    PHP_MINIT(cuda),
    PHP_MSHUTDOWN(cuda),
    PHP_RINIT(cuda),
    PHP_RSHUTDOWN(cuda),
    PHP_MINFO(cuda),
    PHP_CUDA_VERSION,
    STANDARD_MODULE_PROPERTIES
};

#ifdef COMPILE_DL_CUDA
ZEND_GET_MODULE(cuda)
#endif

/* Module initialization */
PHP_MINIT_FUNCTION(cuda)
{
    cudaError_t error = cudaSetDevice(0);
    if (error != cudaSuccess) {
        php_error_docref(NULL, E_WARNING, "CUDA initialization failed: %s", 
            cudaGetErrorString(error));
        return FAILURE;
    }

    /* Register resource types */
    le_cuda_stream = zend_register_list_destructors_ex(cuda_stream_dtor, NULL, 
        "CUDA Stream", module_number);
    le_cuda_memory = zend_register_list_destructors_ex(cuda_memory_dtor, NULL, 
        "CUDA Memory", module_number);

    /* Initialize module globals */
    CUDA_G(allow_async_operations) = 1;
    CUDA_G(default_device) = 0;
    CUDA_G(enable_error_checking) = 1;
    
    return SUCCESS;
}

PHP_MSHUTDOWN_FUNCTION(cuda)
{
    cudaDeviceReset();
    return SUCCESS;
}

PHP_RINIT_FUNCTION(cuda)
{
    CUDA_G(active_streams) = NULL;
    CUDA_G(allocated_memory) = NULL;
    return SUCCESS;
}

PHP_RSHUTDOWN_FUNCTION(cuda)
{
    if (CUDA_G(active_streams)) {
        zend_hash_destroy(CUDA_G(active_streams));
        FREE_HASHTABLE(CUDA_G(active_streams));
    }
    if (CUDA_G(allocated_memory)) {
        zend_hash_destroy(CUDA_G(allocated_memory));
        FREE_HASHTABLE(CUDA_G(allocated_memory));
    }
    return SUCCESS;
}

PHP_MINFO_FUNCTION(cuda)
{
    php_info_print_table_start();
    php_info_print_table_header(2, "CUDA Support", "enabled");
    php_info_print_table_row(2, "CUDA Version", PHP_CUDA_VERSION);
    
    int driver_version = 0;
    cudaDriverGetVersion(&driver_version);
    char driver_version_str[32];
    snprintf(driver_version_str, sizeof(driver_version_str), "%d.%d", 
        driver_version/1000, (driver_version%100)/10);
    php_info_print_table_row(2, "CUDA Driver Version", driver_version_str);
    
    int runtime_version = 0;
    cudaRuntimeGetVersion(&runtime_version);
    char runtime_version_str[32];
    snprintf(runtime_version_str, sizeof(runtime_version_str), "%d.%d", 
        runtime_version/1000, (runtime_version%100)/10);
    php_info_print_table_row(2, "CUDA Runtime Version", runtime_version_str);
    
    php_info_print_table_end();
}

/* Device Management Functions */
PHP_FUNCTION(cuda_device_count)
{
    int count;
    cudaError_t error = cudaGetDeviceCount(&count);
    CUDA_CHECK_ERROR_RET(error);
    RETURN_LONG(count);
}

PHP_FUNCTION(cuda_device_properties)
{
    zend_long device_id;
    cudaDeviceProp props;
    
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_LONG(device_id)
    ZEND_PARSE_PARAMETERS_END();
    
    cudaError_t error = cudaGetDeviceProperties(&props, device_id);
    CUDA_CHECK_ERROR_RET(error);
    
    array_init(return_value);
    add_assoc_string(return_value, "name", props.name);
    add_assoc_long(return_value, "totalGlobalMem", props.totalGlobalMem);
    add_assoc_long(return_value, "maxThreadsPerBlock", props.maxThreadsPerBlock);
    add_assoc_long(return_value, "multiProcessorCount", props.multiProcessorCount);
    add_assoc_long(return_value, "maxThreadsDim[0]", props.maxThreadsDim[0]);
    add_assoc_long(return_value, "maxThreadsDim[1]", props.maxThreadsDim[1]);
    add_assoc_long(return_value, "maxThreadsDim[2]", props.maxThreadsDim[2]);
    add_assoc_long(return_value, "maxGridSize[0]", props.maxGridSize[0]);
    add_assoc_long(return_value, "maxGridSize[1]", props.maxGridSize[1]);
    add_assoc_long(return_value, "maxGridSize[2]", props.maxGridSize[2]);
    add_assoc_long(return_value, "warpSize", props.warpSize);
}

/* Memory Management Functions */
PHP_FUNCTION(cuda_malloc)
{
    zend_long size;
    void* dev_ptr;
    
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_LONG(size)
    ZEND_PARSE_PARAMETERS_END();
    
    cudaError_t error = cudaMalloc(&dev_ptr, size);
    CUDA_CHECK_ERROR_RET(error);
    
    cuda_memory_resource *mem_res = emalloc(sizeof(cuda_memory_resource));
    mem_res->ptr = dev_ptr;
    mem_res->size = size;
    cudaGetDevice(&mem_res->device_id);
    
    RETURN_RES(zend_register_resource(mem_res, le_cuda_memory));
}

PHP_FUNCTION(cuda_free)
{
    zval *res;
    cuda_memory_resource *mem_res;
    
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_RESOURCE(res)
    ZEND_PARSE_PARAMETERS_END();
    
    mem_res = (cuda_memory_resource*)zend_fetch_resource(
        Z_RES_P(res), "CUDA Memory", le_cuda_memory);
    
    cudaError_t error = cudaFree(mem_res->ptr);
    CUDA_CHECK_ERROR_RET(error);
    
    zend_list_close(Z_RES_P(res));
    RETURN_TRUE;
}

/* Matrix Operations */
extern cudaError_t cuda_matrix_multiply_kernel_wrapper(
    const float* a, const float* b, float* c,
    int m, int n, int k,
    cudaStream_t stream
);

PHP_FUNCTION(cuda_matrix_multiply)
{
    zval *matrix_a, *matrix_b, *result;
    HashTable *ht_a, *ht_b;
    float *dev_a, *dev_b, *dev_c;
    float *host_a, *host_b, *host_c;
    int m, n, k;
    
    ZEND_PARSE_PARAMETERS_START(3, 3)
        Z_PARAM_ARRAY(matrix_a)
        Z_PARAM_ARRAY(matrix_b)
        Z_PARAM_ZVAL(result)
    ZEND_PARSE_PARAMETERS_END();
    
    ht_a = Z_ARRVAL_P(matrix_a);
    ht_b = Z_ARRVAL_P(matrix_b);
    
    m = zend_hash_num_elements(ht_a);
    if (m == 0) {
        php_error_docref(NULL, E_WARNING, "Matrix A is empty");
        RETURN_FALSE;
    }
    
    zval *row = zend_hash_index_find(ht_a, 0);
    if (Z_TYPE_P(row) != IS_ARRAY) {
        php_error_docref(NULL, E_WARNING, "Matrix A is not 2D");
        RETURN_FALSE;
    }
    
    n = zend_hash_num_elements(Z_ARRVAL_P(row));
    k = zend_hash_num_elements(ht_b);
    
    /* Validate matrix dimensions */
    if (n == 0 || k == 0) {
        php_error_docref(NULL, E_WARNING, "Invalid matrix dimensions");
        RETURN_FALSE;
    }
    
    /* Allocate host memory */
    host_a = (float*)emalloc(m * n * sizeof(float));
    host_b = (float*)emalloc(n * k * sizeof(float));
    host_c = (float*)emalloc(m * k * sizeof(float));
    
    /* Convert PHP arrays to C arrays */
    zval *element;
    int i, j;
    
    ZEND_HASH_FOREACH_NUM_KEY_VAL(ht_a, i, element) {
        if (Z_TYPE_P(element) != IS_ARRAY) {
            php_error_docref(NULL, E_WARNING, "Matrix A is not 2D");
            goto cleanup;
        }
        
        HashTable *row = Z_ARRVAL_P(element);
        if (zend_hash_num_elements(row) != n) {
            php_error_docref(NULL, E_WARNING, "Inconsistent row length in matrix A");
            goto cleanup;
        }
        
        zval *val;
        ZEND_HASH_FOREACH_NUM_KEY_VAL(row, j, val) {
            host_a[i * n + j] = (float)zval_get_double(val);
        } ZEND_HASH_FOREACH_END();
    } ZEND_HASH_FOREACH_END();
    
    ZEND_HASH_FOREACH_NUM_KEY_VAL(ht_b, i, element) {
        if (Z_TYPE_P(element) != IS_ARRAY) {
            php_error_docref(NULL, E_WARNING, "Matrix B is not 2D");
            goto cleanup;
        }
        
        HashTable *row = Z_ARRVAL_P(element);
        if (zend_hash_num_elements(row) != k) {
            php_error_docref(NULL, E_WARNING, "Inconsistent row length in matrix B");
            goto cleanup;
        }
        
        zval *val;
        ZEND_HASH_FOREACH_NUM_KEY_VAL(row, j, val) {
            host_b[i * k + j] = (float)zval_get_double(val);
        } ZEND_HASH_FOREACH_END();
    } ZEND_HASH_FOREACH_END();
    
    /* Allocate device memory */
    cudaError_t error;
    error = cudaMalloc((void**)&dev_a, m * n * sizeof(float));
    CUDA_CHECK_ERROR_GOTO(error, cleanup);
    
    error = cudaMalloc((void**)&dev_b, n * k * sizeof(float));
    CUDA_CHECK_ERROR_GOTO(error, cleanup_a);
    
    error = cudaMalloc((void**)&dev_c, m * k * sizeof(float));
    CUDA_CHECK_ERROR_GOTO(error, cleanup_b);
    
    /* Copy data to device */
    error = cudaMemcpy(dev_a, host_a, m * n * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR_GOTO(error, cleanup_c);
    
    error = cudaMemcpy(dev_b, host_b, n * k * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR_GOTO(error, cleanup_c);
    
    /* Call CUDA kernel */
    error = cuda_matrix_multiply_kernel_wrapper(dev_a, dev_b, dev_c, m, n, k, 0);
    CUDA_CHECK_ERROR_GOTO(error, cleanup_c);
    
    /* Copy result back to host */
    error = cudaMemcpy(host_c, dev_c, m * k * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR_GOTO(error, cleanup_c);
    
    /* Convert result to PHP array */
    array_init(result);
    for (i = 0; i < m; i++) {
        zval row;
        array_init(&row);
        for (j = 0; j < k; j++) {
            add_next_index_double(&row, (double)host_c[i * k + j]);
        }
        add_next_index_zval(result, &row);
    }
    
    /* Cleanup */
cleanup_c:
    cudaFree(dev_c);
cleanup_b:
    cudaFree(dev_b);
cleanup_a:
    cudaFree(dev_a);
cleanup:
    efree(host_a);
    efree(host_b);
    efree(host_c);
    
    if (error != cudaSuccess) {
        RETURN_FALSE;
    }
    RETURN_TRUE;
}

/* Error Handling Functions */
PHP_FUNCTION(cuda_get_last_error)
{
    cudaError_t error = cudaGetLastError();
    RETURN_LONG(error);
}

PHP_FUNCTION(cuda_get_error_string)
{
    zend_long error_code;
    
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_LONG(error_code)
    ZEND_PARSE_PARAMETERS_END();
    
    RETURN_STRING(cudaGetErrorString((cudaError_t)error_code));
}
