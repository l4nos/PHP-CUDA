--TEST--
CUDA profiling and performance monitoring
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip CUDA extension not loaded');
if (!defined('HAVE_NVTX')) die('skip NVTX support not available');
?>
--FILE--
<?php
// Test basic profiling
function test_basic_profiling() {
    // Start profiler
    if (!cuda_profiler_start()) {
        echo "Failed to start profiler\n";
        return false;
    }
    
    // Create and start event
    $event = cuda_event_create("MatrixMultiply");
    if ($event === false) {
        echo "Failed to create event\n";
        return false;
    }
    
    // Record start
    cuda_event_record_start($event);
    
    // Perform some operations
    $matrix_a = array_fill(0, 1024 * 1024, 1.0);
    $matrix_b = array_fill(0, 1024 * 1024, 2.0);
    $result = [];
    
    cuda_matrix_multiply($matrix_a, $matrix_b, $result);
    
    // Record stop
    cuda_event_record_stop($event);
    
    // Get elapsed time
    $duration = cuda_event_elapsed_time($event);
    echo "Operation took: $duration ms\n";
    
    // Cleanup
    cuda_event_destroy($event);
    
    // Stop profiler
    cuda_profiler_stop();
    
    return true;
}

// Test memory tracking
function test_memory_tracking() {
    // Get initial memory info
    $free_start = 0;
    $total = 0;
    if (!cuda_memory_get_info($free_start, $total)) {
        echo "Failed to get memory info\n";
        return false;
    }
    
    // Allocate some memory
    $ptr = cuda_malloc(1024 * 1024 * 100);  // 100MB
    if ($ptr === false) {
        echo "Memory allocation failed\n";
        return false;
    }
    
    // Get memory info after allocation
    $free_after = 0;
    if (!cuda_memory_get_info($free_after, $total)) {
        echo "Failed to get memory info\n";
        return false;
    }
    
    $used = $free_start - $free_after;
    echo "Memory used: " . ($used / 1024 / 1024) . " MB\n";
    
    // Get peak memory usage
    cuda_memory_get_peak_usage();
    
    // Free memory
    cuda_free($ptr);
    
    return true;
}

// Test kernel metrics
function test_kernel_metrics() {
    // Start profiler
    cuda_profiler_start();
    
    // Create test data
    $size = 1024 * 1024;
    $matrix_a = array_fill(0, $size, 1.0);
    $matrix_b = array_fill(0, $size, 2.0);
    $result = [];
    
    // Profile matrix multiplication
    cuda_matrix_multiply($matrix_a, $matrix_b, $result);
    
    // Get kernel metrics
    cuda_get_kernel_metrics("matrix_multiply_kernel");
    
    // Get device utilization
    cuda_get_device_utilization();
    
    // Get memory utilization
    cuda_get_memory_utilization();
    
    // Stop profiler
    cuda_profiler_stop();
    
    return true;
}

// Run tests
var_dump(test_basic_profiling());
var_dump(test_memory_tracking());
var_dump(test_kernel_metrics());

?>
--EXPECTF--
Operation took: %f ms
Memory used: %f MB
bool(true)
bool(true)
bool(true)
