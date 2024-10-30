--TEST--
Stress testing and error handling
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip CUDA extension not loaded');
?>
--FILE--
<?php
// Stress test matrix operations
function stress_test_matrix_ops() {
    $sizes = [128, 256, 512, 1024];
    $iterations = 100;
    
    foreach ($sizes as $size) {
        $matrix_a = array_fill(0, $size, array_fill(0, $size, 1.0));
        $matrix_b = array_fill(0, $size, array_fill(0, $size, 2.0));
        $result = [];
        
        $start_time = microtime(true);
        
        for ($i = 0; $i < $iterations; $i++) {
            if (!cuda_matrix_multiply($matrix_a, $matrix_b, $result)) {
                echo "Matrix multiplication failed at iteration $i\n";
                return false;
            }
        }
        
        $end_time = microtime(true);
        $duration = $end_time - $start_time;
        
        echo "Size {$size}x{$size}: {$iterations} iterations in {$duration} seconds\n";
    }
    
    return true;
}

// Test error handling and recovery
function test_error_handling() {
    // Test invalid device
    $result = cuda_set_device(999);
    if ($result !== false) {
        echo "Expected failure for invalid device\n";
        return false;
    }
    
    // Test invalid memory allocation
    $result = cuda_malloc(PHP_INT_MAX);
    if ($result !== false) {
        echo "Expected failure for invalid allocation\n";
        return false;
    }
    
    // Test error string
    $error = cuda_get_last_error();
    if ($error === 0) {
        echo "Expected non-zero error code\n";
        return false;
    }
    
    $error_string = cuda_get_error_string($error);
    echo "Error string: $error_string\n";
    
    // Test device reset recovery
    if (!cuda_device_reset()) {
        echo "Device reset failed\n";
        return false;
    }
    
    // Verify we can continue operations
    $matrix_a = [[1.0, 2.0], [3.0, 4.0]];
    $matrix_b = [[5.0, 6.0], [7.0, 8.0]];
    $result = [];
    
    if (!cuda_matrix_multiply($matrix_a, $matrix_b, $result)) {
        echo "Failed to recover after error\n";
        return false;
    }
    
    return true;
}

// Test concurrent operations
function test_concurrent_ops() {
    $streams = [];
    $results = [];
    
    // Create multiple streams
    for ($i = 0; $i < 4; $i++) {
        $stream = cuda_stream_create();
        if ($stream === false) {
            echo "Failed to create stream $i\n";
            return false;
        }
        $streams[] = $stream;
    }
    
    // Launch concurrent operations
    $matrix_a = [[1.0, 2.0], [3.0, 4.0]];
    $matrix_b = [[5.0, 6.0], [7.0, 8.0]];
    
    foreach ($streams as $i => $stream) {
        $result = [];
        if (!cuda_async_matrix_multiply($matrix_a, $matrix_b, $result, $stream)) {
            echo "Async operation failed on stream $i\n";
            return false;
        }
        $results[] = $result;
    }
    
    // Wait for completion
    foreach ($streams as $i => $stream) {
        if (!cuda_stream_synchronize($stream)) {
            echo "Stream synchronization failed for stream $i\n";
            return false;
        }
    }
    
    // Cleanup streams
    foreach ($streams as $stream) {
        cuda_stream_destroy($stream);
    }
    
    return true;
}

// Run stress tests
var_dump(stress_test_matrix_ops());
var_dump(test_error_handling());
var_dump(test_concurrent_ops());

?>
--EXPECTF--
Size 128x128: 100 iterations in %f seconds
Size 256x256: 100 iterations in %f seconds
Size 512x512: 100 iterations in %f seconds
Size 1024x1024: 100 iterations in %f seconds
bool(true)
Error string: %s
bool(true)
bool(true)
