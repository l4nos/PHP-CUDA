--TEST--
cuBLAS operations and performance
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip CUDA extension not loaded');
?>
--FILE--
<?php
// Test basic cuBLAS operations
function test_cublas_basic() {
    // Create cuBLAS handle
    $handle = cuda_cublas_create();
    if ($handle === false) {
        echo "Failed to create cuBLAS handle\n";
        return false;
    }
    
    // Create test matrices
    $m = 1024;
    $n = 1024;
    $k = 1024;
    
    $matrix_a = array_fill(0, $m * k, 1.0);
    $matrix_b = array_fill(0, $k * n, 2.0);
    $matrix_c = array_fill(0, $m * n, 0.0);
    
    // Perform matrix multiplication using cuBLAS
    if (!cuda_cublas_matrix_multiply($handle, $matrix_a, $matrix_b, $matrix_c, $m, $n, $k)) {
        echo "cuBLAS matrix multiplication failed\n";
        return false;
    }
    
    // Test GEMM with different parameters
    $alpha = 2.0;
    $beta = 1.0;
    if (!cuda_cublas_gemm($handle, $matrix_a, $matrix_b, $matrix_c, $m, $n, $k, $alpha, $beta)) {
        echo "cuBLAS GEMM operation failed\n";
        return false;
    }
    
    // Cleanup
    cuda_cublas_destroy($handle);
    
    return true;
}

// Test cuBLAS performance
function test_cublas_performance() {
    $handle = cuda_cublas_create();
    $sizes = [128, 256, 512, 1024, 2048];
    $iterations = 10;
    
    foreach ($sizes as $size) {
        $matrix_a = array_fill(0, $size * $size, 1.0);
        $matrix_b = array_fill(0, $size * $size, 2.0);
        $matrix_c = array_fill(0, $size * $size, 0.0);
        
        $start_time = microtime(true);
        
        for ($i = 0; $i < $iterations; $i++) {
            if (!cuda_cublas_matrix_multiply($handle, $matrix_a, $matrix_b, $matrix_c, $size, $size, $size)) {
                echo "Performance test failed for size $size\n";
                return false;
            }
        }
        
        $end_time = microtime(true);
        $duration = $end_time - $start_time;
        $gflops = (2.0 * $size * $size * $size * $iterations) / ($duration * 1e9);
        
        echo "Size ${size}x${size}: $gflops GFLOPS\n";
    }
    
    cuda_cublas_destroy($handle);
    return true;
}

// Test cuBLAS batch operations
function test_cublas_batch() {
    $handle = cuda_cublas_create();
    
    // Create batch of small matrices
    $batch_size = 1000;
    $matrix_size = 16;
    
    $matrices_a = [];
    $matrices_b = [];
    $matrices_c = [];
    
    for ($i = 0; $i < $batch_size; $i++) {
        $matrices_a[] = array_fill(0, $matrix_size * $matrix_size, 1.0);
        $matrices_b[] = array_fill(0, $matrix_size * $matrix_size, 2.0);
        $matrices_c[] = array_fill(0, $matrix_size * $matrix_size, 0.0);
    }
    
    // Perform batch matrix multiplication
    if (!cuda_batch_gemm($handle, $matrices_a, $matrices_b, $matrices_c, 
                        $matrix_size, $matrix_size, $matrix_size, $batch_size)) {
        echo "Batch GEMM operation failed\n";
        return false;
    }
    
    cuda_cublas_destroy($handle);
    return true;
}

// Run tests
var_dump(test_cublas_basic());
var_dump(test_cublas_performance());
var_dump(test_cublas_batch());

?>
--EXPECTF--
Size 128x128: %f GFLOPS
Size 256x256: %f GFLOPS
Size 512x512: %f GFLOPS
Size 1024x1024: %f GFLOPS
Size 2048x2048: %f GFLOPS
bool(true)
bool(true)
bool(true)
