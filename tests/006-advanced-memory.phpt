--TEST--
Advanced memory operations and unified memory
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip CUDA extension not loaded');
?>
--FILE--
<?php
// Test unified memory operations
function test_unified_memory() {
    // Create memory configuration
    $config = [
        'type' => MEMORY_UNIFIED,
        'read_mostly' => true,
        'preferred_location' => 0  // GPU 0
    ];
    
    // Allocate unified memory
    $ptr = cuda_unified_malloc(1024 * 1024, $config);  // 1MB
    if ($ptr === false) {
        echo "Failed to allocate unified memory\n";
        return false;
    }
    
    // Test memory hints
    if (!cuda_set_memory_hint($ptr, 1024 * 1024, CUDA_MEM_ADVISE_SET_READ_MOSTLY)) {
        echo "Failed to set memory hint\n";
        return false;
    }
    
    // Test prefetching
    if (!cuda_unified_prefetch($ptr, 1024 * 1024, 0)) {  // Prefetch to GPU 0
        echo "Failed to prefetch memory\n";
        return false;
    }
    
    // Test memory access optimization
    if (!cuda_optimize_memory_access($ptr, 1024 * 1024, 0)) {
        echo "Failed to optimize memory access\n";
        return false;
    }
    
    // Measure memory bandwidth
    $bandwidth = 0.0;
    if (!cuda_measure_memory_bandwidth(1024 * 1024, $bandwidth)) {
        echo "Failed to measure memory bandwidth\n";
        return false;
    }
    echo "Memory bandwidth: $bandwidth GB/s\n";
    
    // Cleanup
    cuda_unified_free($ptr);
    
    return true;
}

// Test pinned memory operations
function test_pinned_memory() {
    // Allocate pinned memory
    $ptr = cuda_pinned_malloc(1024 * 1024);  // 1MB
    if ($ptr === false) {
        echo "Failed to allocate pinned memory\n";
        return false;
    }
    
    // Create test data
    $data = array_fill(0, 256 * 1024, 1.0);  // 1MB of floats
    
    // Test async memory operations
    $stream = cuda_stream_create();
    if ($stream === false) {
        echo "Failed to create CUDA stream\n";
        return false;
    }
    
    // Allocate device memory
    $d_ptr = cuda_malloc(1024 * 1024);
    if ($d_ptr === false) {
        echo "Failed to allocate device memory\n";
        return false;
    }
    
    // Perform async copy
    if (!cuda_async_memcpy($d_ptr, $ptr, 1024 * 1024, cudaMemcpyHostToDevice, $stream)) {
        echo "Async memory copy failed\n";
        return false;
    }
    
    // Wait for completion
    if (!cuda_stream_synchronize($stream)) {
        echo "Stream synchronization failed\n";
        return false;
    }
    
    // Cleanup
    cuda_pinned_free($ptr);
    cuda_free($d_ptr);
    cuda_stream_destroy($stream);
    
    return true;
}

// Test memory pool fragmentation handling
function test_memory_fragmentation() {
    // Initialize pool
    $pool = cuda_memory_pool_init(1024 * 1024);  // 1MB
    if ($pool === false) {
        echo "Failed to initialize memory pool\n";
        return false;
    }
    
    // Allocate many small blocks
    $ptrs = [];
    for ($i = 0; $i < 100; $i++) {
        $ptr = cuda_memory_pool_allocate($pool, 1024);  // 1KB each
        if ($ptr === false) {
            echo "Failed to allocate small block\n";
            return false;
        }
        $ptrs[] = $ptr;
    }
    
    // Free every other block to create fragmentation
    for ($i = 0; $i < count($ptrs); $i += 2) {
        if (!cuda_memory_pool_free($pool, $ptrs[$i])) {
            echo "Failed to free memory block\n";
            return false;
        }
    }
    
    // Try to allocate a large block
    $large_ptr = cuda_memory_pool_allocate($pool, 512 * 1024);  // 512KB
    if ($large_ptr === false) {
        echo "Failed to allocate large block after fragmentation\n";
        return false;
    }
    
    // Defragment pool
    if (!cuda_memory_pool_defragment($pool)) {
        echo "Failed to defragment memory pool\n";
        return false;
    }
    
    // Cleanup
    cuda_memory_pool_destroy($pool);
    
    return true;
}

// Run tests
var_dump(test_unified_memory());
var_dump(test_pinned_memory());
var_dump(test_memory_fragmentation());

?>
--EXPECTF--
Memory bandwidth: %f GB/s
bool(true)
bool(true)
bool(true)
