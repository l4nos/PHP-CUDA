--TEST--
Memory management and leak detection
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip CUDA extension not loaded');
?>
--FILE--
<?php
// Test memory allocation and deallocation
function test_memory_management() {
    $sizes = [1024, 2048, 4096, 8192];
    $ptrs = [];
    
    // Allocate memory
    foreach ($sizes as $size) {
        $ptr = cuda_malloc($size);
        if ($ptr === false) {
            echo "Failed to allocate {$size} bytes\n";
            return false;
        }
        $ptrs[] = $ptr;
    }
    
    // Get memory info before free
    $free_before = cuda_memory_get_info()['free'];
    
    // Free memory
    foreach ($ptrs as $ptr) {
        if (!cuda_free($ptr)) {
            echo "Failed to free memory\n";
            return false;
        }
    }
    
    // Get memory info after free
    $free_after = cuda_memory_get_info()['free'];
    
    // Check for memory leaks
    if ($free_after < $free_before) {
        echo "Possible memory leak detected\n";
        return false;
    }
    
    return true;
}

// Test memory pool
function test_memory_pool() {
    // Initialize pool
    $pool = cuda_memory_pool_init(1024 * 1024); // 1MB initial size
    if ($pool === false) {
        echo "Failed to initialize memory pool\n";
        return false;
    }
    
    // Allocate from pool
    $ptrs = [];
    for ($i = 0; $i < 10; $i++) {
        $ptr = cuda_memory_pool_allocate($pool, 1024);
        if ($ptr === false) {
            echo "Failed to allocate from pool\n";
            return false;
        }
        $ptrs[] = $ptr;
    }
    
    // Get pool stats
    $stats = cuda_memory_pool_stats($pool);
    echo "Pool total size: {$stats['total_size']}\n";
    echo "Pool used size: {$stats['used_size']}\n";
    
    // Free some memory
    foreach (array_slice($ptrs, 0, 5) as $ptr) {
        if (!cuda_memory_pool_free($pool, $ptr)) {
            echo "Failed to free memory from pool\n";
            return false;
        }
    }
    
    // Check pool stats after free
    $stats = cuda_memory_pool_stats($pool);
    echo "Pool used size after partial free: {$stats['used_size']}\n";
    
    // Destroy pool
    if (!cuda_memory_pool_destroy($pool)) {
        echo "Failed to destroy memory pool\n";
        return false;
    }
    
    return true;
}

// Run tests
var_dump(test_memory_management());
var_dump(test_memory_pool());

?>
--EXPECT--
bool(true)
Pool total size: 1048576
Pool used size: 10240
Pool used size after partial free: 5120
bool(true)
