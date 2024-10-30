--TEST--
Multi-GPU operations and device management
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip CUDA extension not loaded');
$count = cuda_device_count();
if ($count < 2) die('skip Test requires at least 2 GPUs');
?>
--FILE--
<?php
// Test device affinity and optimal device selection
function test_device_management() {
    // Get device count
    $count = cuda_device_count();
    echo "Available devices: $count\n";
    
    // Get optimal device based on current load
    $optimal_device = cuda_get_optimal_device();
    if ($optimal_device === false) {
        echo "Failed to get optimal device\n";
        return false;
    }
    echo "Optimal device: $optimal_device\n";
    
    // Test device affinity
    if (!cuda_set_device_affinity($optimal_device)) {
        echo "Failed to set device affinity\n";
        return false;
    }
    
    // Get device statistics
    $stats = cuda_get_device_stats($optimal_device);
    if ($stats === false) {
        echo "Failed to get device stats\n";
        return false;
    }
    
    echo "Device memory utilization: {$stats['memory_utilization']}%\n";
    echo "Device compute utilization: {$stats['compute_utilization']}%\n";
    
    return true;
}

// Test multi-GPU matrix multiplication
function test_multi_gpu_computation() {
    $matrix_size = 1024;
    $matrix_a = array_fill(0, $matrix_size, array_fill(0, $matrix_size, 1.0));
    $matrix_b = array_fill(0, $matrix_size, array_fill(0, $matrix_size, 2.0));
    
    // Split computation across available devices
    $device_count = cuda_device_count();
    $rows_per_device = $matrix_size / $device_count;
    
    $streams = [];
    $results = [];
    
    // Create streams for each device
    for ($i = 0; $i < $device_count; $i++) {
        cuda_set_device($i);
        $stream = cuda_stream_create();
        if ($stream === false) {
            echo "Failed to create stream for device $i\n";
            return false;
        }
        $streams[$i] = $stream;
    }
    
    // Launch computations on each device
    for ($i = 0; $i < $device_count; $i++) {
        $start_row = $i * $rows_per_device;
        $sub_matrix_a = array_slice($matrix_a, $start_row, $rows_per_device);
        
        cuda_set_device($i);
        $result = [];
        if (!cuda_async_matrix_multiply($sub_matrix_a, $matrix_b, $result, $streams[$i])) {
            echo "Computation failed on device $i\n";
            return false;
        }
        $results[$i] = $result;
    }
    
    // Synchronize all devices
    if (!cuda_sync_devices()) {
        echo "Failed to synchronize devices\n";
        return false;
    }
    
    // Cleanup
    foreach ($streams as $stream) {
        cuda_stream_destroy($stream);
    }
    
    return true;
}

// Test device locking and thread safety
function test_device_locking() {
    $device = cuda_get_optimal_device();
    
    // Lock device
    if (!cuda_lock_device($device)) {
        echo "Failed to lock device\n";
        return false;
    }
    
    // Verify device is locked
    if (!cuda_is_device_locked($device)) {
        echo "Device should be locked\n";
        return false;
    }
    
    // Try to perform computation while locked
    $matrix_a = [[1.0, 2.0], [3.0, 4.0]];
    $matrix_b = [[5.0, 6.0], [7.0, 8.0]];
    $result = [];
    
    if (!cuda_matrix_multiply($matrix_a, $matrix_b, $result)) {
        echo "Computation failed while device was locked\n";
        return false;
    }
    
    // Unlock device
    if (!cuda_unlock_device($device)) {
        echo "Failed to unlock device\n";
        return false;
    }
    
    return true;
}

// Run tests
var_dump(test_device_management());
var_dump(test_multi_gpu_computation());
var_dump(test_device_locking());

?>
--EXPECTF--
Available devices: %d
Optimal device: %d
Device memory utilization: %f%%
Device compute utilization: %f%%
bool(true)
bool(true)
bool(true)
