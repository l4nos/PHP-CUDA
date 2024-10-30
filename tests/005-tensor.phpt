--TEST--
Tensor operations and manipulations
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip CUDA extension not loaded');
?>
--FILE--
<?php
// Test tensor creation and basic operations
function test_tensor_ops() {
    // Create tensors
    $dims = [2, 3, 4];  // 2x3x4 tensor
    $tensor_a = cuda_tensor_create($dims, CUDA_R_32F);
    $tensor_b = cuda_tensor_create($dims, CUDA_R_32F);
    
    if ($tensor_a === false || $tensor_b === false) {
        echo "Failed to create tensors\n";
        return false;
    }
    
    // Fill with test data
    $data_a = array_fill(0, 24, 1.0);
    $data_b = array_fill(0, 24, 2.0);
    
    if (!cuda_tensor_copy_host_to_device($tensor_a, $data_a) ||
        !cuda_tensor_copy_host_to_device($tensor_b, $data_b)) {
        echo "Failed to copy data to device\n";
        return false;
    }
    
    // Test tensor addition
    $tensor_c = cuda_tensor_create($dims, CUDA_R_32F);
    if (!cuda_tensor_add($tensor_a, $tensor_b, $tensor_c)) {
        echo "Tensor addition failed\n";
        return false;
    }
    
    // Verify results
    $result = [];
    if (!cuda_tensor_copy_device_to_host($tensor_c, $result)) {
        echo "Failed to copy result back to host\n";
        return false;
    }
    
    // All values should be 3.0 (1.0 + 2.0)
    foreach ($result as $val) {
        if (abs($val - 3.0) > 0.0001) {
            echo "Unexpected result value: $val\n";
            return false;
        }
    }
    
    // Test tensor reshape
    $new_dims = [4, 6];  // 4x6 = 24 (same total size)
    if (!cuda_tensor_reshape($tensor_a, $new_dims)) {
        echo "Tensor reshape failed\n";
        return false;
    }
    
    // Test activation functions
    $tensor_d = cuda_tensor_create($dims, CUDA_R_32F);
    
    // ReLU
    if (!cuda_tensor_relu($tensor_a, $tensor_d)) {
        echo "ReLU operation failed\n";
        return false;
    }
    
    // Sigmoid
    if (!cuda_tensor_sigmoid($tensor_a, $tensor_d)) {
        echo "Sigmoid operation failed\n";
        return false;
    }
    
    // Cleanup
    cuda_tensor_destroy($tensor_a);
    cuda_tensor_destroy($tensor_b);
    cuda_tensor_destroy($tensor_c);
    cuda_tensor_destroy($tensor_d);
    
    return true;
}

// Test tensor gradients
function test_tensor_gradients() {
    $dims = [2, 3];  // 2x3 matrix
    $input = cuda_tensor_create($dims, CUDA_R_32F);
    $grad_output = cuda_tensor_create($dims, CUDA_R_32F);
    $grad_input = cuda_tensor_create($dims, CUDA_R_32F);
    
    // Fill with test data
    $input_data = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
    $grad_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    
    cuda_tensor_copy_host_to_device($input, $input_data);
    cuda_tensor_copy_host_to_device($grad_output, $grad_data);
    
    // Test ReLU gradient
    if (!cuda_tensor_backward_relu($input, $grad_output, $grad_input)) {
        echo "ReLU gradient computation failed\n";
        return false;
    }
    
    // Verify gradients
    $result = [];
    cuda_tensor_copy_device_to_host($grad_input, $result);
    
    // ReLU gradient should be 0 for negative inputs
    for ($i = 0; $i < count($input_data); $i++) {
        $expected = $input_data[$i] > 0 ? $grad_data[$i] : 0;
        if (abs($result[$i] - $expected) > 0.0001) {
            echo "Unexpected gradient value at $i: {$result[$i]}, expected $expected\n";
            return false;
        }
    }
    
    // Cleanup
    cuda_tensor_destroy($input);
    cuda_tensor_destroy($grad_output);
    cuda_tensor_destroy($grad_input);
    
    return true;
}

// Run tests
var_dump(test_tensor_ops());
var_dump(test_tensor_gradients());

?>
--EXPECT--
bool(true)
bool(true)
