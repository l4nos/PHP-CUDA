--TEST--
Neural network operations and training
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip CUDA extension not loaded');
if (!defined('HAVE_CUDNN')) die('skip cuDNN not available');
?>
--FILE--
<?php
// Test basic neural network operations
function test_neural_ops() {
    // Create a simple network
    $model = cuda_model_create(0.001); // learning rate
    
    // Add layers
    cuda_model_add_layer($model, LAYER_LINEAR, ['in_features' => 784, 'out_features' => 512]);
    cuda_model_add_layer($model, LAYER_RELU, []);
    cuda_model_add_layer($model, LAYER_LINEAR, ['in_features' => 512, 'out_features' => 10]);
    
    // Create test input
    $batch_size = 32;
    $input = array_fill(0, $batch_size * 784, 0.1);
    $target = array_fill(0, $batch_size * 10, 0.0);
    
    // Forward pass
    $output = [];
    if (!cuda_model_forward($model, $input, $output)) {
        echo "Forward pass failed\n";
        return false;
    }
    
    // Backward pass
    if (!cuda_model_backward($model, $target)) {
        echo "Backward pass failed\n";
        return false;
    }
    
    // Update weights
    if (!cuda_model_update($model)) {
        echo "Weight update failed\n";
        return false;
    }
    
    // Save and load model
    if (!cuda_model_save($model, 'test_model.bin')) {
        echo "Model save failed\n";
        return false;
    }
    
    $loaded_model = cuda_model_load('test_model.bin');
    if ($loaded_model === false) {
        echo "Model load failed\n";
        return false;
    }
    
    // Cleanup
    cuda_model_destroy($model);
    cuda_model_destroy($loaded_model);
    unlink('test_model.bin');
    
    return true;
}

// Test convolution operations
function test_conv_ops() {
    // Create tensors
    $input_shape = [32, 3, 32, 32]; // NCHW format
    $filter_shape = [64, 3, 3, 3];  // OIHW format
    
    $input = array_fill(0, array_product($input_shape), 0.1);
    $filter = array_fill(0, array_product($filter_shape), 0.1);
    $output = [];
    
    // Perform convolution
    if (!cuda_cudnn_convolution_forward(
        $input, $filter, $output,
        32,  // batch_size
        3,   // in_channels
        32,  // height
        32,  // width
        64,  // filters
        3,   // kernel_size
        1,   // stride
        1    // padding
    )) {
        echo "Convolution failed\n";
        return false;
    }
    
    return true;
}

// Test batch processing
function test_batch_ops() {
    $batch_size = 128;
    $matrices = [];
    
    // Create batch of matrices
    for ($i = 0; $i < $batch_size; $i++) {
        $matrices[] = [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]]
        ];
    }
    
    $results = [];
    
    // Process batch
    if (!cuda_batch_matrix_multiply($matrices, $results)) {
        echo "Batch processing failed\n";
        return false;
    }
    
    return count($results) === $batch_size;
}

// Run tests
var_dump(test_neural_ops());
var_dump(test_conv_ops());
var_dump(test_batch_ops());

?>
--EXPECT--
bool(true)
bool(true)
bool(true)
