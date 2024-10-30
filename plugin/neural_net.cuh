#ifndef NEURAL_NET_CUH
#define NEURAL_NET_CUH

#include <cuda_runtime.h>
#include <cudnn.h>
#include "cuda_utils.cuh"
#include "tensor_ops.cuh"

// Neural network layer types
typedef enum {
    LAYER_LINEAR,
    LAYER_CONV2D,
    LAYER_MAXPOOL,
    LAYER_RELU,
    LAYER_DROPOUT,
    LAYER_BATCHNORM
} LayerType;

// Layer descriptor
struct LayerDescriptor {
    LayerType type;
    void* params;        // Layer-specific parameters
    void* forward_desc;  // Forward pass descriptor
    void* backward_desc; // Backward pass descriptor
    TensorDescriptor* weights;
    TensorDescriptor* bias;
    TensorDescriptor* output;
    TensorDescriptor* grad_input;
    TensorDescriptor* grad_weights;
    TensorDescriptor* grad_bias;
};

// Neural network model
struct ModelDescriptor {
    LayerDescriptor** layers;
    int num_layers;
    float learning_rate;
    cudnnHandle_t cudnn_handle;
    cublasHandle_t cublas_handle;
};

extern "C" {
    // Model management
    cudaError_t cuda_model_create(ModelDescriptor** model, float learning_rate);
    cudaError_t cuda_model_destroy(ModelDescriptor* model);
    cudaError_t cuda_model_add_layer(ModelDescriptor* model, LayerType type, void* params);
    
    // Training operations
    cudaError_t cuda_model_forward(ModelDescriptor* model, TensorDescriptor* input);
    cudaError_t cuda_model_backward(ModelDescriptor* model, TensorDescriptor* grad_output);
    cudaError_t cuda_model_update(ModelDescriptor* model);
    
    // Model persistence
    cudaError_t cuda_model_save(ModelDescriptor* model, const char* filename);
    cudaError_t cuda_model_load(ModelDescriptor** model, const char* filename);
    
    // Layer creation helpers
    cudaError_t cuda_create_linear_layer(LayerDescriptor** layer, int in_features, int out_features);
    cudaError_t cuda_create_conv2d_layer(LayerDescriptor** layer, int in_channels, int out_channels, 
                                       int kernel_size, int stride, int padding);
    cudaError_t cuda_create_maxpool_layer(LayerDescriptor** layer, int kernel_size, int stride);
    cudaError_t cuda_create_batchnorm_layer(LayerDescriptor** layer, int num_features, float momentum);
}

#endif // NEURAL_NET_CUH
