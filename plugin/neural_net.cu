#include "neural_net.cuh"
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>

// Model management
extern "C" cudaError_t cuda_model_create(
    ModelDescriptor** model,
    float learning_rate
) {
    *model = new ModelDescriptor;
    (*model)->layers = nullptr;
    (*model)->num_layers = 0;
    (*model)->learning_rate = learning_rate;
    
    cudnnCreate(&(*model)->cudnn_handle);
    cublasCreate(&(*model)->cublas_handle);
    
    return cudaSuccess;
}

extern "C" cudaError_t cuda_model_destroy(ModelDescriptor* model) {
    for (int i = 0; i < model->num_layers; i++) {
        LayerDescriptor* layer = model->layers[i];
        
        // Free layer-specific resources
        switch (layer->type) {
            case LAYER_LINEAR:
                cuda_tensor_destroy(layer->weights);
                cuda_tensor_destroy(layer->bias);
                break;
            case LAYER_CONV2D:
                cudnnDestroyFilterDescriptor((cudnnFilterDescriptor_t)layer->forward_desc);
                cudnnDestroyConvolutionDescriptor((cudnnConvolutionDescriptor_t)layer->backward_desc);
                cuda_tensor_destroy(layer->weights);
                cuda_tensor_destroy(layer->bias);
                break;
            // Add cases for other layer types
        }
        
        cuda_tensor_destroy(layer->output);
        cuda_tensor_destroy(layer->grad_input);
        if (layer->grad_weights) cuda_tensor_destroy(layer->grad_weights);
        if (layer->grad_bias) cuda_tensor_destroy(layer->grad_bias);
        
        delete layer;
    }
    
    delete[] model->layers;
    cudnnDestroy(model->cudnn_handle);
    cublasDestroy(model->cublas_handle);
    delete model;
    
    return cudaSuccess;
}

// Layer creation helpers
extern "C" cudaError_t cuda_create_linear_layer(
    LayerDescriptor** layer,
    int in_features,
    int out_features
) {
    *layer = new LayerDescriptor;
    (*layer)->type = LAYER_LINEAR;
    
    // Create weight tensor
    size_t weight_dims[] = {out_features, in_features};
    cuda_tensor_create(&(*layer)->weights, 2, weight_dims, CUDA_R_32F);
    
    // Create bias tensor
    size_t bias_dims[] = {out_features};
    cuda_tensor_create(&(*layer)->bias, 1, bias_dims, CUDA_R_32F);
    
    // Initialize output and gradient tensors
    size_t output_dims[] = {out_features};
    cuda_tensor_create(&(*layer)->output, 1, output_dims, CUDA_R_32F);
    cuda_tensor_create(&(*layer)->grad_input, 1, weight_dims, CUDA_R_32F);
    cuda_tensor_create(&(*layer)->grad_weights, 2, weight_dims, CUDA_R_32F);
    cuda_tensor_create(&(*layer)->grad_bias, 1, bias_dims, CUDA_R_32F);
    
    return cudaSuccess;
}

extern "C" cudaError_t cuda_create_conv2d_layer(
    LayerDescriptor** layer,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding
) {
    *layer = new LayerDescriptor;
    (*layer)->type = LAYER_CONV2D;
    
    // Create filter descriptor
    cudnnFilterDescriptor_t filter_desc;
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnSetFilter4dDescriptor(
        filter_desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        out_channels,
        in_channels,
        kernel_size,
        kernel_size
    );
    (*layer)->forward_desc = filter_desc;
    
    // Create convolution descriptor
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(
        conv_desc,
        padding, padding,
        stride, stride,
        1, 1,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT
    );
    (*layer)->backward_desc = conv_desc;
    
    // Create weight tensor
    size_t weight_dims[] = {out_channels, in_channels, kernel_size, kernel_size};
    cuda_tensor_create(&(*layer)->weights, 4, weight_dims, CUDA_R_32F);
    
    // Create bias tensor
    size_t bias_dims[] = {out_channels};
    cuda_tensor_create(&(*layer)->bias, 1, bias_dims, CUDA_R_32F);
    
    return cudaSuccess;
}

// Forward pass implementation for different layer types
cudaError_t forward_linear(
    LayerDescriptor* layer,
    TensorDescriptor* input,
    cublasHandle_t handle
) {
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Perform matrix multiplication: output = weights * input + bias
    cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        layer->weights->dims[0],  // m: output features
        input->dims[0],          // n: batch size
        layer->weights->dims[1],  // k: input features
        &alpha,
        (float*)layer->weights->data,
        layer->weights->dims[0],
        (float*)input->data,
        input->dims[0],
        &beta,
        (float*)layer->output->data,
        layer->output->dims[0]
    );
    
    // Add bias
    cublasSaxpy(
        handle,
        layer->output->total_size,
        &alpha,
        (float*)layer->bias->data,
        1,
        (float*)layer->output->data,
        1
    );
    
    return cudaSuccess;
}

// Model forward pass
extern "C" cudaError_t cuda_model_forward(
    ModelDescriptor* model,
    TensorDescriptor* input
) {
    TensorDescriptor* layer_input = input;
    
    for (int i = 0; i < model->num_layers; i++) {
        LayerDescriptor* layer = model->layers[i];
        
        switch (layer->type) {
            case LAYER_LINEAR:
                forward_linear(layer, layer_input, model->cublas_handle);
                break;
            case LAYER_CONV2D:
                // Implement convolution forward pass
                break;
            case LAYER_RELU:
                cuda_tensor_relu(layer_input, layer->output);
                break;
            // Add cases for other layer types
        }
        
        layer_input = layer->output;
    }
    
    return cudaSuccess;
}

// Model persistence
extern "C" cudaError_t cuda_model_save(
    ModelDescriptor* model,
    const char* filename
) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) return cudaErrorInvalidValue;
    
    // Write model metadata
    fwrite(&model->num_layers, sizeof(int), 1, fp);
    fwrite(&model->learning_rate, sizeof(float), 1, fp);
    
    // Write each layer
    for (int i = 0; i < model->num_layers; i++) {
        LayerDescriptor* layer = model->layers[i];
        
        // Write layer type
        fwrite(&layer->type, sizeof(LayerType), 1, fp);
        
        // Write layer weights and biases
        size_t weights_size = layer->weights->total_size * sizeof(float);
        float* host_weights = new float[layer->weights->total_size];
        cudaMemcpy(host_weights, layer->weights->data, weights_size, cudaMemcpyDeviceToHost);
        fwrite(host_weights, sizeof(float), layer->weights->total_size, fp);
        delete[] host_weights;
        
        size_t bias_size = layer->bias->total_size * sizeof(float);
        float* host_bias = new float[layer->bias->total_size];
        cudaMemcpy(host_bias, layer->bias->data, bias_size, cudaMemcpyDeviceToHost);
        fwrite(host_bias, sizeof(float), layer->bias->total_size, fp);
        delete[] host_bias;
    }
    
    fclose(fp);
    return cudaSuccess;
}
