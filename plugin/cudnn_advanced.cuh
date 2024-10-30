#ifndef CUDNN_ADVANCED_CUH
#define CUDNN_ADVANCED_CUH

#include <cuda_runtime.h>
#include <cudnn.h>
#include "cuda_utils.cuh"

// RNN types
enum RNNType {
    RNN_RELU,
    RNN_TANH,
    RNN_LSTM,
    RNN_GRU
};

// RNN configuration
struct RNNConfig {
    RNNType type;
    int input_size;
    int hidden_size;
    int num_layers;
    bool bidirectional;
    float dropout;
};

// Normalization types
enum NormType {
    NORM_BATCH,
    NORM_LAYER,
    NORM_INSTANCE,
    NORM_GROUP
};

extern "C" {
    // RNN operations
    cudaError_t cuda_rnn_forward(
        cudnnHandle_t handle,
        const RNNConfig* config,
        const void* x,
        void* y,
        void* h,
        void* c,
        bool training
    );
    
    cudaError_t cuda_rnn_backward(
        cudnnHandle_t handle,
        const RNNConfig* config,
        const void* dy,
        void* dx,
        void* dh,
        void* dc
    );
    
    // Normalization operations
    cudaError_t cuda_normalization_forward(
        cudnnHandle_t handle,
        NormType type,
        const void* x,
        void* y,
        void* scale,
        void* bias,
        float epsilon,
        bool training
    );
    
    cudaError_t cuda_normalization_backward(
        cudnnHandle_t handle,
        NormType type,
        const void* dy,
        void* dx,
        void* dscale,
        void* dbias
    );
}

#endif // CUDNN_ADVANCED_CUH
