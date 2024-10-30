#ifndef TENSOR_OPS_CUH
#define TENSOR_OPS_CUH

#include <cuda_runtime.h>
#include "cuda_utils.cuh"

// Tensor operations for ML workloads
struct TensorDescriptor {
    size_t* dims;
    int ndims;
    size_t total_size;
    void* data;
    cudaDataType_t dtype;
};

extern "C" {
    // Tensor creation and management
    cudaError_t cuda_tensor_create(TensorDescriptor** tensor, int ndims, size_t* dims, cudaDataType_t dtype);
    cudaError_t cuda_tensor_destroy(TensorDescriptor* tensor);
    cudaError_t cuda_tensor_reshape(TensorDescriptor* tensor, int ndims, size_t* dims);
    
    // Basic operations
    cudaError_t cuda_tensor_add(TensorDescriptor* a, TensorDescriptor* b, TensorDescriptor* c);
    cudaError_t cuda_tensor_multiply(TensorDescriptor* a, TensorDescriptor* b, TensorDescriptor* c);
    cudaError_t cuda_tensor_scale(TensorDescriptor* a, float scale, TensorDescriptor* out);
    
    // Neural network operations
    cudaError_t cuda_tensor_relu(TensorDescriptor* input, TensorDescriptor* output);
    cudaError_t cuda_tensor_sigmoid(TensorDescriptor* input, TensorDescriptor* output);
    cudaError_t cuda_tensor_tanh(TensorDescriptor* input, TensorDescriptor* output);
    cudaError_t cuda_tensor_softmax(TensorDescriptor* input, TensorDescriptor* output);
    
    // Gradient operations
    cudaError_t cuda_tensor_backward_relu(TensorDescriptor* input, TensorDescriptor* grad_output, TensorDescriptor* grad_input);
    cudaError_t cuda_tensor_backward_sigmoid(TensorDescriptor* input, TensorDescriptor* grad_output, TensorDescriptor* grad_input);
    cudaError_t cuda_tensor_backward_tanh(TensorDescriptor* input, TensorDescriptor* grad_output, TensorDescriptor* grad_input);
    cudaError_t cuda_tensor_backward_softmax(TensorDescriptor* input, TensorDescriptor* grad_output, TensorDescriptor* grad_input);
}

#endif // TENSOR_OPS_CUH
