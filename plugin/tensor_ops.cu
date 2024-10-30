#include "tensor_ops.cuh"
#include <cuda_runtime.h>

// Helper function to calculate total size
size_t calculate_total_size(int ndims, size_t* dims) {
    size_t total = 1;
    for (int i = 0; i < ndims; i++) {
        total *= dims[i];
    }
    return total;
}

// Tensor creation and management
extern "C" cudaError_t cuda_tensor_create(
    TensorDescriptor** tensor,
    int ndims,
    size_t* dims,
    cudaDataType_t dtype
) {
    *tensor = new TensorDescriptor;
    (*tensor)->ndims = ndims;
    (*tensor)->dims = new size_t[ndims];
    memcpy((*tensor)->dims, dims, ndims * sizeof(size_t));
    
    (*tensor)->total_size = calculate_total_size(ndims, dims);
    (*tensor)->dtype = dtype;
    
    size_t bytes = (*tensor)->total_size * (dtype == CUDA_R_32F ? sizeof(float) : sizeof(double));
    return cudaMalloc(&(*tensor)->data, bytes);
}

extern "C" cudaError_t cuda_tensor_destroy(TensorDescriptor* tensor) {
    cudaError_t err = cudaFree(tensor->data);
    delete[] tensor->dims;
    delete tensor;
    return err;
}

// CUDA kernels for tensor operations
__global__ void tensor_add_kernel(float* a, float* b, float* c, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void tensor_relu_kernel(float* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] > 0 ? input[idx] : 0;
    }
}

__global__ void tensor_sigmoid_kernel(float* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

__global__ void tensor_tanh_kernel(float* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

// Implementation of tensor operations
extern "C" cudaError_t cuda_tensor_add(
    TensorDescriptor* a,
    TensorDescriptor* b,
    TensorDescriptor* c
) {
    if (a->total_size != b->total_size || a->total_size != c->total_size) {
        return cudaErrorInvalidValue;
    }
    
    int blockSize = 256;
    int numBlocks = (a->total_size + blockSize - 1) / blockSize;
    
    tensor_add_kernel<<<numBlocks, blockSize>>>(
        (float*)a->data,
        (float*)b->data,
        (float*)c->data,
        a->total_size
    );
    
    return cudaGetLastError();
}

extern "C" cudaError_t cuda_tensor_relu(
    TensorDescriptor* input,
    TensorDescriptor* output
) {
    if (input->total_size != output->total_size) {
        return cudaErrorInvalidValue;
    }
    
    int blockSize = 256;
    int numBlocks = (input->total_size + blockSize - 1) / blockSize;
    
    tensor_relu_kernel<<<numBlocks, blockSize>>>(
        (float*)input->data,
        (float*)output->data,
        input->total_size
    );
    
    return cudaGetLastError();
}

// Gradient computation kernels
__global__ void tensor_backward_relu_kernel(
    float* input,
    float* grad_output,
    float* grad_input,
    size_t size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = input[idx] > 0 ? grad_output[idx] : 0;
    }
}

extern "C" cudaError_t cuda_tensor_backward_relu(
    TensorDescriptor* input,
    TensorDescriptor* grad_output,
    TensorDescriptor* grad_input
) {
    if (input->total_size != grad_output->total_size ||
        input->total_size != grad_input->total_size) {
        return cudaErrorInvalidValue;
    }
    
    int blockSize = 256;
    int numBlocks = (input->total_size + blockSize - 1) / blockSize;
    
    tensor_backward_relu_kernel<<<numBlocks, blockSize>>>(
        (float*)input->data,
        (float*)grad_output->data,
        (float*)grad_input->data,
        input->total_size
    );
    
    return cudaGetLastError();
}
