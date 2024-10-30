#ifndef CPU_OPS_CUH
#define CPU_OPS_CUH

extern "C" {
    void cpu_matrix_multiply(
        const float* a,
        const float* b,
        float* c,
        int m, int n, int k
    );
    
    void cpu_convolution(
        const float* input,
        const float* filter,
        float* output,
        int batch_size,
        int channels,
        int height,
        int width,
        int filter_size,
        int stride,
        int padding
    );
}

#endif // CPU_OPS_CUH
