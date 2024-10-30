#include "cpu_ops.cuh"

extern "C" void cpu_matrix_multiply(
    const float* a,
    const float* b,
    float* c,
    int m, int n, int k
) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            float sum = 0.0f;
            for (int l = 0; l < n; l++) {
                sum += a[i * n + l] * b[l * k + j];
            }
            c[i * k + j] = sum;
        }
    }
}

extern "C" void cpu_convolution(
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
) {
    int output_height = (height + 2 * padding - filter_size) / stride + 1;
    int output_width = (width + 2 * padding - filter_size) / stride + 1;
    
    #pragma omp parallel for collapse(4)
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < output_height; h++) {
                for (int w = 0; w < output_width; w++) {
                    float sum = 0.0f;
                    for (int fh = 0; fh < filter_size; fh++) {
                        for (int fw = 0; fw < filter_size; fw++) {
                            int h_in = h * stride + fh - padding;
                            int w_in = w * stride + fw - padding;
                            if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                                sum += input[(b * channels + c) * height * width + h_in * width + w_in] *
                                      filter[c * filter_size * filter_size + fh * filter_size + fw];
                            }
                        }
                    }
                    output[(b * channels + c) * output_height * output_width + h * output_width + w] = sum;
                }
            }
        }
    }
}
