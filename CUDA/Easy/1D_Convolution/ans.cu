#include "solve.h"
#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float tile[];

    for (int i=threadIdx.x; i<kernel_size; i+=blockDim.x) {
        tile[i] = kernel[i];
    }

    __syncthreads();

    if (idx < input_size - kernel_size + 1) {
        float val = 0.0f;

        for (int i=0; i<kernel_size; ++i) {
            val += tile[i] * input[idx + i];
        }

        output[idx] = val;
    }
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    int sharedMemorySize = kernel_size * sizeof(float);

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemorySize>>>(input, kernel, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}

