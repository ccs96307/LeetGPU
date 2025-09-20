#include "kernel.cuh"
#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, 
                                      const float* kernel, 
                                      float* output,
                                      int input_size, 
                                      int kernel_size) {
    __shared__ float kernelData[2048];

    int tId = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tId;

    for (int i = tId; i < kernel_size; i += blockDim.x) {
        kernelData[i] = kernel[i];
    }

    __syncthreads();
    
    if (idx < input_size - kernel_size + 1) {
        for (int k = 0; k < kernel_size; ++k) {
            output[idx] += input[idx + k] * kernelData[k];
        }
    }
}


// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}