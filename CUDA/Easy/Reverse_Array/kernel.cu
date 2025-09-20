#include "kernel.cuh"
#include <cuda_runtime.h>


__global__ void reverse_array(float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int reverse_idx = N - 1 - idx;

    float temp = input[idx];
    input[idx] = input[reverse_idx];
    input[reverse_idx] = temp;
}


// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N / 2 + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}