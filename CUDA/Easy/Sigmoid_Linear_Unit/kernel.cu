#include "kernel.cuh"
#include <cuda_runtime.h>


__forceinline__ __device__ float sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__forceinline__ __device__ float SiLU(float x) {
    return x * sigmoid(x);
}

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    int idxB = idxA + halfN;

    if (idxA < halfN) {
        output[idxA] = SiLU(input[idxA]) * input[idxB];
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}