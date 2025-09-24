#include "kernel.cuh"

#include <float.h>
#include <cuda_runtime.h>


__global__ void histogrammingKernel(const int* input, int* histogram, int N, int num_bins) {
    extern __shared__ int sdata[];

    int tId = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tId;

    for (int i = tId; i < num_bins; tId += blockDim.x) {
        sdata[i] = 0;
    }

    __syncthreads();

    if (idx < N) {
        int val = input[idx];
        atomicAdd(&sdata[val], 1);
    }

    __syncthreads();

    for (int i = tId; i < num_bins; i += blockDim.x) {
        atomicAdd(&histogram[i], sdata[i]);
    }
}


// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    dim3 threadsPerBlock(1024);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);
    size_t smem = threadsPerBlock.x * sizeof(int);

    histogrammingKernel<<<blocksPerGrid, threadsPerBlock, smem>>>(input, histogram, N, num_bins);
    cudaDeviceSynchronize();
}
