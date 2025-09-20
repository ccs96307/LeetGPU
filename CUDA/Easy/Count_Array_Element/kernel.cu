#include "kernel.cuh"
#include <cuda_runtime.h>


__forceinline__ __device__ int warp_sum_func(int val) {
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void count_equal_kernel(const int* input, int* output, int N, int K) {
    extern __shared__ int warpSum[];

    int tId = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tId;
    int warpNum = (blockDim.x + warpSize - 1) / warpSize;
    int warpId = tId / warpSize;
    int laneId = tId % warpSize;

    int val = (idx < N && input[idx] == K) ? 1 : 0;
    val = warp_sum_func(val);

    if (laneId == 0) {
        warpSum[warpId] = val;
    }

    __syncthreads();

    if (warpId == 0) {
        val = (laneId < warpNum) ? warpSum[laneId] : 0;
        val = warp_sum_func(val);

        if (laneId == 0) {
            atomicAdd(output, val);
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int K) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    int _warpSize = 32;
    int warpNum = (threadsPerBlock + _warpSize - 1) / _warpSize;
    size_t smem = warpNum * sizeof(int);

    count_equal_kernel<<<blocksPerGrid, threadsPerBlock, smem>>>(input, output, N, K);
    cudaDeviceSynchronize();
}