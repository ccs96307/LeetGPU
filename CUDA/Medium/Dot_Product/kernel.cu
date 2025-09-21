#include "kernel.cuh"
#include <cuda_runtime.h>


__forceinline__ __device__ float warp_sum_func(float val) {
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void dotProductKernel(const float* A, const float* B, float* result, int N) {
    extern __shared__ float warpSum[];
    
    int tId = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tId;
    int warpId = tId / warpSize;
    int laneId = tId % warpSize;

    float val = (idx < N) ? A[idx] * B[idx] : 0.0f;
    val = warp_sum_func(val);

    if (laneId == 0) {
        warpSum[warpId] = val;
    }

    __syncthreads();

    if (warpId == 0) {
        int warpNum = (blockDim.x + warpSize - 1) / warpSize;
        val = (laneId < warpNum) ? warpSum[laneId] : 0.0f;
        val = warp_sum_func(val);

        if (laneId == 0) {
            atomicAdd(result, val);
        }
    }
}


// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    const int _warpSize = 32;
    const int _warpNum = (threadsPerBlock.x + _warpSize - 1) / _warpSize;
    size_t smem = _warpNum * sizeof(float);

    dotProductKernel<<<blocksPerGrid, threadsPerBlock, smem>>>(A, B, result, N);
    cudaDeviceSynchronize();
}
