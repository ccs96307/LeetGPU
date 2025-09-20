#include "kernel.cuh"
#include <cuda_runtime.h>


__forceinline__ __device__ float warp_sum_func(float val) {
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reductionKernel(const float* input, float* output, const int N) {
    extern __shared__ float warpSum[];

    int tId = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tId;
    int warpNum = (blockDim.x + warpSize - 1) / warpSize;
    int warpId = tId / warpSize;
    int laneId = tId % warpSize;

    float val = (idx < N) ? input[idx] : 0.0f;
    val = warp_sum_func(val);

    if (laneId == 0) {
        warpSum[warpId] = val;
    }

    __syncthreads();

    if (warpId == 0) {
        val = (laneId < warpNum) ? warpSum[laneId] : 0.0f;
        val = warp_sum_func(val);

        if (laneId == 0) {
            atomicAdd(output, val);
        }
    }
}


// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {  
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);
    
    int _warpSize = 32;
    int warpNum = (threadsPerBlock.x + _warpSize - 1) / _warpSize;
    size_t smem = warpNum * sizeof(float);

    reductionKernel<<<blocksPerGrid, threadsPerBlock, smem>>>(input, output, N);
    cudaDeviceSynchronize();
}