#include "kernel.cuh"

#include <float.h>
#include <cuda_runtime.h>



__forceinline__ __device__ int warp_sum_func(int val) {
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    return val;
}


__global__ void SubarraySumKernel(const int* input, int* output, int N, int S, int E) {
    extern __shared__ int warpSum[];

    int tId = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tId + S;
    int warpId = tId / warpSize;
    int laneId = tId % warpSize;

    int val = (idx <= E) ? input[idx] : 0.0f;
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
            atomicAdd(output, val);
        }
    }
}



// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int S, int E) {
    int elements = E - S + 1;
    
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((elements + threadsPerBlock.x - 1) / threadsPerBlock.x);

    int _warpSize = 32;
    int _warpNum = (threadsPerBlock.x + _warpSize - 1) / _warpSize;
    size_t smem = _warpNum * sizeof(int);

    SubarraySumKernel<<<blocksPerGrid, threadsPerBlock, smem>>>(input, output, N, S, E);
    cudaDeviceSynchronize();
}