#include "kernel.cuh"
#include <cuda_runtime.h>


__forceinline__ __device__ int warp_sum_func(int input_val) {
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        input_val += __shfl_down_sync(0xffffffff, input_val, offset);
    }

    return input_val;
}

__global__ void count_2d_equal_kernel(const int* input, int* output, int N, int M, int K) {
    extern __shared__ int warpSum[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * M + x;

    int localId = threadIdx.y * blockDim.x + threadIdx.x;
    int warpId = localId / warpSize;
    int laneId = localId % warpSize;

    int val = (idx < M * N && input[idx] == K) ? 1 : 0;
    val = warp_sum_func(val);

    if (laneId == 0) {
        warpSum[warpId] = val;
    }

    __syncthreads();

    if (warpId == 0) {
        int warpNum = (blockDim.x * blockDim.y + warpSize - 1) / warpSize;
        int partialVal = (laneId < warpNum) ? warpSum[laneId] : 0;
        partialVal = warp_sum_func(partialVal);

        if (laneId == 0) {
            atomicAdd(output, partialVal);
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K) {
    cudaMemset(output, 0, sizeof(int));

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    int _warpSize = 32;
    size_t smem = (threadsPerBlock.x * threadsPerBlock.y + _warpSize - 1) / _warpSize * sizeof(int);

    count_2d_equal_kernel<<<blocksPerGrid, threadsPerBlock, smem>>>(input, output, N, M, K);
    cudaDeviceSynchronize();
}
