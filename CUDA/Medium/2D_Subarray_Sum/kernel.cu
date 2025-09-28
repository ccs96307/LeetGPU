#include "kernel.cuh"

#include <float.h>
#include <cuda_runtime.h>



__forceinline__ __device__ int warp_sum_func(int val) {
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}



__global__ void SubarraySum2DKernel(const int* input, 
                                    int* output, 
                                    int N, 
                                    int M, 
                                    int S_ROW, 
                                    int E_ROW, 
                                    int S_COL, 
                                    int E_COL) {
    extern __shared__ int warpSum[];

    int rowId = blockIdx.x + S_ROW;
    if (rowId > E_ROW) {
        return;
    }

    int tId = threadIdx.x;
    int idx = tId + S_COL;
    int warpId = tId / warpSize;
    int laneId = tId % warpSize;

    int val = 0;
    for (int i = idx; i <= E_COL; i += blockDim.x) {
        val += input[rowId * M + i];
    }
    val = warp_sum_func(val);

    if (laneId == 0) {
        warpSum[warpId] = val;
    }

    __syncthreads();

    if (warpId == 0) {
        int warpNum = (blockDim.x + warpSize - 1) / warpSize;
        val = (laneId < warpNum) ? warpSum[laneId] : 0;
        val = warp_sum_func(val);

        if (laneId == 0) {
            atomicAdd(output, val);
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int S_ROW, int E_ROW, int S_COL, int E_COL) {
    int row = E_ROW - S_ROW + 1;

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid(row);

    int _warpSize = 32;
    int _warpNum = (threadsPerBlock.x + _warpSize - 1) / _warpSize;

    size_t smem = _warpNum * sizeof(int);

    SubarraySum2DKernel<<<blocksPerGrid, threadsPerBlock, smem>>>(input, output, N, M, S_ROW, E_ROW, S_COL, E_COL);
    cudaDeviceSynchronize();
}