#include "kernel.cuh"

#include <float.h>
#include <cuda_runtime.h>


__forceinline__ __device__ float warp_sum_func(float val) {
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}


__global__ void MSEKernel(const float* predictions, const float* targets, float* mse, int N) {
    extern __shared__ float warpSum[];
    
    int tId = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tId;
    int warpId = tId / warpSize;
    int laneId = tId % warpSize;

    float diff = 0.0f;
    if (idx < N) {
        diff = predictions[idx] - targets[idx];
        diff *= diff;
    }

    diff = warp_sum_func(diff);
    if (laneId == 0) {
        warpSum[warpId] = diff; 
    } 

    __syncthreads();

    if (warpId == 0) {
        int warpNum = (blockDim.x + warpSize - 1) / warpSize;
        float val = (laneId < warpNum) ? warpSum[laneId] : 0.0f;
        val = warp_sum_func(val);

        if (laneId == 0) {
            atomicAdd(mse, val);
        }
    }
}



// predictions, targets, mse are device pointers
extern "C" void solve(const float* predictions, const float* targets, float* mse, int N) {
    // Init
    cudaMemset(mse, 0, sizeof(float));

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    int _warpSize = 32;
    int _warpNum = (threadsPerBlock.x + _warpSize - 1) / _warpSize;
    size_t smem = _warpNum * sizeof(float);

    MSEKernel<<<blocksPerGrid, threadsPerBlock, smem>>>(predictions, targets, mse, N);
    cudaDeviceSynchronize();

    float h_mse;
    cudaMemcpy(&h_mse, mse, sizeof(float), cudaMemcpyDeviceToHost);

    h_mse /= N;
    cudaMemcpy(mse, &h_mse, sizeof(float), cudaMemcpyHostToDevice);
}
