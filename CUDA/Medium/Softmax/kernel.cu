#include "kernel.cuh"

#include <float.h>
#include <cuda_runtime.h>


__forceinline__ __device__ float warp_max_func(float val) {
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__forceinline__ __device__ float warp_sum_func(float val) {
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    } 
    return val;
}


__global__ void findMaximumKernel(const float* input, float* MAX, const int N) {
    extern __shared__ float buffer[];

    int tId = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tId;
    int warpNum = (blockDim.x + warpSize - 1) / warpSize;
    int warpId = tId / warpSize;
    int laneId = tId % warpSize;

    float tempMax = -FLT_MAX;
    for (int i = idx; i < N; i += blockDim.x) {
        tempMax = fmaxf(tempMax, input[i]);
    }

    tempMax = warp_max_func(tempMax);
    if (laneId == 0) {
        buffer[warpId] = tempMax;
    }

    __syncthreads();

    if (warpId == 0) {
        tempMax = (laneId < warpNum) ? buffer[laneId] : -FLT_MAX;
        tempMax = warp_max_func(tempMax);

        if (laneId == 0) {
            *MAX = tempMax;
        }
    }
}  


// softmax(x) = y_i = (exp(x_i) - MAX) / sum_{0..j}(exp(x_j) - MAX)
__global__ void softmax_kernel(const float* input, float* output, int N, const float* MAX) {
    extern __shared__ float buffer[];

    int tId = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warpNum = (blockDim.x + warpSize - 1) / warpSize;
    int warpId = tId / warpSize;
    int laneId = tId % warpSize;

    // Step 1: Find the maximum value... we complete it at the previous kernel
    // Step 2: Sum the (exp(x_j) - MAX)
    float val = (idx < N) ? expf(input[idx] - *MAX) : 0.0f;
    val = warp_sum_func(val);

    if (laneId == 0) {
        buffer[warpId] = val;
    }

    __syncthreads();

    __shared__ float SUM;
    if (tId == 0) {
        SUM = 0.0f;
    }
    
    if (warpId == 0) {
        val = (laneId < warpNum) ? buffer[laneId] : 0.0f;
        val = warp_sum_func(val);

        if (laneId == 0) {
            atomicAdd(&SUM, val);
        }
    }

    __syncthreads();

    // Step 3: Calculate every element
    if (idx < N) {
        output[idx] = expf(input[idx] - *MAX) / SUM;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    // Find maximum value of input
    int threadsPerBlock = 1024;
    int blocksPerGrid = 1;
    int _warpSize = 32;
    int _warpNum = (threadsPerBlock + _warpSize - 1) / _warpSize;
    size_t smem = _warpNum * sizeof(float);

    float *MAX;
    cudaMalloc((void**)&MAX, sizeof(float));

    findMaximumKernel<<<blocksPerGrid, threadsPerBlock, smem>>>(
        input,
        MAX,
        N
    );
    cudaDeviceSynchronize();

    // Softmax
    threadsPerBlock = 256;
    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;    
    _warpNum = (threadsPerBlock + _warpSize - 1) / _warpSize;
    smem = _warpNum * sizeof(float);

    softmax_kernel<<<blocksPerGrid, threadsPerBlock, smem>>>(input, output, N, MAX);
    cudaDeviceSynchronize();
}