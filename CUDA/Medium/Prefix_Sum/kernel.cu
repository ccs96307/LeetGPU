#include "kernel.cuh"
#include <cuda_runtime.h>


__global__ void prefixSumKernel(const float* input, 
                                float* output, 
                                float* block_sums, 
                                int N) {
    extern __shared__ float sdata[];

    int tId = threadIdx.x;
    int gId1 = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    int gId2 = gId1 + 1;
    int processNum = 2 * blockDim.x;

    sdata[2 * tId] = (gId1 < N) ? input[gId1] : 0.0f;
    sdata[2 * tId + 1] = (gId2 < N) ? input[gId2] : 0.0f;
    __syncthreads();

    // Up-Sweep
    for (int stride = 1; stride <= blockDim.x; stride <<= 1) {
        int idx = (tId + 1) * 2 * stride - 1;
        if (idx < processNum) {
            sdata[idx] += sdata[idx - stride];
        }
        __syncthreads();
    }

    // Root = 0
    if (tId == 0) {
        block_sums[blockIdx.x] = sdata[processNum - 1];
        sdata[processNum - 1] = 0.0f;
    }

    __syncthreads();

    // Down-Sweep
    for (int stride = blockDim.x; stride > 0; stride >>= 1) {
        int idx = (tId + 1) * 2 * stride - 1;
        if (idx < processNum) {
            float temp = sdata[idx - stride];
            sdata[idx - stride] = sdata[idx];
            sdata[idx] += temp;
        }

        __syncthreads();
    }

    if (gId1 < N) {
        output[gId1] = sdata[2 * tId] + input[gId1];
    }

    if (gId2 < N) {
        output[gId2] = sdata[2 * tId + 1] + input[gId2];
    }
}


__global__ void addBlockSumKernel(float* output, const float* scanned_block_sums, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    if (blockIdx.x > 0) {
        output[idx] += scanned_block_sums[blockIdx.x - 1];
    }
}


// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    if (N == 0) return;

    const int thread_num = 256;
    int process_num = thread_num * 2;
    dim3 threadsPerBlock(thread_num);
    dim3 blocksPerGrid((N + process_num - 1) / process_num);
    size_t smem = process_num * sizeof(float);

    // Malloc block_sums
    float* d_block_sums = nullptr;
    cudaMalloc((void**)&d_block_sums, blocksPerGrid.x * sizeof(float));
    
    // Stage 1: Scan and do prefix sum
    prefixSumKernel<<<blocksPerGrid, threadsPerBlock, smem>>>(input, output, d_block_sums, N);

    // If this task just have only one block, return
    if (blocksPerGrid.x == 1) {
        cudaFree(d_block_sums);
        cudaDeviceSynchronize();
        return;
    }

    // Stage 2: Recursive to Add the prefix sum of block_sums
    float* d_scanned_block_sums = nullptr;
    cudaMalloc((void**)&d_scanned_block_sums, blocksPerGrid.x * sizeof(float));
    solve(d_block_sums, d_scanned_block_sums, blocksPerGrid.x);

    // Stage 3: Add the results
    dim3 add_threads(process_num);
    dim3 add_blocks((N + add_threads.x - 1) / add_threads.x);
    addBlockSumKernel<<<add_blocks, add_threads>>>(output, d_scanned_block_sums, N);

    cudaFree(d_block_sums);
    cudaFree(d_scanned_block_sums);
    cudaDeviceSynchronize();
} 
