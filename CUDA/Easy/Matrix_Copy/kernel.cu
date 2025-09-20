#include "kernel.cuh"
#include <cuda_runtime.h>

__global__ void copy_matrix_kernel(const float4* A, 
                                   float4* B, 
                                   int total, 
                                   int total_vec4,
                                   const int remainder) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_vec4) {
        B[idx] = A[idx];
    }

    int base = total_vec4 * 4;
    int tailIdx = base + idx;

    if (tailIdx < total) {
        reinterpret_cast<float*>(B)[tailIdx] = reinterpret_cast<const float*>(A)[tailIdx];
    }
}

// A, B are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, float* B, int N) {
    int total = N * N;
    int total_vec4 = total / 4;
    int remainder = total % 4;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_vec4 + threadsPerBlock - 1) / threadsPerBlock;
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        reinterpret_cast<const float4*>(A), 
        reinterpret_cast<float4*>(B), 
        total,
        total_vec4,
        remainder
    );
    cudaDeviceSynchronize();
} 

