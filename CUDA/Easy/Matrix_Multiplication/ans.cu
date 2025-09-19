#include "solve.h"
#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(
    const float* A,
    const float* B, 
    float* C, 
    int M, 
    int N, 
    int K
) {   
    int xId = blockIdx.x * blockDim.x + threadIdx.x;
    int yId = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float sharedAreaA[16][16];
    __shared__ float sharedAreaB[16][16];

    float value = 0.0f;

    for (int i=0; i<(N+15)/16; ++i) {
        if (yId < M && (i * 16 + threadIdx.x) < N) {
            sharedAreaA[threadIdx.y][threadIdx.x] = A[yId * N + (i * 16 + threadIdx.x)];
        }
        else {
            sharedAreaA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (xId < K && (i * 16 + threadIdx.y) < N) {
            sharedAreaB[threadIdx.y][threadIdx.x] = B[(i * 16 + threadIdx.y) * K + xId];
        }
        else {
            sharedAreaB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int j=0; j<16; ++j) {
            value += sharedAreaA[threadIdx.y][j] * sharedAreaB[j][threadIdx.x];
        }

        __syncthreads();
    }

    if (xId < K && yId < M) {
        C[yId * K + xId] = value;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

