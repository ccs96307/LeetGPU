#include <cuda_runtime.h>
#define TILE_DIM 16


// A: (M, N)
// B: (N, K)
// C: (M, K)
__global__ void matrix_multiplication_kernel(const float* A,
                                             const float* B, 
                                             float* C, 
                                             int M, 
                                             int N, 
                                             int K) {   
    __shared__ float tileA[TILE_DIM][TILE_DIM + 1];
    __shared__ float tileB[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;

    #pragma unroll
    for (int t = 0; t < (N + TILE_DIM - 1) / TILE_DIM; ++t) {
        int offset = t * TILE_DIM;
        int _x = offset + threadIdx.x;
        int _y = offset + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = (_x < N && y < M) ? A[y * N + _x] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (x < K && _y < N) ? B[_y * K + x] : 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_DIM; ++i) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (x < K && y < M) {
        C[y * K + x] = sum;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
