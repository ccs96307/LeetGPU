#include "solve.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int xId = blockIdx.x * blockDim.x + threadIdx.x;
    int yId = blockIdx.y * blockDim.y + threadIdx.y;

    // Transpose coordinates
    int transposeXId = blockIdx.y * blockDim.y + threadIdx.x;
    int transposeYId = blockIdx.x * blockDim.x + threadIdx.y;

    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE+1];

    if (xId < cols && yId < rows) {
        tile[threadIdx.y][threadIdx.x] = input[yId * cols + xId];
    }

    __syncthreads();

    if (transposeXId < rows && transposeYId < cols) {
        output[transposeYId * rows + transposeXId] = tile[threadIdx.x][threadIdx.y];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}

