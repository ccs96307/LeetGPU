#include "kernel.cuh"
#include <cuda_runtime.h>

#define TILE_DIM 16

__global__ void matrix_transpose_kernel(const float* input, 
                                        float* output, 
                                        int rows, 
                                        int cols) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // Original IDs
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Save elements into tile
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }

    __syncthreads();

    // Transpose IDs
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}


// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
