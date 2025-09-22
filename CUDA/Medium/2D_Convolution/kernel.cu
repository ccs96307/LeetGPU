#include "kernel.cuh"

#include <float.h>
#include <cuda_runtime.h>


__global__ void ConvolutionKernel(const float* input, 
                                    const float* kernel, 
                                    float* output,
                                    int input_rows, 
                                    int input_cols, 
                                    int kernel_rows, 
                                    int kernel_cols,
                                    int out_rows,
                                    int out_cols) {
    extern __shared__ float sdata[];

    int xtId = threadIdx.x;
    int ytId = threadIdx.y;
    int xId = blockIdx.x * blockDim.x + xtId;
    int yId = blockIdx.y * blockDim.y + ytId;

    int kId = ytId * kernel_cols + xtId;

    if (xtId < kernel_cols && ytId < kernel_rows) {
        sdata[kId] = kernel[kId];
    }
    __syncthreads();

    if (xId < out_cols && yId < out_rows) {
        float sum = 0.0f;
        for (int dy = 0; dy < kernel_rows; ++dy) {
            for (int dx = 0; dx < kernel_cols; ++dx) {
                sum += input[(yId + dy) * input_cols + (xId + dx)] * sdata[dy * kernel_cols + dx];
            }
        }

        output[yId * out_cols + xId] = sum;
    }
}


// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    dim3 threadsPerBlock(32, 32);

    int out_rows = input_rows - kernel_rows + 1;
    int out_cols = input_cols - kernel_cols + 1;
    int xblocksPerGrid = (out_cols + threadsPerBlock.x - 1) / threadsPerBlock.x;
    int yblocksPerGrid = (out_rows + threadsPerBlock.y - 1) / threadsPerBlock.y;
    dim3 blocksPerGrid(xblocksPerGrid, yblocksPerGrid);

    size_t smem = kernel_rows * kernel_cols * sizeof(float);

    ConvolutionKernel<<<blocksPerGrid, threadsPerBlock, smem>>>(
        input,
        kernel,
        output,
        input_rows,
        input_cols,
        kernel_rows,
        kernel_cols,
        out_rows,
        out_cols
    );
    cudaDeviceSynchronize();
}
