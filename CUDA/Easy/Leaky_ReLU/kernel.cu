#include "kernel.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

#define ALPHA 0.01f


__global__ void leaky_relu_kernel(const float4* input, 
                                  float4* output, 
                                  const int N,
                                  const int N_vec4, 
                                  const int remainder) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N_vec4) {
        float4 val = input[idx];

        val.x = (val.x <= 0) ? ALPHA * val.x : val.x;
        val.y = (val.y <= 0) ? ALPHA * val.y : val.y;
        val.z = (val.z <= 0) ? ALPHA * val.z : val.z;
        val.w = (val.w <= 0) ? ALPHA * val.w : val.w;

        output[idx] = val;
    }

    int base = N_vec4 * 4;
    int tailIdx = base + idx;

    if (tailIdx < N) {
        float val = reinterpret_cast<const float*>(input)[tailIdx];
        reinterpret_cast<float*>(output)[tailIdx] = (val <= 0) ? ALPHA * val : val;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int N_vec4 = N / 4; 
    int threadsPerBlock = 256;
    int blocksPerGrid = (N_vec4 + threadsPerBlock - 1) / threadsPerBlock;

    int remainder = N % 4;
    
    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        reinterpret_cast<const float4*>(input), 
        reinterpret_cast<float4*>(output), 
        N,
        N_vec4,
        remainder
    );
    cudaDeviceSynchronize();
}
