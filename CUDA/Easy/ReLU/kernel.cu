#include "kernel.cuh"
#include <cuda_runtime.h>
#include <stdio.h>


__global__ void relu_kernel(const float4* input, float4* output, int N_vec4, int remainder) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N_vec4) {
        float4 val = input[idx];
        val.x = fmaxf(0.0f, val.x);
        val.y = fmaxf(0.0f, val.y);
        val.z = fmaxf(0.0f, val.z);
        val.w = fmaxf(0.0f, val.w);
        output[idx] = val;
    }
    
    if (idx == 0 && remainder > 0) {    
        int base = N_vec4 * 4;
        for (int i = 0; i < remainder; ++i) {
            float val = reinterpret_cast<const float*>(input)[base + i];
            reinterpret_cast<float*>(output)[base + i] = fmaxf(0.0f, val);
        }
    }
}


// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int N_vec4 = N / 4;
    int threadsPerBlock = 256;
    int blocksPerGrid = max(1, (N_vec4 + threadsPerBlock - 1) / threadsPerBlock);

    int remainder = N % 4;
    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        reinterpret_cast<const float4*>(input), 
        reinterpret_cast<float4*>(output),
        N_vec4,
        remainder
    );

    cudaDeviceSynchronize();
}
