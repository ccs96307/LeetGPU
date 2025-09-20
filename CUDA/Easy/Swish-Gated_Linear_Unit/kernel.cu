#include "kernel.cuh"
#include <cuda_runtime.h>


__forceinline__ __device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}


__global__ void sigmoid_linear_unit_kernel(const float4* input,
                                           float4* output,
                                           const int N,
                                           const int N_vec4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base = N_vec4 * 4;
    int tailIdx = base + idx;

    if (idx < N_vec4) {
        float4 val = input[idx];
        val.x = val.x * sigmoid(val.x);
        val.y = val.y * sigmoid(val.y);
        val.z = val.z * sigmoid(val.z);
        val.w = val.w * sigmoid(val.w);

        output[idx] = val;
    }

    if (tailIdx < N) {
        float val = reinterpret_cast<const float*>(input)[tailIdx];
        reinterpret_cast<float*>(output)[tailIdx] = val * sigmoid(val);
    }
}


// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int N_vec4 = N >> 2;    
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid(max(1, (N_vec4 + threadsPerBlock.x - 1) / threadsPerBlock.x));

    sigmoid_linear_unit_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        reinterpret_cast<const float4*>(input),
        reinterpret_cast<float4*>(output),
        N,
        N_vec4
    );

    cudaDeviceSynchronize();
}
