#include <cuda_runtime.h>


__global__ void sum_kernel(const float* input, float* output, int N) {
    // Shared memory
    extern __shared__ float shared[];

    // Index
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Local sum (per every thread)
    float thread_sum = 0.0f;
    int stride = gridDim.x * blockDim.x;
    for (int i=idx; i<N; i+=stride) {
        thread_sum += input[i];
    }

    // Put the sum of thread into shared memory
    shared[tid] = thread_sum;
    __syncthreads();

    // Reduction in shared memory (optimal)
    for (int i=blockDim.x>>1; i>32; i>>=1) {
        if (tid < i) {
            shared[tid] += shared[tid + i];
        }
        __syncthreads();
    }

    // The last warp
    if (tid < 32) {
        volatile float* v = shared;
        v[tid] += v[tid + 32];
        v[tid] += v[tid + 16];
        v[tid] += v[tid + 8];
        v[tid] += v[tid + 4];
        v[tid] += v[tid + 2];
        v[tid] += v[tid + 1];
    }

    // Write the result into output
    if (tid == 0) {
        atomicAdd(output, shared[0]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {  
    const int threads = 256;
    const int maxBlocks = 1024;
    const int blocks = min((N + threads - 1) / threads, maxBlocks);
    size_t shmem = threads * sizeof(float);

    // Init
    cudaMemset(output, 0, sizeof(float));

    // Kernel
    sum_kernel<<<blocks, threads, shmem>>>(input, output, N);
    cudaDeviceSynchronize();
}
