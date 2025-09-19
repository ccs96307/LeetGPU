#pragma once

#include <iostream>
#include <format>
#include <cuda_runtime.h>


#define checkCudaError(call) do {                                                           \
    cudaError_t err = call;                                                                 \
    if (err != cudaSuccess) {                                                               \
        std::cerr << std::format("CUDA Error at {} : {}", __FILE__, __LINE__) << std::endl; \
        std::cerr << std::format(" - {}", cudaGetErrorString(err)) << std::endl;            \
        exit(EXIT_FAILURE);                                                                 \
    }                                                                                       \
} while (0)
