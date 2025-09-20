#include "common/utils.cuh"
#include "kernel.cuh"

#include <iostream>
#include <format>
#include <vector>
#include <string>
#include <cmath> // For abs() in float comparison

#include <cuda_runtime.h>

// ===================================================================
// MAIN FUNCTION WITH MULTIPLE TEST CASES
// ===================================================================

// A simple struct to hold all information for a single test case.
// Modified for matrix multiplication with dimensions M, N, K.
struct TestCase {
    std::string description;
    std::vector<float> A;
    std::vector<float> B;
    int M;
    int N;
    int K;
    std::vector<float> expected_C;
};


int main() {
    // Updated test cases for matrix multiplication
    std::vector<TestCase> test_cases = {
        {
            "Example 1: 2x2 by 2x2",
            {1.0, 2.0, 3.0, 4.0},      // Matrix A (2x2)
            {5.0, 6.0, 7.0, 8.0},      // Matrix B (2x2)
            2, 2, 2,                   // M, N, K
            {19.0, 22.0, 43.0, 50.0}   // Expected C (2x2)
        },
        {
            "Example 2: 1x3 by 3x1",
            {1.0, 2.0, 3.0},           // Matrix A (1x3)
            {4.0, 5.0, 6.0},           // Matrix B (3x1)
            1, 3, 1,                   // M, N, K
            {32.0}                     // Expected C (1x1)
        }
    };

    int tests_passed = 0;
    for (int i = 0; i < test_cases.size(); ++i) {
        const auto& tc = test_cases[i];
        std::cout << std::format("--- Running Test Case: {} (A:{}x{}, B:{}x{}) ---",
                               tc.description, tc.M, tc.N, tc.N, tc.K) << std::endl;

        // Init data length for each matrix
        size_t size_A = (size_t)tc.M * tc.N * sizeof(float);
        size_t size_B = (size_t)tc.N * tc.K * sizeof(float);
        size_t size_C = (size_t)tc.M * tc.K * sizeof(float);

        // Host and Device memory pointers
        float* d_A = nullptr;
        float* d_B = nullptr;
        float* d_C = nullptr;
        std::vector<float> h_C(tc.M * tc.K, 0.0f); // Use std::vector for safer memory management

        // Allocate memory on the GPU
        checkCudaError(cudaMalloc((void**)&d_A, size_A));
        checkCudaError(cudaMalloc((void**)&d_B, size_B));
        checkCudaError(cudaMalloc((void**)&d_C, size_C));

        // Copy input data from Host (CPU) to Device (GPU)
        checkCudaError(cudaMemcpy(d_A, tc.A.data(), size_A, cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpy(d_B, tc.B.data(), size_B, cudaMemcpyHostToDevice));
        
        // It's good practice to zero out the output buffer on the device
        checkCudaError(cudaMemset(d_C, 0, size_C));

        // Call the solver function with matrix dimensions
        // IMPORTANT: Assumes your solve function signature is:
        // void solve(const float* A, const float* B, float* C, int M, int N, int K);
        solve(d_A, d_B, d_C, tc.M, tc.N, tc.K);

        // Copy result from Device back to Host
        checkCudaError(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));

        // Print results and verify
        bool test_passed_local = true;
        for (int j = 0; j < tc.M * tc.K; ++j) {
            std::cout << std::format("Expected: {}, Got: {}", tc.expected_C[j], h_C[j]) << std::endl;
            // Use a small tolerance (epsilon) for floating-point comparisons
            if (std::abs(h_C[j] - tc.expected_C[j]) > 1e-5) {
                test_passed_local = false;
            }
        }

        if (test_passed_local) {
            std::cout << "Result: PASSED." << std::endl;
            ++tests_passed;
        } else {
            std::cout << "Result: FAILED!" << std::endl;
        }
        
        // Free GPU memory for the current test case
        checkCudaError(cudaFree(d_A));
        checkCudaError(cudaFree(d_B));
        checkCudaError(cudaFree(d_C));
        std::cout << std::endl;
    }

    std::cout << "========================================" << std::endl;
    std::cout << std::format("Test Summary: {} / {} passed.", tests_passed, test_cases.size()) << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}