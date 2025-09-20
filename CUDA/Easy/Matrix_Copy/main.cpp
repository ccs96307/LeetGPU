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
// Modified for matrix copy.
struct TestCase {
    std::string description;
    std::vector<float> input_A;
    int N;
    std::vector<float> expected_B;
};


int main() {
    // Updated test cases for Matrix Copy
    std::vector<TestCase> test_cases = {
        {
            "Example 1: 2x2 matrix",
            {1.0f, 2.0f, 3.0f, 4.0f},
            2, // N
            {1.0f, 2.0f, 3.0f, 4.0f}
        },
        {
            "Example 2: 3x3 matrix",
            {5.5f, 6.6f, 7.7f, 8.8f, 9.9f, 10.1f, 11.2f, 12.3f, 13.4f},
            3, // N
            {5.5f, 6.6f, 7.7f, 8.8f, 9.9f, 10.1f, 11.2f, 12.3f, 13.4f}
        }
    };

    int tests_passed = 0;
    for (int i = 0; i < test_cases.size(); ++i) {
        const auto& tc = test_cases[i];
        
        std::cout << std::format("--- Running Test Case: {} (N={}) ---",
                               tc.description, tc.N) << std::endl;

        // Calculate data size in bytes for an N x N matrix
        size_t num_elements = (size_t)tc.N * tc.N;
        size_t data_size = num_elements * sizeof(float);

        // Host and Device memory pointers
        float* d_A = nullptr;
        float* d_B = nullptr; // Separate buffer for the output
        std::vector<float> h_B(num_elements);

        // Allocate memory on the GPU for both input and output
        checkCudaError(cudaMalloc((void**)&d_A, data_size));
        checkCudaError(cudaMalloc((void**)&d_B, data_size));

        // Copy input data from Host (CPU) to Device (GPU)
        checkCudaError(cudaMemcpy(d_A, tc.input_A.data(), data_size, cudaMemcpyHostToDevice));
        
        // Call the solver function
        // IMPORTANT: Assumes your solve function signature is:
        // void solve(const float* A, float* B, int N);
        solve(d_A, d_B, tc.N);

        // Copy the result from d_B on Device back to Host
        checkCudaError(cudaMemcpy(h_B.data(), d_B, data_size, cudaMemcpyDeviceToHost));

        // Print results and verify
        bool test_passed_local = true;
        for (int j = 0; j < num_elements; ++j) {
            std::cout << std::format("Expected: {}, Got: {}", tc.expected_B[j], h_B[j]) << std::endl;
            if (std::abs(h_B[j] - tc.expected_B[j]) > 1e-5) {
                test_passed_local = false;
            }
        }

        if (test_passed_local) {
            std::cout << "Result: PASSED." << std::endl;
            ++tests_passed;
        } else {
            std::cout << "Result: FAILED!" << std::endl;
        }
        
        // Free both GPU memory buffers
        checkCudaError(cudaFree(d_A));
        checkCudaError(cudaFree(d_B));
        std::cout << std::endl;
    }

    std::cout << "========================================" << std::endl;
    std::cout << std::format("Test Summary: {} / {} passed.", tests_passed, test_cases.size()) << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}