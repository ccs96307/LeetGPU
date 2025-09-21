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
// Modified for dot product.
struct TestCase {
    std::string description;
    std::vector<float> A;
    std::vector<float> B;
    float expected_output; // Output is a single value
};


int main() {
    // Updated test cases for Dot Product
    std::vector<TestCase> test_cases = {
        {
            "Example 1",
            {1.0f, 2.0f, 3.0f, 4.0f},
            {5.0f, 6.0f, 7.0f, 8.0f},
            70.0f
        },
        {
            "Example 2",
            {0.5f, 1.5f, 2.5f},
            {2.0f, 3.0f, 4.0f},
            15.5f
        }
    };

    int tests_passed = 0;
    for (int i = 0; i < test_cases.size(); ++i) {
        const auto& tc = test_cases[i];
        int N = tc.A.size();
        
        std::cout << std::format("--- Running Test Case: {} (N={}) ---",
                               tc.description, N) << std::endl;

        // Calculate data sizes in bytes
        size_t vector_data_size = (size_t)N * sizeof(float);
        size_t output_data_size = sizeof(float); // Output is just a single float

        // Host and Device memory pointers
        float* d_A = nullptr;
        float* d_B = nullptr;
        float* d_output = nullptr;
        float h_output = 0.0f;

        // Allocate memory on the GPU for two input vectors and one output value
        checkCudaError(cudaMalloc((void**)&d_A, vector_data_size));
        checkCudaError(cudaMalloc((void**)&d_B, vector_data_size));
        checkCudaError(cudaMalloc((void**)&d_output, output_data_size));

        // Copy input data from Host (CPU) to Device (GPU)
        checkCudaError(cudaMemcpy(d_A, tc.A.data(), vector_data_size, cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpy(d_B, tc.B.data(), vector_data_size, cudaMemcpyHostToDevice));
        
        // CRITICAL: For reduction problems, always initialize the output memory on the device to 0.
        checkCudaError(cudaMemset(d_output, 0, output_data_size));

        // Call the solver function
        // IMPORTANT: Assumes your solve function signature is:
        // void solve(const float* A, const float* B, float* output, int N);
        solve(d_A, d_B, d_output, N);

        // Copy the single float result from Device back to Host
        checkCudaError(cudaMemcpy(&h_output, d_output, output_data_size, cudaMemcpyDeviceToHost));

        // Print results and verify
        std::cout << std::format("Expected: {}, Got: {}", tc.expected_output, h_output) << std::endl;
        if (std::abs(h_output - tc.expected_output) < 1e-5) {
            std::cout << "Result: PASSED." << std::endl;
            ++tests_passed;
        } else {
            std::cout << "Result: FAILED!" << std::endl;
        }
        
        // Free all three GPU memory buffers
        checkCudaError(cudaFree(d_A));
        checkCudaError(cudaFree(d_B));
        checkCudaError(cudaFree(d_output));
        std::cout << std::endl;
    }

    std::cout << "========================================" << std::endl;
    std::cout << std::format("Test Summary: {} / {} passed.", tests_passed, test_cases.size()) << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}