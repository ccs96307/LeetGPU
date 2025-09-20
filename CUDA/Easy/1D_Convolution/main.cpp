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
// Modified for 1D convolution.
struct TestCase {
    std::string description;
    std::vector<float> input_array;
    std::vector<float> kernel_array;
    std::vector<float> expected_output;
};


int main() {
    // Updated test cases for 1D convolution
    std::vector<TestCase> test_cases = {
        {
            "Example 1",
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f},
            {1.0f, 0.0f, -1.0f},
            {-2.0f, -2.0f, -2.0f}
        },
        {
            "Example 2",
            {2.0f, 4.0f, 6.0f, 8.0f},
            {0.5f, 0.2f},
            {1.8f, 3.2f, 4.6f}
        }
    };

    int tests_passed = 0;
    for (int i = 0; i < test_cases.size(); ++i) {
        const auto& tc = test_cases[i];
        int input_size = tc.input_array.size();
        int kernel_size = tc.kernel_array.size();
        int output_size = input_size - kernel_size + 1;
        
        std::cout << std::format("--- Running Test Case: {} (Input: {}, Kernel: {}) ---",
                               tc.description, input_size, kernel_size) << std::endl;

        // Calculate data sizes in bytes
        size_t size_input = (size_t)input_size * sizeof(float);
        size_t size_kernel = (size_t)kernel_size * sizeof(float);
        size_t size_output = (size_t)output_size * sizeof(float);

        // Host and Device memory pointers
        float* d_input = nullptr;
        float* d_kernel = nullptr;
        float* d_output = nullptr;
        std::vector<float> h_output(output_size);

        // Allocate memory on the GPU
        checkCudaError(cudaMalloc((void**)&d_input, size_input));
        checkCudaError(cudaMalloc((void**)&d_kernel, size_kernel));
        checkCudaError(cudaMalloc((void**)&d_output, size_output));

        // Copy input data from Host (CPU) to Device (GPU)
        checkCudaError(cudaMemcpy(d_input, tc.input_array.data(), size_input, cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpy(d_kernel, tc.kernel_array.data(), size_kernel, cudaMemcpyHostToDevice));
        
        // Call the solver function
        // IMPORTANT: Assumes your solve function signature is:
        // void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size);
        solve(d_input, d_kernel, d_output, input_size, kernel_size);

        // Copy result from Device back to Host
        checkCudaError(cudaMemcpy(h_output.data(), d_output, size_output, cudaMemcpyDeviceToHost));

        // Print results and verify
        bool test_passed_local = true;
        for (int j = 0; j < output_size; ++j) {
            std::cout << std::format("Expected: {}, Got: {}", tc.expected_output[j], h_output[j]) << std::endl;
            // Use a small tolerance (epsilon) for floating-point comparisons
            if (std::abs(h_output[j] - tc.expected_output[j]) > 1e-5) {
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
        checkCudaError(cudaFree(d_input));
        checkCudaError(cudaFree(d_kernel));
        checkCudaError(cudaFree(d_output));
        std::cout << std::endl;
    }

    std::cout << "========================================" << std::endl;
    std::cout << std::format("Test Summary: {} / {} passed.", tests_passed, test_cases.size()) << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}