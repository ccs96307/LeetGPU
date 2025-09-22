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
// Modified for 2D convolution.
struct TestCase {
    std::string description;
    std::vector<float> input_matrix;
    std::vector<float> kernel_matrix;
    int input_rows, input_cols;
    int kernel_rows, kernel_cols;
    std::vector<float> expected_output;
};


int main() {
    // Updated test cases for 2D Convolution
    std::vector<TestCase> test_cases = {
        {
            "Example 1: 3x3 input, 2x2 kernel",
            {1, 2, 3, 4, 5, 6, 7, 8, 9}, // input
            {0, 1, 1, 0},                // kernel
            3, 3,                        // input_rows, input_cols
            2, 2,                        // kernel_rows, kernel_cols
            {6, 8, 12, 14}               // expected_output
        },
        {
            "Example 2: 4x4 input, 1x3 kernel",
            {1, 1, 1, 1, 1, 2, 3, 1, 1, 4, 5, 1, 1, 1, 1, 1}, // input
            {1, 0, 1},                                       // kernel
            4, 4,                                            // input_rows, input_cols
            1, 3,                                            // kernel_rows, kernel_cols
            {2, 2, 4, 3, 6, 5, 2, 2}                         // expected_output
        }
    };

    int tests_passed = 0;
    for (int i = 0; i < test_cases.size(); ++i) {
        const auto& tc = test_cases[i];
        
        int output_rows = tc.input_rows - tc.kernel_rows + 1;
        int output_cols = tc.input_cols - tc.kernel_cols + 1;

        std::cout << std::format("--- Running Test Case: {} ---", tc.description) << std::endl;
        std::cout << std::format("    Input: {}x{}, Kernel: {}x{}, Output: {}x{}", 
                               tc.input_rows, tc.input_cols, tc.kernel_rows, tc.kernel_cols, output_rows, output_cols) << std::endl;

        // Calculate data sizes in bytes
        size_t size_input = (size_t)tc.input_rows * tc.input_cols * sizeof(float);
        size_t size_kernel = (size_t)tc.kernel_rows * tc.kernel_cols * sizeof(float);
        size_t size_output = (size_t)output_rows * output_cols * sizeof(float);

        // Host and Device memory pointers
        float* d_input = nullptr;
        float* d_kernel = nullptr;
        float* d_output = nullptr;
        std::vector<float> h_output(output_rows * output_cols);

        // Allocate memory on the GPU
        checkCudaError(cudaMalloc((void**)&d_input, size_input));
        checkCudaError(cudaMalloc((void**)&d_kernel, size_kernel));
        checkCudaError(cudaMalloc((void**)&d_output, size_output));

        // Copy input data from Host (CPU) to Device (GPU)
        checkCudaError(cudaMemcpy(d_input, tc.input_matrix.data(), size_input, cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpy(d_kernel, tc.kernel_matrix.data(), size_kernel, cudaMemcpyHostToDevice));
        
        // Call the solver function
        // IMPORTANT: Assumes your solve function signature is:
        // void solve(const float* input, const float* kernel, float* output, int input_rows, int input_cols, int kernel_rows, int kernel_cols);
        solve(d_input, d_kernel, d_output, tc.input_rows, tc.input_cols, tc.kernel_rows, tc.kernel_cols);

        // Copy the result from d_output on Device back to Host
        checkCudaError(cudaMemcpy(h_output.data(), d_output, size_output, cudaMemcpyDeviceToHost));

        // Print results and verify
        bool test_passed_local = true;
        for (int j = 0; j < h_output.size(); ++j) {
            std::cout << std::format("Expected: {}, Got: {}", tc.expected_output[j], h_output[j]) << std::endl;
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
        
        // Free all three GPU memory buffers
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