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
// Modified for matrix transpose.
struct TestCase {
    std::string description;
    std::vector<float> input_matrix;
    int rows;
    int cols;
    std::vector<float> expected_output;
};


int main() {
    // Updated test cases for matrix transpose
    std::vector<TestCase> test_cases = {
        {
            "Example 1: 2x3 matrix",
            {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, // Input (2x3 matrix)
            2, 3,                            // rows, cols
            {1.0, 4.0, 2.0, 5.0, 3.0, 6.0}   // Expected Output (3x2 matrix)
        },
        {
            "Example 2: 3x1 matrix",
            {1.0, 2.0, 3.0},                 // Input (3x1 matrix)
            3, 1,                            // rows, cols
            {1.0, 2.0, 3.0}                  // Expected Output (1x3 matrix)
        }
    };

    int tests_passed = 0;
    for (int i = 0; i < test_cases.size(); ++i) {
        const auto& tc = test_cases[i];
        std::cout << std::format("--- Running Test Case: {} (Input: {}x{}) ---",
                               tc.description, tc.rows, tc.cols) << std::endl;

        // The number of elements is the same for the input and output matrices
        size_t num_elements = (size_t)tc.rows * tc.cols;
        size_t data_size = num_elements * sizeof(float);

        // Host and Device memory pointers
        float* d_input = nullptr;
        float* d_output = nullptr;
        std::vector<float> h_output(num_elements, 0.0f);

        // Allocate memory on the GPU
        checkCudaError(cudaMalloc((void**)&d_input, data_size));
        checkCudaError(cudaMalloc((void**)&d_output, data_size));

        // Copy input data from Host (CPU) to Device (GPU)
        checkCudaError(cudaMemcpy(d_input, tc.input_matrix.data(), data_size, cudaMemcpyHostToDevice));
        
        // Call the solver function with matrix dimensions
        // IMPORTANT: Assumes your solve function signature is:
        // void solve(const float* input, float* output, int rows, int cols);
        solve(d_input, d_output, tc.rows, tc.cols);

        // Copy result from Device back to Host
        checkCudaError(cudaMemcpy(h_output.data(), d_output, data_size, cudaMemcpyDeviceToHost));

        // Print results and verify
        bool test_passed_local = true;
        for (int j = 0; j < num_elements; ++j) {
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
        checkCudaError(cudaFree(d_output));
        std::cout << std::endl;
    }

    std::cout << "========================================" << std::endl;
    std::cout << std::format("Test Summary: {} / {} passed.", tests_passed, test_cases.size()) << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}