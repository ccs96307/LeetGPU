#include "common/utils.cuh"
#include "kernel.cuh"

#include <iostream>
#include <format>
#include <vector>
#include <string>

#include <cuda_runtime.h>

// ===================================================================
// MAIN FUNCTION WITH MULTIPLE TEST CASES
// ===================================================================

// A simple struct to hold all information for a single test case.
// Modified for Subarray Sum.
struct TestCase {
    std::string description;
    std::vector<int> input_array;
    int S; // Start index
    int E; // End index
    int expected_output;
};


int main() {
    // Updated test cases for Subarray Sum
    std::vector<TestCase> test_cases = {
        {
            "Example 1",
            {1, 2, 1, 3, 4},
            1, 3, // S, E
            6
        },
        {
            "Example 2",
            {1, 2, 3, 4},
            0, 3, // S, E
            10
        }
    };

    int tests_passed = 0;
    for (int i = 0; i < test_cases.size(); ++i) {
        const auto& tc = test_cases[i];
        int N = tc.input_array.size();
        
        std::cout << std::format("--- Running Test Case: {} (N={}, S={}, E={}) ---",
                               tc.description, N, tc.S, tc.E) << std::endl;

        // Calculate data sizes in bytes
        size_t input_data_size = (size_t)N * sizeof(int);
        size_t output_data_size = sizeof(int); // Output is a single integer

        // Host and Device memory pointers
        int* d_input = nullptr;
        int* d_output = nullptr;
        int h_output = 0;

        // Allocate memory on the GPU
        checkCudaError(cudaMalloc((void**)&d_input, input_data_size));
        checkCudaError(cudaMalloc((void**)&d_output, output_data_size));

        // Copy input data from Host (CPU) to Device (GPU)
        checkCudaError(cudaMemcpy(d_input, tc.input_array.data(), input_data_size, cudaMemcpyHostToDevice));
        
        // CRITICAL: For reduction problems, always initialize the output memory on the device to 0.
        checkCudaError(cudaMemset(d_output, 0, output_data_size));

        // Call the solver function
        // IMPORTANT: Assumes your solve function signature is:
        // void solve(const int* input, int* output, int N, int S, int E);
        solve(d_input, d_output, N, tc.S, tc.E);

        // Copy the single integer result from Device back to Host
        checkCudaError(cudaMemcpy(&h_output, d_output, output_data_size, cudaMemcpyDeviceToHost));

        // Print results and verify
        std::cout << std::format("Expected: {}, Got: {}", tc.expected_output, h_output) << std::endl;
        if (h_output == tc.expected_output) {
            std::cout << "Result: PASSED." << std::endl;
            ++tests_passed;
        } else {
            std::cout << "Result: FAILED!" << std::endl;
        }
        
        // Free both GPU memory buffers
        checkCudaError(cudaFree(d_input));
        checkCudaError(cudaFree(d_output));
        std::cout << std::endl;
    }

    std::cout << "========================================" << std::endl;
    std::cout << std::format("Test Summary: {} / {} passed.", tests_passed, test_cases.size()) << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}