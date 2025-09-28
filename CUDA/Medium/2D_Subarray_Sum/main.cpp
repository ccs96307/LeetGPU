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
// Modified for 2D Subarray Sum.
struct TestCase {
    std::string description;
    std::vector<int> input_array;
    int N, M; // Matrix dimensions
    int S_ROW, E_ROW, S_COL, E_COL; // Subarray indices
    int expected_output;
};


int main() {
    // Updated test cases for 2D Subarray Sum
    std::vector<TestCase> test_cases = {
        {
            "Example 1",
            {1, 2, 3, 4, 5, 1},
            2, 3,       // N, M
            0, 1, 1, 2, // S_ROW, E_ROW, S_COL, E_COL
            11
        },
        {
            "Example 2",
            {5, 10, 5, 2},
            2, 2,       // N, M
            0, 0, 1, 1, // S_ROW, E_ROW, S_COL, E_COL
            10
        },
        {
            "Example 3",
            {5, 10, 5, 2, 5, 3, 5, 14},
            4, 2,       // N, M
            0, 3, 0, 0, // S_ROW, E_ROW, S_COL, E_COL
            20
        }
    };

    int tests_passed = 0;
    for (int i = 0; i < test_cases.size(); ++i) {
        const auto& tc = test_cases[i];
        
        std::cout << std::format("--- Running Test Case: {} (Matrix: {}x{}) ---",
                               tc.description, tc.N, tc.M) << std::endl;

        // Calculate data sizes in bytes
        size_t input_data_size = (size_t)tc.N * tc.M * sizeof(int);
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
        // void solve(const int* input, int* output, int N, int M, int S_ROW, int E_ROW, int S_COL, int E_COL);
        solve(d_input, d_output, tc.N, tc.M, tc.S_ROW, tc.E_ROW, tc.S_COL, tc.E_COL);

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