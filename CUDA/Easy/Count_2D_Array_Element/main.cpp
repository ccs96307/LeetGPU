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
struct TestCase {
    std::string description;
    std::vector<int> input;
    int N;
    int M;
    int K;
    int expected_output;
};


int main() {
    std::vector<TestCase> test_cases = {
        {
            "Example 1",
            {1, 2, 3, 4, 5, 1},
            2,
            3,
            1,
            2
        },
        {
            "Example 2",
            {5, 10, 5, 2},
            2,
            2,
            1,
            0
        }
    };

    int tests_passed = 0;
    for (int i = 0; i < test_cases.size(); ++i) {
        const auto& tc = test_cases[i];
        std::cout << std::format("--- Running Test Case: {} ---", tc.description) << std::endl;

        // Init data length
        size_t input_data_size = tc.input.size() * sizeof(int);
        size_t output_data_size = sizeof(int);

        // Host and Device memory pointers
        int* d_input = nullptr;
        int* d_output = nullptr;
        int h_output = 0;

        // Allocate memory on the GPU
        checkCudaError(cudaMalloc((void**)&d_input, input_data_size));
        checkCudaError(cudaMalloc((void**)&d_output, output_data_size));

        // Copy input data from Host (CPU) to Device (GPU)
        checkCudaError(cudaMemcpy(d_input, tc.input.data(), input_data_size, cudaMemcpyHostToDevice));

        // Call the solver function
        solve(d_input, d_output, tc.N, tc.M, tc.K);

        // Copy result from Device back to Host
        checkCudaError(cudaMemcpy(&h_output, d_output, output_data_size, cudaMemcpyDeviceToHost));

        // Print results and verify
        std::cout << std::format("Dimensions: {}x{}, K: {}", tc.N, tc.M, tc.K) << std::endl;
        std::cout << std::format("Expected: {}, Got: {}", tc.expected_output, h_output) << std::endl;
        if (h_output == tc.expected_output) {
            std::cout << "Result: PASSED." << std::endl;
            ++tests_passed;
        }
        else {
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
