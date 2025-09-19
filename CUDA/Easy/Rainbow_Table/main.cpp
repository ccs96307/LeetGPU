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
    std::vector<int> numbers;
    int N;
    int R;
    std::vector<unsigned int> expected_hashes;
};


int main() {
    std::vector<TestCase> test_cases = {
        {
            "Test Case 1",
            {123, 456, 789},
            3,
            2,
            {1636807824, 1273011621, 2193987222}
        },
        {
            "Test Case 2",
            {0, 1, 2147483647},
            3,
            3,
            {96754810, 3571711400, 2006156166}
        }
    };

    int tests_passed = 0;
    for (int i = 0; i < test_cases.size(); ++i) {
        const auto& tc = test_cases[i];
        int N = tc.numbers.size();
        std::cout << std::format("--- Running Test Case: {} ---", tc.description) << std::endl;

        // Init data length
        size_t input_data_size = N * sizeof(int);
        size_t output_data_size = N * sizeof(unsigned int);

        // Host and Device memory pointers
        int* d_input = nullptr;
        unsigned int* d_output = nullptr;
        unsigned int* h_output = new unsigned int[N];

        // Initialize host output with 0
        for (int j = 0; j < N; ++j) {
            h_output[j] = 0;
        }

        //size_t output_data_size = sizeof(int);
        //int h_output = 0;

        // Allocate memory on the GPU
        checkCudaError(cudaMalloc((void**)&d_input, input_data_size));
        checkCudaError(cudaMalloc((void**)&d_output, output_data_size));

        // Copy input data from Host (CPU) to Device (GPU)
        checkCudaError(cudaMemcpy(d_input, tc.numbers.data(), input_data_size, cudaMemcpyHostToDevice));

        // Call the solver function
        solve(d_input, d_output, N, tc.R);

        // Copy result from Device back to Host
        checkCudaError(cudaMemcpy(h_output, d_output, output_data_size, cudaMemcpyDeviceToHost));

        // Print results and verify
        std::cout << std::format("N: {}, R: {}", N, tc.R) << std::endl;

        bool test_passed_local = true;
        for (int j = 0; j < N; ++j) {
            std::cout << std::format("Expected: {}, Got: {}", tc.expected_hashes[j], h_output[j]) << std::endl;
            if (h_output[j] != tc.expected_hashes[j]) {
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
        delete[] h_output;
        checkCudaError(cudaFree(d_input));
        checkCudaError(cudaFree(d_output));
        std::cout << std::endl;
    }

    std::cout << "========================================" << std::endl;
    std::cout << std::format("Test Summary: {} / {} passed.", tests_passed, test_cases.size()) << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
