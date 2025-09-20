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
    std::vector<float> A;
    std::vector<float> B;
    std::vector<float> C;
    std::vector<float> expected_outputs;
};


int main() {
    std::vector<TestCase> test_cases = {
        {
            "Test Case 1",
            {1.0, 2.0, 3.0, 4.0},
            {5.0, 6.0, 7.0, 8.0},
            {0.0, 0.0, 0.0, 0.0},
            {6.0, 8.0, 10.0, 12.0}
        },
        {
            "Test Case 2",
            {1.5, 1.5, 1.5},
            {2.3, 2.3, 2.3},
            {0.0, 0.0, 0.0},
            {3.8, 3.8, 3.8}
        }
    };

    int tests_passed = 0;
    for (int i = 0; i < test_cases.size(); ++i) {
        const auto& tc = test_cases[i];
        int N = tc.A.size();
        std::cout << std::format("--- Running Test Case: {} ---", tc.description) << std::endl;

        // Init data length
        size_t data_size = N * sizeof(float);

        // Host and Device memory pointers
        float* d_A = nullptr;
        float* d_B = nullptr;
        float* d_C = nullptr;
        float* h_output = new float[N];

        // Initialize host output with 0
        for (int j = 0; j < N; ++j) {
            h_output[j] = 0.0f;
        }

        // Allocate memory on the GPU
        checkCudaError(cudaMalloc((void**)&d_A, data_size));
        checkCudaError(cudaMalloc((void**)&d_B, data_size));
        checkCudaError(cudaMalloc((void**)&d_C, data_size));

        // Copy input data from Host (CPU) to Device (GPU)
        checkCudaError(cudaMemcpy(d_A, tc.A.data(), data_size, cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpy(d_B, tc.B.data(), data_size, cudaMemcpyHostToDevice));

        // Call the solver function
        solve(d_A, d_B, d_C, N);

        // Copy result from Device back to Host
        checkCudaError(cudaMemcpy(h_output, d_C, data_size, cudaMemcpyDeviceToHost));

        // Print results and verify
        bool test_passed_local = true;
        for (int j = 0; j < N; ++j) {
            std::cout << std::format("Expected: {}, Got: {}", tc.expected_outputs[j], h_output[j]) << std::endl;
            if (h_output[j] != tc.expected_outputs[j]) {
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
