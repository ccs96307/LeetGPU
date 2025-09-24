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
// Modified for Histogramming.
struct TestCase {
    std::string description;
    std::vector<int> input_array;
    int num_bins;
    std::vector<int> expected_histogram;
};


int main() {
    // Updated test cases for Histogramming
    std::vector<TestCase> test_cases = {
        {
            "Example 1",
            {0, 1, 2, 1, 0},
            3, // num_bins
            {2, 2, 1}
        },
        {
            "Example 2",
            {3, 3, 3, 3},
            5, // num_bins
            {0, 0, 0, 4, 0}
        }
    };

    int tests_passed = 0;
    for (int i = 0; i < test_cases.size(); ++i) {
        const auto& tc = test_cases[i];
        int N = tc.input_array.size();
        
        std::cout << std::format("--- Running Test Case: {} (N={}, num_bins={}) ---",
                               tc.description, N, tc.num_bins) << std::endl;

        // Calculate data sizes in bytes
        size_t input_data_size = (size_t)N * sizeof(int);
        size_t histogram_data_size = (size_t)tc.num_bins * sizeof(int);

        // Host and Device memory pointers
        int* d_input = nullptr;
        int* d_histogram = nullptr;
        std::vector<int> h_histogram(tc.num_bins);

        // Allocate memory on the GPU
        checkCudaError(cudaMalloc((void**)&d_input, input_data_size));
        checkCudaError(cudaMalloc((void**)&d_histogram, histogram_data_size));

        // Copy input data from Host (CPU) to Device (GPU)
        checkCudaError(cudaMemcpy(d_input, tc.input_array.data(), input_data_size, cudaMemcpyHostToDevice));
        
        // CRITICAL: The histogram bins must be initialized to 0 before counting.
        checkCudaError(cudaMemset(d_histogram, 0, histogram_data_size));

        // Call the solver function
        // IMPORTANT: Assumes your solve function signature is:
        // void solve(const int* input, int* histogram, int N, int num_bins);
        solve(d_input, d_histogram, N, tc.num_bins);

        // Copy the result from Device back to Host
        checkCudaError(cudaMemcpy(h_histogram.data(), d_histogram, histogram_data_size, cudaMemcpyDeviceToHost));

        // Print results and verify
        bool test_passed_local = true;
        for (int j = 0; j < tc.num_bins; ++j) {
            std::cout << std::format("Bin[{}]: Expected: {}, Got: {}", j, tc.expected_histogram[j], h_histogram[j]) << std::endl;
            if (h_histogram[j] != tc.expected_histogram[j]) {
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
        checkCudaError(cudaFree(d_input));
        checkCudaError(cudaFree(d_histogram));
        std::cout << std::endl;
    }

    std::cout << "========================================" << std::endl;
    std::cout << std::format("Test Summary: {} / {} passed.", tests_passed, test_cases.size()) << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}