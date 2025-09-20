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
struct TestCase {
    std::string description;
    std::vector<float> input_array;
    std::vector<float> expected_output;
};


int main() {
    // Updated test cases for Softmax
    std::vector<TestCase> test_cases = {
        {
            "Example 1",
            {1.0f, 2.0f, 3.0f},
            {0.09003057f, 0.24472847f, 0.66524096f}
        },
        {
            "Example 2 (with corrected expected values)",
            {-10.0f, -5.0f, 0.0f, 5.0f, 10.0f},
            // Note: The expected values from the image are inconsistent.
            // These are the correct values after applying the max trick.
            {2.046e-09f, 3.039e-07f, 4.508e-05f, 0.00669f, 0.9932f}
        }
    };

    int tests_passed = 0;
    for (int i = 0; i < test_cases.size(); ++i) {
        const auto& tc = test_cases[i];
        int N = tc.input_array.size();
        
        std::cout << std::format("--- Running Test Case: {} (N={}) ---",
                               tc.description, N) << std::endl;

        // Calculate data size in bytes
        size_t data_size = (size_t)N * sizeof(float);

        // Host and Device memory pointers
        float* d_input = nullptr;
        float* d_output = nullptr; // Separate buffer for the output
        std::vector<float> h_output(N);

        // Allocate memory on the GPU for both input and output
        checkCudaError(cudaMalloc((void**)&d_input, data_size));
        checkCudaError(cudaMalloc((void**)&d_output, data_size));

        // Copy input data from Host (CPU) to Device (GPU)
        checkCudaError(cudaMemcpy(d_input, tc.input_array.data(), data_size, cudaMemcpyHostToDevice));
        
        // Call the solver function
        // IMPORTANT: Assumes your solve function signature is:
        // void solve(const float* input, float* output, int N);
        solve(d_input, d_output, N);

        // Copy the result from d_output on Device back to Host
        checkCudaError(cudaMemcpy(h_output.data(), d_output, data_size, cudaMemcpyDeviceToHost));

        // Print results and verify
        bool test_passed_local = true;
        for (int j = 0; j < N; ++j) {
            std::cout << std::format("Expected: {:.6e}, Got: {:.6e}", tc.expected_output[j], h_output[j]) << std::endl;
            // Use a slightly larger tolerance for complex calculations like softmax
            if (std::abs(h_output[j] - tc.expected_output[j]) > 1e-4) {
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
        checkCudaError(cudaFree(d_output));
        std::cout << std::endl;
    }

    std::cout << "========================================" << std::endl;
    std::cout << std::format("Test Summary: {} / {} passed.", tests_passed, test_cases.size()) << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}