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
// Modified for Mean Squared Error.
struct TestCase {
    std::string description;
    std::vector<float> predictions;
    std::vector<float> targets;
    float expected_mse; // Output is a single value
};


int main() {
    // Updated test cases for Mean Squared Error
    std::vector<TestCase> test_cases = {
        {
            "Example 1",
            {1.0f, 2.0f, 3.0f, 4.0f},
            {1.5f, 2.5f, 3.5f, 4.5f},
            0.25f
        },
        {
            "Example 2",
            {10.0f, 20.0f, 30.0f},
            {12.0f, 18.0f, 33.0f},
            5.666667f // 17.0 / 3.0
        }
    };

    int tests_passed = 0;
    for (int i = 0; i < test_cases.size(); ++i) {
        const auto& tc = test_cases[i];
        int N = tc.predictions.size();
        
        std::cout << std::format("--- Running Test Case: {} (N={}) ---",
                               tc.description, N) << std::endl;

        // Calculate data sizes in bytes
        size_t vector_data_size = (size_t)N * sizeof(float);
        size_t output_data_size = sizeof(float); // Output is just a single float

        // Host and Device memory pointers
        float* d_predictions = nullptr;
        float* d_targets = nullptr;
        float* d_mse = nullptr;
        float h_mse = 0.0f;

        // Allocate memory on the GPU
        checkCudaError(cudaMalloc((void**)&d_predictions, vector_data_size));
        checkCudaError(cudaMalloc((void**)&d_targets, vector_data_size));
        checkCudaError(cudaMalloc((void**)&d_mse, output_data_size));

        // Copy input data from Host (CPU) to Device (GPU)
        checkCudaError(cudaMemcpy(d_predictions, tc.predictions.data(), vector_data_size, cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpy(d_targets, tc.targets.data(), vector_data_size, cudaMemcpyHostToDevice));
        
        // CRITICAL: For reduction problems, always initialize the output memory on the device to 0.
        checkCudaError(cudaMemset(d_mse, 0, output_data_size));

        // Call the solver function
        // IMPORTANT: Assumes your solve function signature is:
        // void solve(const float* predictions, const float* targets, float* mse, int N);
        solve(d_predictions, d_targets, d_mse, N);

        // Copy the single float result from Device back to Host
        checkCudaError(cudaMemcpy(&h_mse, d_mse, output_data_size, cudaMemcpyDeviceToHost));

        // Print results and verify
        std::cout << std::format("Expected: {}, Got: {}", tc.expected_mse, h_mse) << std::endl;
        if (std::abs(h_mse - tc.expected_mse) < 1e-5) {
            std::cout << "Result: PASSED." << std::endl;
            ++tests_passed;
        } else {
            std::cout << "Result: FAILED!" << std::endl;
        }
        
        // Free all three GPU memory buffers
        checkCudaError(cudaFree(d_predictions));
        checkCudaError(cudaFree(d_targets));
        checkCudaError(cudaFree(d_mse));
        std::cout << std::endl;
    }

    std::cout << "========================================" << std::endl;
    std::cout << std::format("Test Summary: {} / {} passed.", tests_passed, test_cases.size()) << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}