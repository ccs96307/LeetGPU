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
// Modified for color inversion.
struct TestCase {
    std::string description;
    std::vector<unsigned char> image;
    int width;
    int height;
    std::vector<unsigned char> expected_output;
};


int main() {
    // Updated test cases for color inversion
    std::vector<TestCase> test_cases = {
        {
            "Example 1: 1x2 image",
            {255, 0, 128, 255, 0, 255, 0, 255}, // Input image
            1, 2,                                // width, height
            {0, 255, 127, 255, 255, 0, 255, 255} // Expected output
        },
        {
            "Example 2: 2x1 image",
            {10, 20, 30, 255, 100, 150, 200, 255}, // Input image
            2, 1,                                  // width, height
            {245, 235, 225, 255, 155, 105, 55, 255} // Expected output
        }
    };

    int tests_passed = 0;
    for (int i = 0; i < test_cases.size(); ++i) {
        const auto& tc = test_cases[i];
        std::cout << std::format("--- Running Test Case: {} (Image: {}x{}) ---",
                               tc.description, tc.width, tc.height) << std::endl;

        // Calculate total number of bytes in the image
        size_t data_size = (size_t)tc.width * tc.height * 4 * sizeof(unsigned char);

        // Host and Device memory pointers
        unsigned char* d_image = nullptr;
        std::vector<unsigned char> h_output(tc.image.size());

        // Allocate memory on the GPU
        checkCudaError(cudaMalloc((void**)&d_image, data_size));

        // Copy input data from Host (CPU) to Device (GPU)
        checkCudaError(cudaMemcpy(d_image, tc.image.data(), data_size, cudaMemcpyHostToDevice));
        
        // Call the solver function
        // IMPORTANT: Assumes your solve function signature is:
        // void solve(unsigned char* image, int width, int height);
        // The operation is in-place, so the result is stored back into d_image.
        solve(d_image, tc.width, tc.height);

        // Copy result from Device back to Host
        checkCudaError(cudaMemcpy(h_output.data(), d_image, data_size, cudaMemcpyDeviceToHost));

        // Print results and verify
        bool test_passed_local = true;
        for (int j = 0; j < h_output.size(); ++j) {
            // Note: std::format doesn't directly support unsigned char, so we cast to int for printing.
            std::cout << std::format("Expected: {}, Got: {}", (int)tc.expected_output[j], (int)h_output[j]) << std::endl;
            if (h_output[j] != tc.expected_output[j]) {
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
        checkCudaError(cudaFree(d_image));
        std::cout << std::endl;
    }

    std::cout << "========================================" << std::endl;
    std::cout << std::format("Test Summary: {} / {} passed.", tests_passed, test_cases.size()) << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}