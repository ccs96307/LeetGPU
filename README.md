# LeetGPU

This repository contains my CUDA C++ solutions to the [LeetGPU](https://leetgpu.com/challenges) challenges.

I aim to use optimized methods for each problem and document my thought process.

Additionally, I have created test cases for every problem. You can verify the correctness and robustness of your solution by running:

```bash
./run.sh <PROBLEM_PATH> [kernel.cu]
```

For example, if you want to test the solution for "Easy/Vector_Addition":

```bash
./run.sh ./CUDA/Easy/Vector_Addition 
```

If you do not provide a kernel name, the default is `kernel.cu`.

---

## TODO List (Updated 2025/09/20)
- [ ] Easy
    - [X] Vector Addition
    - [X] Matrix Multiplication
    - [ ] Matrix Transpose
    - [ ] Color Inversion
    - [ ] 1D Convolution
    - [ ] Reverse Array
    - [ ] ReLU
    - [ ] Leaky ReLU
    - [ ] Rainbow Table
    - [ ] Matrix Copy
    - [ ] Count Array Element
    - [ ] Count 2D Array Element
    - [ ] Sigmoid Linear Unit
    - [ ] Swish-Gated Linear Unit
- [ ] Medium
    - [ ] Reduction
    - [ ] Softmax
    - [ ] Softmax Attention
    - [ ] 2D Convolution
    - [ ] Histogramming
    - [ ] Sorting
    - [ ] Prefix Sum
    - [ ] Dot Product
    - [ ] Sparse Matrix-Vector Multiplication
    - [ ] General Matrix Multiplication
    - [ ] Categorical Cross Entropy Loss
    - [ ] Mean Square Error
    - [ ] Gaussian Blur
    - [ ] Ordinary Least Squares
    - [ ] Logistic Regression
    - [ ] Monte Carlo Integration
    - [ ] Radix Sort
    - [ ] Matrix Power
    - [ ] Nearest Neighbor
    - [ ] Batch Normalization
    - [ ] 2D Max Pooling
    - [ ] Count 3D Array Element
    - [ ] BFS Shortest Path
    - [ ] Subarray Sum
    - [ ] 2D Subarray Sum
    - [ ] 3D Subarray Sum
    - [ ] RMS Normalization
    - [ ] Max Subarray Sum
    - [ ] Attention with Linear Biases
- [ ] Hard
    - [ ] 3D Convolution
    - [ ] Multi-Head Attention
    - [ ] Multi-Agent Simulation
    - [ ] K-Means Clustering
    - [ ] Fast Fourier Transform
    - [ ] Causal Self-Attention
    - [ ] Linear Self-Attention
