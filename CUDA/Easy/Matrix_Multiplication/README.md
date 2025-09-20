# Matrix Multiplication
Write a program that multiplies two matrices of 32-bit floating point numbers on a GPU. Given matrix *A* of dimensions *M* x *N* and matrix *B* of dimensions *N* x *K*, compute the product matrix *C* = *A* x *B*, which will have dimensions *M* x *K*. All matrices are stored in row-major format.

## Implementation Requirements

* Use only native features (external libraries are not permitted)
* The `solve` function signature must remain unchanged
* The final result must be stored in matrix `C`

## Example 1:

Input:
Matrix *A* (2 x 2):

    [1.0 2.0]
    [3.0 4.0]

Matrix *B* (2 x 2):

    [5.0 6.0]
    [7.0 8.0]

Output:
Matrix *C* (2 x 2):

    [19.0 22.0]
    [43.0 50.0]

## Example 2:

Input:
Matrix *A* (1 x 3):

    [1.0 2.0 3.0]

Matrix *B* (3 x 1):

    [4.0]
    [5.0]
    [6.0]

Output:
Matrix *C* (1 x 1):

    [32.0]

## Constraints

* 1 <= `M`, `N`, `K` <= 8192
* Performance is measured with `M` = 8192, `N` = 6144, `K` = 4096

---

# Solution

In matrix multiplication, we deal with three dimensions: `M`, `N`, and `K`.

    A: (M, N)
    B: (N, K)
    C: (M, K)

Each thread computes one element `C[row, x]`, where `row` (`y`) maps to `M` and `col` (`x`) maps to `K`.

The computation involves iterating over the `N` axis (the reduction dimension).

To improve performance, we use shared memory to cache tiles of `A` and `B`. This avoids repeatedly loading the same elements from global memory (even though they may already be cached in L1/L2). 

Since many elements of `A` and `B` are reused across multiple dot products, shared memory provides significant speedups by reducing redundant global memory accesses.
