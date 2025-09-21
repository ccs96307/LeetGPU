# Dot Product

**Medium**

Implement a CUDA program that computes the dot product of two vectors containing 32-bit floating point numbers. The dot product is the sum of the products of the corresponding elements of two vectors.

Mathematically, the dot product of two vectors **A** and **B** of length *n* is defined as:
$$
\mathbf{A} \cdot \mathbf{B} = \sum_{i=0}^{n-1} A_i \cdot B_i = A_0 \cdot B_0 + A_1 \cdot B_1 + \dots + A_{n-1} \cdot B_{n-1}
$$

## Implementation Requirements
* Use only CUDA native features (external libraries are not permitted)
* The `solve` function signature must remain unchanged
* The final result must be stored in the `output` variable

## Example 1:

Input:
`A = [1.0, 2.0, 3.0, 4.0]`
`B = [5.0, 6.0, 7.0, 8.0]`

Output: `result = 70.0` (1.0*5.0 + 2.0*6.0 + 3.0*7.0 + 4.0*8.0)

## Example 2:

Input:
`A = [0.5, 1.5, 2.5]`
`B = [2.0, 3.0, 4.0]`

Output: `result = 16.0` (0.5*2.0 + 1.5*3.0 + 2.5*4.0)

## Constraints
* `A` and `B` have identical lengths
* 1 ≤ `N` ≤ 100,000,000