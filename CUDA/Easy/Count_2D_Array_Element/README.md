# Count 2D Array Element

**Easy**

Write a GPU program that counts the number of elements with the integer value `k` in an 2D array of 32-bit integers. The program should count the number of elements with `k` in an array. You are given an input 2D array `input` of length `N` x `M` and integer `k`.

## Implementation Requirements

* Use only native features (external libraries are not permitted)
* The `solve` function signature must remain unchanged
* The final result must be stored in the `output` variable

## Example 1:

Input:
$$
input = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 1 \end{bmatrix}, \quad k = 1
$$
Output: `output` = 2

## Example 2:

Input:
$$
input = \begin{bmatrix} 5 & 10 \\ 5 & 2 \end{bmatrix}, \quad k = 1
$$
Output: `output` = 0

## Constraints

* 1 ≤ `N`, `M` ≤ 10,000
* 1 ≤ `input[i]`, `k` ≤ 100