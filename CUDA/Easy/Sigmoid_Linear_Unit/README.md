# Swish-Gated Linear Unit

**Easy**

Implement the Swish-Gated Linear Unit (SwiGLU) activation function forward pass for 1D input vectors. Given an input tensor of shape [N] where N is the number of elements, compute the output using the elementwise formula. The input and output tensor must be of type `float32`.

SwiGLU is defined as:

1.  Split input `x` into two halves: $x_1$ and $x_2$.
2.  Compute SiLU on the first half:
    $$
    \text{SiLU}(x_1) = x_1 \cdot \sigma(x_1), \quad \text{where} \quad \sigma(x) = \frac{1}{1 + e^{-x}}
    $$
3.  Compute the SwiGLU output:
    $$
    \text{SwiGLU}(x_1, x_2) = \text{SiLU}(x_1) \cdot x_2
    $$

## Implementation Requirements

* Use only native features (external libraries are not permitted)
* The `solve` function signature must remain unchanged
* The final result must be stored in the `output` tensor

## Example 1:

    Input: `[1.0, 2.0, 3.0, 4.0]` (N=4)
    Output: `[2.1931758, 7.0463767]`

## Example 2:

    Input: `[0.5, 1.0]` (N=2)
    Output: `[0.31122968]`
*(Note: The image likely contains a typo for N in this example; `N=2` is consistent with the operation.)*