#pragma once

#include <c10/util/complex.h>
#include <cstdlib>

namespace torchscience::kernel::polynomial {

// Backward pass for Chebyshev W polynomial multiply-by-x.
//
// Forward operation:
//   output[0] -= 0.5 * coeffs[0]                   (from W_0 term)
//   output[1] += 0.5 * coeffs[0]                   (from W_0 term)
//   output[k+1] += 0.5 * coeffs[k]  for k >= 1    (from W_k terms)
//   output[k-1] += 0.5 * coeffs[k]  for k >= 1    (from W_k terms)
//
// Backward pass reverses this:
//   grad_coeffs[0] = -0.5 * grad_output[0] + 0.5 * grad_output[1]
//   grad_coeffs[k] = 0.5 * (grad_output[k+1] + grad_output[k-1])  for k >= 1
//
// Parameters:
//   grad_coeffs: output gradient for coeffs, size N
//   grad_output: incoming gradient, size output_size
//   coeffs: original input coefficients, size N (unused but kept for API consistency)
//   N: number of input coefficients
//   output_size: size of grad_output
template <typename T>
void chebyshev_polynomial_w_mulx_backward(
    T* grad_coeffs,
    const T* grad_output,
    const T* coeffs,
    int64_t N,
    int64_t output_size
) {
    // Initialize gradients to zero
    for (int64_t k = 0; k < N; ++k) {
        grad_coeffs[k] = T(0);
    }

    if (N == 0) {
        return;
    }

    // Backward for: output[0] -= 0.5 * coeffs[0] and output[1] += 0.5 * coeffs[0]
    if (0 < output_size) {
        grad_coeffs[0] -= T(0.5) * grad_output[0];
    }
    if (1 < output_size) {
        grad_coeffs[0] += T(0.5) * grad_output[1];
    }

    // Backward for: output[k+1] += 0.5 * coeffs[k] and output[k-1] += 0.5 * coeffs[k]
    for (int64_t k = 1; k < N; ++k) {
        T grad_sum = T(0);
        if (k + 1 < output_size) {
            grad_sum += grad_output[k + 1];
        }
        if (k - 1 >= 0 && k - 1 < output_size) {
            grad_sum += grad_output[k - 1];
        }
        grad_coeffs[k] += T(0.5) * grad_sum;
    }
}

// Complex specialization
template <typename T>
void chebyshev_polynomial_w_mulx_backward(
    c10::complex<T>* grad_coeffs,
    const c10::complex<T>* grad_output,
    const c10::complex<T>* coeffs,
    int64_t N,
    int64_t output_size
) {
    const c10::complex<T> zero(T(0), T(0));
    const c10::complex<T> half(T(0.5), T(0));

    // Initialize gradients to zero
    for (int64_t k = 0; k < N; ++k) {
        grad_coeffs[k] = zero;
    }

    if (N == 0) {
        return;
    }

    // Backward for: output[0] -= 0.5 * coeffs[0] and output[1] += 0.5 * coeffs[0]
    if (0 < output_size) {
        grad_coeffs[0] -= half * grad_output[0];
    }
    if (1 < output_size) {
        grad_coeffs[0] += half * grad_output[1];
    }

    // Backward for: output[k+1] += 0.5 * coeffs[k] and output[k-1] += 0.5 * coeffs[k]
    for (int64_t k = 1; k < N; ++k) {
        c10::complex<T> grad_sum = zero;
        if (k + 1 < output_size) {
            grad_sum += grad_output[k + 1];
        }
        if (k - 1 >= 0 && k - 1 < output_size) {
            grad_sum += grad_output[k - 1];
        }
        grad_coeffs[k] += half * grad_sum;
    }
}

} // namespace torchscience::kernel::polynomial
