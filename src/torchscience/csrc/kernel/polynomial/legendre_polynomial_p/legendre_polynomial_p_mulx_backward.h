#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Backward pass for Legendre polynomial mulx
//
// Forward:
//   output[1] += coeffs[0]
//   For k >= 1:
//     output[k-1] += (k / (2k+1)) * coeffs[k]
//     output[k+1] += ((k+1) / (2k+1)) * coeffs[k]
//
// Backward:
//   grad_coeffs[0] += grad_output[1]
//   For k >= 1:
//     grad_coeffs[k] += (k / (2k+1)) * grad_output[k-1]
//     grad_coeffs[k] += ((k+1) / (2k+1)) * grad_output[k+1]
//
// Parameters:
//   grad_coeffs: output gradient w.r.t. coefficients, size N
//   grad_output: incoming gradient, size output_size (N+1)
//   N: number of original coefficients
//   output_size: size of grad_output
template <typename T>
void legendre_polynomial_p_mulx_backward(
    T* grad_coeffs,
    const T* grad_output,
    int64_t N,
    int64_t output_size
) {
    if (N == 0) {
        return;
    }

    // Initialize grad_coeffs to zero
    for (int64_t k = 0; k < N; ++k) {
        grad_coeffs[k] = T(0);
    }

    // Gradient from output[1] += coeffs[0]
    if (output_size > 1) {
        grad_coeffs[0] = grad_coeffs[0] + grad_output[1];
    }

    // Gradient from k >= 1 terms
    for (int64_t k = 1; k < N; ++k) {
        const T denom = T(2 * k + 1);

        // From output[k-1] += (k / denom) * coeffs[k]
        if (k - 1 < output_size) {
            grad_coeffs[k] = grad_coeffs[k] + (T(k) / denom) * grad_output[k - 1];
        }
        // From output[k+1] += ((k+1) / denom) * coeffs[k]
        if (k + 1 < output_size) {
            grad_coeffs[k] = grad_coeffs[k] + (T(k + 1) / denom) * grad_output[k + 1];
        }
    }
}

// Complex specialization
template <typename T>
void legendre_polynomial_p_mulx_backward(
    c10::complex<T>* grad_coeffs,
    const c10::complex<T>* grad_output,
    int64_t N,
    int64_t output_size
) {
    if (N == 0) {
        return;
    }

    const c10::complex<T> zero(T(0), T(0));

    // Initialize grad_coeffs to zero
    for (int64_t k = 0; k < N; ++k) {
        grad_coeffs[k] = zero;
    }

    // Gradient from output[1] += coeffs[0]
    if (output_size > 1) {
        grad_coeffs[0] = grad_coeffs[0] + grad_output[1];
    }

    // Gradient from k >= 1 terms
    for (int64_t k = 1; k < N; ++k) {
        const c10::complex<T> denom(T(2 * k + 1), T(0));

        if (k - 1 < output_size) {
            grad_coeffs[k] = grad_coeffs[k] + (c10::complex<T>(T(k), T(0)) / denom) * grad_output[k - 1];
        }
        if (k + 1 < output_size) {
            grad_coeffs[k] = grad_coeffs[k] + (c10::complex<T>(T(k + 1), T(0)) / denom) * grad_output[k + 1];
        }
    }
}

} // namespace torchscience::kernel::polynomial
