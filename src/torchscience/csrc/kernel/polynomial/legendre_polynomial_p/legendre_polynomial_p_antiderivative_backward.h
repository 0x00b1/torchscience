#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Backward pass for Legendre polynomial antiderivative
//
// Forward:
//   output[1] += coeffs[0]
//   For k >= 1: output[k+1] += coeffs[k] / (2k+1)
//               output[k-1] -= coeffs[k] / (2k+1)
//
// Backward:
//   grad_coeffs[0] += grad_output[1]
//   For k >= 1: grad_coeffs[k] += grad_output[k+1] / (2k+1)
//               grad_coeffs[k] -= grad_output[k-1] / (2k+1)
//
// Parameters:
//   grad_coeffs: output gradient w.r.t. coefficients, size N
//   grad_output: incoming gradient, size output_size (N+1)
//   N: number of original coefficients
//   output_size: size of grad_output
template <typename T>
void legendre_polynomial_p_antiderivative_backward(
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
        const T factor = T(1) / T(2 * k + 1);
        // From output[k+1] += coeffs[k] * factor
        if (k + 1 < output_size) {
            grad_coeffs[k] = grad_coeffs[k] + grad_output[k + 1] * factor;
        }
        // From output[k-1] -= coeffs[k] * factor
        if (k - 1 < output_size) {
            grad_coeffs[k] = grad_coeffs[k] - grad_output[k - 1] * factor;
        }
    }
}

// Complex specialization
template <typename T>
void legendre_polynomial_p_antiderivative_backward(
    c10::complex<T>* grad_coeffs,
    const c10::complex<T>* grad_output,
    int64_t N,
    int64_t output_size
) {
    if (N == 0) {
        return;
    }

    const c10::complex<T> zero(T(0), T(0));
    const c10::complex<T> one(T(1), T(0));

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
        const c10::complex<T> factor = one / c10::complex<T>(T(2 * k + 1), T(0));
        if (k + 1 < output_size) {
            grad_coeffs[k] = grad_coeffs[k] + grad_output[k + 1] * factor;
        }
        if (k - 1 < output_size) {
            grad_coeffs[k] = grad_coeffs[k] - grad_output[k - 1] * factor;
        }
    }
}

} // namespace torchscience::kernel::polynomial
