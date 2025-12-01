#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Backward for Hermite H polynomial derivative
//
// Forward: output[j] = 2*(j+1) * coeffs[j+1]
// So: d(output[j])/d(coeffs[k]) = 2*(j+1) if k == j+1, 0 otherwise
//
// grad_coeffs[k] = sum_j grad_output[j] * d(output[j])/d(coeffs[k])
//                = grad_output[k-1] * 2*k  for k >= 1
// grad_coeffs[0] = 0 (no dependence)
//
// Parameters:
//   grad_coeffs: output gradient w.r.t. coeffs, size N
//   grad_output: incoming gradient, size output_size
//   N: size of coeffs
//   output_size: size of grad_output
template <typename T>
void hermite_polynomial_h_derivative_backward(
    T* grad_coeffs,
    const T* grad_output,
    int64_t N,
    int64_t output_size
) {
    // Initialize grad_coeffs to zero
    for (int64_t k = 0; k < N; ++k) {
        grad_coeffs[k] = T(0);
    }

    if (N <= 1) {
        return;
    }

    // grad_coeffs[k] = 2*k * grad_output[k-1] for k >= 1
    for (int64_t k = 1; k < N; ++k) {
        if (k - 1 < output_size) {
            grad_coeffs[k] = T(2 * k) * grad_output[k - 1];
        }
    }
}

// Complex specialization
template <typename T>
void hermite_polynomial_h_derivative_backward(
    c10::complex<T>* grad_coeffs,
    const c10::complex<T>* grad_output,
    int64_t N,
    int64_t output_size
) {
    const c10::complex<T> zero(T(0), T(0));

    for (int64_t k = 0; k < N; ++k) {
        grad_coeffs[k] = zero;
    }

    if (N <= 1) {
        return;
    }

    for (int64_t k = 1; k < N; ++k) {
        if (k - 1 < output_size) {
            grad_coeffs[k] = c10::complex<T>(T(2 * k), T(0)) * grad_output[k - 1];
        }
    }
}

} // namespace torchscience::kernel::polynomial
