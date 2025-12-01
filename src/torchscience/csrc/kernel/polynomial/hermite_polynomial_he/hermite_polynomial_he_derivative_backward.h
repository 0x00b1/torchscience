#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Backward for Hermite He polynomial derivative
//
// Forward: output[j] = (j+1) * coeffs[j+1] for j = 0, ..., N-2
// Backward: grad_coeffs[k] = sum_j grad_output[j] * d(output[j])/d(coeffs[k])
//
// d(output[j])/d(coeffs[k]) = (j+1) if k == j+1, else 0
// So grad_coeffs[j+1] = (j+1) * grad_output[j]
//
// Parameters:
//   grad_coeffs: output gradient w.r.t. coeffs, size N
//   grad_output: incoming gradient, size output_size
//   N: size of coeffs
//   output_size: size of grad_output
template <typename T>
void hermite_polynomial_he_derivative_backward(
    T* grad_coeffs,
    const T* grad_output,
    int64_t N,
    int64_t output_size
) {
    // Initialize to zero
    for (int64_t k = 0; k < N; ++k) {
        grad_coeffs[k] = T(0);
    }

    // grad_coeffs[j+1] = (j+1) * grad_output[j]
    for (int64_t j = 0; j < output_size && j + 1 < N; ++j) {
        grad_coeffs[j + 1] = T(j + 1) * grad_output[j];
    }
}

// Complex specialization
template <typename T>
void hermite_polynomial_he_derivative_backward(
    c10::complex<T>* grad_coeffs,
    const c10::complex<T>* grad_output,
    int64_t N,
    int64_t output_size
) {
    const c10::complex<T> zero(T(0), T(0));

    for (int64_t k = 0; k < N; ++k) {
        grad_coeffs[k] = zero;
    }

    for (int64_t j = 0; j < output_size && j + 1 < N; ++j) {
        grad_coeffs[j + 1] = c10::complex<T>(T(j + 1), T(0)) * grad_output[j];
    }
}

} // namespace torchscience::kernel::polynomial
