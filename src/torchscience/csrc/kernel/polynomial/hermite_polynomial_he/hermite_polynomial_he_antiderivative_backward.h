#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Backward for Hermite He polynomial antiderivative
//
// Forward: output[k+1] = coeffs[k] / (k+1) for k = 0, ..., N-1
//          output[0] = 0 (or adjusted for constant)
//
// Backward: grad_coeffs[k] = grad_output[k+1] / (k+1)
//
// Parameters:
//   grad_coeffs: output gradient w.r.t. coeffs, size N
//   grad_output: incoming gradient, size output_size
//   coeffs: original input coefficients (unused)
//   N: size of coeffs
//   output_size: size of grad_output
template <typename T>
void hermite_polynomial_he_antiderivative_backward(
    T* grad_coeffs,
    const T* grad_output,
    const T* coeffs,
    int64_t N,
    int64_t output_size
) {
    (void)coeffs;

    for (int64_t k = 0; k < N; ++k) {
        if (k + 1 < output_size) {
            grad_coeffs[k] = grad_output[k + 1] / T(k + 1);
        } else {
            grad_coeffs[k] = T(0);
        }
    }
}

// Complex specialization
template <typename T>
void hermite_polynomial_he_antiderivative_backward(
    c10::complex<T>* grad_coeffs,
    const c10::complex<T>* grad_output,
    const c10::complex<T>* coeffs,
    int64_t N,
    int64_t output_size
) {
    (void)coeffs;
    const c10::complex<T> zero(T(0), T(0));

    for (int64_t k = 0; k < N; ++k) {
        if (k + 1 < output_size) {
            grad_coeffs[k] = grad_output[k + 1] / c10::complex<T>(T(k + 1), T(0));
        } else {
            grad_coeffs[k] = zero;
        }
    }
}

} // namespace torchscience::kernel::polynomial
