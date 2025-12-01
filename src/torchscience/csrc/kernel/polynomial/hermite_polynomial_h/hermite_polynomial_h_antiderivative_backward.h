#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Backward for Hermite H polynomial antiderivative
//
// Forward: output[k+1] = coeffs[k] / (2*(k+1))
// So: d(output[j])/d(coeffs[k]) = 1/(2*(k+1)) if j == k+1, 0 otherwise
//
// grad_coeffs[k] = grad_output[k+1] / (2*(k+1))
//
// Parameters:
//   grad_coeffs: output gradient w.r.t. coeffs, size N
//   grad_output: incoming gradient, size output_size
//   coeffs: original input coefficients (unused)
//   N: size of coeffs
//   output_size: size of grad_output
template <typename T>
void hermite_polynomial_h_antiderivative_backward(
    T* grad_coeffs,
    const T* grad_output,
    const T* coeffs,
    int64_t N,
    int64_t output_size
) {
    (void)coeffs;
    (void)output_size;

    for (int64_t k = 0; k < N; ++k) {
        grad_coeffs[k] = grad_output[k + 1] / T(2 * (k + 1));
    }
}

// Complex specialization
template <typename T>
void hermite_polynomial_h_antiderivative_backward(
    c10::complex<T>* grad_coeffs,
    const c10::complex<T>* grad_output,
    const c10::complex<T>* coeffs,
    int64_t N,
    int64_t output_size
) {
    (void)coeffs;
    (void)output_size;

    for (int64_t k = 0; k < N; ++k) {
        grad_coeffs[k] = grad_output[k + 1] / c10::complex<T>(T(2 * (k + 1)), T(0));
    }
}

} // namespace torchscience::kernel::polynomial
