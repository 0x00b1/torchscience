#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Backward for Laguerre polynomial antiderivative
//
// Forward:
//   output[0] = coeffs[0]
//   output[k] = coeffs[k] - coeffs[k-1]  for 1 <= k < N
//   output[N] = -coeffs[N-1]
//
// Gradients:
//   d(output[0])/d(coeffs[0]) = 1
//   d(output[k])/d(coeffs[k]) = 1     for 1 <= k < N
//   d(output[k])/d(coeffs[k-1]) = -1  for 1 <= k < N
//   d(output[N])/d(coeffs[N-1]) = -1
//
// So:
//   grad_coeffs[k] = grad_output[k] - grad_output[k+1]  for 0 <= k < N-1
//   grad_coeffs[N-1] = grad_output[N-1] - grad_output[N]
//
// Parameters:
//   grad_coeffs: output gradient w.r.t. coeffs, size N
//   grad_output: incoming gradient, size N+1
//   coeffs: original input coefficients (unused)
//   N: size of coeffs
//   output_size: size of grad_output (N+1)
template <typename T>
void laguerre_polynomial_l_antiderivative_backward(
    T* grad_coeffs,
    const T* grad_output,
    const T* coeffs,
    int64_t N,
    int64_t output_size
) {
    (void)coeffs;
    (void)output_size;

    for (int64_t k = 0; k < N; ++k) {
        grad_coeffs[k] = grad_output[k] - grad_output[k + 1];
    }
}

// Complex specialization
template <typename T>
void laguerre_polynomial_l_antiderivative_backward(
    c10::complex<T>* grad_coeffs,
    const c10::complex<T>* grad_output,
    const c10::complex<T>* coeffs,
    int64_t N,
    int64_t output_size
) {
    (void)coeffs;
    (void)output_size;

    for (int64_t k = 0; k < N; ++k) {
        grad_coeffs[k] = grad_output[k] - grad_output[k + 1];
    }
}

} // namespace torchscience::kernel::polynomial
